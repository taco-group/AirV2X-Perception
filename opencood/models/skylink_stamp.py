""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import torch
import torch.nn as nn
import numpy as np
from icecream import ic
import torchvision
from collections import OrderedDict, Counter
from opencood.models.common_modules.skylink_base_model import SkylinkBase
from opencood.models.common_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.task_heads.segmentation_head import BevSegHead
from opencood.models.fuse_modules.adapter import Adapter, Reverter
import importlib
# from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn

class SkylinkSTAMP(SkylinkBase):
    def __init__(self, args):
        super(SkylinkSTAMP, self).__init__(args)
        
        self.args = args

        # here we use image encoder LSS instead of lidar
        self.collaborators = args["collaborators"]
        self.active_sensors = args["active_sensors"]
        self.max_cav_num = args["max_cav_num"]
        
        self.init_encoders(args)
        modality_args = args["modality_fusion"]
        self.backbone = ResNetBEVBackbone(modality_args["base_bev_backbone"], 64)
        
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if "shrink_header" in modality_args and modality_args["shrink_header"]["use"]:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(modality_args["shrink_header"])
        self.compression = False

        if modality_args["compression"] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args["compression"])
            
        self.build_adapter_and_reverter(args)
            
        self.pyramid_backbone = PyramidFusion(args["fusion_backbone"])

        """
        Shared Heads, Would load from pretrain base.
        """
        if args["task"] == "det":
            self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'] * args["num_class"],
                                    kernel_size=1)
            self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                    kernel_size=1)
            if args["obj_head"]:
                self.obj_head = nn.Conv2d(
                    args['in_head'], args["anchor_number"], kernel_size=1
                )
        elif args["task"] == "seg":
            self.seg_head = BevSegHead(
                args["seg_hw"], args["seg_hw"], args['in_head'], args["dynamic_class"], args["static_class"],
                seg_res=args["seg_res"], cav_range=args["cav_range"]
            )
        # self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
        #                           kernel_size=1) # BIN_NUM = 2
        
        if args["backbone_fix"]:
            self.backbone_fix()
            
    def build_adapter_and_reverter(self, args):

        if "vehicle" in self.collaborators:
            self.adapter_vehicle = Adapter(args['vehicle']["adapter"])
            # self.reverter_vehicle = Reverter(args['vehicle']["reverter"])
        if "rsu" in self.collaborators:
            self.adapter_rsu = Adapter(args['rsu']["adapter"])
            # self.reverter_rsu = Reverter(args['rsu']["reverter"])
        if "drone" in self.collaborators:
            self.adapter_drone = Adapter(args['drone']["adapter"])
            # self.reverter_drone = Reverter(args['drone']["reverter"])
            
    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        if "vehicle" in self.collaborators:
            for p in self.veh_models.parameters():
                p.requires_grad = False
        if "rsu" in self.collaborators:
            for p in self.rsu_models.parameters():
                p.requires_grad = False
        if "drone" in self.collaborators:
            for p in self.drone_models.parameters():
                p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False
                
        for p in self.pyramid_backbone.parameters():
            p.requires_grad = False

        if self.args["task"] == "det":
            for p in self.cls_head.parameters():
                p.requires_grad = False
            for p in self.reg_head.parameters():
                p.requires_grad = False
            if self.args["obj_head"]:
                for p in self.obj_head.parameters():
                    p.requires_grad = False

        elif self.args["task"] == "seg":
            for p in self.seg_head.parameters():
                p.requires_grad = False
                
                
    def extract_features(self, data_dict):
        """
        Extract and aggregate features from each collaborating agent.

        This method processes the input `data_dict` by forwarding each agent's data
        through its respective sub-model (vehicle, RSU, drone). It gathers the
        intermediate outputs into a unified batch structure.

        Only the agents specified in `self.collaborators` and with non-empty `batch_idxs`
        will be processed. Each agent must have its corresponding sub-model initialized
        before feature extraction.

        Parameters
        ----------
        data_dict : dict
            A dictionary containing batched input data for each agent type. Each agent's
            data must include at least 'batch_idxs' and 'record_len' fields.

        Returns
        -------
        batch_output_dict : dict
            The merged feature outputs from all processed agents, structured batch-wise.

        batch_record_len : torch.Tensor
            A tensor indicating the number of records aggregated per batch sample.
        """
        batch_dicts = OrderedDict()
        if "vehicle" in self.collaborators and len(data_dict["vehicle"]["batch_idxs"]) > 0:
            assert self.veh_models is not None, "Vehicle model is not initialized."
            output_dict_veh = []
            for v_model in self.veh_models:
                output_dict_veh.append(v_model(data_dict))
            output_dict_veh = self.fuse_bev(output_dict_veh)
            
            batch_output_dict = self.backbone(output_dict_veh)
            
            feat = batch_output_dict["spatial_features_2d"]
            adapted_feat = self.adapter_vehicle(feat)
            batch_output_dict["spatial_features_2d"] = adapted_feat
            
            batch_dicts["vehicle"] = output_dict_veh
            
        if "rsu" in self.collaborators and len(data_dict["rsu"]["batch_idxs"]) > 0:
            assert self.rsu_models is not None, "RSU model is not initialized."
            output_dict_rsu = []
            for r_model in self.rsu_models:
                output_dict_rsu.append(r_model(data_dict))
            output_dict_rsu = self.fuse_bev(output_dict_rsu)
            
            batch_output_dict = self.backbone(output_dict_rsu)
            
            feat = batch_output_dict["spatial_features_2d"]
            adapted_feat = self.adapter_rsu(feat)
            batch_output_dict["spatial_features_2d"] = adapted_feat
            
            batch_dicts["rsu"] = output_dict_rsu
            
        if "drone" in self.collaborators and len(data_dict["drone"]["batch_idxs"]) > 0:
            assert self.drone_models is not None, "Drone model is not initialized."
            output_dict_drone = []
            for d_model in self.drone_models:
                output_dict_drone.append(d_model(data_dict))
            output_dict_drone = self.fuse_bev(output_dict_drone)
            
            batch_output_dict = self.backbone(output_dict_drone)
            
            feat = batch_output_dict["spatial_features_2d"]
            adapted_feat = self.adapter_drone(feat)
            batch_output_dict["spatial_features_2d"] = adapted_feat
            
            batch_dicts["drone"] = output_dict_drone

        # Normally, all agent types has the same batch size, but if one type of agent is not in the collaborator list,
        # the batch size of that agent type will be 0. So we need to find the maximum batch size among all agent types.
        B = max(len(data_dict["vehicle"]["batch_idxs"]), 
                    len(data_dict["rsu"]["batch_idxs"]),
                    len(data_dict["drone"]["batch_idxs"]))

        batch_output_dict, batch_record_len = self.repack_batch(batch_dicts, data_dict, B)

        assert (
            batch_output_dict["spatial_features"].shape[0]
            == batch_record_len.sum().item()
        ), f"{batch_output_dict['spatial_features'].shape}, {batch_record_len}"
        return batch_output_dict, batch_record_len


    def forward(self, data_dict):
        output_dict = {'pyramid': 'single'}
        
        batch_output_dict, batch_record_len = self.extract_features(data_dict)
        comm_rates = batch_output_dict["spatial_features"].count_nonzero().item()
        batch_spatial_features_2d = batch_output_dict["spatial_features_2d"]
        # camera features are still in its own coordinate system
        pairwise_t_matrix = data_dict["img_pairwise_t_matrix_collab"]
        
        fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
                                        batch_spatial_features_2d,
                                        batch_record_len, 
                                        pairwise_t_matrix[:, :, :, [0, 1], :][
                                            :, :, :, :, [0, 1, 3]], 
                                    )
        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        if self.args["task"] == "det":
            psm = self.cls_head(fused_feature)
            rm = self.reg_head(fused_feature)

            if self.args["obj_head"]:
                obj = self.obj_head(fused_feature)
                output_dict.update({"obj": obj})
            output_dict.update(
                {
                    "psm": psm,
                    "rm": rm,
                    "comm_rate": comm_rates,
                }
            )

        elif self.args["task"] == "seg":
            seg_logits = self.seg_head(fused_feature)
            output_dict.update(
                {
                    "comm_rate": comm_rates,
                }
            )
            output_dict.update(seg_logits)
       
        return output_dict
        
        
        
        
    