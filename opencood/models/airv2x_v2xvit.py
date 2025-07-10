# Modified by: Yuheng Wu <yuhengwu@kaist.ac.kr>


import torch
import torch.nn as nn

from opencood.models.common_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.fuse_utils import regroup
from opencood.models.common_modules.naive_compress import NaiveCompressor
from opencood.models.common_modules.airv2x_base_model import Airv2xBase
from opencood.models.common_modules.airv2x_encoder import LiftSplatShootEncoder
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.common_modules.airv2x_pillar_vfe import PillarVFE
from opencood.models.v2xvit_modules.v2xvit_basic import V2XTransformer
from opencood.models.task_heads.segmentation_head import BevSegHead 


class Airv2xV2XVit(Airv2xBase):
    def __init__(self, args):
        super().__init__(args)

        self.args = args

        # here we use image encoder LSS instead of lidar
        self.collaborators = args["collaborators"]
        self.active_sensors = args["active_sensors"]
        max_cav = args["max_cav"]
        self.max_cav_num = sum(max_cav.values())

        # if "vehicle" in self.collaborators:
        #     self.veh_model = LiftSplatShootEncoder(args, agent_type="vehicle")

        # if "rsu" in self.collaborators:
        #     self.rsu_model = LiftSplatShootEncoder(args, agent_type="rsu")

        # if "drone" in self.collaborators:
        #     self.drone_model = LiftSplatShootEncoder(args, agent_type="drone")
        
        self.init_encoders(args)

        modality_args = args["modality_fusion"]
        self.backbone = BaseBEVBackbone(modality_args["base_bev_backbone"], 64)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if "shrink_header" in modality_args and modality_args["shrink_header"]["use"]:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(modality_args["shrink_header"])
        self.compression = False

        if modality_args["compression"] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args["compression"])

        self.fusion_net = V2XTransformer(args["transformer"])

        self.outC = args["outC"]
        if args["task"] == "det":
            self.cls_head = nn.Conv2d(
                self.outC, args["anchor_number"] * args["num_class"], kernel_size=1
            )
            self.reg_head = nn.Conv2d(
                self.outC, 7 * args["anchor_number"], kernel_size=1
            )
            if args["obj_head"]:
                self.obj_head = nn.Conv2d(
                    self.outC, args["anchor_number"], kernel_size=1
                )
        elif args["task"] == "seg":
            self.seg_head = BevSegHead(
                args["seg_branch"], args["seg_hw"], args["seg_hw"], self.outC, args["dynamic_class"], args["static_class"],
                seg_res=args["seg_res"], cav_range=args["cav_range"]
            )

        if args["backbone_fix"]:
            self.backbone_fix()
            

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
                

    def forward(self, data_dict):
        batch_output_dict, batch_record_len = self.extract_features(data_dict)

        comm_rates = batch_output_dict["spatial_features"].count_nonzero().item()
        spatial_correction_matrix = data_dict["spatial_correction_matrix"]

        prior_encoding = data_dict["prior_encoding"].unsqueeze(-1).unsqueeze(-1)

        batch_output_dict = self.backbone(batch_output_dict)
        spatial_features_2d = batch_output_dict["spatial_features_2d"]

        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        feat = spatial_features_2d[0].mean(0).detach().cpu().numpy()
        import cv2; import numpy as np
        cv2.imwrite("debug/debug_image_bevfeat_drone_lidar.png", ((feat - feat.min()) / (feat.max() - feat.min()) * 255).astype(np.uint8),)
        

        regroup_feature, mask = regroup(
            spatial_features_2d, batch_record_len, self.max_cav_num
        )

        prior_encoding = prior_encoding.repeat(
            1, 1, 1, regroup_feature.shape[3], regroup_feature.shape[4]
        )
        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)

        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2).contiguous()
        
        
        # feat = spatial_features_2d[0].mean(0).detach().cpu().numpy()
        # import cv2; import numpy as np
        # cv2.imwrite("/home/xiangbog/Folder/Research/airv2x/debug/debug_image_bevfeat.png", ((feat - feat.min()) / (feat.max() - feat.min()) * 255).astype(np.uint8),)
        
        # dynamic = data_dict['label_dict']['dynamic_seg_label'][0]
        # dynamic = dynamic.detach().cpu().numpy()
        # cv2.imwrite("/home/xiangbog/Folder/Research/airv2x/debug/debug_image_dynamic.png", ((dynamic - dynamic.min()) / (dynamic.max() - dynamic.min()) * 255).astype(np.uint8),)
        # import pdb; pdb.set_trace()

        # transformer fusion
        # TODO(YH): fix window size here
        # current regroup_feature is [N, C, 100, 352, D] which is not divided by [4, 8, 16] window size
        #  (1) change window size or (2) change voxel config or (3) change cav_lidar_range to produce regular shape
        fused_feature = self.fusion_net(
            regroup_feature, mask, spatial_correction_matrix
        )
        # b h w c -> b c h w
        fused_feature = fused_feature.permute(0, 3, 1, 2).contiguous()


        # feat_fused = fused_feature[0].mean(0).detach().cpu().numpy()
        # cv2.imwrite("/home/xiangbog/Folder/Research/SkyLink/airv2x/debug/debug_image_fusedfeat.png", ((feat_fused - feat_fused.min()) / (feat_fused.max() - feat_fused.min()) * 255).astype(np.uint8),)

        

        ret_dict = {}
        if self.args["task"] == "det":
            psm = self.cls_head(fused_feature)
            rm = self.reg_head(fused_feature)

            if self.args["obj_head"]:
                obj = self.obj_head(fused_feature)
                ret_dict.update({"obj": obj})
            ret_dict.update(
                {
                    "psm": psm,
                    "rm": rm,
                    "comm_rate": comm_rates,
                }
            )

        elif self.args["task"] == "seg":
            # seg_logits = self.seg_head(fused_feature)
            seg_output_dict = self.seg_head(fused_feature)
            ret_dict.update(
                {
                    **seg_output_dict,
                    "comm_rate": comm_rates,
                }
            )

        return ret_dict
