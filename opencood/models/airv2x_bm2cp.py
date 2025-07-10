"""
# Author: Binyu Zhao <byzhao@stu.hit.edu.cn>
"""

from collections import defaultdict, OrderedDict

import torch
from einops import rearrange
from numpy import record
from torch import nn
from torchvision.models.resnet import resnet18

from opencood.models.bm2cp_modules.attentioncomm import (
    AttenComm,
    ScaledDotProductAttention,
)
from opencood.models.bm2cp_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.bm2cp_modules.sensor_blocks import (
    LidarCamBM2CPEncoder,
    LiftSplatShootEncoder,
)
from opencood.models.common_modules.base_bev_backbone import (
    BaseBEVBackbone as PCBaseBEVBackbone,
)
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor
from opencood.models.common_modules.airv2x_base_model_bk import Airv2xBase
from opencood.models.task_heads.segmentation_head import BevSegHead 
from opencood.utils.camera_utils import (
    QuickCumsum,
    cumsum_trick,
    depth_discretization,
    gen_dx_bx,
)


class Airv2xBM2CP(Airv2xBase):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self.collaborators = args["collaborators"]
        self.active_sensors = args["active_sensors"]

        # create veh model
        if "vehicle" in self.collaborators:
            self.veh_model = LidarCamBM2CPEncoder(args, agent_type="vehicle")

        # create rsu model
        if "rsu" in self.collaborators:
            self.rsu_model = LidarCamBM2CPEncoder(args, agent_type="rsu")

        # create drone model -> LSS
        if "drone" in self.collaborators:
            self.drone_model = LiftSplatShootEncoder(args, agent_type="drone")

        self.supervise_single = args["supervise_single"]

        # multi-modal fusion
        modality_args = args["modality_fusion"]
        self.modal_multi_scale = modality_args["bev_backbone"]["multi_scale"]
        self.num_levels = len(modality_args["bev_backbone"]["layer_nums"])
        img_args = args["vehicle"]["img_params"]
        pc_args = args["vehicle"]["pc_params"]
        assert img_args["bev_dim"] == pc_args["point_pillar_scatter"]["num_features"]
        # self.fusion = MultiModalFusion(img_args["bev_dim"])
        # print(
        #     "Number of parameter modal fusion: %d"
        #     % (sum([param.nelement() for param in self.fusion.parameters()]))
        # )
        self.backbone = ResNetBEVBackbone(
            modality_args["bev_backbone"],
            input_channels=pc_args["point_pillar_scatter"]["num_features"],
        )
        # print(
        #     "Number of parameter bevbackbone: %d"
        #     % (sum([param.nelement() for param in self.backbone.parameters()]))
        # )

        self.shrink_flag = False
        if "shrink_header" in modality_args:
            self.shrink_flag = modality_args["shrink_header"]["use"]
            self.shrink_conv = DownsampleConv(modality_args["shrink_header"])
            # print(
            #     "Number of parameter shrink_conv: %d"
            #     % (sum([param.nelement() for param in self.shrink_conv.parameters()]))
            # )

        self.compression = False
        if "compression" in modality_args and modality_args["compression"] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, modality_args["compression"])

        # collaborative fusion network
        self.multi_scale = args["collaborative_fusion"]["multi_scale"]
        self.fusion_net = AttenComm(args["collaborative_fusion"])
        # print(
        #     "Number of fusion_net parameter: %d"
        #     % (sum([param.nelement() for param in self.fusion_net.parameters()]))
        # )

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
            for p in self.veh_model.parameters():
                p.requires_grad = False
        if "rsu" in self.collaborators:
            for p in self.rsu_model.parameters():
                p.requires_grad = False
        if "drone" in self.collaborators:
            for p in self.drone_model.parameters():
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
        # extract features from each agent
        # spatial_features(bev feat): [N1+N2+..., C, H, W], N is num of egents of each sample
        # and N is in sequential order (veh, rsu, drone)
        batch_output_dict, batch_record_len = self.extract_features(data_dict)

        # encode and compress before sharing if necessary
        batch_output_dict = self.backbone(batch_output_dict)
        batch_spatial_features_2d = batch_output_dict["spatial_features_2d"]
        if self.shrink_flag:
            batch_spatial_features_2d = self.shrink_conv(batch_spatial_features_2d)
        if self.compression:
            batch_spatial_features_2d = self.naive_compressor(batch_spatial_features_2d)

        ret_dict = {}

        if self.args["task"] == "det":
            # cross-agent fusion
            if self.multi_scale:
                fused_feature, communication_rates, result_dict = self.fusion_net(
                    batch_output_dict["spatial_features"],
                    self.cls_head(batch_spatial_features_2d),
                    batch_output_dict["thres_map"],
                    batch_record_len,
                    data_dict["pairwise_t_matrix_collab"],
                    self.backbone,
                    [self.shrink_conv, self.cls_head, self.reg_head],
                )
                # downsample feature to reduce memory
                if self.shrink_flag:
                    fused_feature = self.shrink_conv(fused_feature)
            else:
                fused_feature, communication_rates, result_dict = self.fusion_net(
                    batch_spatial_features_2d,
                    self.cls_head(batch_spatial_features_2d),
                    batch_output_dict["thres_map"],
                    batch_record_len,
                    data_dict["pairwise_t_matrix_collab"],
                )

            psm = self.cls_head(fused_feature)
            rm = self.reg_head(fused_feature)
            if self.args["obj_head"]:
                obj = self.obj_head(fused_feature)
                ret_dict.update({"obj": obj})
            ret_dict.update(
                {
                    "psm": psm,
                    "rm": rm,
                    "mask": batch_output_dict["mask"],
                    "each_mask": batch_output_dict["each_mask"],
                    "comm_rate": communication_rates,
                }
            )

            if not self.supervise_single:
                return ret_dict
            return ret_dict

        elif self.args["task"] == "seg":
            _, cls_feat = self.seg_head(batch_spatial_features_2d, True)
            if self.multi_scale:
                fused_feature, communication_rates, result_dict = self.fusion_net(
                    batch_output_dict["spatial_features"],
                    cls_feat,
                    batch_output_dict["thres_map"],
                    batch_record_len,
                    data_dict["pairwise_t_matrix_collab"],
                    self.backbone,
                    # [self.shrink_conv, self.cls_head, self.reg_head],
                )
                # downsample feature to reduce memory
                if self.shrink_flag:
                    fused_feature = self.shrink_conv(fused_feature)
            else:
                fused_feature, communication_rates, result_dict = self.fusion_net(
                    batch_spatial_features_2d,
                    cls_feat,
                    batch_output_dict["thres_map"],
                    batch_record_len,
                    data_dict["pairwise_t_matrix_collab"],
                )

            seg_logits = self.seg_head(fused_feature)
            ret_dict.update(
                {
                    "seg_logits": seg_logits,
                    "mask": batch_output_dict["mask"],
                    "each_mask": batch_output_dict["each_mask"],
                    "comm_rate": communication_rates,
                }
            )

            if not self.supervise_single:
                return ret_dict
            return ret_dict

    def merge_output_dict_list(self, output_dict_list):
        accumulated_dict = defaultdict(list)

        # First pass: accumulate features
        for output_dict in output_dict_list:
            for feat_name, feat in output_dict.items():
                accumulated_dict[feat_name].append(feat)

        # Second pass: create the final merged output
        merged_output_dict = OrderedDict()
        for feat_name, feat_list in accumulated_dict.items():
            # For bm2cp, it's a bit different
            if feat_name == "each_mask":
                # For each_mask, we need to concatenate along the first dimension
                merged_output_dict[feat_name] = torch.cat(feat_list, dim=1)
            else:
                merged_output_dict[feat_name] = torch.cat(feat_list, dim=0)

        return merged_output_dict
    