# -*- coding: utf-8 -*-
# Author: Deyuan Qu <deyuanqu@my.unt.edu>
# Modified by: Yuheng Wu <yuhengwu@kaist.ac.kr>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import torch
import torch.nn as nn

from opencood.models.common_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor
from opencood.models.sicpfuse_modules.multiagent_sicp_fuse import MultiSpatialFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.common_modules.airv2x_base_model import Airv2xBase
from opencood.models.common_modules.airv2x_encoder import LiftSplatShootEncoder
from opencood.models.task_heads.segmentation_head import BevSegHead


class Airv2xSiCP(Airv2xBase):
    """
    SiCP implementation with point pillar backbone.
    """

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

        self.voxel_size = args["voxel_size"]

        self.backbone = BaseBEVBackbone(args["base_bev_backbone"], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if "shrink_header" in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args["shrink_header"])
        self.compression = False

        if args["compression"] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args["compression"])

        self.fusion_net = MultiSpatialFusion(
            in_channels=args["fusion"]["in_channels"],
            out_channels=args["fusion"]["out_channels"],
        )

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

        # calculate pairwise affine transformation matrix
        _, _, H0, W0 = batch_output_dict["spatial_features"].shape
        t_matrix = normalize_pairwise_tfm(
            data_dict["pairwise_t_matrix_collab"], H0, W0, self.voxel_size[0]
        )
        batch_dict = self.backbone(batch_output_dict)

        spatial_features_2d = batch_dict["spatial_features_2d"]
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        output_dict = {}
        # in the original v2x-r, they only works on 2 collab agents
        # here we range from [1, max_cav_num]
        if self.args["task"] == "det":
            if self.training:
                ego_indices = torch.cat([torch.tensor([0], device=spatial_features_2d.device), batch_record_len.cumsum(0)[:-1]])
                ego = spatial_features_2d[ego_indices]  # shape [B, C, H, W]
                fused_feature = self.fusion_net(
                    spatial_features_2d, batch_record_len, t_matrix
                )

                psm1 = self.cls_head(ego)
                rm1 = self.reg_head(ego)
                obj1 = self.obj_head(ego)
                psm2 = self.cls_head(fused_feature)
                rm2 = self.reg_head(fused_feature)
                obj2 = self.obj_head(fused_feature)
                output_dict.update({
                    "psm1": psm1,
                    "rm1": rm1,
                    "psm2": psm2,
                    "rm2": rm2,
                    "obj1": obj1,
                    "obj2": obj2,
                })
            else:
                if batch_record_len == 1:
                    ego = spatial_features_2d[0, :, :, :].unsqueeze(0)
                    psm = self.cls_head(ego)
                    rm = self.reg_head(ego)
                    obj = self.obj_head(ego)
                else:
                    fused_feature = self.fusion_net(
                        spatial_features_2d, batch_record_len, t_matrix
                    )
                    psm = self.cls_head(fused_feature)
                    rm = self.reg_head(fused_feature)
                    obj = self.obj_head(fused_feature)
                output_dict.update({"psm": psm, "rm": rm, "obj": obj})

        elif self.args["task"] == "seg":
            if self.training:
                import pdb; pdb.set_trace()
                ego_indices = torch.cat([torch.tensor([0], device=spatial_features_2d.device), batch_record_len.cumsum(0)[:-1]])
                ego = spatial_features_2d[ego_indices]  # shape [B, C, H, W]
                fused_feature = self.fusion_net(
                    spatial_features_2d, batch_record_len, t_matrix
                )
                assert len(ego.shape) == 4, f"ego shape 1: {ego.shape}"
                assert len(fused_feature.shape) == 4, (
                    f"fused_feature shape 2: {fused_feature.shape}"
                )
                # seg_logits1 = self.seg_head(ego)
                # seg_logits2 = self.seg_head(fused_feature)
                seg_output_dict1 = self.seg_head(ego)
                seg_output_dict2 = self.seg_head(fused_feature)
                output_dict = {"dynamic_seg1": seg_output_dict1["dynamic_seg"], "static_seg1": seg_output_dict1["static_seg"], "dynamic_seg2": seg_output_dict2["dynamic_seg"], "static_seg2": seg_output_dict2["static_seg"]}
            else:
                if batch_record_len == 1:
                    ego = spatial_features_2d[0, :, :, :].unsqueeze(0)
                    assert len(ego.shape) == 4, f"ego shape 3: {ego.shape}"
                    seg_output_dict = self.seg_head(ego)
                else:
                    fused_feature = self.fusion_net(
                        spatial_features_2d, batch_record_len, t_matrix
                    )
                    assert len(fused_feature.shape) == 4, (
                        f"fused_feature shape 4: {fused_feature.shape}"
                    )
                    # seg_logits = self.seg_head(fused_feature)
                    seg_output_dict = self.seg_head(fused_feature)
                # output_dict.update({"seg_logits": seg_logits})
                output_dict.update(seg_output_dict)

        return output_dict
