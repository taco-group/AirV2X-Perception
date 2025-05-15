# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# Modified by: Yuheng Wu <yuhengwu@kaist.ac.kr>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn

from opencood.models.common_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor
from opencood.models.common_modules.skylink_encoder import LiftSplatShootEncoder
from opencood.models.common_modules.skylink_base_model import SkylinkBase
from opencood.models.when2com_modules.when2com import When2comFusion
from opencood.models.task_heads.segmentation_head import BevSegHead 


class SkylinkWhen2com(SkylinkBase):
    def __init__(self, args):
        super().__init__(args)

        self.args = args

        # here we use image encoder LSS instead of lidar
        self.collaborators = args["collaborators"]
        self.active_sensors = args["active_sensors"]

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

        self.fusion_net = When2comFusion(args["when2com_fusion"])

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
        batch_output_dict, batch_record_len = self.extract_features(data_dict)
        batch_output_dict = self.backbone(batch_output_dict)

        batch_spatial_features_2d = batch_output_dict["spatial_features_2d"]
        if self.shrink_flag:
            batch_spatial_features_2d = self.shrink_conv(batch_spatial_features_2d)
        if self.compression:
            batch_spatial_features_2d = self.naive_compressor(batch_spatial_features_2d)

        # camera features are still in its own coordinate system
        pairwise_t_matrix = data_dict["img_pairwise_t_matrix_collab"]

        fused_feature, comm_rates = self.fusion_net(
            batch_spatial_features_2d, batch_record_len, pairwise_t_matrix
        )

        output_dict = {}
        if self.args["task"] == "det":
            psm = self.cls_head(fused_feature)
            rm = self.reg_head(fused_feature)

            output_dict.update({"psm": psm, "rm": rm})

            if self.args["obj_head"]:
                obj = self.obj_head(fused_feature)
                output_dict.update({"obj": obj})
            output_dict.update({"mask": 0, "comm_rate": comm_rates})

        elif self.args["task"] == "seg":
            # seg_logits = self.seg_head(fused_feature)
            # output_dict.update({"seg_logits": seg_logits})
            seg_output_dict = self.seg_head(fused_feature)
            output_dict.update(seg_output_dict)
            output_dict.update({"mask": 0, "comm_rate": comm_rates})

        return output_dict
