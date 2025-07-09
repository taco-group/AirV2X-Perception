import torch
import torch.nn as nn
from numpy import record

from opencood.models.common_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.common_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor
from opencood.models.common_modules.naive_decoder import NaiveDecoder
from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.common_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.how2comm_modules.how2comm_deformable import How2comm


def transform_feature(feature_list, delay):
    return feature_list[delay]


class PointPillarHow2comm(nn.Module):
    def __init__(self, args):
        super(PointPillarHow2comm, self).__init__()

        self.pillar_vfe = PillarVFE(
            args["pillar_vfe"],
            num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=args["lidar_range"],
        )
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])
        if "resnet" in args["base_bev_backbone"]:
            self.backbone = ResNetBEVBackbone(args["base_bev_backbone"], 64)
        else:
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

        self.dcn = False

        self.fusion_net = How2comm(args["fusion_args"], args)
        self.frame = args["fusion_args"]["frame"]
        self.delay = 1
        self.discrete_ratio = args["fusion_args"]["voxel_size"][0]
        self.downsample_rate = args["fusion_args"]["downsample_rate"]
        self.multi_scale = args["fusion_args"]["multi_scale"]

        self.detection = False
        if "detection" in args["task"]:
            self.detection = True
            # detection task
            self.cls_head = nn.Conv2d(
                128 * 2, args["task"]["detection"]["anchor_num"], kernel_size=1
            )
            self.reg_head = nn.Conv2d(
                128 * 2, 7 * args["task"]["detection"]["anchor_num"], kernel_size=1
            )
            if "dir_args" in args["task"].keys():
                self.use_dir = True
                self.dir_head = nn.Conv2d(
                    128 * 2,
                    args["task"]["detection"]["dir_args"]["num_bins"]
                    * args["task"]["detection"]["dir_args"]["anchor_num"],
                    kernel_size=1,
                )  # BIN_NUM = 2
            else:
                self.use_dir = False

        self.segmentation, self.lane, self.trajectory = False, False, False
        if "segmentation" in args["task"]:
            self.segmentation = True
            self.seg_head = nn.Conv2d(
                args["task"]["segmentation"]["head_dim"],
                args["task"]["segmentation"]["output_class"],
                kernel_size=3,
                padding=1,
            )
            self.seg_rot = (
                args["task"]["segmentation"]["rotation"]
                if "rotation" in args["task"]["segmentation"]
                else False
            )
            self.seg_clip = (
                args["task"]["segmentation"]["clip_size"]
                if "clip_size" in args["task"]["segmentation"]
                else False
            )

        if "lane" in args["task"]:
            self.lane = True
            self.lane_head = nn.Conv2d(
                args["task"]["lane"]["head_dim"],
                args["task"]["lane"]["output_class"],
                kernel_size=3,
                padding=1,
            )
            self.lane_rot = (
                args["task"]["lane"]["rotation"]
                if "rotation" in args["task"]["lane"]
                else False
            )
            self.lane_clip = (
                args["task"]["lane"]["clip_size"]
                if "clip_size" in args["task"]["lane"]
                else False
            )

        self.decoder = None
        if self.segmentation or self.lane or self.trajectory:
            assert "decoder" in args["task"]
            self.decoder = NaiveDecoder(args["task"]["decoder"])

        if args["backbone_fix"]:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        if self.detection:
            for p in self.cls_head.parameters():
                p.requires_grad = False
            for p in self.reg_head.parameters():
                p.requires_grad = False

        if self.segmentation:
            for p in self.seg_head.parameters():
                p.requires_grad = False
        if self.lane:
            for p in self.lane_head.parameters():
                p.requires_grad = False
        if self.trajectory:
            for p in self.traj_head.parameters():
                p.requires_grad = False

        if self.decoder is not None:
            for p in self.decoder.parameters():
                p.requires_grad = False

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, data_dict_list):
        batch_dict_list = []
        feature_list = []
        feature_2d_list = []
        matrix_list = []
        regroup_feature_list = []
        regroup_feature_list_large = []

        for origin_data in data_dict_list:
            data_dict = origin_data["ego"]
            voxel_features = data_dict["processed_lidar"]["voxel_features"]
            voxel_coords = data_dict["processed_lidar"]["voxel_coords"]
            voxel_num_points = data_dict["processed_lidar"]["voxel_num_points"]
            record_len = data_dict["record_len"]

            pairwise_t_matrix = data_dict["pairwise_t_matrix"]
            batch_dict = {
                "voxel_features": voxel_features,
                "voxel_coords": voxel_coords,
                "voxel_num_points": voxel_num_points,
                "record_len": record_len,
            }
            # n, 4 -> n, c encoding voxel feature using point-pillar method
            batch_dict = self.pillar_vfe(batch_dict)
            # n, c -> N, C, H, W
            batch_dict = self.scatter(batch_dict)
            batch_dict = self.backbone(batch_dict)
            # N, C, H', W'
            spatial_features_2d = batch_dict["spatial_features_2d"]

            # downsample feature to reduce memory
            if self.shrink_flag:
                spatial_features_2d = self.shrink_conv(spatial_features_2d)
            # compressor
            if self.compression:
                spatial_features_2d = self.naive_compressor(spatial_features_2d)
            # dcn
            if self.dcn:
                spatial_features_2d = self.dcn_net(spatial_features_2d)

            batch_dict_list.append(batch_dict)
            spatial_features = batch_dict["spatial_features"]
            feature_list.append(spatial_features)
            feature_2d_list.append(spatial_features_2d)
            matrix_list.append(pairwise_t_matrix)
            # regroup_feature_list.append(self.regroup(spatial_features_2d, record_len))
            regroup_feature_list_large.append(
                self.regroup(spatial_features, record_len)
            )

        pairwise_t_matrix = matrix_list[0].clone().detach()
        history_feature = transform_feature(regroup_feature_list_large, self.delay)
        spatial_features = feature_list[0]
        spatial_features_2d = feature_2d_list[0]
        batch_dict = batch_dict_list[0]
        record_len = batch_dict["record_len"]
        psm_single = self.cls_head(spatial_features_2d)

        if self.delay == 0:
            (
                fused_feature,
                communication_rates,
                result_dict,
                offset_loss,
                commu_loss,
                _,
                _,
            ) = self.fusion_net(
                spatial_features,
                psm_single,
                record_len,
                pairwise_t_matrix,
                self.backbone,
                [self.shrink_conv, self.cls_head, self.reg_head],
            )
        elif self.delay > 0:
            (
                fused_feature,
                communication_rates,
                result_dict,
                offset_loss,
                commu_loss,
                _,
                _,
            ) = self.fusion_net(
                spatial_features,
                psm_single,
                record_len,
                pairwise_t_matrix,
                self.backbone,
                [self.shrink_conv, self.cls_head, self.reg_head],
                history=history_feature,
            )
        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        output_dict = {}
        output_dict.update(result_dict)
        output_dict.update(
            {
                "comm_rate": communication_rates,
                "offset_loss": offset_loss,
                "commu_loss": commu_loss,
            }
        )

        if self.detection:
            psm = self.cls_head(fused_feature)
            rm = self.reg_head(fused_feature)

            output_dict.update({"psm": psm, "rm": rm})

        if self.decoder is not None:
            x = self.decoder(fused_feature)

        if self.segmentation:
            # torch.Size([4, 64, 192, 704])
            seg_map = self.seg_head(x)
            if self.seg_rot:
                seg_map = torch.rot90(seg_map, k=1, dims=[-2, -1])
            if self.seg_clip:
                _, _, h, w = seg_map.shape
                h_down, h_up = (h - self.seg_clip[0]) // 2, (h + self.seg_clip[0]) // 2
                w_down, w_up = (w - self.seg_clip[1]) // 2, (w + self.seg_clip[1]) // 2
                seg_map = seg_map[:, :, h_down:h_up, w_down:w_up]
            output_dict.update({"seg": seg_map})

        if self.lane:
            lane_map = self.lane_head(x)
            if self.lane_rot:
                lane_map = torch.rot90(lane_map, k=1, dims=[-2, -1])
            if self.lane_clip:
                _, _, h, w = lane_map.shape
                h_down, h_up = (
                    (h - self.lane_clip[0]) // 2,
                    (h + self.lane_clip[0]) // 2,
                )
                w_down, w_up = (
                    (w - self.lane_clip[1]) // 2,
                    (w + self.lane_clip[1]) // 2,
                )
                lane_map = lane_map[:, :, h_down:h_up, w_down:w_up]
            output_dict.update({"lane": lane_map})

        if self.trajectory:
            pass
            # output_dict.update({'seg_lane': self.lane_head(x)})

        return output_dict
