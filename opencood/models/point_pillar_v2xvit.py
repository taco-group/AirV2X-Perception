import torch
import torch.nn as nn

from opencood.models.common_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.fuse_utils import regroup
from opencood.models.common_modules.naive_compress import NaiveCompressor
from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.common_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.v2xvit_modules.v2xvit_basic import V2XTransformer


class PointPillarV2XVit(nn.Module):
    def __init__(self, args):
        super(PointPillarV2XVit, self).__init__()

        self.modality = "processed_lidar"
        if "use_modality" in args.keys():
            self.modality = args["use_modality"]

        self.max_cav = args["max_cav"]
        # PIllar VFE
        self.pillar_vfe = PillarVFE(
            args["pillar_vfe"],
            num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=args["lidar_range"],
        )
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])
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

        self.fusion_net = V2XTransformer(args["transformer"])

        self.cls_head = nn.Conv2d(128 * 2, args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args["anchor_number"], kernel_size=1)
        self.discrete_ratio = args["voxel_size"][0]

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

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        # voxel_features = data_dict['processed_lidar']['voxel_features']
        # voxel_coords = data_dict['processed_lidar']['voxel_coords']
        # voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        voxel_features = data_dict[self.modality]["voxel_features"]
        voxel_coords = data_dict[self.modality]["voxel_coords"]
        voxel_num_points = data_dict[self.modality]["voxel_num_points"]
        record_len = data_dict["record_len"]
        pairwise_t_matrix = data_dict["pairwise_t_matrix"]
        """
        We don't consider the time delay
        So spatial_correction_matrix is just identity matrix
        """
        # B, L, 4, 4
        spatial_correction_matrix = (
            torch.eye(4)
            .expand(len(record_len), self.max_cav, 4, 4)
            .to(record_len.device)
        )

        # B, max_cav, 3(dt dv infra), 1, 1
        # prior_encoding =\
        #     data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)
        prior_encoding = torch.zeros(len(record_len), self.max_cav, 3, 1, 1).to(
            record_len.device
        )

        # print(voxel_features.shape)
        batch_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
            "record_len": record_len,
        }
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        comm_rates = batch_dict["spatial_features"].count_nonzero().item()
        _, _, H, W = batch_dict["spatial_features"].shape
        self.downsample_rate = 1

        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict["spatial_features_2d"]
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        # comm_rates = spatial_features_2d.count_nonzero().item()
        # N, C, H, W -> B, L, C, H, W
        regroup_feature, mask = regroup(spatial_features_2d, record_len, self.max_cav)

        # prior encoding added
        prior_encoding = prior_encoding.repeat(
            1, 1, 1, regroup_feature.shape[3], regroup_feature.shape[4]
        )

        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)

        ######## Added by Yifan Lu 2022.9.5
        regroup_feature_new = []
        B = regroup_feature.shape[0]
        pairwise_t_matrix = pairwise_t_matrix[:, :, :, [0, 1], :][
            :, :, :, :, [0, 1, 3]
        ]  # [B, L, L, 2, 3]
        pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
        pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
        pairwise_t_matrix[..., 0, 2] = (
            pairwise_t_matrix[..., 0, 2]
            / (self.downsample_rate * self.discrete_ratio * W)
            * 2
        )
        pairwise_t_matrix[..., 1, 2] = (
            pairwise_t_matrix[..., 1, 2]
            / (self.downsample_rate * self.discrete_ratio * H)
            * 2
        )
        H_, W_ = spatial_features_2d.shape[-2:]
        for b in range(B):
            # (B,L,L,2,3)
            ego = 0
            regroup_feature_new.append(
                warp_affine_simple(
                    regroup_feature[b], pairwise_t_matrix[b, ego], (H_, W_)
                )
            )
        regroup_feature = torch.stack(regroup_feature_new)
        ###################################

        # b l c h w -> b l h w c
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        # transformer fusion
        fused_feature = self.fusion_net(
            regroup_feature, mask, spatial_correction_matrix
        )
        # b h w c -> b c h w
        fused_feature = fused_feature.permute(0, 3, 1, 2)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {"psm": psm, "rm": rm}
        output_dict.update({"mask": 0, "each_mask": 0, "comm_rate": comm_rates})
        # print(comm_rates)
        return output_dict
