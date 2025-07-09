from turtle import update

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import batch_norm, einsum
from torch.nn import functional as F

from opencood.models.common_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.how2comm_modules.communication import Communication
from opencood.models.how2comm_modules.how2comm_preprocess import How2commPreprocess
from opencood.models.how2comm_modules.stcformer import STCFormer


class How2comm(nn.Module):
    def __init__(self, args, args_pre):
        super(How2comm, self).__init__()

        self.max_cav = 5
        self.communication = False
        self.round = 1
        if "communication" in args:
            self.communication = True
            self.naive_communication = Communication(args["communication"])
            if "round" in args["communication"]:
                self.round = args["communication"]["round"]
        print("communication:", self.communication)
        self.communication_flag = args["communication_flag"]
        self.discrete_ratio = args["voxel_size"][0]
        self.downsample_rate = args["downsample_rate"]
        self.async_flag = True
        self.channel_fuse = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3
        )

        self.agg_mode = args["agg_operator"]["mode"]
        self.multi_scale = args["multi_scale"]
        self.how2comm = How2commPreprocess(args_pre, channel=64, delay=1)
        if self.multi_scale:
            layer_nums = args["layer_nums"]
            num_filters = args["num_filters"]
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                if self.agg_mode == "STCFormer":
                    fuse_network = STCFormer(
                        channel=num_filters[idx], args=args["temporal_fusion"], idx=idx
                    )
                self.fuse_modules.append(fuse_network)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(
        self,
        x,
        psm,
        record_len,
        pairwise_t_matrix,
        backbone=None,
        heads=None,
        history=None,
    ):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        pairwise_t_matrix = pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]
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

        if history and self.async_flag:
            feat_final, offset_loss = self.how2comm(
                x, history, record_len, backbone, heads
            )
            x = feat_final
        else:
            offset_loss = torch.zeros(1).to(x.device)
        neighbor_psm_list = []
        if history:
            # his = history[0]
            his = torch.concat(history, 0)
        else:
            his = x

        if self.multi_scale:
            ups = []
            ups_temporal = []
            ups_exclusive = []
            ups_common = []
            with_resnet = True if hasattr(backbone, "resnet") else False
            if with_resnet:
                feats = backbone.resnet(x)
                history_feats = backbone.resnet(his)

            for i in range(self.num_levels):
                x = feats[i] if with_resnet else backbone.blocks[i](x)
                his = history_feats[i] if with_resnet else backbone.blocks[i](his)

                if i == 0:
                    if self.communication:
                        batch_confidence_maps = self.regroup(psm, record_len)
                        _, _, confidence_maps = self.naive_communication(
                            batch_confidence_maps
                        )

                        batch_temp_features = self.regroup(x, record_len)
                        batch_temp_features_his = self.regroup(his, record_len)
                        temp_list = []
                        temp_psm_list = []
                        history_list = []
                        for b in range(B):
                            N = record_len[b]
                            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                            temp_features = batch_temp_features[b]
                            C, H, W = temp_features.shape[1:]
                            neighbor_feature = warp_affine_simple(
                                temp_features, t_matrix[0, :, :, :], (H, W)
                            )
                            temp_list.append(neighbor_feature)

                            temp_features_his = batch_temp_features_his[b]
                            C, H, W = temp_features_his.shape[1:]
                            neighbor_feature_his = warp_affine_simple(
                                temp_features_his, t_matrix[0, :, :, :], (H, W)
                            )
                            history_list.append(neighbor_feature_his)

                            temp_psm_list.append(
                                warp_affine_simple(
                                    confidence_maps[b], t_matrix[0, :, :, :], (H, W)
                                )
                            )
                        x = torch.cat(temp_list, dim=0)
                        his = torch.cat(history_list, dim=0)
                        if self.communication_flag:
                            (
                                sparse_feats,
                                commu_loss,
                                communication_rates,
                                sparse_history,
                            ) = self.how2comm.communication(
                                x, record_len, history_list, temp_psm_list
                            )
                            x = F.interpolate(
                                sparse_feats,
                                scale_factor=1,
                                mode="bilinear",
                                align_corners=False,
                            )
                            x = self.channel_fuse(x)
                            his = F.interpolate(
                                sparse_history,
                                scale_factor=1,
                                mode="bilinear",
                                align_corners=False,
                            )
                            his = self.channel_fuse(his)
                        else:
                            communication_rates = torch.tensor(0).to(x.device)
                            commu_loss = torch.zeros(1).to(x.device)
                    else:
                        communication_rates = torch.tensor(0).to(x.device)

                batch_node_features = self.regroup(x, record_len)
                batch_node_features_his = self.regroup(his, record_len)

                x_fuse = []
                x_temporal = []
                x_exclusive = []
                x_common = []
                for b in range(B):
                    N = record_len[b]
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    node_features = batch_node_features[b]
                    node_features_his = batch_node_features_his[b]
                    if i == 0:
                        neighbor_feature = node_features
                        neighbor_feature_his = node_features_his
                        neighbor_psm = warp_affine_simple(
                            confidence_maps[b], t_matrix[0, :, :, :], (H, W)
                        )

                    else:
                        C, H, W = node_features.shape[1:]
                        neighbor_feature = warp_affine_simple(
                            node_features, t_matrix[0, :, :, :], (H, W)
                        )
                        neighbor_feature_his = warp_affine_simple(
                            node_features_his, t_matrix[0, :, :, :], (H, W)
                        )

                    feature_shape = neighbor_feature.shape
                    padding_len = self.max_cav - feature_shape[0]
                    padding_feature = torch.zeros(
                        padding_len,
                        feature_shape[1],
                        feature_shape[2],
                        feature_shape[3],
                    )
                    padding_feature = padding_feature.to(neighbor_feature.device)
                    neighbor_feature = torch.cat(
                        [neighbor_feature, padding_feature], dim=0
                    )

                    if i == 0:
                        padding_map = torch.zeros(
                            padding_len, 1, feature_shape[2], feature_shape[3]
                        )
                        padding_map = padding_map.to(neighbor_feature.device)
                        neighbor_psm = torch.cat([neighbor_psm, padding_map], dim=0)
                        neighbor_psm_list.append(neighbor_psm)

                    if self.agg_mode == "STCFormer":
                        fusion, output_list = self.fuse_modules[i](
                            neighbor_feature,
                            neighbor_psm_list[b],
                            neighbor_feature_his,
                            i,
                        )
                        x_fuse.append(fusion)
                        x_temporal.append(output_list[0])
                        x_exclusive.append(output_list[1])
                        x_common.append(output_list[2])

                x_fuse = torch.stack(x_fuse)
                x_temporal = torch.stack(x_temporal)
                x_exclusive = torch.stack(x_exclusive)
                x_common = torch.stack(x_common)

                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                    ups_temporal.append(backbone.deblocks[i](x_temporal))
                    ups_exclusive.append(backbone.deblocks[i](x_exclusive))
                    ups_common.append(backbone.deblocks[i](x_common))
                else:
                    ups.append(x_fuse)

            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
                x_temporal = torch.cat(ups_temporal, dim=1)
                x_exclusive = torch.cat(ups_exclusive, dim=1)
                x_common = torch.cat(ups_common, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]

            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)

        return (
            x_fuse,
            communication_rates,
            {},
            offset_loss,
            commu_loss,
            None,
            [x_temporal, x_exclusive, x_common],
        )
