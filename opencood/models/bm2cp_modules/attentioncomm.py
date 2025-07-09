# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
# Modified from Yue Hu <18671129361@sjtu.edu.cn>
# Author: Binyu Zhao <byzhao@stu.hit.edu.cn>


from turtle import update

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.common_modules.torch_transformation_utils import warp_affine_simple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


class AttenFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttenFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(
            2, 0, 1
        )  #  (H*W, cav_num, C), perform self attention on each pixel.
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x


def communication(batch_confidence_maps, threshold_maps, record_len, pairwise_t_matrix):
    # batch_confidence_maps:[(L1, H, W), (L2, H, W), ...]
    # pairwise_t_matrix: (B,L,L,2,3)
    # thre: threshold of objectiveness
    # a_ji = (1 - q_i)*q_ji
    B, L, _, _, _ = pairwise_t_matrix.shape
    _, _, H, W = batch_confidence_maps[0].shape

    communication_masks = []
    communication_rates = []
    for b in range(B):
        N = record_len[b]

        ori_communication_maps = (
            batch_confidence_maps[b].sigmoid().max(dim=1)[0].unsqueeze(1)
        )  # dim1=2 represents the confidence of two anchors

        communication_maps = ori_communication_maps

        ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
        zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
        communication_mask = torch.where(
            communication_maps > threshold_maps[b], ones_mask, zeros_mask
        )

        communication_rate = communication_mask[0].sum() / (H * W)

        communication_mask_nodiag = communication_mask.clone()
        ones_mask = torch.ones_like(communication_mask).to(communication_mask.device)
        communication_mask_nodiag[::2] = ones_mask[::2]

        communication_masks.append(communication_mask_nodiag)
        communication_rates.append(communication_rate)

    communication_rates = sum(communication_rates) / B
    communication_masks = torch.concat(communication_masks, dim=0)
    return {}, communication_masks, communication_rates


class AttenComm(nn.Module):
    def __init__(self, args):
        super(AttenComm, self).__init__()
        self.discrete_ratio = args["voxel_size"][0]  # voxel_size[0]=0.4
        self.downsample_rate = args[
            "downsample_rate"
        ]  # 2/4, downsample rate from original feature map [200, 704]

        self.multi_scale = args["multi_scale"]
        if self.multi_scale:
            layer_nums = args["layer_nums"]
            num_filters = args["num_filters"]
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                fuse_network = AttenFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
        else:
            self.fuse_modules = AttenFusion(args["agg_operator"]["feature_dim"])

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(
        self, x, rm, thres_map, record_len, pairwise_t_matrix, backbone=None, heads=None
    ):
        """
        Fusion forwarding.

        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)

        record_len : list
            shape: (B)

        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego,
            shape: (B, L, L, 4, 4)

        Returns
        -------
        Fused feature.
        """
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        # (B,L,L,2,3)
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

        if self.multi_scale:
            ups = []
            # backbone.__dict__()
            with_resnet = True if hasattr(backbone, "resnet") else False
            if with_resnet:
                feats = backbone.resnet(x)

            for i in range(self.num_levels):
                x = feats[i] if with_resnet else backbone.blocks[i](x)

                if rm.shape != thres_map.shape:
                    level_thres_map = F.interpolate(
                        thres_map, size=rm.shape[2:], mode="bilinear"
                    )
                else:
                    level_thres_map = thres_map

                if i == 0:
                    batch_confidence_maps = self.regroup(rm, record_len)
                    batch_level_thres_map = self.regroup(level_thres_map, record_len)
                    _, communication_masks, communication_rates = communication(
                        batch_confidence_maps,
                        batch_level_thres_map,
                        record_len,
                        pairwise_t_matrix,
                    )
                    if x.shape[-1] != communication_masks.shape[-1]:
                        communication_masks = F.interpolate(
                            communication_masks,
                            size=(x.shape[-2], x.shape[-1]),
                            mode="bilinear",
                            align_corners=False,
                        )
                    x = x * communication_masks

                # split x:[(L1, C, H, W), (L2, C, H, W), ...]
                # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
                batch_node_features = self.regroup(x, record_len)

                x_fuse = []
                for b in range(B):
                    # number of valid agent
                    N = record_len[b]
                    # (N,N,4,4)
                    # t_matrix[i, j]-> from i to j
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    node_features = batch_node_features[b]
                    C, H, W = node_features.shape[1:]
                    neighbor_feature = warp_affine_simple(
                        node_features, t_matrix[0, :, :, :], (H, W)
                    )
                    x_fuse.append(self.fuse_modules[i](neighbor_feature))
                x_fuse = torch.stack(x_fuse)

                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)

            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]

            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
        else:
            if x.shape != thres_map.shape:
                thres_map = F.interpolate(thres_map, size=x.shape[2:], mode="bilinear")
            # split x:[(L1, C, H, W), (L2, C, H, W), ...]
            # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
            batch_node_features = self.regroup(x, record_len)
            batch_confidence_maps = self.regroup(rm, record_len)
            batch_thres_map = self.regroup(thres_map, record_len)

            _, communication_masks, communication_rates = communication(
                batch_confidence_maps, batch_thres_map, record_len, pairwise_t_matrix
            )

            x_fuse = []
            for b in range(B):
                N = record_len[b]
                # (N,N,4,4)
                # t_matrix[i, j]-> from i to j
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                node_features = batch_node_features[b]
                if self.communication:
                    node_features = node_features * communication_masks[b]
                neighbor_feature = warp_affine_simple(
                    node_features, t_matrix[0, :, :, :], (H, W)
                )
                x_fuse.append(self.fuse_modules(neighbor_feature))
            x_fuse = torch.stack(x_fuse)

        return x_fuse, communication_rates, {}
