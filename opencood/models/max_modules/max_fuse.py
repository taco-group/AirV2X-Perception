# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from opencood.models.common_modules.torch_transformation_utils import (
    get_discretized_transformation_matrix,
    get_rotated_roi,
    get_transformation_matrix,
    warp_affine_simple,
)
from opencood.models.max_modules.convgru import ConvGRU


class MaxFusion(nn.Module):
    def __init__(self, args):
        super(MaxFusion, self).__init__()
        self.discrete_ratio = args["voxel_size"][0]  # voxel_size[0]=0.4
        self.downsample_rate = args[
            "downsample_rate"
        ]  # 2/4, downsample rate from original feature map [200, 704]

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len, pairwise_t_matrix):
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

        # split x:[(L1, C, H, W), (L2, C, H, W), ...]
        # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
        split_x = self.regroup(x, record_len)

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

        # (B*L,L,1,H,W)
        roi_mask = torch.zeros((B, L, L, 1, H, W)).to(x)
        for b in range(B):
            N = record_len[b]
            for i in range(N):
                one_tensor = torch.ones((L, 1, H, W)).to(x)
                roi_mask[b, i] = warp_affine_simple(
                    one_tensor, pairwise_t_matrix[b][i, :, :, :], (H, W)
                )

        batch_node_features = split_x
        # iteratively update the features for num_iteration times

        out = []
        # iterate each batch
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            updated_node_features = []

            # update each node i
            i = 0  # ego
            # (N,1,H,W)
            mask = roi_mask[b, i, :N, ...]
            # (N,C,H,W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
            # Notice we put i one the first dim of t_matrix. Different from original.
            # t_matrix[i,j] = Tji
            neighbor_feature = warp_affine_simple(
                batch_node_features[b], t_matrix[i, :, :, :], (H, W)
            )
            out.append(torch.max(neighbor_feature, dim=0)[0])
        out = torch.stack(out)

        return out

    def forward_debug(self, x, origin_x, record_len, pairwise_t_matrix):
        """
        Fusion forwarding
        Used for debug and visualization


        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)

        origin_x: torch.Tensor
            pillars (sum(n_cav), C, H * downsample_rate, W * downsample_rate)

        record_len : list
            shape: (B)

        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego,
            shape: (B, L, L, 4, 4)

        Returns
        -------
        Fused feature.
        """
        from matplotlib import pyplot as plt

        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        split_x = self.regroup(x, record_len)
        split_origin_x = self.regroup(origin_x, record_len)

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

        # (B*L,L,1,H,W)
        roi_mask = torch.zeros((B, L, L, 1, H, W)).to(x)
        for b in range(B):
            N = record_len[b]
            for i in range(N):
                one_tensor = torch.ones((L, 1, H, W)).to(x)
                roi_mask[b, i] = warp_affine_simple(
                    one_tensor, pairwise_t_matrix[b][i, :, :, :], (H, W)
                )

        batch_node_features = split_x
        # iteratively update the features for num_iteration times

        # visualize warped feature map
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            # update each node i
            i = 0  # ego
            mask = roi_mask[b, i, :N, ...]
            # (N,C,H,W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
            # Notice we put i one the first dim of t_matrix. Different from original.
            # t_matrix[i,j] = Tji
            neighbor_feature = warp_affine_simple(
                batch_node_features[b], t_matrix[i, :, :, :], (H, W)
            )
            for idx in range(N):
                plt.imshow(
                    torch.max(neighbor_feature[idx], 0)[0].detach().cpu().numpy()
                )
                plt.savefig(
                    f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/vis_result/debug_warp_feature/feature_{b}_{idx}"
                )
                plt.clf()
                plt.imshow(mask[idx][0].detach().cpu().numpy())
                plt.savefig(
                    f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/vis_result/debug_warp_feature/mask_feature_{b}_{idx}"
                )
                plt.clf()

        # visualize origin pillar feature
        origin_node_features = split_origin_x

        for b in range(B):
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            i = 0  # ego
            # (N,C,H,W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
            # Notice we put i one the first dim of t_matrix. Different from original.
            # t_matrix[i,j] = Tji
            neighbor_feature = warp_affine_simple(
                origin_node_features[b],
                t_matrix[i, :, :, :],
                (H * self.downsample_rate, W * self.downsample_rate),
            )

            for idx in range(N):
                plt.imshow(
                    torch.max(neighbor_feature[idx], 0)[0].detach().cpu().numpy()
                )
                plt.savefig(
                    f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/vis_result/debug_warp_feature/origin_{b}_{idx}"
                )
                plt.clf()
