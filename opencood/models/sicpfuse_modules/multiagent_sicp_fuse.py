# -*- coding: utf-8 -*-
# Author: Deyuan Qu <deyuanqu@my.unt.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import math
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# from opencood.models.sub_modules.torch_transformation_utils import \
#     warp_affine_simple
from opencood.models.common_modules.torch_transformation_utils import warp_affine_simple


class MultiSpatialFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiSpatialFusion, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid(),
        )
        self.compChannels1 = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
        )
        self.compChannels2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

    def generate_overlap_selector(self, selector):
        overlap_sel = torch.mean(selector, 1).unsqueeze(0).cuda()
        return overlap_sel

    def generate_nonoverlap_selector(self, overlap_sel):
        non_overlap_sel = torch.tensor(np.where(overlap_sel.cpu() > 0, 0, 1)).cuda()
        return non_overlap_sel

    def forward(self, x, batch_record_len, pairwise_t_matrix):
        # TODO(YH): we should support multiagent collab
        batch_size = batch_record_len.shape[0]
        batch_fused_feats = []
        ptr = 0  # pointer to track position in x
        for i in range(batch_size):
            num_agents = batch_record_len[i].item()
            cur_x = x[ptr : ptr + num_agents]
            fused_feats = []

            if num_agents == 1:  # only contains one vehicle
                batch_fused_feats.append(cur_x[0])
                ptr += num_agents
                continue

            rec_feature = cur_x[0].unsqueeze(0)  # (1, C, H, W) for ego
            for j in range(1, num_agents):
                # split x to receiver feature and sender feature
                sed_feature = cur_x[j].unsqueeze(0)  # (1, C, H, W) for cav

                # transfer sed to rec's space
                t_matrix = pairwise_t_matrix[i, 0, j]  # (2, 3)
                t_sed_feature = warp_affine_simple(
                    sed_feature,
                    t_matrix.unsqueeze(0),
                    (cur_x.shape[2], cur_x.shape[3]),
                )

                # generate overlap selector and non-overlap selector
                selector = torch.ones_like(sed_feature)
                selector = warp_affine_simple(
                    selector,
                    t_matrix.unsqueeze(0),
                    (cur_x.shape[2], cur_x.shape[3]),
                )
                overlap_sel = self.generate_overlap_selector(
                    selector
                )  # overlap area selector
                non_overlap_sel = self.generate_nonoverlap_selector(
                    overlap_sel
                )  # non-overlap area selector

                # generate the weight map
                cat_feature = torch.cat((rec_feature, t_sed_feature), dim=1)
                comp_feature = self.compChannels1(cat_feature)
                f1 = self.conv1(comp_feature)
                f2 = self.conv2(f1)
                weight_map = comp_feature + f2

                # normalize the weight map to [0,1]
                normalize_weight_map = (weight_map - torch.min(weight_map)) / (
                    torch.max(weight_map) - torch.min(weight_map)
                )

                # apply normalized weight map to rec_feature and t_sed_feature
                weight_to_rec = rec_feature * (
                    normalize_weight_map * overlap_sel + non_overlap_sel
                )
                weight_to_t_sed = t_sed_feature * (1 - normalize_weight_map)

                fused_feat = torch.cat((weight_to_rec, weight_to_t_sed), dim=1)
                fused_feat = self.compChannels2(fused_feat)

                fused_feats.append(fused_feat)

            fused_feats = torch.cat(fused_feats, dim=0)
            fused_feats = fused_feats.mean(dim=0)
            batch_fused_feats.append(fused_feats)
            ptr += num_agents

        # NOTE: SiCP only consider 2 collab agents, here consider more general case
        # we take the mean of all fused features which is slightly different from
        # the original paper
        fused_feats = torch.stack(batch_fused_feats, dim=0)

        return fused_feats
