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


class SpatialFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialFusion, self).__init__()

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

    def forward(self, x, record_len, pairwise_t_matrix):
        # split x to receiver feature and sender feature
        rec_feature = x[0, :, :, :].unsqueeze(0)
        sed_feature = x[1, :, :, :].unsqueeze(0)

        # transfer sed to rec's space
        t_matrix = pairwise_t_matrix[0][:2, :2, :, :]
        t_sed_feature = warp_affine_simple(
            sed_feature, t_matrix[0, 1, :, :].unsqueeze(0), (x.shape[2], x.shape[3])
        )

        # generate overlap selector and non-overlap selector
        selector = torch.ones_like(sed_feature)
        selector = warp_affine_simple(
            selector, t_matrix[0, 1, :, :].unsqueeze(0), (x.shape[2], x.shape[3])
        )
        overlap_sel = self.generate_overlap_selector(selector)  # overlap area selector
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

        x = torch.cat((weight_to_rec, weight_to_t_sed), dim=1)
        x = self.compChannels2(x)

        return x
