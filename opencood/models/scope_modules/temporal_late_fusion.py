import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import batch_norm, einsum


class LateFusion(nn.Module):
    def __init__(self, args):
        super(LateFusion, self).__init__()
        self.thre = args["thre"]
        if "gaussian_smooth" in args:
            self.smooth = True
            kernel_size = args["gaussian_smooth"]["k_size"]
            c_sigma = args["gaussian_smooth"]["c_sigma"]
            self.gaussian_filter = nn.Conv2d(
                1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2
            )
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False

    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = (
                1
                / (2 * np.pi * sigma)
                * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            )
            return g

        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = (
            torch.Tensor(gaussian_kernel)
            .to(self.gaussian_filter.weight.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.gaussian_filter.bias.data.zero_()

    def forward(self, input, psm_temporal, psm_ego, psm_agent):
        B, C, H, W = input[0].shape
        fusion_list = []
        for b in range(B):
            confi_temporal = (
                psm_temporal[b : b + 1, :].sigmoid().max(dim=1)[0].unsqueeze(1)
            )
            confi_ego = psm_ego[b : b + 1, :].sigmoid().max(dim=1)[0].unsqueeze(1)
            confi_agent = psm_agent[b : b + 1, :].sigmoid().max(dim=1)[0].unsqueeze(1)
            if self.smooth:
                confi_temporal = self.gaussian_filter(confi_temporal)
                confi_ego = self.gaussian_filter(confi_ego)
                confi_agent = self.gaussian_filter(confi_agent)
            total_confi = torch.cat([confi_temporal, confi_ego, confi_agent], dim=1)
            total_confi = torch.softmax(total_confi, dim=1)
            feat_temporal = input[0][b : b + 1, :] * total_confi[0:1, 0:1, :, :]
            feat_ego = input[1][b : b + 1, :] * total_confi[0:1, 1:2, :, :]
            feat_agent = input[2][b : b + 1, :] * total_confi[0:1, 2:3, :, :]
            fusion_list.append(feat_temporal + feat_ego + feat_agent)
        final_feat = torch.cat(fusion_list, dim=0)
        return final_feat
