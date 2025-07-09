import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import batch_norm, einsum


class SyncLSTM(nn.Module):
    def __init__(
        self, channel_size=256, spatial_size=32, compressed_size=64, height=32, width=32
    ):
        super(SyncLSTM, self).__init__()
        self.spatial_size = spatial_size
        self.channel_size = channel_size
        self.compressed_size = compressed_size
        self.lstmcell = MotionLSTM(32, self.compressed_size, height=height, width=width)
        self.init_c = nn.parameter.Parameter(
            torch.rand(self.compressed_size, height, width)
        )

        self.ratio = int(math.sqrt(channel_size / compressed_size))
        self.conv_pre_1 = nn.Conv2d(
            self.channel_size,
            self.ratio * self.compressed_size,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_pre_2 = nn.Conv2d(
            self.ratio * self.compressed_size,
            self.compressed_size,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn_pre_1 = nn.BatchNorm2d(self.ratio * self.compressed_size)
        self.bn_pre_2 = nn.BatchNorm2d(self.compressed_size)
        self.conv_after_1 = nn.Conv2d(
            self.compressed_size,
            self.ratio * self.compressed_size,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_after_2 = nn.Conv2d(
            self.ratio * self.compressed_size,
            self.channel_size,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn_after_1 = nn.BatchNorm2d(self.ratio * self.compressed_size)
        self.bn_after_2 = nn.BatchNorm2d(self.channel_size)

    def forward(self, x_raw):
        frame_num, C, H, W = x_raw.shape
        if self.compressed_size != self.channel_size:
            x = F.relu(self.bn_pre_1(self.conv_pre_1(x_raw.view(-1, C, H, W))))
            x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))
            x = x.view(frame_num, self.compressed_size, H, W)
        else:
            x = x_raw

        h = x[-1, :].unsqueeze(0)
        c = self.init_c
        for i in range(frame_num - 2, -1, -1):
            h, c = self.lstmcell(x[i, :].unsqueeze(0), (h, c))
        res = h
        if self.compressed_size != self.channel_size:
            res = F.relu(self.bn_after_1(self.conv_after_1(res)))
            res = F.relu(self.bn_after_2(self.conv_after_2(res)))
        else:
            res = res
        return res


class MotionLSTM(nn.Module):
    def __init__(
        self, spatial_size, input_channel_size, hidden_size=0, height=32, width=32
    ):
        super().__init__()
        self.input_channel_size = input_channel_size
        self.hidden_size = hidden_size
        self.spatial_size = spatial_size

        self.U_i = STPN_MotionLSTM(height_feat_size=self.input_channel_size)
        self.V_i = STPN_MotionLSTM(height_feat_size=self.input_channel_size)
        self.b_i = nn.Parameter(torch.Tensor(1, self.input_channel_size, height, width))

        self.U_f = STPN_MotionLSTM(height_feat_size=self.input_channel_size)
        self.V_f = STPN_MotionLSTM(height_feat_size=self.input_channel_size)
        self.b_f = nn.Parameter(torch.Tensor(1, self.input_channel_size, height, width))

        self.U_c = STPN_MotionLSTM(height_feat_size=self.input_channel_size)
        self.V_c = STPN_MotionLSTM(height_feat_size=self.input_channel_size)
        self.b_c = nn.Parameter(torch.Tensor(1, self.input_channel_size, height, width))

        self.U_o = STPN_MotionLSTM(height_feat_size=self.input_channel_size)
        self.V_o = STPN_MotionLSTM(height_feat_size=self.input_channel_size)
        self.b_o = nn.Parameter(torch.Tensor(1, self.input_channel_size, height, width))

    def forward(self, x, init_states=None):
        h, c = init_states
        i = torch.sigmoid(self.U_i(x) + self.V_i(h) + self.b_i)
        f = torch.sigmoid(self.U_f(x) + self.V_f(h) + self.b_f)
        g = torch.tanh(self.U_c(x) + self.V_c(h) + self.b_c)
        o = torch.sigmoid(self.U_o(x) + self.V_o(x) + self.b_o)
        c_out = f * c + i * g
        h_out = o * torch.tanh(c_out)

        return (h_out, c_out)


class STPN_MotionLSTM(nn.Module):
    def __init__(self, height_feat_size=16):
        super(STPN_MotionLSTM, self).__init__()

        self.conv1_1 = nn.Conv2d(
            height_feat_size, 2 * height_feat_size, kernel_size=3, stride=2, padding=1
        )
        self.conv1_2 = nn.Conv2d(
            2 * height_feat_size,
            2 * height_feat_size,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2_1 = nn.Conv2d(
            2 * height_feat_size,
            4 * height_feat_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2_2 = nn.Conv2d(
            4 * height_feat_size,
            4 * height_feat_size,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv7_1 = nn.Conv2d(
            6 * height_feat_size,
            2 * height_feat_size,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv7_2 = nn.Conv2d(
            2 * height_feat_size,
            2 * height_feat_size,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv8_1 = nn.Conv2d(
            3 * height_feat_size, height_feat_size, kernel_size=3, stride=1, padding=1
        )
        self.conv8_2 = nn.Conv2d(
            height_feat_size, height_feat_size, kernel_size=3, stride=1, padding=1
        )

        self.bn1_1 = nn.BatchNorm2d(2 * height_feat_size)
        self.bn1_2 = nn.BatchNorm2d(2 * height_feat_size)

        self.bn2_1 = nn.BatchNorm2d(4 * height_feat_size)
        self.bn2_2 = nn.BatchNorm2d(4 * height_feat_size)

        self.bn7_1 = nn.BatchNorm2d(2 * height_feat_size)
        self.bn7_2 = nn.BatchNorm2d(2 * height_feat_size)

        self.bn8_1 = nn.BatchNorm2d(1 * height_feat_size)
        self.bn8_2 = nn.BatchNorm2d(1 * height_feat_size)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))
        x_7 = F.relu(
            self.bn7_1(
                self.conv7_1(
                    torch.cat((F.interpolate(x_2, scale_factor=(2, 2)), x_1), dim=1)
                )
            )
        )
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))
        x_8 = F.relu(
            self.bn8_1(
                self.conv8_1(
                    torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1)
                )
            )
        )
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))

        return res_x


class SpatialAttention_mtf(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_mtf, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.Tan = nn.Tanh()

    def forward(self, curr, prev):
        curr_avg_out = torch.mean(curr, dim=1, keepdim=True)
        curr_max_out, _ = torch.max(curr, dim=1, keepdim=True)
        curr_merge = torch.cat([curr_avg_out, curr_max_out], dim=1)

        prev_avg_out = torch.mean(
            torch.sum(prev, dim=0, keepdim=True), dim=1, keepdim=True
        )
        prev_max_out, _ = torch.max(
            torch.sum(prev, dim=0, keepdim=True), dim=1, keepdim=True
        )
        prev_merge = torch.cat([prev_avg_out, prev_max_out], dim=1)
        merge = self.sigmoid(self.conv1(curr_merge + prev_merge))
        final_out = (1 - merge) * self.Tan(curr) + merge * prev

        return final_out


class TemporalFusion_lstm(nn.Module):
    def __init__(self, args):
        super(TemporalFusion_lstm, self).__init__()
        self.channel = args["temporal_fusion"]["channel"]
        self.height = args["temporal_fusion"]["height"]
        self.width = args["temporal_fusion"]["width"]
        self.mtf_attention = SpatialAttention_mtf()
        self.sync_lstm = SyncLSTM(
            channel_size=self.channel, height=self.height, width=self.width
        )

    def forward(self, origin_input):
        x_curr = origin_input[0:1, :]
        x_prev = origin_input[1:, :]

        x_prev_cat = self.mtf_attention(x_curr, x_prev)
        x_raw = torch.cat([x_curr, x_prev_cat], dim=0)

        x_fuse = self.sync_lstm(x_raw)

        return x_fuse
