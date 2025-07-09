import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from opencood.models.how2comm_modules.how2comm_deformable_transformer import (
    RPN_transformer_deformable_mtf_singlescale,
)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


class TemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super(TemporalAttention, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)
        self.hidden_dim = feature_dim * 2
        self.conv_query = nn.Conv2d(
            feature_dim, self.hidden_dim, kernel_size=3, padding=1
        )
        self.conv_key = nn.Conv2d(
            feature_dim, self.hidden_dim, kernel_size=3, padding=1
        )
        self.conv_value = nn.Conv2d(
            feature_dim, self.hidden_dim, kernel_size=3, padding=1
        )
        self.conv_temporal_key = nn.Conv1d(
            self.hidden_dim, self.hidden_dim, kernel_size=1, stride=1
        )
        self.conv_temporal_value = nn.Conv1d(
            self.hidden_dim, self.hidden_dim, kernel_size=1, stride=1
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_feat = nn.Conv2d(
            self.hidden_dim, feature_dim, kernel_size=3, padding=1
        )

    def forward(self, x):
        frame, C, H, W = x.shape
        ego = x[:1]
        query = self.conv_query(ego)
        query = query.view(1, self.hidden_dim, -1).permute(2, 0, 1)

        key = self.conv_key(x)
        key_avg = key
        value = self.conv_value(x)
        val_avg = value
        key = key.view(frame, self.hidden_dim, -1).permute(2, 0, 1)
        value = value.view(frame, self.hidden_dim, -1).permute(2, 0, 1)

        key_avg = self.pool(key_avg).squeeze(-1).squeeze(-1)
        val_avg = self.pool(val_avg).squeeze(-1).squeeze(-1)
        key_avg = self.conv_temporal_key(key_avg.unsqueeze(0).permute(0, 2, 1))
        val_avg = self.conv_temporal_value(val_avg.unsqueeze(0).permute(0, 2, 1))
        key_avg = key_avg.permute(0, 2, 1)
        val_avg = val_avg.permute(0, 2, 1)
        key = key * key_avg
        value = value * val_avg

        x = self.att(query, key, value)
        x = x.permute(1, 2, 0).view(1, self.hidden_dim, H, W)
        out = self.conv_feat(x)

        return out


class LateFusion(nn.Module):
    def __init__(self, channel):
        super(LateFusion, self).__init__()
        self.channel = channel
        self.gate_1 = nn.Conv2d(self.channel, 1, kernel_size=3, stride=1, padding=1)
        self.gate_2 = nn.Conv2d(self.channel, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, exc, com):
        weight_1 = self.gate_1(exc)
        weight_2 = self.gate_2(com)
        weights = torch.cat([weight_1, weight_2], dim=1)
        weights = torch.softmax(weights, dim=1)
        final = weights[:, :1, :, :] * exc + weights[:, 1:, :, :] * com  #

        return final


class Decoupling(nn.Module):
    def __init__(self):
        super(Decoupling, self).__init__()
        self.exclusive_thre = 0.01
        self.common_thre = 0.01

    def forward(self, feat, confidence):
        ego_confi = confidence[:1]
        exclusive_list = []
        exclusive_map_list = [ego_confi]
        common_list = []
        common_map_list = [ego_confi]
        for n in range(1, feat.shape[0]):
            exclusive_map = (1 - ego_confi) * confidence[n].unsqueeze(0)  #
            exclusive_map_list.append(exclusive_map)
            common_map = ego_confi * confidence[n].unsqueeze(0)
            common_map_list.append(common_map)
            ones_mask = torch.ones_like(exclusive_map).to(exclusive_map.device)
            zeros_mask = torch.zeros_like(exclusive_map).to(exclusive_map.device)
            exclusive_mask = torch.where(
                exclusive_map > self.exclusive_thre, ones_mask, zeros_mask
            )
            common_mask = torch.where(
                common_map > self.common_thre, ones_mask, zeros_mask
            )

            exclusive_list.append(feat[n].unsqueeze(0) * exclusive_mask)
            common_list.append(feat[n].unsqueeze(0) * common_mask)

        return (
            torch.cat(exclusive_list, dim=0),
            torch.cat(common_list, dim=0),
            torch.cat(exclusive_map_list, dim=0),
            torch.cat(common_map_list, dim=0),
        )


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class STCFormer(nn.Module):
    def __init__(self, channel, args, idx):
        super(STCFormer, self).__init__()

        self.decoupling = Decoupling()
        self.scale = [1, 0.5, 0.25]
        self.temporal_self_attention = TemporalAttention(channel)
        self.layer_norm = nn.LayerNorm(
            [channel, args["height"][idx], args["width"][idx]]
        )
        self.exclusive_encoder = RPN_transformer_deformable_mtf_singlescale(
            channel=channel, points=9
        )
        self.common_encoder = RPN_transformer_deformable_mtf_singlescale(
            channel=channel, points=3
        )
        self.late_fusion = LateFusion(channel=channel)
        self.time_embedding = nn.Linear(1, channel)

    def forward(self, neighbor_feat, neighbor_confidence, history_feat, level):
        if level > 0:
            neighbor_confidence = F.interpolate(
                neighbor_confidence, scale_factor=self.scale[level]
            )
        exclusive_feat, common_feat, exclusive_map, common_map = self.decoupling(
            neighbor_feat, neighbor_confidence
        )

        ego_feat = neighbor_feat[:1]
        history_feat = torch.cat([ego_feat, history_feat], dim=0)

        delay = [0.0] + [-1.0] * (history_feat.shape[0] - 1)
        delay = torch.tensor([delay]).to(ego_feat.device)
        time_embed = self.time_embedding(delay[:, :, None])
        time_embed = time_embed.reshape(history_feat.shape[0], -1, 1, 1)
        history_feat = history_feat + time_embed

        x = self.temporal_self_attention(history_feat)
        ego_feat = x
        temporal_feat = ego_feat

        exclusive_feat = torch.cat([ego_feat, exclusive_feat], dim=0)
        common_feat = torch.cat([ego_feat, common_feat], dim=0)
        ego_exclusive_feat = self.exclusive_encoder(
            exclusive_feat, exclusive_map
        ).unsqueeze(0)
        ego_common_feat = self.common_encoder(common_feat, common_map).unsqueeze(0)

        x = self.late_fusion(ego_exclusive_feat, ego_common_feat)
        ego_feat = x

        return ego_feat[0], [
            temporal_feat[0],
            ego_exclusive_feat[0],
            ego_common_feat[0],
        ]
