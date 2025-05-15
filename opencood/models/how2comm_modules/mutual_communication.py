import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Channel_Request_Attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Channel_Request_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class Spatial_Request_Attention(nn.Module):
    def __init__(self, kernel_size=3):
        super(Spatial_Request_Attention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class StatisticsNetwork(nn.Module):
    def __init__(self, img_feature_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=img_feature_channels,
            out_channels=img_feature_channels * 2,
            kernel_size=1,
            stride=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=img_feature_channels * 2,
            out_channels=img_feature_channels * 2,
            kernel_size=1,
            stride=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=img_feature_channels * 2,
            out_channels=1,
            kernel_size=1,
            stride=1,
        )
        self.relu = nn.ReLU()

    def forward(self, concat_feature: torch.Tensor) -> torch.Tensor:
        x = self.conv1(concat_feature)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        local_statistics = self.conv3(x)
        return local_statistics


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, loss_coeff=1) -> None:
        super().__init__()
        self.loss_coeff = loss_coeff

    def __call__(self, T: torch.Tensor, T_prime: torch.Tensor) -> float:
        joint_expectation = (-F.softplus(-T)).mean()
        marginal_expectation = F.softplus(T_prime).mean()
        mutual_info = joint_expectation - marginal_expectation

        return -mutual_info * self.loss_coeff


class Communication(nn.Module):
    def __init__(self, args, in_planes):
        super(Communication, self).__init__()
        self.channel_request = Channel_Request_Attention(in_planes)
        self.spatial_request = Spatial_Request_Attention()
        self.channel_fusion = nn.Conv2d(in_planes * 2, in_planes, 1, bias=False)
        self.spatial_fusion = nn.Conv2d(2, 1, 1, bias=False)
        self.statisticsNetwork = StatisticsNetwork(in_planes * 2)
        self.mutual_loss = DeepInfoMaxLoss()
        self.request_flag = args["request_flag"]

        self.smooth = False
        self.thre = args["thre"]
        if "gaussian_smooth" in args:
            self.smooth = True
            kernel_size = args["gaussian_smooth"]["k_size"]
            self.kernel_size = kernel_size
            c_sigma = args["gaussian_smooth"]["c_sigma"]
            self.gaussian_filter = nn.Conv2d(
                1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2
            )
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False

        x = torch.arange(
            -(kernel_size - 1) // 2, (kernel_size + 1) // 2, dtype=torch.float32
        )
        d1_gaussian_filter = torch.exp(-(x**2) / (2 * c_sigma**2))
        d1_gaussian_filter /= d1_gaussian_filter.sum()

        self.d1_gaussian_filter = d1_gaussian_filter.view(1, 1, kernel_size).cuda()

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
        gaussian_kernel = (
            torch.Tensor(gaussian_kernel)
            .to(self.gaussian_filter.weight.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.bias.data.zero_()

    def forward(self, feat_list, confidence_map_list=None):
        send_feats = []
        comm_rate_list = []
        sparse_mask_list = []
        total_loss = torch.zeros(1).to(feat_list[0].device)
        for bs in range(len(feat_list)):
            agent_feature = feat_list[bs]
            cav_num, C, H, W = agent_feature.shape
            if cav_num == 1:
                send_feats.append(agent_feature)
                ones_mask = torch.ones(cav_num, C, H, W).to(feat_list[0].device)
                sparse_mask_list.append(ones_mask)
                continue

            collaborator_feature = torch.tensor([]).to(agent_feature.device)
            sparse_batch_mask = torch.tensor([]).to(agent_feature.device)

            agent_channel_attention = self.channel_request(agent_feature)
            agent_spatial_attention = self.spatial_request(agent_feature)
            agent_activation = torch.mean(agent_feature, dim=1, keepdims=True).sigmoid()
            agent_activation = self.gaussian_filter(agent_activation)

            ego_channel_request = (1 - agent_channel_attention[0,]).unsqueeze(0)
            ego_spatial_request = (1 - agent_spatial_attention[0,]).unsqueeze(0)

            for i in range(cav_num - 1):
                if self.request_flag:
                    channel_coefficient = self.channel_fusion(
                        torch.cat(
                            [
                                ego_channel_request,
                                agent_channel_attention[i + 1,].unsqueeze(0),
                            ],
                            dim=1,
                        )
                    )
                    spatial_coefficient = self.spatial_fusion(
                        torch.cat(
                            [
                                ego_spatial_request,
                                agent_spatial_attention[i + 1,].unsqueeze(0),
                            ],
                            dim=1,
                        )
                    )
                else:
                    channel_coefficient = agent_channel_attention[i + 1,].unsqueeze(0)
                    spatial_coefficient = agent_spatial_attention[i + 1,].unsqueeze(0)

                spatial_coefficient = spatial_coefficient.sigmoid()
                channel_coefficient = channel_coefficient.sigmoid()
                smoth_channel_coefficient = F.conv1d(
                    channel_coefficient.reshape(1, 1, C),
                    self.d1_gaussian_filter,
                    padding=(self.kernel_size - 1) // 2,
                )
                channel_coefficient = smoth_channel_coefficient.reshape(1, C, 1, 1)

                spatial_coefficient = self.gaussian_filter(spatial_coefficient)
                sparse_matrix = channel_coefficient * spatial_coefficient
                temp_activation = agent_activation[i + 1,].unsqueeze(0)
                sparse_matrix = sparse_matrix * temp_activation

                if self.thre > 0:
                    ones_mask = torch.ones_like(sparse_matrix).to(sparse_matrix.device)
                    zeros_mask = torch.zeros_like(sparse_matrix).to(
                        sparse_matrix.device
                    )
                    sparse_mask = torch.where(
                        sparse_matrix > self.thre, ones_mask, zeros_mask
                    )
                else:
                    K = int(C * H * W * random.uniform(0, 0.3))
                    communication_maps = sparse_matrix.reshape(1, C * H * W)
                    _, indices = torch.topk(communication_maps, k=K, sorted=False)
                    communication_mask = torch.zeros_like(communication_maps).to(
                        communication_maps.device
                    )
                    ones_fill = torch.ones(
                        1,
                        K,
                        dtype=communication_maps.dtype,
                        device=communication_maps.device,
                    )
                    sparse_mask = torch.scatter(
                        communication_mask, -1, indices, ones_fill
                    ).reshape(1, C, H, W)

                comm_rate = sparse_mask.sum() / (C * H * W)
                comm_rate_list.append(comm_rate)

                collaborator_feature = torch.cat(
                    [
                        collaborator_feature,
                        agent_feature[i + 1,].unsqueeze(0) * sparse_mask,
                    ],
                    dim=0,
                )
                sparse_batch_mask = torch.cat([sparse_batch_mask, sparse_mask], dim=0)

            org_feature = agent_feature.clone()
            sparse_feature = torch.cat([agent_feature[:1], collaborator_feature], dim=0)
            send_feats.append(sparse_feature)
            ego_mask = torch.ones_like(agent_feature[:1]).to(agent_feature[:1].device)
            sparse_batch_mask = torch.cat([ego_mask, sparse_batch_mask], dim=0)
            sparse_mask_list.append(sparse_batch_mask)

            org_feature_prime = torch.cat(
                [org_feature[1:], org_feature[0].unsqueeze(0)], dim=0
            )
            local_mutual = self.statisticsNetwork(
                torch.cat([org_feature, sparse_feature], dim=1)
            )
            local_mutual_prime = self.statisticsNetwork(
                torch.cat([org_feature_prime, sparse_feature], dim=1)
            )
            loss = self.mutual_loss(local_mutual, local_mutual_prime)
            total_loss += loss

        if len(comm_rate_list) > 0:
            mean_rate = sum(comm_rate_list) / len(comm_rate_list)
        else:
            mean_rate = torch.tensor(0).to(feat_list[0].device)
        sparse_mask = torch.cat(sparse_mask_list, dim=0)

        return send_feats, total_loss, mean_rate, sparse_mask
