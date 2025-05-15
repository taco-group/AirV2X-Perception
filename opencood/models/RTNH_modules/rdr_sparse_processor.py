# '''
# * Copyright (c) AVELab, KAIST. All rights reserved.
# * author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
# * e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
# '''

import torch
import torch.nn as nn
from spconv.pytorch.utils import PointToVoxel


class RadarSparseProcessor(nn.Module):
    def __init__(self, in_channels=4, out_channel=64):
        super().__init__()
        x_min, y_min, z_min, x_max, y_max, z_max = [-140.8, -40, -3, 140.8, 40, 1]
        self.min_roi = [x_min, y_min, z_min]
        self.grid_size = 0.4
        self.input_dim = in_channels
        self.simplified_pointnet = nn.Linear(self.input_dim, out_channel, bias=False)
        self.pooling_method = "max"

        max_vox_percentage = 0.25
        x_size = int(round((x_max - x_min) / self.grid_size))
        y_size = int(round((y_max - y_min) / self.grid_size))
        z_size = int(round((z_max - z_min) / self.grid_size))

        max_num_vox = int(x_size * y_size * z_size * max_vox_percentage)

        self.gen_voxels = PointToVoxel(
            vsize_xyz=[self.grid_size, self.grid_size, self.grid_size],
            coors_range_xyz=[-140.8, -40, -3, 140.8, 40, 1],
            num_point_features=self.input_dim,
            max_num_voxels=max_num_vox,
            max_num_points_per_voxel=4,
            device=torch.device("cuda"),
        )

    def forward(self, dict_item):
        voxel_features, voxel_num_points, voxel_coords = (
            dict_item["voxel_features"],
            dict_item["voxel_num_points"],
            dict_item["voxel_coords"],
        )

        voxel_features = self.simplified_pointnet(voxel_features)
        if self.pooling_method == "max":
            voxel_features = torch.max(voxel_features, dim=1, keepdim=False)[0]
        elif self.pooling_method == "mean":
            voxel_features = voxel_features.sum(dim=1, keepdim=False)
            normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(
                voxel_features
            )
            voxel_features = voxel_features / normalizer
        else:
            voxel_features = voxel_features.sum(dim=1, keepdim=False)
            normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(
                voxel_features
            )
            voxel_features = voxel_features / normalizer

        dict_item["spatial_features"] = voxel_features.contiguous()
        dict_item["spatial_indices"] = voxel_coords.int()
        dict_item["batch_size"] = voxel_coords[:, 0].max().int().item() + 1
        return dict_item
