# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn

from opencood.models.common_modules.att_bev_backbone import AttBEVBackbone

# from opencood.models.sub_modules.pillar_vfe import PillarVFE
# from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
# from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone
from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter


class PointPillarIntermediateLRF(nn.Module):
    def __init__(self, args):
        super(PointPillarIntermediateLRF, self).__init__()

        # PIllar VFE

        self.lidar_pillar_vfe = PillarVFE(
            args["pillar_vfe"],
            num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=args["lidar_range"],
        )
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])
        self.radar_pillar_vfe = PillarVFE(
            args["pillar_vfe"],
            num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=args["lidar_range"],
        )
        self.backbone = AttBEVBackbone(args["base_bev_backbone"], 128)

        self.cls_head = nn.Conv2d(128 * 3, args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args["anchor_num"], kernel_size=1)

    def forward(self, data_dict):
        lidar_voxel_features = data_dict["processed_lidar"]["voxel_features"]
        lidar_voxel_coords = data_dict["processed_lidar"]["voxel_coords"]
        lidar_voxel_num_points = data_dict["processed_lidar"]["voxel_num_points"]
        record_len = data_dict["record_len"]

        lidar_batch_dict = {
            "voxel_features": lidar_voxel_features,
            "voxel_coords": lidar_voxel_coords,
            "voxel_num_points": lidar_voxel_num_points,
            "record_len": record_len,
        }

        radar_voxel_features = data_dict["processed_radar"]["voxel_features"]
        radar_voxel_coords = data_dict["processed_radar"]["voxel_coords"]
        radar_voxel_num_points = data_dict["processed_radar"]["voxel_num_points"]
        record_len = data_dict["record_len"]

        radar_batch_dict = {
            "voxel_features": radar_voxel_features,
            "voxel_coords": radar_voxel_coords,
            "voxel_num_points": radar_voxel_num_points,
            "record_len": record_len,
        }

        lidar_batch_dict = self.lidar_pillar_vfe(lidar_batch_dict)
        lidar_batch_dict = self.scatter(lidar_batch_dict)

        radar_batch_dict = self.radar_pillar_vfe(radar_batch_dict)
        radar_batch_dict = self.scatter(radar_batch_dict)

        batch_dict = {
            "spatial_features": torch.cat(
                [
                    lidar_batch_dict["spatial_features"],
                    radar_batch_dict["spatial_features"],
                ],
                dim=1,
            ),
            "record_len": record_len,
        }

        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict["spatial_features_2d"]

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)
        output_dict = {"psm": psm, "rm": rm}

        return output_dict
