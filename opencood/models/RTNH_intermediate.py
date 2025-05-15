import torch
import torch.nn as nn

from opencood.models.RTNH_modules.rdr_sp_pw import RadarSparseBackbone
from opencood.models.RTNH_modules.rdr_sparse_processor import RadarSparseProcessor
from opencood.models.RTNH_modules.rdr_spcube_head import RdrSpcubeHead


class RTNHIntermediate(nn.Module):
    def __init__(self, cfg):
        super(RTNHIntermediate, self).__init__()
        self.cfg = cfg

        # 直接初始化各模块，并传入配置文件 cfg
        self.pre_processor = RadarSparseProcessor()
        self.backbone = RadarSparseBackbone(cfg)
        self.head = RdrSpcubeHead(cfg)

        # # 从head中拆分出分类和回归头
        # self.cls_head = self.head.cls_head
        # self.reg_head = self.head.reg_head

    def forward(self, data_dict):
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

        # radar_batch_dict = self.radar_pillar_vfe(radar_batch_dict)
        # radar_batch_dict = self.scatter(radar_batch_dict)
        radar_batch_dict = self.pre_processor(radar_batch_dict)
        batch_dict = {
            "sp_features": radar_batch_dict["spatial_features"],
            "sp_indices": radar_batch_dict["spatial_indices"],
            "batch_size": radar_batch_dict["batch_size"],
            "record_len": record_len,
        }

        # batch_dict = self.backbone(batch_dict)
        batch_dict = self.backbone(batch_dict)

        # bev_features = batch_dict['bev_feat'] # dict_item['bev_feat'] = bev_features # B, C, Y, X

        output_dict = self.head(batch_dict)

        # psm = self.cls_head(spatial_features_2d)
        # rm = self.reg_head(spatial_features_2d)
        # output_dict = {'psm': psm, 'rm': rm}

        return output_dict
