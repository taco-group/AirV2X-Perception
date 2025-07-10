# -*- coding: utf-8 -*-
# Author: Binyu Zhao <byzhao@stu.hit.edu.cn>
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.datasets.dair.early_fusion_dataset import (
    EarlyFusionDatasetDAIR,
)
from opencood.data_utils.datasets.dair.intermediate_fusion_dataset import (
    IntermediateFusionDatasetDAIR,
)
from opencood.data_utils.datasets.dair.intermediate_fusion_dataset_multi_frame import (
    IntermediateFusionDatasetDAIR as IntermediateFusionDatasetDAIR_MULTI,
)
from opencood.data_utils.datasets.dair.late_fusion_dataset import LateFusionDatasetDAIR
from opencood.data_utils.datasets.dair.lidar_camera_intermediate_fusion_dataset import (
    LiDARCameraIntermediateFusionDatasetDAIR,
)
from opencood.data_utils.datasets.dair.lidar_camera_intermediate_fusion_dataset_v2 import (
    LiDARCameraIntermediateFusionDatasetDAIR as LiDARCameraIntermediateFusionDatasetDAIR_V2,
)
from opencood.data_utils.datasets.opv2v.early_fusion_dataset import (
    EarlyFusionDataset as EarlyFusionDatasetOPV2V,
)
from opencood.data_utils.datasets.opv2v.intermediate_fusion_dataset import (
    IntermediateFusionDataset as IntermediateFusionDatasetV2XR,
)
from opencood.data_utils.datasets.opv2v.intermediate_fusion_dataset_multi_frame import (
    IntermediateFusionDataset as IntermediateFusionDatasetV2XR_MULTI,
)
from opencood.data_utils.datasets.opv2v.intermediate_fusion_dataset_multi_frame_how2comm import (
    IntermediateFusionDataset as IntermediateFusionDatasetOPV2V_MULTI_HOW2COMM,
)
from opencood.data_utils.datasets.opv2v.intermediate_fusion_dataset_v2 import (
    IntermediateFusionDatasetV2 as IntermediateFusionDatasetOPV2V_V2,
)
from opencood.data_utils.datasets.opv2v.intermediate_fusion_sicp_dataset import (
    IntermediateFusionSicpDataset as IntermediateFusionSicpDatasetV2XR,
)
from opencood.data_utils.datasets.opv2v.late_fusion_dataset import (
    LateFusionDataset as LateFusionDatasetOPV2V,
)
from opencood.data_utils.datasets.opv2v.lidar_camera_intermediate_fusion_dataset import (
    LiDARCameraIntermediateFusionDataset as LiDARCameraIntermediateFusionDatasetOPV2V,
)
from opencood.data_utils.datasets.opv2v.lidar_camera_intermediate_fusion_dataset_v2 import (
    LiDARCameraIntermediateFusionDataset as LiDARCameraIntermediateFusionDatasetOPV2V_V2,
)
from opencood.data_utils.datasets.airv2x.early_fusion_dataset import (
    EarlyFusionDatasetAirv2x,
)
from opencood.data_utils.datasets.airv2x.intermediate_fusion_dataset import (
    IntermediateFusionDatasetAirv2x,
)
from opencood.data_utils.datasets.airv2x.intermediate_fusion_dataset_bm2cp import (
    IntermediateFusionDatasetAirv2xBM2CP,
)
from opencood.data_utils.datasets.airv2x.intermediate_fusion_dataset_sicp import (
    IntermediateFusionDatasetAirv2xSiCP,
)

__all__ = {
    "EarlyFusionDatasetOPV2V": EarlyFusionDatasetOPV2V,
    "IntermediateFusionDatasetV2XR": IntermediateFusionDatasetV2XR,
    "IntermediateFusionDatasetOPV2V_V2": IntermediateFusionDatasetOPV2V_V2,
    "IntermediateFusionDatasetV2XR_Multi": IntermediateFusionDatasetV2XR_MULTI,
    "IntermediateFusionDatasetOPV2V_Multi_How2comm": IntermediateFusionDatasetOPV2V_MULTI_HOW2COMM,
    "IntermediateFusionSicpDatasetV2XR": IntermediateFusionSicpDatasetV2XR,
    "LateFusionDatasetOPV2V": LateFusionDatasetOPV2V,
    "LiDARCameraIntermediateFusionDatasetOPV2V": LiDARCameraIntermediateFusionDatasetOPV2V,
    "LiDARCameraIntermediateFusionDatasetOPV2V_V2": LiDARCameraIntermediateFusionDatasetOPV2V_V2,
    "EarlyFusionDatasetDAIR": EarlyFusionDatasetDAIR,
    "IntermediateFusionDatasetDAIR": IntermediateFusionDatasetDAIR,
    "IntermediateFusionDatasetDAIR_Multi": IntermediateFusionDatasetDAIR_MULTI,
    "LateFusionDatasetDAIR": LateFusionDatasetDAIR,
    "LiDARCameraIntermediateFusionDatasetDAIR": LiDARCameraIntermediateFusionDatasetDAIR,
    "LiDARCameraIntermediateFusionDatasetDAIR_V2": LiDARCameraIntermediateFusionDatasetDAIR_V2,
    "EarlyFusionDatasetAirv2x": EarlyFusionDatasetAirv2x,
    "IntermediateFusionDatasetAirv2x": IntermediateFusionDatasetAirv2x,
    "IntermediateFusionDatasetAirv2xBM2CP": IntermediateFusionDatasetAirv2xBM2CP,
    "IntermediateFusionDatasetAirv2xSiCP": IntermediateFusionDatasetAirv2xSiCP,
}

# the final range for evaluation
GT_RANGE_OPV2V = [-140, -40, -3, 140, 40, 1]
GT_RANGE_V2XSIM = [-32, -32, -3, 32, 32, 1]
GT_RANGE_SKYLINK = [-140.8, -40, -3, 140.8, 40, 1]  # TODO(YH): check with xiangbo
# The communication range for cavs
VEH_COM_RANGE = 120
RSU_COM_RANGE = 120
DRONE_COM_RANGE = 180


def build_dataset(dataset_cfg, visualize=False, train=True):
    dataset_name = dataset_cfg["fusion"]["core_method"]
    error_message = (
        f"{dataset_name} is not found. "
        f"Please add your processor file's name in opencood/"
        f"data_utils/datasets/init.py"
    )

    dataset = __all__[dataset_name](
        params=dataset_cfg, visualize=visualize, train=train
    )

    return dataset
