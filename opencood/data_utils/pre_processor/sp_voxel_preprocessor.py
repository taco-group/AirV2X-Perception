# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Transform points to voxels using sparse conv library
"""

import sys

import numpy as np
import torch
from cumm import tensorview as tv

from opencood.data_utils.pre_processor.base_preprocessor import BasePreprocessor
from typing import Sequence, Mapping, Dict
import warnings

# Set up warning filter before defining the warning class
warnings.filterwarnings(
    "once",                      # Show warning only once
    category=UserWarning,        # Match any UserWarning (including subclasses)
    module="sp_voxel_preprocessor",  # Only for warnings from this module
    message="Warning: empty point cloud. Add dummy points.*"  # Match the warning message pattern
)

class EmptyPointCloudWarning(UserWarning):
    pass

class SpVoxelPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super(SpVoxelPreprocessor, self).__init__(preprocess_params, train)
        self.spconv = 1
        try:
            # spconv v1.x
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
        except:
            # spconv v2.x
            from spconv.utils import Point2VoxelCPU3d as VoxelGenerator

            self.spconv = 2
        ego_type = self.params["ego_type"]
        if ego_type == "vehicle":
            self.lidar_range = self.params["cav_lidar_range"]
        elif ego_type == "rsu":
            self.lidar_range = self.params["rsu_lidar_range"]
        elif ego_type == "drone":
            self.lidar_range = self.params["drone_lidar_range"]
        self.voxel_size = self.params["args"]["voxel_size"]
        self.max_points_per_voxel = self.params["args"]["max_points_per_voxel"]

        if train:
            self.max_voxels = self.params["args"]["max_voxel_train"]
        else:
            self.max_voxels = self.params["args"]["max_voxel_test"]

        grid_size = (
            np.array(self.lidar_range[3:6]) - np.array(self.lidar_range[0:3])
        ) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

        # use sparse conv library to generate voxel
        if self.spconv == 1:
            self.voxel_generator = VoxelGenerator(
                voxel_size=self.voxel_size,
                point_cloud_range=self.lidar_range,
                max_num_points=self.max_points_per_voxel,
                max_voxels=self.max_voxels,
            )
        else:
            self.voxel_generator = VoxelGenerator(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.lidar_range,
                max_num_points_per_voxel=self.max_points_per_voxel,
                num_point_features=4,
                max_num_voxels=self.max_voxels,
            )

    def preprocess(self, pcd_np):
        data_dict = {}
        
        # Note: Here we handle the case of empty lidar points (mostly due to system error).
        # Add dummpy point clouds. This will essentially ignore this sample in the training 
        # since we will also mask the gt bbox in this case.
        if len(pcd_np) == 0:
            pcd_np1 = np.zeros((1, 4), dtype=np.float32)
            # Drone mode will remove the dummpy points that is too close to the origin, so we add a dummy point lower.
            pcd_np2 = np.array([-0.218277, -11.13425732, -80.05884552, 1.230595649e-38], dtype=np.float32).reshape(1, 4)
            pcd_np = np.concatenate((pcd_np1, pcd_np2), axis=0)
            # print("Warning: empty point cloud, add dummy points")
            with warnings.catch_warnings():
                warnings.warn("Warning: empty point cloud. Add dummy points. Add dummy points. "
                              "This is because of some package loss during the data collection. "
                              "It will be solved in the future dataset version.", 
                              EmptyPointCloudWarning)
            
        
        if self.spconv == 1:
            voxel_output = self.voxel_generator.generate(pcd_np)
        else:
            pcd_tv = tv.from_numpy(pcd_np)
            voxel_output = self.voxel_generator.point_to_voxel(pcd_tv)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = (
                voxel_output["voxels"],
                voxel_output["coordinates"],
                voxel_output["num_points_per_voxel"],
            )
        else:
            voxels, coordinates, num_points = voxel_output

        if self.spconv == 2:
            voxels = voxels.numpy()
            coordinates = coordinates.numpy()
            num_points = num_points.numpy()

        data_dict["voxel_features"] = voxels
        data_dict["voxel_coords"] = coordinates
        data_dict["voxel_num_points"] = num_points

        return data_dict

    def collate_batch(self, batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list or dict
            List or dictionary.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """

        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            sys.exit("Batch has too be a list or a dictionarn")



    def collate_batch(self, batch):
        """
        Collate function for a list-of-dict or dict-of-list batch.

        Accepted input
        --------------
        list  : [ {'voxel_features': ..., 'voxel_coords': ..., ...}, … ]
        dict  : { 'voxel_features': [arr0, arr1, …], 'voxel_coords': [...] }

        Returns
        -------
        dict with torch.Tensor values.
        """
        if isinstance(batch, Sequence):           # list‐of‐dict
            batch = _transpose_to_dict(batch)
        elif not isinstance(batch, Mapping):      # neither list nor dict
            raise TypeError("batch must be list or dict, got {}".format(type(batch)))

        vf   = torch.from_numpy(np.concatenate(batch["voxel_features"]))
        vnp  = torch.from_numpy(np.concatenate(batch["voxel_num_points"]))

        # add batch-index as leading column in coords (vectorised)
        coords_with_idx = np.concatenate(
            [
                np.hstack((np.full((c.shape[0], 1), i, dtype=c.dtype), c))
                for i, c in enumerate(batch["voxel_coords"])
            ],
            axis=0,
        )
        vc = torch.from_numpy(coords_with_idx)

        return {"voxel_features": vf,
                "voxel_coords":   vc,
                "voxel_num_points": vnp}


    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _transpose_to_dict(batch_list: Sequence[Dict]) -> Dict[str, list]:
        """
        Turn list-of-dict into dict-of-list *without extra copies*.
        """
        keys = batch_list[0].keys()
        out  = {k: [] for k in keys}
        for sample in batch_list:
            for k in keys:
                out[k].append(sample[k])
        return out