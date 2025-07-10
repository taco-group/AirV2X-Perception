# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# Modifier: Yuheng Wu <yuhengwu@kaist.ac.kr>, Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for intermediate fusion for bm2cp

It will project the depth from lidar to image and pad noncover area with predicted depth
"""

import bisect
import math
import random
import time
import warnings
from collections import OrderedDict
from itertools import chain
from more_itertools import unique_everseen

import matplotlib
import numpy as np
import torch
import cv2
from collections import defaultdict

import opencood.data_utils.datasets
import opencood.data_utils.post_processor as post_processor
from opencood.data_utils.datasets.airv2x import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import (
    points_in_boxes_cpu,
)
from opencood.utils import (
    box_utils,
    camera_utils,
    pcd_utils,
    sensor_transformation_utils,
    airv2x_utils,
    transformation_utils,
)
from opencood.utils.pcd_utils import (
    downsample_lidar_minimum,
    mask_ego_points,
    mask_points_by_range,
    shuffle_points,
)
from opencood.utils.transformation_utils import x1_to_x2

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev


class IntermediateFusionDatasetAirv2x(basedataset.BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """

    def __init__(self, params, visualize, train=True):
        super(IntermediateFusionDatasetAirv2x, self).__init__(params, visualize, train)

        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        if "proj_first" in params["fusion"]["args"]:
            self.proj_first = params["fusion"]["args"]["proj_first"]
        # self.proj_first = False
        print("proj_first: ", self.proj_first)

        self.N = 0  # used for debug depth map

        # which sensors to use, can be cam_only, lidar_only and cam+lidar
        self.use_cam = True if "cam" in params["active_sensors"] else False
        self.use_lidar = True if "lidar" in params["active_sensors"] else False

        # which device to use
        self.use_rsu = True if "rsu" in params["collaborators"] else False
        self.use_drone = True if "drone" in params["collaborators"] else False

        self.collaborators = params["collaborators"]
        self.active_sensors = params["active_sensors"]
        # get max number of CAVs regardless of the type
        self.max_cav_num = 0
        for collaborator in self.collaborators:
            self.max_cav_num += params["train_params"]["max_cav"][collaborator]

        self.fov = params["bevcam_fov"]

        self.veh_data_aug_conf = params["fusion"]["args"]["veh_data_aug_conf"]
        self.rsu_data_aug_conf = params["fusion"]["args"]["rsu_data_aug_conf"]
        self.drone_data_aug_conf = params["fusion"]["args"]["drone_data_aug_conf"]
        
        # self.veh_grid_conf = params["fusion"]["args"]["grid_conf"]
        # self.depth_discre = camera_utils.depth_discretization(
        #     self.grid_conf["ddiscr"][0],
        #     self.grid_conf["ddiscr"][1],
        #     self.grid_conf["ddiscr"][2],
        #     self.grid_conf["mode"],
        # )

        # if self.use_cam:
        #     self.data_aug_conf_cam = params["image_modality"]["data_aug_conf"]

        # if self.use_lidar:
        #     self.data_aug_conf_lidar = params["lidar_modality"]["data_aug_conf"]

        self.training = train

        # whether there is a time delay between the time that cav project
        # lidar to ego and the ego receive the delivered feature
        self.cur_ego_pose_flag = (
            True
            if "cur_ego_pose_flag" not in params["fusion"]["args"]
            else params["fusion"]["args"]["cur_ego_pose_flag"]
        )

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = post_processor.build_postprocessor(
            params["postprocess"], dataset="airv2x", train=train
        )
        
        if self.ego_type == "vehicle":
            self.agent_order = ["vehicle", "rsu", "drone"]
        elif self.ego_type == "rsu":
            self.agent_order = ["rsu", "vehicle", "drone"]
        elif self.ego_type == "drone":
            self.agent_order = ["drone", "vehicle", "rsu"]
            
    
    def __getitem__(self, idx):
        # Default order if none specified
        agent_order = self.agent_order
        
        base_data_dict, scenario_index, timestamp_key = self.retrieve_base_data(
            idx, cur_ego_pos_flag=self.cur_ego_pose_flag
        )

        processed_data_dict = OrderedDict()
        processed_data_dict["ego"] = {}

        # Find ego vehicle
        ego_id = -1
        ego_lidar_pose = []
        for cav_id, cav_content in base_data_dict.items():
            if cav_content["ego"]:
                ego_id = cav_id
                ego_lidar_pose = cav_content["params"]["delay_ego_lidar_pose"]
                break
        processed_data_dict["ego"].update({"ego_id": ego_id})

        # Define agent configurations
        agent_configs = {
            "vehicle": {
                "max_cav": self.max_cav_veh,
                "com_range": opencood.data_utils.datasets.VEH_COM_RANGE,
                "infra_value": 0,
            },
            "rsu": {
                "max_cav": self.max_cav_rsu,
                "com_range": opencood.data_utils.datasets.RSU_COM_RANGE,
                "infra_value": 1,
            },
            "drone": {
                "max_cav": self.max_cav_drone, 
                "com_range": opencood.data_utils.datasets.DRONE_COM_RANGE,
                "infra_value": 1,
            }
        }
        
        # Initialize data storage for each agent type
        agent_data = {}
        for agent_type in agent_configs:
            agent_data[agent_type] = {
                **agent_configs[agent_type],
                "object_stack": [],
                "object_id_stack": [],
                "class_id_stack": [],
                "velocity": [],
                "time_delay": [],
                "distance_to_ego": [],
                "spatial_correction_matrix": [],
                "processed_lidar_features": [],
                "cam_inputs": [],
                "original_lidar_vis": [],
                "num_agents": 0,
                "retain_ids": []
            }

        dynamic_seg_label = None
        static_seg_label = None
        metadata_path = None

        # Collect data for each agent
        for cav_id, selected_cav_base in base_data_dict.items():
            agent_type = selected_cav_base["agent_type"]
            distance = selected_cav_base["distance_to_ego"]
            
            # Skip if out of communication range
            if distance > agent_data[agent_type]["com_range"]:
                continue
                
            selected_cav_processed = self.get_item_single_car(
                selected_cav_base, ego_lidar_pose
            )
            
            # Add data to the appropriate agent collection
            current_agent = agent_data[agent_type]
            
            current_agent["object_stack"].append(selected_cav_processed["object_bbx_center"])
            current_agent["object_id_stack"].append(selected_cav_processed["object_ids"])
            current_agent["class_id_stack"].append(selected_cav_processed["class_ids"])
            current_agent["velocity"].append(selected_cav_base["params"]["odometry"]["ego_speed"] / 30.0)
            current_agent["time_delay"].append(float(selected_cav_base["time_delay"]))
            current_agent["distance_to_ego"].append((cav_id, distance))
            current_agent["spatial_correction_matrix"].append(selected_cav_base["params"]["spatial_correction_matrix"])
            current_agent["processed_lidar_features"].append(selected_cav_processed["processed_lidar_features"])
            current_agent["cam_inputs"].append(selected_cav_processed["cam_inputs"])
            current_agent["original_lidar_vis"].append(selected_cav_processed["lidar_np"])
            
            # Get seg_label from ego vehicle
            if selected_cav_base["ego"]:
                metadata_path = selected_cav_base["metadata_path"]
                metadata_path = selected_cav_base["metadata_path"]
                dynamic_seg_label = selected_cav_base["dynamic_seg_label"]
                static_seg_label = selected_cav_base["static_seg_label"]
        
        # Common data collectors for all agent types
        all_object_stack = []
        all_object_id_stack = []
        all_class_id_stack = []
        all_velocity = []
        all_time_delay = []
        all_infra = []
        all_spatial_correction_matrix = []
        retain_cav_ids = []
        
        # Process each agent type in specified order
        for agent_type in agent_order:
            current_agent = agent_data[agent_type]
            
            # Skip if no agents of this type
            if not current_agent["distance_to_ego"]:
                current_agent["num_agents"] = 0
                current_agent["merged_lidar_features_dict"] = {}
                current_agent["merged_cam_inputs_dict"] = {}
                continue
            
            # Filter to closest agents
            indices, retain_ids = airv2x_utils.get_smallest_k_indices(current_agent["distance_to_ego"], current_agent["max_cav"])
            
            # Apply filtering to all data fields
            for field in ["object_stack", "object_id_stack", "class_id_stack", 
                        "velocity", "time_delay", "spatial_correction_matrix",
                        "processed_lidar_features", "cam_inputs", "original_lidar_vis"]:
                current_agent[field] = [current_agent[field][i] for i in indices]
                if field in ['object_stack', 'object_id_stack', 'class_id_stack']:
                    current_agent[field] = list(chain.from_iterable(current_agent[field]))
            
            # Add to combined data collections in requested order
            all_object_stack.extend(current_agent["object_stack"])
            all_object_id_stack.extend(current_agent["object_id_stack"])
            all_class_id_stack.extend(current_agent["class_id_stack"])
            all_velocity.extend(current_agent["velocity"])
            all_time_delay.extend(current_agent["time_delay"])
            all_infra.extend([current_agent["infra_value"]] * len(indices))
            all_spatial_correction_matrix.extend(current_agent["spatial_correction_matrix"])
            
            retain_cav_ids.extend(retain_ids)
            current_agent["retain_ids"] = retain_ids
            current_agent["num_agents"] = len(retain_ids)
            
            # Merge features for this agent type if available
            if current_agent["num_agents"] > 0:
                current_agent["merged_lidar_features_dict"] = self.merge_features_to_dict(
                    current_agent["processed_lidar_features"], None, "lidar"
                )
                current_agent["merged_cam_inputs_dict"] = self.merge_features_to_dict(
                    current_agent["cam_inputs"], "stack", "cam"
                )
            else:
                current_agent["merged_lidar_features_dict"] = {}
                current_agent["merged_cam_inputs_dict"] = {}
        
        # Ensure unique IDs (preserve order)
        retain_cav_ids = list(unique_everseen(retain_cav_ids))
        
        # Get pairwise transformation matrix
        retain_base_data_dict = OrderedDict(
            (cav_id, base_data_dict[cav_id]) for cav_id in retain_cav_ids
        )
        
        pairwise_t_matrix_collab, img_pairwise_t_matrix_collab = self.get_pairwise_transformation(
            retain_base_data_dict, self.max_cav_num, self.collaborators
        )
        
        # Pad to max_cav_num
        all_velocity = all_velocity + (self.max_cav_num - len(all_velocity)) * [0.0]
        all_time_delay = all_time_delay + (self.max_cav_num - len(all_time_delay)) * [0.0]
        all_infra = all_infra + (self.max_cav_num - len(all_infra)) * [0.0]
        
        # Process spatial correction matrix
        try:
            all_spatial_correction_matrix = np.stack(all_spatial_correction_matrix)
        except:
            import traceback; traceback.print_exc()
            import pdb; pdb.set_trace()
        padding_eye = np.tile(
            np.eye(4)[None], (self.max_cav_num - len(all_spatial_correction_matrix), 1, 1)
        )
        all_spatial_correction_matrix = np.concatenate(
            [all_spatial_correction_matrix, padding_eye], axis=0
        )
        
        # Process object data (remove duplicates)
        unique_elements = list(unique_everseen(all_object_id_stack))
        unique_indices = [all_object_id_stack.index(x) for x in unique_elements]
        
        try:
            
            if len(all_object_stack):
                object_stack = np.vstack(all_object_stack)
            else:
                object_stack = np.zeros((0, 7))
        except:
            import traceback; traceback.print_exc()
            import pdb; pdb.set_trace()
        
        object_stack = object_stack[unique_indices]
        
        ret_class_ids = all_class_id_stack
        all_class_id_stack = np.array(all_class_id_stack)
        all_class_id_stack = all_class_id_stack[unique_indices]
        
        # Create padded object boxes and mask
        object_bbx_center = np.zeros((self.params["postprocess"]["max_num"], 7))
        mask = np.zeros(self.params["postprocess"]["max_num"])
        object_bbx_center[: object_stack.shape[0], :] = object_stack
        mask[: object_stack.shape[0]] = 1
        
        # Pad class IDs
        class_ids_padded = np.full(self.params["postprocess"]["max_num"], 0)
        class_ids_padded[: object_stack.shape[0]] = all_class_id_stack
        
        # Generate labels
        anchor_box = self.post_processor.generate_anchor_box()
        label_dict = self.post_processor.generate_label_airv2x(
            gt_box_center=object_bbx_center,
            anchors=anchor_box,
            mask=mask,
            class_ids_padded=class_ids_padded,
        )
        
        assert dynamic_seg_label is not None, "seg_label should not be None"
        assert static_seg_label is not None, "seg_label should not be None"
        
        # Build final output dictionary
        total_agents = sum(agent_data[agent_type]["num_agents"] for agent_type in agent_order)
        
        processed_data_dict["ego"].update({
            # Common data
            "object_bbx_center": object_bbx_center,
            "object_bbx_mask": mask,
            "object_ids": [all_object_id_stack[i] for i in unique_indices],
            "class_ids": [ret_class_ids[i] for i in unique_indices],
            "dynamic_seg_label": dynamic_seg_label,
            "static_seg_label": static_seg_label,
            "anchor_box": anchor_box,
            "num_cavs": total_agents,
            "label_dict": label_dict,
            "pairwise_t_matrix_collab": pairwise_t_matrix_collab,
            "img_pairwise_t_matrix_collab": img_pairwise_t_matrix_collab,
            "spatial_correction_matrix": all_spatial_correction_matrix,
            "velocity": all_velocity,
            "time_delay": all_time_delay,
            "infra": all_infra,
            "scenario_index": scenario_index,
            "timestamp_key": timestamp_key,
            "metadata_path": metadata_path,
            "ego_lidar_pose": ego_lidar_pose,
        })
        
        # Add agent-specific data
        for agent_type in agent_order:
            current_agent = agent_data[agent_type]
            
            type_abb_map = {
                "vehicle": "veh",
                "rsu": "rsu",
                "drone": "drone"
            }
            # Add count and data lists
            processed_data_dict["ego"][f"num_{type_abb_map[agent_type]}"] = current_agent["num_agents"]
            processed_data_dict["ego"][f"processed_lidar_features_{type_abb_map[agent_type]}_list"] = current_agent["processed_lidar_features"]
            processed_data_dict["ego"][f"merged_lidar_features_dict_{type_abb_map[agent_type]}"] = current_agent["merged_lidar_features_dict"]
            processed_data_dict["ego"][f"cam_inputs_{type_abb_map[agent_type]}"] = current_agent["cam_inputs"]
            processed_data_dict["ego"][f"merged_cam_inputs_dict_{type_abb_map[agent_type]}"] = current_agent["merged_cam_inputs_dict"]
            processed_data_dict["ego"][f"original_lidar_vis_{type_abb_map[agent_type]}_list"] = current_agent["original_lidar_vis"]
            
            # Add concatenated lidar data if available
            if current_agent["num_agents"] > 0:
                processed_data_dict["ego"][f"origin_lidar_{type_abb_map[agent_type]}"] = np.concatenate(
                    current_agent["original_lidar_vis"], axis=0
                )
            else:
                processed_data_dict["ego"][f"origin_lidar_{type_abb_map[agent_type]}"] = np.zeros((0, 4))
        
        return processed_data_dict


    def mask_ego_fov_flag(self, selected_cav_base, lidar, ego_params):
        """
        Args:
            lidar: lidar point clouds in ego lidar pose
            ego_params : epo params
        Returns:
            mask of fov lidar point clouds <<in ego coords>>
        """
        xyz = lidar[:, :3]

        xyz_hom = np.concatenate(
            [xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1
        )

        # print(ego_params['camera0'])
        intrinsic = np.array(ego_params["camera0"]["intrinsic"])
        # img_shape = selected_cav_base['camera0'].shape
        # 如果selected_cav_base['camera0']是图像对象，使用size获取图像尺寸
        img_size = selected_cav_base["camera0"].size  # 返回 (宽度, 高度)
        img_shape = (img_size[1], img_size[0])  # 转换为 (高度, 宽度)
        ext_matrix = ego_params["c2e_transformation_matrix"]
        ext_matrix = np.linalg.inv(ext_matrix)[:3, :4]
        img_pts = (intrinsic @ ext_matrix @ xyz_hom.T).T
        depth = img_pts[:, 2]
        uv = img_pts[:, :2] / depth[:, None]

        val_flag_1 = np.logical_and(uv[:, 0] >= 0, uv[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(uv[:, 1] >= 0, uv[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(depth > 0, val_flag_merge)
        return lidar[pts_valid_flag]

    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.
        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        # selected_cav_base: ['ego', 'agent_type', 'distance_to_ego', 'params', 'cameras', 'lidar_np'])
        # rsu/vehicle, drone different in `params` and 'cameras'
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = selected_cav_base["params"][
            "transformation_matrix"
        ]  # cav lidar/drone -> ego lidar

        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids, class_ids = (
            self.post_processor.generate_object_center_airv2x(
                [selected_cav_base], ego_pose
            )
        )
        
        agent_type = selected_cav_base["agent_type"]
        selected_cav_processed.update({"agent_type": agent_type})
        
        camera_data_list = selected_cav_base["cameras"]
        depth_data_list = selected_cav_base.get("depth", [])
        params = selected_cav_base["params"]

        N = len(camera_data_list)
        camera_to_lidar_matrix = params["delay_extrinsic"].reshape(N, 4, 4)
        camera_intrinsics = params["delay_intrinsic"].reshape(N, 3, 3)

        post_trans = torch.zeros(N, 3, dtype=torch.float32)
        post_rots = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)

        imgs = []
        rots = []
        trans = []
        intrins = []
        extrinsics = []
        post_rots = []
        post_trans = []

        for idx, img in enumerate(camera_data_list):
            camera_to_lidar = camera_to_lidar_matrix[idx]
            camera_to_lidar = camera_utils.ue4_to_lss(camera_to_lidar)
            camera_intrinsic = camera_intrinsics[idx]
            intrin = torch.from_numpy(camera_intrinsic)
            rot = torch.from_numpy(camera_to_lidar[:3, :3])  # R_wc, we consider world-coord is the lidar-coord
            tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            if depth_data_list:
                depth_img = camera_utils.decode_depth_carla(depth_data_list[idx], to_PIL=True)
                # import pdb; pdb.set_trace()
                img_src = [img, depth_img]
            else:
                img_src = [img]
                
            
            if agent_type == "vehicle":
                resize, resize_dims, crop, flip, rotate = camera_utils.sample_augmentation(
                    self.veh_data_aug_conf, self.train
                )
            elif agent_type == "rsu":
                resize, resize_dims, crop, flip, rotate = camera_utils.sample_augmentation(
                    self.rsu_data_aug_conf, self.train
                )
            elif agent_type == "drone":
                resize, resize_dims, crop, flip, rotate = camera_utils.sample_augmentation(
                    self.drone_data_aug_conf, self.train
                )
            
            img_src, post_rot2, post_tran2 = camera_utils.img_transform(
                img_src,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            
            
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            img_src[0] = camera_utils.normalize_img(img_src[0])
            if depth_data_list:
                img_src[1] = camera_utils.pil_depth_to_tensor(img_src[-1]).unsqueeze(0)
                

            imgs.append(torch.cat(img_src, dim=0))
            intrins.append(intrin)
            extrinsics.append(torch.from_numpy(camera_to_lidar))
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            
        selected_cav_processed.update(
            {
            "cam_inputs": 
                {
                    "imgs": torch.stack(imgs), # [Ncam, 3or4, H, W]
                    "intrinsics": torch.stack(intrins),
                    "extrinsics": torch.stack(extrinsics),
                    "rots": torch.stack(rots),
                    "trans": torch.stack(trans),
                    "post_rots": torch.stack(post_rots),
                    "post_trans": torch.stack(post_trans),
                }
            }
        )


        
        # if agent_type == "drone":
        #     lidar_np = []
        #     processed_lidar = []
        # else:
        lidar_np = selected_cav_base["lidar_np"]
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        if self.proj_first:
            lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(
                lidar_np[:, :3], transformation_matrix
            )
        lidar_np = mask_points_by_range(
            lidar_np, self.params["preprocess"]["cav_lidar_range"]
        )
        
        # Note: Here we handle the case of empty lidar points (mostly due to system error).
        # No supervision under such case.
        if len(lidar_np) == 0:
            object_bbx_mask = np.zeros_like(object_bbx_mask)
            
        processed_lidar = self.pre_processor.preprocess(lidar_np)

        selected_cav_processed.update(
            {
                "lidar_np": lidar_np,
                "processed_lidar_features": processed_lidar,
                "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                "object_ids": object_ids,
                "class_ids": class_ids,
            }
        )
        return selected_cav_processed

    def collate_batch_train(self, batch):
        # Intermediate fusion is different the other two
        output_dict = {"ego": {}}

        # commom
        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        class_ids = []
        record_len = []
        record_len_veh = []
        record_len_rsu = []
        record_len_drone = []
        label_dict_list = []
        anchor_box_list = []
        pairwise_t_matrix_collab_list = []
        img_pairwise_t_matrix_collab_list = []
        dynamic_seg_label_list = []
        static_seg_label_list = []

        # used for prior encoding
        velocity = []
        time_delay = []
        infra = []

        # used to correct time delay
        spatial_correction_matrix_list = []

        # veh
        processed_lidar_features_veh_lists = []
        merged_lidar_features_dict_veh_list = []
        cam_inputs_veh_list = []
        merged_cam_inputs_dict_veh_list = []
        batch_idxs_veh = []

        # rsu
        processed_lidar_features_rsu_lists = []
        merged_lidar_features_dict_rsu_list = []
        cam_inputs_rsu_list = []
        merged_cam_inputs_dict_rsu_list = []
        batch_idxs_rsu = []

        # drone
        processed_lidar_features_drone_lists = []
        merged_lidar_features_dict_drone_list = []
        cam_inputs_drone_list = []
        merged_cam_inputs_dict_drone_list = []
        batch_idxs_drone = []

        # vis
        origin_lidar_veh_list = []
        origin_lidar_rsu_list = []
        origin_lidar_drone_list = []
        
        # metadata
        scenario_index_list = []
        timestamp_key_list = []
        metadata_path_list = []
        ego_lidar_pose_list = []

        for i in range(len(batch)):
            ego_dict = batch[i]["ego"]
            object_bbx_center.append(ego_dict["object_bbx_center"])
            object_bbx_mask.append(ego_dict["object_bbx_mask"])
            object_ids.append(ego_dict["object_ids"])
            class_ids.append(ego_dict["class_ids"])
            dynamic_seg_label_list.append(ego_dict["dynamic_seg_label"])
            static_seg_label_list.append(ego_dict["static_seg_label"])
            anchor_box_list.append(ego_dict["anchor_box"])
            record_len.append(ego_dict["num_cavs"])
            record_len_veh.append(ego_dict["num_veh"])
            record_len_rsu.append(ego_dict["num_rsu"])
            record_len_drone.append(ego_dict["num_drone"])
            label_dict_list.append(ego_dict["label_dict"])
            pairwise_t_matrix_collab_list.append(ego_dict["pairwise_t_matrix_collab"])
            img_pairwise_t_matrix_collab_list.append(
                ego_dict["img_pairwise_t_matrix_collab"]
            )

            velocity.append(ego_dict["velocity"])
            time_delay.append(ego_dict["time_delay"])
            infra.append(ego_dict["infra"])
            spatial_correction_matrix_list.append(ego_dict["spatial_correction_matrix"])

            # veh
            if ego_dict["num_veh"] > 0:
                processed_lidar_features_veh_lists.append(
                    ego_dict["processed_lidar_features_veh_list"]
                )
                merged_lidar_features_dict_veh_list.append(
                    ego_dict["merged_lidar_features_dict_veh"]
                )
                cam_inputs_veh_list.append(ego_dict["cam_inputs_veh"])
                merged_cam_inputs_dict_veh_list.append(
                    ego_dict["merged_cam_inputs_dict_veh"]
                )
                batch_idxs_veh.append(i)
                
            # rsu
            if ego_dict["num_rsu"] > 0:
                processed_lidar_features_rsu_lists.append(
                    ego_dict["processed_lidar_features_rsu_list"]
                )
                merged_lidar_features_dict_rsu_list.append(
                    ego_dict["merged_lidar_features_dict_rsu"]
                )
                cam_inputs_rsu_list.append(ego_dict["cam_inputs_rsu"])
                merged_cam_inputs_dict_rsu_list.append(
                    ego_dict["merged_cam_inputs_dict_rsu"]
                )
                batch_idxs_rsu.append(i)
                
            # drone
            if ego_dict["num_drone"] > 0:
                processed_lidar_features_drone_lists.append(
                    ego_dict["processed_lidar_features_drone_list"]
                )
                merged_lidar_features_dict_drone_list.append(
                    ego_dict["merged_lidar_features_dict_drone"]
                )
                cam_inputs_drone_list.append(ego_dict["cam_inputs_drone"])
                merged_cam_inputs_dict_drone_list.append(
                    ego_dict["merged_cam_inputs_dict_drone"]
                )
                batch_idxs_drone.append(i)

            # vis
            origin_lidar_veh_list.append(ego_dict["origin_lidar_veh"])
            # origin_lidar_rsu_list.append(ego_dict["origin_lidar_rsu"])

        dynamic_seg_label_torch = torch.from_numpy(np.array(dynamic_seg_label_list))
        static_seg_label_torch = torch.from_numpy(np.array(static_seg_label_list))
        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        batch_merged_lidar_features_veh = self.merge_features_to_dict(
            merged_lidar_features_dict_veh_list, None, "lidar"
        )
        batch_merged_cam_inputs_veh = self.merge_features_to_dict(
            merged_cam_inputs_dict_veh_list, "cat", "cam"
        )

        batch_merged_lidar_features_rsu = self.merge_features_to_dict(
            merged_lidar_features_dict_rsu_list, None, "lidar"
        )
        batch_merged_cam_inputs_rsu = self.merge_features_to_dict(
            merged_cam_inputs_dict_rsu_list, "cat", "cam"
        )

        batch_merged_lidar_features_drone = self.merge_features_to_dict(
            merged_lidar_features_dict_drone_list, None, "lidar"
        )
        batch_merged_cam_inputs_drone = self.merge_features_to_dict(
            merged_cam_inputs_dict_drone_list, "cat", "cam"
        )

        
        

        # at least we have one ego veh
        batch_merged_lidar_features_veh_torch = (
            self.pre_processor.collate_batch(batch_merged_lidar_features_veh)
            if len(merged_lidar_features_dict_veh_list) > 0
            else None
        )

        batch_merged_lidar_features_rsu_torch = (
            self.pre_processor.collate_batch(batch_merged_lidar_features_rsu)
            if len(merged_lidar_features_dict_rsu_list) > 0
            else None
        )
        
        batch_merged_lidar_features_drone_torch = (
            self.pre_processor.collate_batch(batch_merged_lidar_features_drone)
            if len(merged_lidar_features_dict_drone_list) > 0
            else None
        )

        label_dict_torch = self.post_processor.collate_batch_airv2x(label_dict_list)
        label_dict_torch.update({"dynamic_seg_label": dynamic_seg_label_torch, "static_seg_label": static_seg_label_torch})

        pairwise_t_matrix_collab_torch = torch.from_numpy(
            np.array(pairwise_t_matrix_collab_list)
        ).float()
        img_pairwise_t_matrix_collab_torch = torch.from_numpy(
            np.array(img_pairwise_t_matrix_collab_list)
        ).float()

        record_len = torch.from_numpy(np.array(record_len, dtype=np.int32))
        record_len_veh = torch.from_numpy(np.array(record_len_veh, dtype=np.int32))
        record_len_rsu = torch.from_numpy(np.array(record_len_rsu, dtype=np.int32))
        record_len_drone = torch.from_numpy(np.array(record_len_drone, dtype=np.int32))

        # (B, max_cav)
        velocity = torch.from_numpy(np.array(velocity))
        time_delay = torch.from_numpy(np.array(time_delay))
        infra = torch.from_numpy(np.array(infra))
        spatial_correction_matrix_list = torch.from_numpy(
            np.array(spatial_correction_matrix_list)
        )
        # (B, max_cav, 3)
        prior_encoding = torch.stack([velocity, time_delay, infra], dim=-1).float()

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict["ego"].update(
            {
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": object_bbx_mask,
                "label_dict": label_dict_torch,
                "object_ids": object_ids,
                "class_ids": class_ids,
                "record_len": record_len,
                "pairwise_t_matrix_collab": pairwise_t_matrix_collab_torch,
                "img_pairwise_t_matrix_collab": img_pairwise_t_matrix_collab_torch,
                "prior_encoding": prior_encoding,
                "spatial_correction_matrix": spatial_correction_matrix_list,
                "vehicle": {
                    "batch_merged_lidar_features_torch": batch_merged_lidar_features_veh_torch,
                    "batch_merged_cam_inputs": batch_merged_cam_inputs_veh,
                    "record_len": record_len_veh,
                    "batch_idxs": batch_idxs_veh,  # len as record_len and each corresponding to the idx in the batch
                },
                "rsu": {
                    "batch_merged_lidar_features_torch": batch_merged_lidar_features_rsu_torch,
                    "batch_merged_cam_inputs": batch_merged_cam_inputs_rsu,
                    "record_len": record_len_rsu,
                    "batch_idxs": batch_idxs_rsu,
                },
                "drone": {
                    "batch_merged_lidar_features_torch": batch_merged_lidar_features_drone_torch,
                    "batch_merged_cam_inputs": batch_merged_cam_inputs_drone,
                    "record_len": record_len_drone,
                    "batch_idxs": batch_idxs_drone,
                },
                "scenario_index_list": scenario_index_list,
                "timestamp_key_list": timestamp_key_list,
                "metadata_path_list": metadata_path_list,
                "ego_lidar_pose_list": ego_lidar_pose_list,
            }
        )

        if self.visualize:
            origin_lidars_veh = np.array(
                downsample_lidar_minimum(pcd_np_list=origin_lidar_veh_list)
            )
            origin_lidars_veh = torch.from_numpy(origin_lidars_veh)
            origin_lidars_rsu = np.array(
                downsample_lidar_minimum(pcd_np_list=origin_lidar_rsu_list)
            )
            origin_lidars_rsu = torch.from_numpy(origin_lidars_rsu)
            origin_lidars_drone = np.array(
                downsample_lidar_minimum(pcd_np_list=origin_lidar_drone_list)
            )
            origin_lidars_drone = torch.from_numpy(origin_lidars_drone)
            
            output_dict["ego"].update({"origin_lidar": origin_lidars_veh})
            output_dict["ego"].update({"origin_lidar_rsu": origin_lidars_rsu})
            output_dict["ego"].update({"origin_lidar_drone": origin_lidars_drone})

        return output_dict





    # def collate_batch_train(self, batch):
    #     """
    #     Collate function with inline profiling.  
    #     日志格式: [collate] <阶段名称> ...... <耗时>s
    #     """
    #     # ---------- 简易计时工具 ----------
    #     t0 = time.perf_counter()
    #     def log(stage: str):
    #         nonlocal t0
    #         now = time.perf_counter()
    #         print(f"[collate] {stage:<40} {now - t0:7.3f}s")
    #         t0 = now
    #     # ---------------------------------

    #     output_dict = {"ego": {}}

    #     # ---------- 1. 预分配容器 ----------
    #     object_bbx_center, object_bbx_mask = [], []
    #     object_ids, class_ids = [], []
    #     record_len = []
    #     record_len_veh, record_len_rsu, record_len_drone = [], [], []
    #     label_dict_list, anchor_box_list = [], []
    #     pairwise_t_matrix_collab_list, img_pairwise_t_matrix_collab_list = [], []
    #     dynamic_seg_label_list, static_seg_label_list = [], []

    #     velocity, time_delay, infra = [], [], []
    #     spatial_correction_matrix_list = []

    #     processed_lidar_features_veh_lists, merged_lidar_features_dict_veh_list = [], []
    #     cam_inputs_veh_list, merged_cam_inputs_dict_veh_list, batch_idxs_veh = [], [], []

    #     processed_lidar_features_rsu_lists, merged_lidar_features_dict_rsu_list = [], []
    #     cam_inputs_rsu_list, merged_cam_inputs_dict_rsu_list, batch_idxs_rsu = [], [], []

    #     processed_lidar_features_drone_lists, merged_lidar_features_dict_drone_list = [], []
    #     cam_inputs_drone_list, merged_cam_inputs_dict_drone_list, batch_idxs_drone = [], [], []

    #     origin_lidar_veh_list, origin_lidar_rsu_list, origin_lidar_drone_list = [], [], []

    #     scenario_index_list, timestamp_key_list = [], []
    #     metadata_path_list, ego_lidar_pose_list = [], []
    #     log("init containers")
    #     # ---------------------------------

    #     # ---------- 2. 遍历 batch ----------
    #     for i, sample in enumerate(batch):
    #         ego_dict = sample["ego"]

    #         # 基础字段
    #         object_bbx_center.append(ego_dict["object_bbx_center"])
    #         object_bbx_mask.append(ego_dict["object_bbx_mask"])
    #         object_ids.append(ego_dict["object_ids"])
    #         class_ids.append(ego_dict["class_ids"])
    #         dynamic_seg_label_list.append(ego_dict["dynamic_seg_label"])
    #         static_seg_label_list.append(ego_dict["static_seg_label"])
    #         anchor_box_list.append(ego_dict["anchor_box"])
    #         record_len.append(ego_dict["num_cavs"])
    #         record_len_veh.append(ego_dict["num_veh"])
    #         record_len_rsu.append(ego_dict["num_rsu"])
    #         record_len_drone.append(ego_dict["num_drone"])
    #         label_dict_list.append(ego_dict["label_dict"])
    #         pairwise_t_matrix_collab_list.append(ego_dict["pairwise_t_matrix_collab"])
    #         img_pairwise_t_matrix_collab_list.append(
    #             ego_dict["img_pairwise_t_matrix_collab"]
    #         )

    #         velocity.append(ego_dict["velocity"])
    #         time_delay.append(ego_dict["time_delay"])
    #         infra.append(ego_dict["infra"])
    #         spatial_correction_matrix_list.append(ego_dict["spatial_correction_matrix"])

    #         # veh
    #         if ego_dict["num_veh"] > 0:
    #             processed_lidar_features_veh_lists.append(
    #                 ego_dict["processed_lidar_features_veh_list"]
    #             )
    #             merged_lidar_features_dict_veh_list.append(
    #                 ego_dict["merged_lidar_features_dict_veh"]
    #             )
    #             cam_inputs_veh_list.append(ego_dict["cam_inputs_veh"])
    #             merged_cam_inputs_dict_veh_list.append(
    #                 ego_dict["merged_cam_inputs_dict_veh"]
    #             )
    #             batch_idxs_veh.append(i)

    #         # rsu
    #         if ego_dict["num_rsu"] > 0:
    #             processed_lidar_features_rsu_lists.append(
    #                 ego_dict["processed_lidar_features_rsu_list"]
    #             )
    #             merged_lidar_features_dict_rsu_list.append(
    #                 ego_dict["merged_lidar_features_dict_rsu"]
    #             )
    #             cam_inputs_rsu_list.append(ego_dict["cam_inputs_rsu"])
    #             merged_cam_inputs_dict_rsu_list.append(
    #                 ego_dict["merged_cam_inputs_dict_rsu"]
    #             )
    #             batch_idxs_rsu.append(i)

    #         # drone
    #         if ego_dict["num_drone"] > 0:
    #             processed_lidar_features_drone_lists.append(
    #                 ego_dict["processed_lidar_features_drone_list"]
    #             )
    #             merged_lidar_features_dict_drone_list.append(
    #                 ego_dict["merged_lidar_features_dict_drone"]
    #             )
    #             cam_inputs_drone_list.append(ego_dict["cam_inputs_drone"])
    #             merged_cam_inputs_dict_drone_list.append(
    #                 ego_dict["merged_cam_inputs_dict_drone"]
    #             )
    #             batch_idxs_drone.append(i)

    #         # vis
    #         origin_lidar_veh_list.append(ego_dict["origin_lidar_veh"])
    #     log("loop over batch")
    #     # ---------------------------------

    #     # ---------- 3. numpy → torch ----------
    #     dynamic_seg_label_torch = torch.from_numpy(np.array(dynamic_seg_label_list))
    #     static_seg_label_torch  = torch.from_numpy(np.array(static_seg_label_list))
    #     object_bbx_center       = torch.from_numpy(np.array(object_bbx_center))
    #     object_bbx_mask         = torch.from_numpy(np.array(object_bbx_mask))
    #     log("to torch (basic tensors)")
    #     # ---------------------------------

    #     # ---------- 4. merge features ----------
    #     batch_merged_lidar_features_veh  = self.merge_features_to_dict(
    #         merged_lidar_features_dict_veh_list, None, "lidar"
    #     )
    #     batch_merged_cam_inputs_veh      = self.merge_features_to_dict(
    #         merged_cam_inputs_dict_veh_list, "cat", "cam"
    #     )
    #     batch_merged_lidar_features_rsu  = self.merge_features_to_dict(
    #         merged_lidar_features_dict_rsu_list, None, "lidar"
    #     )
    #     batch_merged_cam_inputs_rsu      = self.merge_features_to_dict(
    #         merged_cam_inputs_dict_rsu_list, "cat", "cam"
    #     )
    #     batch_merged_lidar_features_drone = self.merge_features_to_dict(
    #         merged_lidar_features_dict_drone_list, None, "lidar"
    #     )
    #     batch_merged_cam_inputs_drone     = self.merge_features_to_dict(
    #         merged_cam_inputs_dict_drone_list, "cat", "cam"
    #     )
    #     log("merge feature dicts")
    #     # ---------------------------------

    #     # ---------- 5. pre-process (collate) ----------
    #     batch_merged_lidar_features_veh_torch = (
    #         self.pre_processor.collate_batch(batch_merged_lidar_features_veh)
    #         if merged_lidar_features_dict_veh_list else None
    #     )
    #     batch_merged_lidar_features_rsu_torch = (
    #         self.pre_processor.collate_batch(batch_merged_lidar_features_rsu)
    #         if merged_lidar_features_dict_rsu_list else None
    #     )
    #     batch_merged_lidar_features_drone_torch = (
    #         self.pre_processor.collate_batch(batch_merged_lidar_features_drone)
    #         if merged_lidar_features_dict_drone_list else None
    #     )
    #     log("preprocess.collate_batch")
    #     # ---------------------------------

    #     # ---------- 6. 其余 tensor 构造 ----------
    #     label_dict_torch = self.post_processor.collate_batch_airv2x(label_dict_list)
    #     label_dict_torch.update({
    #         "dynamic_seg_label": dynamic_seg_label_torch,
    #         "static_seg_label" : static_seg_label_torch,
    #     })

    #     pairwise_t_matrix_collab_torch = torch.from_numpy(
    #         np.array(pairwise_t_matrix_collab_list)
    #     ).float()
    #     img_pairwise_t_matrix_collab_torch = torch.from_numpy(
    #         np.array(img_pairwise_t_matrix_collab_list)
    #     ).float()

    #     record_len      = torch.from_numpy(np.array(record_len,      dtype=np.int32))
    #     record_len_veh  = torch.from_numpy(np.array(record_len_veh,  dtype=np.int32))
    #     record_len_rsu  = torch.from_numpy(np.array(record_len_rsu,  dtype=np.int32))
    #     record_len_drone= torch.from_numpy(np.array(record_len_drone,dtype=np.int32))

    #     velocity = torch.from_numpy(np.array(velocity))
    #     time_delay = torch.from_numpy(np.array(time_delay))
    #     infra = torch.from_numpy(np.array(infra))
    #     spatial_correction_matrix_list = torch.from_numpy(
    #         np.array(spatial_correction_matrix_list)
    #     )

    #     prior_encoding = torch.stack([velocity, time_delay, infra], dim=-1).float()
    #     log("build remaining tensors")
    #     # ---------------------------------

    #     # ---------- 7. 填充 output_dict ----------
    #     output_dict["ego"].update({
    #         "object_bbx_center": object_bbx_center,
    #         "object_bbx_mask"  : object_bbx_mask,
    #         "label_dict"       : label_dict_torch,
    #         "object_ids"       : object_ids,
    #         "class_ids"        : class_ids,
    #         "record_len"       : record_len,
    #         "pairwise_t_matrix_collab"   : pairwise_t_matrix_collab_torch,
    #         "img_pairwise_t_matrix_collab": img_pairwise_t_matrix_collab_torch,
    #         "prior_encoding"   : prior_encoding,
    #         "spatial_correction_matrix": spatial_correction_matrix_list,

    #         "vehicle": {
    #             "batch_merged_lidar_features_torch": batch_merged_lidar_features_veh_torch,
    #             "batch_merged_cam_inputs"          : batch_merged_cam_inputs_veh,
    #             "record_len"                       : record_len_veh,
    #             "batch_idxs"                       : batch_idxs_veh,
    #         },
    #         "rsu": {
    #             "batch_merged_lidar_features_torch": batch_merged_lidar_features_rsu_torch,
    #             "batch_merged_cam_inputs"          : batch_merged_cam_inputs_rsu,
    #             "record_len"                       : record_len_rsu,
    #             "batch_idxs"                       : batch_idxs_rsu,
    #         },
    #         "drone": {
    #             "batch_merged_lidar_features_torch": batch_merged_lidar_features_drone_torch,
    #             "batch_merged_cam_inputs"          : batch_merged_cam_inputs_drone,
    #             "record_len"                       : record_len_drone,
    #             "batch_idxs"                       : batch_idxs_drone,
    #         },
    #         "scenario_index_list": scenario_index_list,
    #         "timestamp_key_list" : timestamp_key_list,
    #         "metadata_path_list" : metadata_path_list,
    #         "ego_lidar_pose_list": ego_lidar_pose_list,
    #     })

    #     # 可视化附加
    #     if self.visualize:
    #         origin_lidars_veh = torch.from_numpy(
    #             np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar_veh_list))
    #         )
    #         output_dict["ego"]["origin_lidar"] = origin_lidars_veh
    #     log("build output_dict")
    #     # ---------------------------------

    #     log("TOTAL")
    #     return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)

        # check if anchor box in the batch
        if batch[0]["ego"]["anchor_box"] is not None:
            output_dict["ego"].update(
                {
                    "anchor_box": torch.from_numpy(
                        np.array(batch[0]["ego"]["anchor_box"])
                    )
                }
            )

        # save the transformation matrix (4, 4) to ego vehicle
        transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()
        output_dict["ego"].update(
            {"transformation_matrix": transformation_matrix_torch}
        )

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score, pred_labels, pred_boxes3d = (
            self.post_processor.post_process_airv2x(data_dict, output_dict)
        )
        gt_box_tensor, gt_class_label_list, gt_track_list = self.post_processor.generate_gt_bbx_airv2x(data_dict)
        gt_box_tensor, gt_class_label_list, gt_track_list = self.post_processor.generate_gt_bbx_airv2x(data_dict)

        return pred_box_tensor, pred_score, pred_labels, pred_boxes3d, gt_box_tensor, gt_class_label_list, gt_track_list
    
    def post_process_seg(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_dynamic_seg_map, pred_static_seg_map, gt_dynamic_seg_map, gt_static_seg_map = (
            self.post_processor.post_process_segmentation_airv2x(data_dict, output_dict)
        )

        return pred_dynamic_seg_map, pred_static_seg_map, gt_dynamic_seg_map, gt_static_seg_map

    
    
    def get_pairwise_transformation(self, base_data_dict, max_cav, cop_agent_type):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        identity_pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))
        # this one is used for images when proj_first
        # proj_first only converts lidar to ego coordinate
        # but the cameras features are still in its own coordinate
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))

        # if lidar projected to ego first, then the pairwise matrix
        identity_pairwise_t_matrix[:, :] = np.identity(4)

        # if self.proj_first:
        #     # if lidar projected to ego first, then the pairwise matrix
        #     # becomes identity
        #     identity_pairwise_t_matrix[:, :] = np.identity(4)
        # else:
        t_list = []

        # save all transformation matrix in a list in order first.
        for cav_id, cav_content in base_data_dict.items():
            agent_type = cav_content["agent_type"]
            if agent_type in cop_agent_type:
                t_list.append(
                    cav_content["params"]["transformation_matrix"]
                )  # cur center -> ego

        for i in range(len(t_list)):
            for j in range(len(t_list)):
                # identity matrix to self
                if i == j:
                    t_matrix = np.eye(4)
                    pairwise_t_matrix[i, j] = t_matrix
                    continue
                # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                pairwise_t_matrix[i, j] = t_matrix
        if self.proj_first:
            return identity_pairwise_t_matrix, pairwise_t_matrix
        else:
            return pairwise_t_matrix, pairwise_t_matrix

    # @staticmethod
    # def merge_features_to_dict(processed_feature_list, merge=None, sensor="lidar"):
    #     """
    #     Merge the preprocessed features from different cavs to the same
    #     dictionary.

    #     Parameters
    #     ----------
    #     processed_feature_list : list
    #         A list of dictionary containing all processed features from
    #         different cavs.

    #     Returns
    #     -------
    #     merged_feature_dict: dict
    #         key: feature names, value: list of features.
    #     """

    #     merged_feature_dict = OrderedDict()

    #     if sensor == "lidar":
    #         for i in range(len(processed_feature_list)):
    #             for feature_name, feature in processed_feature_list[i].items():
    #                 if feature_name not in merged_feature_dict:
    #                     merged_feature_dict[feature_name] = []
    #                 if isinstance(feature, list):
    #                     merged_feature_dict[feature_name] += feature
    #                 else:
    #                     merged_feature_dict[feature_name].append(feature)
    #         return merged_feature_dict

    #     elif sensor == "cam":
    #         for i in range(len(processed_feature_list)):
    #             for feature_name, feature in processed_feature_list[i].items():
    #                 if feature_name not in merged_feature_dict:
    #                     merged_feature_dict[feature_name] = []
    #                 if isinstance(feature, list):
    #                     merged_feature_dict[feature_name] += feature
    #                 else:
    #                     merged_feature_dict[feature_name].append(feature)

    #         # stack them
    #         # it usually happens when merging cavs images -> v.shape = [N, Ncam, C, H, W]
    #         # cat them
    #         # it usually happens when merging batches cav images -> v is a list [(N1+N2+...Nn, Ncam, C, H, W))]
    #         if merge == "stack":
    #             for feature_name, features in merged_feature_dict.items():
    #                 merged_feature_dict[feature_name] = torch.stack(features, dim=0)
    #         elif merge == "cat":
    #             for feature_name, features in merged_feature_dict.items():
    #                 merged_feature_dict[feature_name] = torch.cat(features, dim=0)

    #         return merged_feature_dict


    @staticmethod
    def merge_features_to_dict(processed_feature_list, merge=None, sensor="lidar"):
        """
        Merge pre-processed features from multiple CAVs into a single dictionary.

        Parameters
        ----------
        processed_feature_list : list[dict]
            Each element is a dict of features for one CAV.
        merge : {None, "stack", "cat"}, optional
            For camera data:
                * None   – keep each value as a list of tensors.
                * "stack" – torch.stack() over dim 0.
                * "cat"   – torch.cat()   over dim 0.
            Ignored for lidar data (always kept as lists).
        sensor : {"lidar", "cam"}
            Sensor type that determines how features are merged.

        Returns
        -------
        dict
            Keys are feature names. Values are lists or tensors depending on
            `sensor` and `merge`.
        """
        if not processed_feature_list:          # Empty input → empty dict
            return {}

        # 1) Collect features: defaultdict avoids repeated key checks
        buckets = defaultdict(list)
        for cav_feat in processed_feature_list:          # outer loop: CAVs
            for name, feat in cav_feat.items():          # inner loop: features
                buckets[name].extend(feat if isinstance(feat, list) else [feat])

        # 2) Lidar: return lists directly; Cam: optionally stack/cat
        if sensor == "lidar" or merge is None:
            return buckets

        if merge == "stack":
            return {k: torch.stack(v, dim=0) for k, v in buckets.items()}
        if merge == "cat":
            return {k: torch.cat(v, dim=0) for k, v in buckets.items()}

        raise ValueError("merge must be None, 'stack', or 'cat'")


if __name__ == "__main__":
    params = load_yaml(
        "/code/opencood/hypes_yaml/airv2x/airv2x_intermediate_cobevt.yaml"
    )
    airv2x_database = IntermediateFusionDatasetAirv2x(params, None)

    batch_dict = airv2x_database.collate_batch_train(
        [
            airv2x_database.__getitem__(100),
            airv2x_database.__getitem__(3),
            airv2x_database.__getitem__(1),
        ]
    )

    # check the correctness of pairwise_t_matrix
    # dict_keys(['object_bbx_center', 'object_bbx_mask', 'processed_lidar_inputs', 'lidar_indexing_list', 'processed_cam_inputs', 'cam_indexing_list', 'record_len', 'label_dict', 'object_ids', 'pairwise_t_matrix', 'train'])
    print(batch_dict.keys())
