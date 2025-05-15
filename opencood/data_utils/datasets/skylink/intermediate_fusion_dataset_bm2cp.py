# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
# Modifier: Yuheng Wu <yuhengwu@kaist.ac.kr>

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

import opencood.data_utils.datasets
import opencood.data_utils.post_processor as post_processor
from opencood.data_utils.datasets.skylink import basedataset
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
    skylink_utils,
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


class IntermediateFusionDatasetSkylinkBM2CP(basedataset.BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """

    def __init__(self, params, visualize, train=True):
        super(IntermediateFusionDatasetSkylinkBM2CP, self).__init__(
            params, visualize, train
        )

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

        self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]
        self.grid_conf = params["fusion"]["args"]["grid_conf"]
        self.depth_discre = camera_utils.depth_discretization(
            self.grid_conf["ddiscr"][0],
            self.grid_conf["ddiscr"][1],
            self.grid_conf["ddiscr"][2],
            self.grid_conf["mode"],
        )

        if self.use_cam:
            self.data_aug_conf_cam = params["image_modality"]["data_aug_conf"]

        if self.use_lidar:
            self.data_aug_conf_lidar = params["lidar_modality"]["data_aug_conf"]

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
            params["postprocess"], dataset="skylink", train=train
        )

    def __getitem__(self, idx):
        base_data_dict, _, _ = self.retrieve_base_data(
            idx, cur_ego_pos_flag=self.cur_ego_pose_flag
        )

        processed_data_dict = OrderedDict()
        processed_data_dict["ego"] = {}

        ego_id = -1
        ego_lidar_pose = []
        num_cav = 0

        for cav_id, cav_content in base_data_dict.items():
            if cav_content["ego"]:
                ego_id = cav_id
                ego_lidar_pose = cav_content["params"]["delay_ego_lidar_pose"]
                break

        processed_data_dict["ego"].update({"ego_id": ego_id})

        # =================================================================
        #                      return data attributes
        # =================================================================
        # common for this timestamp
        object_stack = []
        object_id_stack = []
        class_id_stack = []
        seg_label = None

        # used for temp collection, will be filtered out later
        object_stack_veh = []
        object_id_stack_veh = []
        class_id_stack_veh = []
        distance_to_ego_veh = []
        num_veh = 0

        object_stack_rsu = []
        object_id_stack_rsu = []
        class_id_stack_rsu = []
        num_rsu = 0
        distance_to_ego_rsu = []

        object_stack_drone = []
        object_id_stack_drone = []
        class_id_stack_drone = []
        num_drone = 0
        distance_to_ego_drone = []

        retain_cav_ids = []
        # =================================================================
        #                       retrieve information
        # =================================================================

        # veh
        processed_lidar_features_veh_list = []
        cam_inputs_veh_list = []
        depth_maps_veh_list = []
        original_lidar_vis_veh_list = []

        # rsu
        processed_lidar_features_rsu_list = []
        cam_inputs_rsu_list = []
        depth_maps_rsu_list = []
        original_lidar_vis_rsu_list = []

        # drone
        cam_inputs_drone_list = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = selected_cav_base["distance_to_ego"]
            agent_type = selected_cav_base["agent_type"]

            if (
                agent_type == "vehicle"
                and distance > opencood.data_utils.datasets.VEH_COM_RANGE
            ):
                continue
            elif (
                agent_type == "rsu"
                and distance > opencood.data_utils.datasets.RSU_COM_RANGE
            ):
                continue
            elif (
                agent_type == "drone"
                and distance > opencood.data_utils.datasets.DRONE_COM_RANGE
            ):
                continue

            # num_cav += 1
            selected_cav_processed = self.get_item_single_car(
                selected_cav_base, ego_lidar_pose
            )

            # agent_type = selected_cav_processed["agent_type"]

            if agent_type == "drone":
                object_stack_drone.append(selected_cav_processed["object_bbx_center"])
                object_id_stack_drone.append(selected_cav_processed["object_ids"])
                class_id_stack_drone.append(selected_cav_processed["class_ids"])
                distance_to_ego_drone.append((cav_id, distance))
                num_drone += 1

                cam_inputs_drone_list.append(selected_cav_processed["cam_inputs"])

            elif agent_type == "rsu":
                object_stack_rsu.append(selected_cav_processed["object_bbx_center"])
                object_id_stack_rsu.append(selected_cav_processed["object_ids"])
                class_id_stack_rsu.append(selected_cav_processed["class_ids"])
                distance_to_ego_rsu.append((cav_id, distance))
                num_rsu += 1

                processed_lidar_features_rsu_list.append(
                    selected_cav_processed["processed_lidar_features"]
                )
                cam_inputs_rsu_list.append(selected_cav_processed["cam_inputs"])

                depth_maps_rsu_list.append(selected_cav_processed["depth_maps"])
                original_lidar_vis_rsu_list.append(selected_cav_processed["lidar_np"])
            else:
                # veh
                object_stack_veh.append(selected_cav_processed["object_bbx_center"])
                object_id_stack_veh.append(selected_cav_processed["object_ids"])
                class_id_stack_veh.append(selected_cav_processed["class_ids"])
                distance_to_ego_veh.append((cav_id, distance))
                num_veh += 1

                if selected_cav_base["ego"]:
                    seg_label = selected_cav_base["seg_label"]

                processed_lidar_features_veh_list.append(
                    selected_cav_processed["processed_lidar_features"]
                )
                cam_inputs_veh_list.append(selected_cav_processed["cam_inputs"])
                depth_maps_veh_list.append(selected_cav_processed["depth_maps"])
                original_lidar_vis_veh_list.append(selected_cav_processed["lidar_np"])

        # ========================================================================
        # remove out of max agents
        # metric: distance to ego, priortize the closer ones
        # in some cases, there is possible that missing  rsu/drone
        # when encounter num_rsu/drone == 0, handle differently
        # ========================================================================

        # veh
        veh_indices, retain_veh_ids = skylink_utils.get_smallest_k_indices(
            distance_to_ego_veh, self.max_cav_veh
        )
        object_stack_veh = [object_stack_veh[i] for i in veh_indices]
        object_id_stack_veh = [object_id_stack_veh[i] for i in veh_indices]
        class_id_stack_veh = [class_id_stack_veh[i] for i in veh_indices]
        processed_lidar_features_veh_list = [
            processed_lidar_features_veh_list[i] for i in veh_indices
        ]
        cam_inputs_veh_list = [cam_inputs_veh_list[i] for i in veh_indices]
        depth_maps_veh_list = [depth_maps_veh_list[i] for i in veh_indices]
        original_lidar_vis_veh_list = [
            original_lidar_vis_veh_list[i] for i in veh_indices
        ]
        retain_cav_ids += retain_veh_ids
        num_veh = len(retain_veh_ids)

        # rsu
        rsu_indices, retain_rsu_ids = skylink_utils.get_smallest_k_indices(
            distance_to_ego_rsu, self.max_cav_rsu
        )
        object_stack_rsu = [object_stack_rsu[i] for i in rsu_indices]
        object_id_stack_rsu = [object_id_stack_rsu[i] for i in rsu_indices]
        class_id_stack_rsu = [class_id_stack_rsu[i] for i in rsu_indices]
        processed_lidar_features_rsu_list = [
            processed_lidar_features_rsu_list[i] for i in rsu_indices
        ]
        cam_inputs_rsu_list = [cam_inputs_rsu_list[i] for i in rsu_indices]
        depth_maps_rsu_list = [depth_maps_rsu_list[i] for i in rsu_indices]
        original_lidar_vis_rsu_list = [
            original_lidar_vis_rsu_list[i] for i in rsu_indices
        ]
        retain_cav_ids += retain_rsu_ids
        num_rsu = len(retain_rsu_ids)

        # drone
        drone_indices, retain_drone_ids = skylink_utils.get_smallest_k_indices(
            distance_to_ego_drone, self.max_cav_drone
        )
        object_stack_drone = [object_stack_drone[i] for i in drone_indices]
        object_id_stack_drone = [object_id_stack_drone[i] for i in drone_indices]
        class_id_stack_drone = [class_id_stack_drone[i] for i in drone_indices]
        cam_inputs_drone_list = [cam_inputs_drone_list[i] for i in drone_indices]
        retain_cav_ids += retain_drone_ids
        num_drone = len(retain_drone_ids)

        # clean visible objects
        object_stack = list(
            chain.from_iterable(
                object_stack_veh + object_stack_rsu + object_stack_drone
            )
        )
        object_id_stack = list(
            chain.from_iterable(
                object_id_stack_veh + object_id_stack_rsu + object_id_stack_drone
            )
        )
        class_id_stack = list(
            chain.from_iterable(
                class_id_stack_veh + class_id_stack_rsu + class_id_stack_drone
            )
        )
        # retain_cav_ids = list(set(retain_cav_ids))
        # base_data_dict = {
        #     cav_id: base_data_dict[cav_id] for cav_id in retain_cav_ids
        # }  # in the order [veh, rsu, drone]

        retain_cav_ids = list(unique_everseen(retain_cav_ids))

        retain_base_data_dict = OrderedDict(
            (retain_cav_id, base_data_dict[retain_cav_id])
            for retain_cav_id in retain_cav_ids
        )

        # ============================================================================
        # get the pairwise transformation matrix
        # ============================================================================

        pairwise_t_matrix_collab, img_pairwise_t_matrix_collab = (
            self.get_pairwise_transformation(
                retain_base_data_dict, self.max_cav_num, self.collaborators
            )
        )

        # ============================================================================
        # merge the features
        # ============================================================================

        # we always have at least one ego veh, so we ensure veh here
        merged_processed_lidar_dict_veh = self.merge_features_to_dict(
            processed_lidar_features_veh_list,
            None,
            "lidar",  # here, the voxel has been projected with self.proj_first so that they can be merged
        )
        merged_cam_inputs_dict_veh = self.merge_features_to_dict(
            cam_inputs_veh_list, "stack", "cam"
        )
        depth_maps_veh_torch = torch.stack(depth_maps_veh_list, dim=0)

        # handle cases when no rsu/drone
        merged_processed_lidar_dict_rsu = (
            self.merge_features_to_dict(
                processed_lidar_features_rsu_list, None, "lidar"
            )
            if num_rsu > 0
            else {}
        )
        merged_cam_inputs_dict_rsu = (
            self.merge_features_to_dict(cam_inputs_rsu_list, "stack", "cam")
            if num_rsu > 0
            else {}
        )
        depth_maps_rsu_torch = (
            torch.stack(depth_maps_rsu_list, dim=0) if num_rsu > 0 else None
        )
        # drone
        merged_cam_inputs_dict_drone = (
            self.merge_features_to_dict(cam_inputs_drone_list, "stack", "cam")
            if num_drone > 0
            else {}
        )

        # exclude all repetitive objects
        # unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
        # NOTE: don't use set() which may shift the order
        unique_elements = list(unique_everseen(object_id_stack))
        unique_indices = [object_id_stack.index(x) for x in unique_elements]

        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        ret_class_ids = class_id_stack
        class_id_stack = np.array(class_id_stack)
        class_id_stack = class_id_stack[unique_indices]

        # make sure bbx across all frames have the same number
        object_bbx_center = np.zeros((self.params["postprocess"]["max_num"], 7))
        mask = np.zeros(self.params["postprocess"]["max_num"])
        object_bbx_center[: object_stack.shape[0], :] = object_stack
        mask[: object_stack.shape[0]] = 1

        # pad class_ids to max num
        class_ids_padded = np.full(
            self.params["postprocess"]["max_num"], 0
        )  # pad 0 as "background"
        class_ids_padded[: object_stack.shape[0]] = class_id_stack

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = self.post_processor.generate_label_skylink(
            gt_box_center=object_bbx_center,
            anchors=anchor_box,
            mask=mask,
            class_ids_padded=class_ids_padded,
        )

        assert seg_label is not None, "seg_label should not be None"

        processed_data_dict["ego"].update(
            {
                # common
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": mask,
                "object_ids": [object_id_stack[i] for i in unique_indices],
                "class_ids": [ret_class_ids[i] for i in unique_indices],
                "seg_label": seg_label,
                "anchor_box": anchor_box,
                "num_cavs": num_veh + num_rsu + num_drone,
                "num_veh": num_veh,
                "num_rsu": num_rsu,
                "num_drone": num_drone,
                "label_dict": label_dict,
                "pairwise_t_matrix_collab": pairwise_t_matrix_collab,
                "img_pairwise_t_matrix_collab": img_pairwise_t_matrix_collab,
                "original_lidar_vis_veh_list": original_lidar_vis_veh_list,
                "original_lidar_vis_rsu_list": original_lidar_vis_rsu_list,
                # veh
                "processed_lidar_features_veh_list": processed_lidar_features_veh_list,
                "merged_lidar_features_dict_veh": merged_processed_lidar_dict_veh,
                "depth_maps_veh_torch": depth_maps_veh_torch,
                "cam_inputs_veh": cam_inputs_veh_list,
                "merged_cam_inputs_dict_veh": merged_cam_inputs_dict_veh,
                # rsu
                "processed_lidar_features_rsu_list": processed_lidar_features_rsu_list,
                "merged_lidar_features_dict_rsu": merged_processed_lidar_dict_rsu,
                "depth_maps_rsu_torch": depth_maps_rsu_torch,
                "cam_inputs_rsu": cam_inputs_rsu_list,
                "merged_cam_inputs_dict_rsu": merged_cam_inputs_dict_rsu,
                # drone
                "cam_inputs_drone": cam_inputs_drone_list,
                "merged_cam_inputs_dict_drone": merged_cam_inputs_dict_drone,
                # vis
                "origin_lidar_veh": np.concatenate(original_lidar_vis_veh_list, axis=0),
                # "origin_lidar_rsu": np.concatenate(original_lidar_vis_rsu_list, axis=0),
            }
        )
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
            self.post_processor.generate_object_center_skylink(
                [selected_cav_base], ego_pose
            )
        )
        agent_type = selected_cav_base["agent_type"]
        selected_cav_processed.update({"agent_type": agent_type})
        image_inputs = None
        if agent_type == "drone":
            # refer to: https://github.com/ylwhxht/V2X-R/blob/master/opencood/data_utils/datasets/opv2v/lidar_camera_intermediate_fusion_dataset.py#L624

            # process image to bev feature
            camera_data_list = selected_cav_base["cameras"]
            params = selected_cav_base["params"]

            N = len(camera_data_list)

            # here the cam2lidar = bevcam -> center of drone (mocked lidar)
            camera_to_lidar_matrix = params["delay_extrinsic"].reshape(N, 4, 4)
            camera_intrinsic = params["delay_intrinsic"].reshape(N, 3, 3)

            # no augmentatio for now, consider how to add aug?
            post_trans = torch.zeros(N, 3, dtype=torch.float32)
            post_rots = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)

            reH, reW = (
                self.data_aug_conf["final_dim"][0],
                self.data_aug_conf["final_dim"][1],
            )

            img_src = []

            for img in camera_data_list:
                # TODO(YH): should we aug?
                imgH, imgW = img.height, img.width
                resized_img = img.resize((reW, reH))
                img_src.append(camera_utils.normalize_img(resized_img))

            image_inputs = {
                "cam_inputs": {
                    "imgs": torch.stack(img_src, dim=0).float(),
                    "rots": torch.from_numpy(camera_to_lidar_matrix[:, :3, :3]).float(),
                    "trans": torch.from_numpy(camera_to_lidar_matrix[:, :3, 3]).float(),
                    "intrinsics": torch.from_numpy(camera_intrinsic).float(),
                    "extrinsics": torch.from_numpy(
                        camera_to_lidar_matrix
                    ).float(),  # to drone center
                    "post_rots": post_rots,
                    "post_trans": post_trans,
                }
            }

            selected_cav_processed.update(image_inputs)

            # object_bbx_center, object_bbx_mask, object_ids, class_ids
            selected_cav_processed.update(
                {
                    "lidar_np": [],  # placeholder, drone does not have lidar
                    "processed_lidar_features": [],  # placeholder
                    "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                    "object_ids": object_ids,
                    "class_ids": class_ids,
                }
            )
        else:
            # deal with rsu/vehicle, both has cam and lidar

            # filter lidar
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

            processed_lidar = self.pre_processor.preprocess(lidar_np)

            # refer to: https://github.com/ylwhxht/V2X-R/blob/master/opencood/data_utils/datasets/opv2v/lidar_camera_intermediate_fusion_dataset.py#L624

            # process image to bev feature
            camera_data_list = selected_cav_base["cameras"]
            params = selected_cav_base["params"]

            N = len(camera_data_list)
            camera_to_lidar_matrix = params["delay_extrinsic"].reshape(N, 4, 4)
            camera_intrinsic = params["delay_intrinsic"].reshape(N, 3, 3)

            post_trans = torch.zeros(N, 3, dtype=torch.float32)
            post_rots = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)

            reH, reW = (
                self.data_aug_conf["final_dim"][0],
                self.data_aug_conf["final_dim"][1],
            )

            img_src = []

            for img in camera_data_list:
                # TODO(YH): should we aug?
                imgH, imgW = img.height, img.width
                resized_img = img.resize((reW, reH))
                img_src.append(camera_utils.normalize_img(resized_img))

            image_inputs = {
                "cam_inputs": {
                    "imgs": torch.stack(img_src, dim=0).float(),
                    "rots": torch.from_numpy(camera_to_lidar_matrix[:, :3, :3]).float(),
                    "trans": torch.from_numpy(camera_to_lidar_matrix[:, :3, 3]).float(),
                    "intrinsics": torch.from_numpy(camera_intrinsic).float(),
                    "extrinsics": torch.from_numpy(
                        camera_to_lidar_matrix
                    ).float(),  # cam2cur ego
                    "post_rots": post_rots,
                    "post_trans": post_trans,
                }
            }

            # update
            selected_cav_processed.update(image_inputs)
            selected_cav_processed.update(
                {
                    "lidar_np": lidar_np,
                    "processed_lidar_features": processed_lidar,
                    "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                    "object_ids": object_ids,
                    "class_ids": class_ids,
                }
            )

            # bm2cp depth generation
            depth_maps = []
            for idx, camera_data in enumerate(selected_cav_base["cameras"]):
                # because drone doesn't have lidar, we only apply this to rsu/vehicle
                depth_map = self.generate_depth_map(
                    idx,
                    np.array(camera_data),
                    lidar_np,  # in the ego coordinate system
                    image_inputs["cam_inputs"]["intrinsics"][idx],
                    imgH,
                    imgW,
                    params["delay_cams_abs_pos"][idx],
                    params["delay_lidar_ego_abs_pos"]
                    if self.proj_first
                    else params["delay_cav_lidar_pose"],
                    False,
                )  # (imgH, imgW)
                depth_maps.append(depth_map)
            depth_maps = torch.cat(depth_maps, dim=0)  # (Ncam, imgH, imgW)

            selected_cav_processed.update(
                {"depth_maps": depth_maps}
            )  # project lidar point height back to LSS frustum

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
        seg_label_list = []

        # veh
        processed_lidar_features_veh_lists = []
        merged_lidar_features_dict_veh_list = []
        depth_maps_veh_torch_list = []
        cam_inputs_veh_list = []
        merged_cam_inputs_dict_veh_list = []
        batch_idxs_veh = []

        # rsu
        processed_lidar_features_rsu_lists = []
        merged_lidar_features_dict_rsu_list = []
        depth_maps_rsu_torch_list = []
        cam_inputs_rsu_list = []
        merged_cam_inputs_dict_rsu_list = []
        batch_idxs_rsu = []

        # drone
        cam_inputs_drone_list = []
        merged_cam_inputs_dict_drone_list = []
        batch_idxs_drone = []

        # vis
        origin_lidar_veh_list = []
        origin_lidar_rsu_list = []

        for i in range(len(batch)):
            ego_dict = batch[i]["ego"]
            object_bbx_center.append(ego_dict["object_bbx_center"])
            object_bbx_mask.append(ego_dict["object_bbx_mask"])
            object_ids.append(ego_dict["object_ids"])
            class_ids.append(ego_dict["class_ids"])
            seg_label_list.append(ego_dict["seg_label"])
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

            # veh
            processed_lidar_features_veh_lists.append(
                ego_dict["processed_lidar_features_veh_list"]
            )
            merged_lidar_features_dict_veh_list.append(
                ego_dict["merged_lidar_features_dict_veh"]
            )
            depth_maps_veh_torch_list.append(ego_dict["depth_maps_veh_torch"])
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
                depth_maps_rsu_torch_list.append(ego_dict["depth_maps_rsu_torch"])
                cam_inputs_rsu_list.append(ego_dict["cam_inputs_rsu"])
                merged_cam_inputs_dict_rsu_list.append(
                    ego_dict["merged_cam_inputs_dict_rsu"]
                )
                batch_idxs_rsu.append(i)
            # drone
            if ego_dict["num_drone"] > 0:
                cam_inputs_drone_list.append(ego_dict["cam_inputs_drone"])
                merged_cam_inputs_dict_drone_list.append(
                    ego_dict["merged_cam_inputs_dict_drone"]
                )
                batch_idxs_drone.append(i)

            # vis
            origin_lidar_veh_list.append(ego_dict["origin_lidar_veh"])
            # origin_lidar_rsu_list.append(ego_dict["origin_lidar_rsu"])

        seg_label_torch = torch.from_numpy(np.array(seg_label_list))  # (B, H, W)
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

        batch_merged_cam_inputs_drone = self.merge_features_to_dict(
            merged_cam_inputs_dict_drone_list, "cat", "cam"
        )

        # at least we have one ego veh
        batch_merged_lidar_features_veh_torch = self.pre_processor.collate_batch(
            batch_merged_lidar_features_veh
        )

        batch_merged_lidar_features_rsu_torch = (
            self.pre_processor.collate_batch(batch_merged_lidar_features_rsu)
            if len(merged_lidar_features_dict_rsu_list) > 0
            else None
        )

        label_dict_torch = self.post_processor.collate_batch_skylink(label_dict_list)
        label_dict_torch.update({"seg_label": seg_label_torch})

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

        depth_maps_veh_torch = torch.cat(depth_maps_veh_torch_list, dim=0)
        # FIXME(YH): sometimes, here is bug, possibly sometimes this list is empty
        depth_maps_rsu_torch = (
            torch.cat(depth_maps_rsu_torch_list, dim=0)
            if len(merged_lidar_features_dict_rsu_list) > 0
            else None
        )

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict["ego"].update(
            {
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": object_bbx_mask,
                "label_dict": label_dict_torch,
                "object_ids": object_ids[0],
                "record_len": record_len,
                "pairwise_t_matrix_collab": pairwise_t_matrix_collab_torch,
                "img_pairwise_t_matrix_collab": img_pairwise_t_matrix_collab_torch,
                "vehicle": {
                    "batch_merged_lidar_features_torch": batch_merged_lidar_features_veh_torch,
                    "batch_merged_cam_inputs": batch_merged_cam_inputs_veh,
                    "depth_maps_torch": depth_maps_veh_torch,
                    "record_len": record_len_veh,
                    "batch_idxs": batch_idxs_veh,  # len as record_len and each corresponding to the idx in the batch
                },
                "rsu": {
                    "batch_merged_lidar_features_torch": batch_merged_lidar_features_rsu_torch,
                    "batch_merged_cam_inputs": batch_merged_cam_inputs_rsu,
                    "depth_maps_torch": depth_maps_rsu_torch,
                    "record_len": record_len_rsu,
                    "batch_idxs": batch_idxs_rsu,
                },
                "drone": {
                    "batch_merged_cam_inputs": batch_merged_cam_inputs_drone,
                    "record_len": record_len_drone,
                    "batch_idxs": batch_idxs_drone,
                },
            }
        )

        if self.visualize:
            origin_lidars_veh = np.array(
                downsample_lidar_minimum(pcd_np_list=origin_lidar_veh_list)
            )
            origin_lidars_veh = torch.from_numpy(origin_lidars_veh)
            output_dict["ego"].update({"origin_lidar": origin_lidars_veh})

            # origin_lidars_rsu = np.array(
            #     downsample_lidar_minimum(pcd_np_list=origin_lidar_rsu_list)
            # )
            # origin_lidars_rsu = torch.from_numpy(origin_lidars_rsu)
            # output_dict["ego"].update({"origin_lidars_rsu": origin_lidars_rsu})

        return output_dict

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
        pred_box_tensor, pred_score, pred_labels = (
            self.post_processor.post_process_skylink(data_dict, output_dict)
        )
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor

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

    @staticmethod
    def merge_features_to_dict(processed_feature_list, merge=None, sensor="lidar"):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        if sensor == "lidar":
            for i in range(len(processed_feature_list)):
                for feature_name, feature in processed_feature_list[i].items():
                    if feature_name not in merged_feature_dict:
                        merged_feature_dict[feature_name] = []
                    if isinstance(feature, list):
                        merged_feature_dict[feature_name] += feature
                    else:
                        merged_feature_dict[feature_name].append(feature)
            return merged_feature_dict

        elif sensor == "cam":
            for i in range(len(processed_feature_list)):
                for feature_name, feature in processed_feature_list[i].items():
                    if feature_name not in merged_feature_dict:
                        merged_feature_dict[feature_name] = []
                    if isinstance(feature, list):
                        merged_feature_dict[feature_name] += feature
                    else:
                        merged_feature_dict[feature_name].append(feature)

            # stack them
            # it usually happens when merging cavs images -> v.shape = [N, Ncam, C, H, W]
            # cat them
            # it usually happens when merging batches cav images -> v is a list [(N1+N2+...Nn, Ncam, C, H, W))]
            if merge == "stack":
                for feature_name, features in merged_feature_dict.items():
                    merged_feature_dict[feature_name] = torch.stack(features, dim=0)
            elif merge == "cat":
                for feature_name, features in merged_feature_dict.items():
                    merged_feature_dict[feature_name] = torch.cat(features, dim=0)

            return merged_feature_dict

    def generate_depth_map(
        self,
        index,
        image,
        lidar_np,
        cam_intrinsic,
        imgH,
        imgW,
        cam_pos,
        lidar_pos,
        draws=False,
    ):
        vis_image = np.zeros(image.shape, dtype=np.float32)
        vis_image, points_2d = skylink_utils.project_lidar_to_cam_single(
            lidar_np, cam_intrinsic, imgH, imgW, cam_pos, lidar_pos, vis_image
        )

        # project lidar to cam coordinates
        depth = points_2d[:, 2]
        uv = points_2d[:, :2]
        uv_int = (np.ceil(uv) - ((uv - np.floor(uv)) < 0.5).astype(np.int32)).astype(
            np.int32
        )
        uv_int = uv_int[:, ::-1]

        valid_mask = (
            (depth >= self.grid_conf["ddiscr"][0])
            & (uv_int[:, 0] >= 0)
            & (uv_int[:, 0] < imgH)
            & (uv_int[:, 1] >= 0)
            & (uv_int[:, 1] < imgW)
        )
        valid_uvint, valid_depth = uv_int[valid_mask], depth[valid_mask]

        depth_map = -1 * np.ones((imgH, imgW), dtype=np.float32)
        for idx, valid_coord in enumerate(valid_uvint):
            u, v = valid_coord[0], valid_coord[1]
            depth_level = bisect.bisect_left(self.depth_discre, valid_depth[idx])
            if depth_level == 0:
                depth_level = 1
            depth_map[u, v] = (
                depth_level - 1
                if depth_map[u, v] < 0
                else min(depth_map[u, v], depth_level - 1)
            )
            # depth_map[u,v] = valid_depth[idx] if depth_map[u,v]<0 else min(depth_map[u,v], valid_depth[idx])

        # downsample via pooling to match network input
        assert imgH % self.data_aug_conf["final_dim"][0] == 0
        assert imgW % self.data_aug_conf["final_dim"][1] == 0
        scaleH, scaleW = (
            imgH // self.data_aug_conf["final_dim"][0],
            imgW // self.data_aug_conf["final_dim"][1],
        )

        max_depth_level = np.max(depth_map)
        depth_map[depth_map < 0] = max_depth_level + 1
        depth_map = torch.FloatTensor(-1 * depth_map).unsqueeze(0)
        pool_layer = torch.nn.MaxPool2d(
            kernel_size=(scaleH, scaleW), stride=(scaleH, scaleW)
        )
        depth_map = -1 * pool_layer(depth_map)
        depth_map[depth_map > max_depth_level] = -1

        if draws:
            self.N += 1
            plt.imshow(image)
            plt.savefig(f"/code/dummy_images/image_{self.N}.png")
            plt.close()

            plt.imshow(vis_image)
            plt.savefig(f"/code/dummy_images/proj_image_{self.N}.png")
            plt.close()

            plt.imshow(depth_map.numpy().transpose(1, 2, 0))
            plt.savefig(f"/code/dummy_images/depth_{self.N}.png")
            plt.close()

        return depth_map


if __name__ == "__main__":
    params = load_yaml(
        "/code/opencood/hypes_yaml/skylink/skylink_intermediate_bm2cp.yaml"
    )
    skylink_database = IntermediateFusionDatasetSkylinkBM2CP(params, None)

    batch_dict = skylink_database.collate_batch_train(
        [
            skylink_database.__getitem__(100),
            skylink_database.__getitem__(3),
            skylink_database.__getitem__(1),
        ]
    )

    # check the correctness of pairwise_t_matrix
    # dict_keys(['object_bbx_center', 'object_bbx_mask', 'processed_lidar_inputs', 'lidar_indexing_list', 'processed_cam_inputs', 'cam_indexing_list', 'record_len', 'label_dict', 'object_ids', 'pairwise_t_matrix', 'train'])
    print(batch_dict.keys())
