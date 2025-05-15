# -*- coding: utf-8 -*-
# Author: Yuheng Wu <yuhengwu@kaist.ac.kr>

"""
Basedataset class for all kinds of fusion.
"""

import math
import os
import pickle
import random
from collections import OrderedDict

import numpy as np
import torch
from pyparsing import Or
from torch.utils.data import Dataset
from PIL import Image

import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_pickle, load_yaml
from opencood.utils.camera_utils import load_camera_data
from opencood.utils.keypoint_utils import bev_sample, get_keypoints
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.skylink_utils import get_ex_intrinsic, parse_seq, filter_objects
from opencood.utils.transformation_utils import (
    get_abs_world_pose,
    x1_to_x2,
    x_to_world,
)
from tqdm import tqdm


class BaseDataset(Dataset):
    """
    Base dataset for all kinds of fusion. Mainly used to initialize the
    database and associate the __get_item__ index with the correct timestamp
    and scenario.

    Parameters
    __________
    params : dict
        The dictionary contains all parameters for training/testing.

    visualize : false
        If set to true, the raw point cloud will be saved in the memory
        for visualization.

    Attributes
    ----------
    scenario_database : OrderedDict
        A structured dictionary contains all file information.

    len_record : list
        The list to record each scenario's data length. This is used to
        retrieve the correct index during training.

    pre_processor : opencood.pre_processor
        Used to preprocess the raw data.

    post_processor : opencood.post_processor
        Used to generate training labels and convert the model outputs to
        bbx formats.

    data_augmentor : opencood.data_augmentor
        Used to augment data.

    """

    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = None
        self.post_processor = None
        self.data_augmentor = DataAugmentor(params["data_augment"], train)
        # if the training/testing include noisy setting
        if "wild_setting" in params:
            self.seed = params["wild_setting"]["seed"]
            # whether to add time delay
            self.async_flag = params["wild_setting"]["async"]
            self.async_mode = (
                "sim"
                if "async_mode" not in params["wild_setting"]
                else params["wild_setting"]["async_mode"]
            )
            self.async_overhead = params["wild_setting"]["async_overhead"]

            # localization error
            self.loc_err_flag = params["wild_setting"]["loc_err"]
            self.xyz_noise_std = params["wild_setting"]["xyz_std"]
            self.ryp_noise_std = params["wild_setting"]["ryp_std"]

            # transmission data size
            self.data_size = (
                params["wild_setting"]["data_size"]
                if "data_size" in params["wild_setting"]
                else 0
            )
            self.transmission_speed = (
                params["wild_setting"]["transmission_speed"]
                if "transmission_speed" in params["wild_setting"]
                else 27
            )
            self.backbone_delay = (
                params["wild_setting"]["backbone_delay"]
                if "backbone_delay" in params["wild_setting"]
                else 0
            )

        else:
            self.async_flag = False
            self.async_overhead = 0  # ms
            self.async_mode = "sim"
            self.loc_err_flag = False
            self.xyz_noise_std = 0
            self.ryp_noise_std = 0
            self.data_size = 0  # Mb (Megabits)
            self.transmission_speed = 27  # Mbps
            self.backbone_delay = 0  # ms

        # if 'select_kp' in params:
        #     self.select_keypoint = params['select_kp']
        # else:
        #     self.select_keypoint = None

        assert "proj_first" in params["fusion"]["args"]
        if params["fusion"]["args"]["proj_first"]:
            self.proj_first = True
        else:
            self.proj_first = False

        if self.train:
            root_dir = params["root_dir"]  # dataset/skylink/<seq>
        else:
            root_dir = params["validate_dir"]

        print("Dataset dir:", root_dir)

        if "train_params" not in params or "max_cav" not in params["train_params"]:
            self.max_cav_veh = 10
            self.max_cav_rsu = 5
            self.max_cav_drone = 5
        else:
            max_cav = params["train_params"]["max_cav"]
            self.max_cav_veh = max_cav["vehicle"]
            self.max_cav_rsu = max_cav["rsu"]
            self.max_cav_drone = max_cav["drone"]

        # first load all paths of different scenarios
        scenario_folders = sorted(
            [
                os.path.join(root_dir, x)
                for x in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, x))
            ]
        )
        scenario_folders_name = sorted(
            [
                x
                for x in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, x))
            ]
        )
        self.scenario_database = OrderedDict()
        self.len_record = []
        
        self.ego_type = params.get("ego_type", "vehicle")
        assert self.ego_type in ["vehicle", "rsu", "drone"], f"ego type {self.ego_type} not supported"

        # loop over all scenarios
        i = 0
        for scenario_folder in tqdm(
            scenario_folders,
            total=len(scenario_folders),
            desc="Loading %s dataset" % ("training" if train else "validation/testing"),
        ):
            # if "2025_04_27_17_22_41" in scenario_folder:
            #     continue  # 2025_04_27_17_22_41 has some bugs
            scenario_dict = parse_seq(scenario_folder)

            self.scenario_database.update({i: scenario_dict})

            # rsu and drone can't be ego, move them to the tail
            # make sure the first agent is always the vehicle with smallest idx
            while True:
                first = list(self.scenario_database[i].keys())[0]
                if next(iter(self.scenario_database[i][first].items()))[1][
                    "agent_type"
                ] != self.ego_type:
                    self.scenario_database[i].move_to_end(first)
                else:
                    # set the first as ego by default
                    # self.scenario_database[i][first]['ego'] = True
                    record_length = len(list(self.scenario_database[i][first].keys()))
                    if not self.len_record:
                        self.len_record.append(record_length)
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + record_length)
                    break
            i += 1

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

    def retrieve_base_data(self, idx, cur_ego_pos_flag=True):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]
        # check the timestamp index
        timestamp_index = (
            idx if scenario_index == 0 else idx - self.len_record[scenario_index - 1]
        )
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database, timestamp_index)
        # shuffle ego: traininig -> random, val -> no shuffle
        self.shuffle_ego()

        ego_cav_content = self.calc_dist_to_ego(scenario_database, timestamp_key)
        data = OrderedDict()

        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            # print(f"cav id: {cav_id}, ego: {cav_content['ego']}, agent type: {cav_content[0]['agent_type']}, cameras {cav_content[timestamp_key]['cameras']}")

            data[cav_id] = OrderedDict()
            data[cav_id]["ego"] = cav_content["ego"]
            data[cav_id]["agent_type"] = cav_content[timestamp_key]["agent_type"]
            data[cav_id]["distance_to_ego"] = cav_content["distance_to_ego"]

            agent_type = data[cav_id]["agent_type"]
            # calculate delay for this vehicle
            timestamp_delay = self.time_delay_calculation(cav_content["ego"])
            if timestamp_index - timestamp_delay <= 0:
                timestamp_delay = timestamp_index
            timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            timestamp_key_delay = self.return_timestamp_key(
                scenario_database, timestamp_index_delay
            )
            data[cav_id]["time_delay"] = timestamp_delay

            data[cav_id]["params"] = self.reform_param(
                cav_content,
                ego_cav_content,
                timestamp_key,
                timestamp_key_delay,
                cur_ego_pos_flag,
            )

            data[cav_id]["cameras"] = load_camera_data(
                cav_content[timestamp_key_delay]["cameras"]
            )
            
            data[cav_id]["depth"] = load_camera_data(
                cav_content[timestamp_key_delay]["depth"]
            )
            
            data[cav_id]["lidar_np"] = (
                pcd_utils.pcd_to_np(
                    cav_content[timestamp_key_delay]["lidars"][0]
                )  # TODO(YH): don't consider semantic lidar for now
                # if agent_type != "drone"
                # else []
            )  # placeholder for drone which doesn't have lidar
            data[cav_id]["dynamic_seg_label"] = self._wrap_segmentation_map(
                cav_content[timestamp_key_delay]["map"][-7:], "dynamic"
            )
            data[cav_id]["static_seg_label"] = self._wrap_segmentation_map(
                cav_content[timestamp_key_delay]["map"][-10:-7], "static" # static segmentation map
            )
            data[cav_id]['metadata_path'] = cav_content[timestamp_key_delay]['metadata_path']

        return data, scenario_index, timestamp_key

    def reform_param(
        self,
        cav_content,
        ego_content,
        timestamp_cur,
        timestamp_delay,
        cur_ego_pos_flag,
    ):
        """
        Reform the data params with current timestamp object groundtruth and
        delay timestamp LiDAR pose for other CAVs.

        Parameters
        ----------
        cav_content : dict
            Dictionary that contains all file paths in the current cav/rsu.

        ego_content : dict
            Ego vehicle content.

        timestamp_cur : str
            The current timestamp.

        timestamp_delay : str
            The delayed timestamp.

        cur_ego_pos_flag : bool
            Whether use current ego pose to calculate transformation matrix.

        Return
        ------
        The merged parameters.
        """
        cur_params = load_pickle(cav_content[timestamp_cur]["metadata_path"])
        delay_params = load_pickle(cav_content[timestamp_delay]["metadata_path"])

        cur_ego_params = load_pickle(ego_content[timestamp_cur]["metadata_path"])
        delay_ego_params = load_pickle(ego_content[timestamp_delay]["metadata_path"])

        # get abs pose of current ego lidar
        cur_ego_lidar_rel_pose = cur_ego_params["lidar"]["lidar_pose"]
        cur_ego_vehicle_pose = cur_ego_params["odometry"]["ego_pos"]
        cur_ego_lidar_pose = get_abs_world_pose(
            cur_ego_lidar_rel_pose, cur_ego_vehicle_pose
        )

        # get abs pose of current ego lidar
        delay_ego_lidar_rel_pose = delay_ego_params["lidar"]["lidar_pose"]
        delay_ego_vehicle_pose = delay_ego_params["odometry"]["ego_pos"]
        delay_ego_lidar_pose = get_abs_world_pose(
            delay_ego_lidar_rel_pose, delay_ego_vehicle_pose
        )

        delay_params["cur_ego_lidar_pose"] = cur_ego_lidar_pose
        delay_params["delay_ego_lidar_pose"] = delay_ego_lidar_pose

        # load current timestamp world objects and then filter
        cur_objects = load_pickle(cav_content[timestamp_cur]["objects"])
        delay_objects = load_pickle(cav_content[timestamp_delay]["objects"])
        cur_ego_objects = load_pickle(ego_content[timestamp_cur]["objects"])
        delay_ego_objects = load_pickle(ego_content[timestamp_delay]["objects"])

        cur_objects = filter_objects(cur_objects)

        # print(f"cav keys reform: {cav_content.keys()}, {ego_content.keys()}")
        agent_type = cav_content[timestamp_cur]["agent_type"]
        # if agent_type == "drone":
        #     cur_bevcam_intrinsic = np.array(
        #         cur_params["bev_camera"]["intrinsic"], dtype=np.float32
        #     )
        #     delay_bevcam_intrinsic = np.array(
        #         delay_params["bev_camera"]["intrinsic"], dtype=np.float32
        #     )

        #     # get abs bevcam pose under world
        #     cur_bevcam_rel_pose = cur_params["bev_camera"]["cords"]
        #     cur_drone_pose = cur_params["odometry"]["ego_pos"]
        #     cur_bevcam_pose = get_abs_world_pose(cur_bevcam_rel_pose, cur_drone_pose)

        #     delay_bevcam_rel_pose = delay_params["bev_camera"]["cords"]
        #     delay_drone_pose = delay_params["odometry"]["ego_pos"]
        #     delay_bevcam_pose = get_abs_world_pose(
        #         delay_bevcam_rel_pose, delay_drone_pose
        #     )

        #     # TODO(YH) check here, we use the drone world pose as the ego of the drone
        #     cur_extrinsic = x1_to_x2(cur_bevcam_pose, cur_drone_pose)
        #     delay_extrinsic = x1_to_x2(delay_bevcam_pose, delay_drone_pose)

        #     # don't consider add loc noise for drone for now
        #     if self.loc_err_flag:
        #         delay_bevcam_pose = self.add_loc_noise(
        #             delay_bevcam_pose, self.xyz_noise_std, self.ryp_noise_std
        #         )
        #         cur_bevcam_pose = self.add_loc_noise(
        #             cur_bevcam_pose, self.xyz_noise_std, self.ryp_noise_std
        #         )

        #     if cur_ego_pos_flag:
        #         transformation_matrix = x1_to_x2(
        #             delay_drone_pose, cur_ego_lidar_pose
        #         )  # ego drone -> cur ego lidar
        #         spatial_correction_matrix = np.eye(4)
        #     else:
        #         transformation_matrix = x1_to_x2(
        #             delay_drone_pose, delay_ego_lidar_pose
        #         )  # delay drone -> delay ego
        #         spatial_correction_matrix = x1_to_x2(
        #             delay_ego_lidar_pose, cur_ego_lidar_pose
        #         )  # delay ego -> cur ego

        #     delay_params["transformation_matrix"] = (
        #         transformation_matrix  # delay drone -> delay ego lidar
        #     )
        #     delay_params["spatial_correction_matrix"] = (
        #         spatial_correction_matrix  # delay_lidar -> cur lidar
        #     )

        #     delay_params["cur_intrinsic"] = np.expand_dims(
        #         cur_bevcam_intrinsic, axis=0
        #     )  # (Ncam=1, 3, 3)
        #     delay_params["delay_intrinsic"] = np.expand_dims(
        #         delay_bevcam_intrinsic, axis=0
        #     )  # (Ncam=1, 3, 3)
        #     delay_params["cur_extrinsic"] = np.expand_dims(
        #         cur_extrinsic, axis=0
        #     )  # (Ncam=1, 4, 4)
        #     delay_params["delay_extrinsic"] = np.expand_dims(
        #         delay_extrinsic, axis=0
        #     )  # (Ncam=1, 4, 4)
        #     delay_params["cur_bevcam_pos"] = cur_bevcam_pose
        #     delay_params["delay_bevcam_pos"] = delay_bevcam_pose

        # else:  # rsu/vehicle
            # get in/extrinsics of cams
        cur_intrinsics, cur_extrinsics = get_ex_intrinsic(
            cur_params
        )  # cam2img (Ncam, 3, 3), lidar2cam (Ncam, 4, 4)
        delay_intrinsics, delay_extrinsics = get_ex_intrinsic(delay_params)

        # we need to calculate the transformation matrix from cav to ego
        # at the delayed timestamp
        delay_cav_lidar_rel_pose = delay_params["lidar"]["lidar_pose"]
        delay_cav_vehicle_pose = delay_params["odometry"]["ego_pos"]
        delay_cav_lidar_pose = get_abs_world_pose(
            delay_cav_lidar_rel_pose, delay_cav_vehicle_pose
        )

        cur_cav_lidar_rel_pose = cur_params["lidar"]["lidar_pose"]
        cur_cav_vehicle_pose = cur_params["odometry"]["ego_pos"]
        cur_cav_lidar_pose = get_abs_world_pose(
            cur_cav_lidar_rel_pose, cur_cav_vehicle_pose
        )

        delay_params["cur_cav_lidar_pose"] = cur_cav_lidar_pose
        delay_params["delay_cav_lidar_pose"] = delay_cav_lidar_pose

        if not cav_content["ego"] and self.loc_err_flag:
            delay_cav_lidar_pose = self.add_loc_noise(
                delay_cav_lidar_pose, self.xyz_noise_std, self.ryp_noise_std
            )
            cur_cav_lidar_pose = self.add_loc_noise(
                cur_cav_lidar_pose, self.xyz_noise_std, self.ryp_noise_std
            )

        if cur_ego_pos_flag:
            transformation_matrix = x1_to_x2(
                delay_cav_lidar_pose, cur_ego_lidar_pose
            )
            spatial_correction_matrix = np.eye(4)
        else:
            transformation_matrix = x1_to_x2(
                delay_cav_lidar_pose, delay_ego_lidar_pose
            )
            spatial_correction_matrix = x1_to_x2(
                delay_ego_lidar_pose, cur_ego_lidar_pose
            )
        delay_params["transformation_matrix"] = (
            transformation_matrix  # delay cav lidar -> delay ego lidar
        )
        delay_params["spatial_correction_matrix"] = (
            spatial_correction_matrix  # delay ego -> cur ego
        )
        delay_params["cur_intrinsic"] = cur_intrinsics  # cam2img
        delay_params["cur_extrinsic"] = cur_extrinsics  # cam2lidar
        delay_params["delay_intrinsic"] = delay_intrinsics  # (Ncam, 3, 3)
        delay_params["delay_extrinsic"] = delay_extrinsics  # (Ncam, 4, 4)

        # This is only used for late fusion, as it did the transformation
        # in the postprocess, so we want the gt object transformation use
        # the correct one
        gt_transformation_matrix = x1_to_x2(cur_cav_lidar_pose, cur_ego_lidar_pose)

        # get all cams world coordinates
        delay_cams_abs_pos = []
        for k, v in delay_params.items():
            if "camera" in k:
                delay_cam_rel = v["cords"]
                delay_cam_abs_pos = get_abs_world_pose(
                    delay_cam_rel, delay_ego_vehicle_pose
                )
                delay_cams_abs_pos.append(delay_cam_abs_pos)
        delay_params["delay_cams_abs_pos"] = np.array(delay_cams_abs_pos)

        cur_cams_abs_pos = []
        for k, v in cur_params.items():
            if "camera" in k:
                cur_cam_rel = v["cords"]
                cur_cam_abs_pos = get_abs_world_pose(
                    cur_cam_rel, cur_ego_vehicle_pose
                )
                cur_cams_abs_pos.append(cur_cam_abs_pos)
        cur_params["cur_cams_abs_pos"] = np.array(cur_cams_abs_pos)  # (Ncam, 6)

        # # we always use current timestamp's gt bbx to gain a fair evaluation

        delay_params["gt_transformation_matrix"] = gt_transformation_matrix
        delay_params["delay_lidar_ego_abs_pos"] = (
            delay_ego_lidar_pose  # use for bm2cp
        )
        delay_params["cur_lidar_ego_abs_pos"] = cur_ego_lidar_pose
        
        
        

        delay_params["objects"] = cur_objects

        return delay_params

    def shuffle_ego(self):
        for scenario_idx, scenario_dict in self.scenario_database.items():
            # collect vehicles that could be ego
            ego_type_idx = []
            for agent_idx, agent_dict in scenario_dict.items():
                agent_dict["ego"] = False
                if next(iter(agent_dict.items()))[1]["agent_type"] == self.ego_type:
                    ego_type_idx.append(agent_idx)
            if self.train:
                # random choose one vehicle as ego
                ego_idx = random.choice(ego_type_idx)
                self.scenario_database[scenario_idx][ego_idx]["ego"] = True
                self.scenario_database[scenario_idx].move_to_end(ego_idx, last=False)
            else:
                first = list(self.scenario_database[scenario_idx].keys())[0]
                self.scenario_database[scenario_idx][first]["ego"] = True

    def calc_dist_to_ego(self, scenario_database, timestamp_key):
        """
        Calculate the distance to ego for each cav.
        """
        ego_lidar_pose = None
        ego_cav_content = None
        # Find ego pose first
        for cav_id, cav_content in scenario_database.items():
            if cav_content["ego"]:
                ego_cav_content = cav_content
                ego_lidar_pose = load_pickle(
                    cav_content[timestamp_key]["metadata_path"]
                )["odometry"]["ego_pos"]
                break

        assert ego_lidar_pose is not None

        # calculate the distance, we use odometry (world coord) here
        for cav_id, cav_content in scenario_database.items():
            # # for drone, we doesn't consider distance for now, ensure connection with cav
            # if cav_content[timestamp_key]["agent_type"] == "drone":
            #     distance = 0  # placeholder
            #     cav_content["distance_to_ego"] = distance
            #     scenario_database.update({cav_id: cav_content})
            # #     continue
            try:
                cur_lidar_pose = load_pickle(
                    cav_content[timestamp_key]["metadata_path"]
                )["odometry"]["ego_pos"]
            except:
                import traceback

                traceback.print_exc()
                import pdb

                pdb.set_trace()

            distance = math.sqrt(
                (cur_lidar_pose[0] - ego_lidar_pose[0]) ** 2
                + (cur_lidar_pose[1] - ego_lidar_pose[1]) ** 2
                + (cur_lidar_pose[2] - ego_lidar_pose[2]) ** 2
            )
            cav_content["distance_to_ego"] = distance
            scenario_database.update({cav_id: cav_content})

        return ego_cav_content

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split("/")[-1]

            timestamp = res.replace(".yaml", "")
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

    # @staticmethod
    # def load_camera_files(cav_path, timestamp):
    #     """
    #     Retrieve the paths to all camera files.

    #     Parameters
    #     ----------
    #     cav_path : str
    #         The full file path of current cav.

    #     timestamp : str
    #         Current timestamp

    #     Returns
    #     -------
    #     camera_files : list
    #         The list containing all camera png file paths.
    #     """

    #     camera0_file = os.path.join(cav_path,
    #                                 timestamp + '_camera0.png')
    #     camera1_file = os.path.join(cav_path,
    #                                 timestamp + '_camera1.png')
    #     camera2_file = os.path.join(cav_path,
    #                                 timestamp + '_camera2.png')
    #     camera3_file = os.path.join(cav_path,
    #                                 timestamp + '_camera3.png')
    #     return [camera0_file, camera1_file, camera2_file, camera3_file]

    # def project_points_to_bev_map(self, points, ratio=0.1):
    #     """
    #     Project points to BEV occupancy map with default ratio=0.1.

    #     Parameters
    #     ----------
    #     points : np.ndarray
    #         (N, 3) / (N, 4)

    #     ratio : float
    #         Discretization parameters. Default is 0.1.

    #     Returns
    #     -------
    #     bev_map : np.ndarray
    #         BEV occupancy map including projected points
    #         with shape (img_row, img_col).

    #     """
    #     return self.pre_processor.project_points_to_bev_map(points, ratio)

    def add_loc_noise(self, pose, xyz_std, ryp_std):
        """
        Add localization noise to the pose.

        Parameters
        ----------
        pose : list
            x,y,z,roll,yaw,pitch

        xyz_std : float
            std of the gaussian noise on xyz

        ryp_std : float
            std of the gaussian noise
        """
        np.random.seed(self.seed)
        xyz_noise = np.random.normal(0, xyz_std, 3)
        ryp_std = np.random.normal(0, ryp_std, 3)
        noise_pose = [
            pose[0] + xyz_noise[0],
            pose[1] + xyz_noise[1],
            pose[2] + xyz_noise[2],
            pose[3],
            pose[4] + ryp_std[1],
            pose[5],
        ]
        return noise_pose

    def time_delay_calculation(self, ego_flag):
        """
        Calculate the time delay for a certain vehicle.

        Parameters
        ----------
        ego_flag : boolean
            Whether the current cav is ego.

        Return
        ------
        time_delay : int
            The time delay quantization.
        """
        # there is not time delay for ego vehicle
        if ego_flag:
            return 0
        # time delay real mode
        if self.async_mode == "real":
            # in the real mode, time delay = systematic async time + data
            # transmission time + backbone computation time
            overhead_noise = np.random.uniform(0, self.async_overhead)
            tc = self.data_size / self.transmission_speed * 1000
            time_delay = int(overhead_noise + tc + self.backbone_delay)
        elif self.async_mode == "sim":
            # in the simulation mode, the time delay is constant
            time_delay = np.abs(self.async_overhead)

        # the data is 10 hz for both opv2v and v2x-set
        # todo: it may not be true for other dataset like DAIR-V2X and V2X-Sim
        time_delay = time_delay // 100
        return time_delay if self.async_flag else 0

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {
            "lidar_np": lidar_np,
            "object_bbx_center": object_bbx_center,
            "object_bbx_mask": object_bbx_mask,
        }
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict["lidar_np"]
        object_bbx_center = tmp_dict["object_bbx_center"]
        object_bbx_mask = tmp_dict["object_bbx_mask"]

        return lidar_np, object_bbx_center, object_bbx_mask

    def collate_batch_train(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for early and late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # during training, we only care about ego.
        output_dict = {"ego": {}}

        object_bbx_center = []
        object_bbx_mask = []
        processed_lidar_list = []
        label_dict_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]["ego"]
            object_bbx_center.append(ego_dict["object_bbx_center"])
            object_bbx_mask.append(ego_dict["object_bbx_mask"])
            processed_lidar_list.append(ego_dict["processed_lidar"])
            label_dict_list.append(ego_dict["label_dict"])

            if self.visualize:
                origin_lidar.append(ego_dict["origin_lidar"])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        processed_lidar_torch_dict = self.pre_processor.collate_batch(
            processed_lidar_list
        )
        label_torch_dict = self.post_processor.collate_batch_skylink(label_dict_list)
        output_dict["ego"].update(
            {
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": object_bbx_mask,
                "processed_lidar": processed_lidar_torch_dict,
                "anchor_box": torch.from_numpy(ego_dict["anchor_box"]),
                "label_dict": label_torch_dict,
            }
        )
        if self.visualize:
            origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict["ego"].update({"origin_lidar": origin_lidar})

        return output_dict

    def visualize_result(
        self, pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None
    ):
        # visualize the model output
        self.post_processor.visualize(
            pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=dataset
        )

    def generate_object_center(self, cav_contents, reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_object_center(
            cav_contents, reference_lidar_pose
        )

    def _wrap_segmentation_map(self, seg_bev_imgs_list, seg_map_type=None):
        if seg_map_type == "dynamic":
            assert len(seg_bev_imgs_list) == 7, (
                f"we should have 10 seg bev imgs, but got {len(seg_bev_imgs_list)}"
            )

        elif seg_map_type == "static":
            assert len(seg_bev_imgs_list) == 3, (
                f"we should have 3 seg bev imgs, but got {len(seg_bev_imgs_list)}"
            )
        else:
            raise ValueError(
                f"seg_map_type should be dynamic or static, but got {seg_map_type}"
            )

        mask = []
        for filename in seg_bev_imgs_list:
            img = Image.open(filename).convert("L")  # convert to grayscale
            img_np = np.array(img)
            binary_mask = (img_np > 10).astype(np.uint8)  # thresholding
            mask.append(binary_mask)
        mask = np.array(mask)
        label_map = np.zeros((mask.shape[1], mask.shape[2]), dtype=np.uint8)  # (H, W)
        for class_idx in range(mask.shape[0]):
            label_map[mask[class_idx] == 1] = class_idx
        
        label_map = label_map.T
        label_map = label_map[:, ::-1]
        

        # # save img for debug label_map visualization
        # # Define a color map (random or predefined)
        # colors = np.array([
        #     [0, 0, 0],         # class 0 -> black
        #     [255, 0, 0],       # class 1 -> red
        #     [0, 255, 0],       # class 2 -> green
        #     [0, 0, 255],       # class 3 -> blue
        #     [255, 255, 0],     # class 4 -> yellow
        #     [255, 0, 255],     # class 5 -> magenta
        #     [0, 255, 255],     # class 6 -> cyan
        #     [128, 128, 0],     # class 7 -> olive
        #     [128, 0, 128],     # class 8 -> purple
        #     [0, 128, 128],     # class 9 -> teal
        # ], dtype=np.uint8)

        # # Map each label to a color
        # color_image = colors[label_map]  # shape: [H, W, 3]

        # # Save color image
        # img = Image.fromarray(color_image)
        # img.save('dummy_images/label_map_color.png')
        return label_map


if __name__ == "__main__":
    params = load_yaml("/code/opencood/hypes_yaml/skylink/skylink_early.yaml")
    skylink_database = BaseDataset(params, None)
    skylink_database.retrieve_base_data(5)
