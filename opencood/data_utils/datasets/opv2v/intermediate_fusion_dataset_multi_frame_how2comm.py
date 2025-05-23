"""
Dataset class for early fusion
"""

import math
from collections import OrderedDict

import numpy as np
import torch

import opencood
import opencood.data_utils.post_processor as post_processor
from opencood.data_utils.datasets.opv2v import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils import box_utils
from opencood.utils.pcd_utils import (
    downsample_lidar_minimum,
    mask_ego_points,
    mask_points_by_range,
    shuffle_points,
)


class IntermediateFusionDataset(basedataset.BaseDataset):
    def __init__(self, params, visualize, train=True):
        super(IntermediateFusionDataset, self).__init__(params, visualize, train)
        self.cur_ego_pose_flag = params["fusion"]["args"]["cur_ego_pose_flag"]
        self.frame = params["train_params"]["frame"]
        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = post_processor.build_postprocessor(
            params["postprocess"], dataset="opv2v", train=train
        )

        self.proj_first = False
        if "proj_first" in params["fusion"]["args"]:
            self.proj_first = params["fusion"]["args"]["proj_first"]
        print("proj_first: ", self.proj_first)

    def retrieve_base_data_before(
        self, scenario_index, idx, cur_timestamp_key, cur_ego_pose_flag=True
    ):
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = (
            idx if scenario_index == 0 else idx - self.len_record[scenario_index - 1]
        )
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database, timestamp_index)
        # calculate distance to ego for each cav for time delay estimation
        ego_cav_content = self.calc_dist_to_ego(scenario_database, timestamp_key)

        data = OrderedDict()
        # load files for all CAVs self.scenario_database[i][cav_id]['ego'] = True
        for cav_id, cav_content in scenario_database.items():
            if True:
                data[cav_id] = OrderedDict()
                data[cav_id]["ego"] = cav_content["ego"]

                # calculate delay for this vehicle
                timestamp_delay = self.time_delay_calculation(cav_content["ego"])

                if timestamp_index - timestamp_delay <= 0:
                    timestamp_delay = timestamp_index
                timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
                timestamp_key_delay = self.return_timestamp_key(
                    scenario_database, timestamp_index_delay
                )
                # add time delay to vehicle parameters
                data[cav_id]["time_delay"] = timestamp_delay
                # load the corresponding data into the dictionary
                data[cav_id]["params"] = self.reform_param(
                    cav_content,
                    ego_cav_content,
                    cur_timestamp_key,
                    timestamp_key_delay,
                    cur_ego_pose_flag,
                )
                data[cav_id]["lidar_np"] = pcd_utils.pcd_to_np(
                    cav_content[timestamp_key_delay]["lidar"]
                )
                data[cav_id]["folder_name"] = cav_content[timestamp_key_delay][
                    "lidar"
                ].split("/")[-3]
                data[cav_id]["index"] = timestamp_index
                data[cav_id]["cav_id"] = int(cav_id)
                data[cav_id]["frame_id"] = (
                    data[cav_id]["folder_name"],
                    cav_content[timestamp_key_delay]["lidar"].split("/")[-1][:-4],
                )
        return data

    def __getitem__(self, idx):
        select_num = self.frame
        select_dict, scenario_index, index_list, timestamp_index = (
            self.retrieve_multi_data(
                idx, select_num, cur_ego_pose_flag=self.cur_ego_pose_flag
            )
        )
        if timestamp_index < select_num:
            idx += select_num
        try:
            assert idx == list(select_dict.keys())[0], (
                "The first element in the multi frame must be current index"
            )
        except AssertionError as aeeor:
            print(
                "assert error dataset", list(select_dict.keys()), idx, timestamp_index
            )

        processed_data_list = []
        ego_id = -1
        ego_lidar_pose = []
        ego_id_list = []
        for index, base_data_dict in select_dict.items():
            processed_data_dict = OrderedDict()
            processed_data_dict["ego"] = {}

            if index == idx:
                # first find the ego vehicle's lidar pose
                for cav_id, cav_content in base_data_dict.items():
                    if cav_content["ego"]:
                        ego_id = cav_id
                        ego_lidar_pose = cav_content["params"]["lidar_pose"]
                        break
                assert cav_id == list(base_data_dict.keys())[0], (
                    "The first element in the OrderedDict must be ego"
                )
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0
            ego_id_list.append(ego_id)
            # this is used for v2vnet and disconet
            pairwise_t_matrix = self.get_pairwise_transformation(
                base_data_dict, self.params["train_params"]["max_cav"]
            )

            processed_features = []
            object_stack = []
            object_id_stack = []

            # prior knowledge for time delay correction and indicating data type
            # (V2V vs V2i)
            velocity = []
            time_delay = []
            infra = []
            spatial_correction_matrix = []

            if self.visualize:
                projected_lidar_stack = []

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                distance = math.sqrt(
                    (selected_cav_base["params"]["lidar_pose"][0] - ego_lidar_pose[0])
                    ** 2
                    + (selected_cav_base["params"]["lidar_pose"][1] - ego_lidar_pose[1])
                    ** 2
                )
                # if distance > opencood.data_utils.datasets.COM_RANGE and index == idx:
                #    continue

                selected_cav_processed, void_lidar = self.get_item_single_car(
                    selected_cav_base, ego_lidar_pose
                )

                if void_lidar:
                    continue

                object_stack.append(selected_cav_processed["object_bbx_center"])
                object_id_stack += selected_cav_processed["object_ids"]
                processed_features.append(selected_cav_processed["processed_features"])

                velocity.append(selected_cav_processed["velocity"])
                time_delay.append(float(selected_cav_base["time_delay"]))
                spatial_correction_matrix.append(
                    selected_cav_base["params"]["spatial_correction_matrix"]
                )
                infra.append(1 if int(cav_id) < 0 else 0)

                if self.visualize:
                    projected_lidar_stack.append(
                        selected_cav_processed["projected_lidar"]
                    )

            # exclude all repetitive objects
            unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack)
            object_stack = object_stack[unique_indices]

            # make sure bounding boxes across all frames have the same number
            object_bbx_center = np.zeros((self.params["postprocess"]["max_num"], 7))
            mask = np.zeros(self.params["postprocess"]["max_num"])
            object_bbx_center[: object_stack.shape[0], :] = object_stack
            mask[: object_stack.shape[0]] = 1

            # merge preprocessed features from different cavs into the same dict
            cav_num = len(processed_features)
            merged_feature_dict = self.merge_features_to_dict(processed_features)

            # generate the anchor boxes
            anchor_box = self.post_processor.generate_anchor_box()

            # generate targets label
            label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center, anchors=anchor_box, mask=mask
            )

            # pad dv, dt, infra to max_cav
            velocity = velocity + (self.max_cav - len(velocity)) * [0.0]
            time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.0]
            infra = infra + (self.max_cav - len(infra)) * [0.0]
            spatial_correction_matrix = np.stack(spatial_correction_matrix)
            padding_eye = np.tile(
                np.eye(4)[None], (self.max_cav - len(spatial_correction_matrix), 1, 1)
            )
            spatial_correction_matrix = np.concatenate(
                [spatial_correction_matrix, padding_eye], axis=0
            )

            processed_data_dict["ego"].update(
                {
                    "object_bbx_center": object_bbx_center,
                    "object_bbx_mask": mask,
                    "object_ids": [object_id_stack[i] for i in unique_indices],
                    "anchor_box": anchor_box,
                    "processed_lidar": merged_feature_dict,
                    "label_dict": label_dict,
                    "cav_num": cav_num,
                    "velocity": velocity,
                    "time_delay": time_delay,
                    "infra": infra,
                    "spatial_correction_matrix": spatial_correction_matrix,
                    "pairwise_t_matrix": pairwise_t_matrix,
                }
            )

            if self.visualize:
                processed_data_dict["ego"].update(
                    {"origin_lidar": np.vstack(projected_lidar_stack)}
                )
            processed_data_list.append(processed_data_dict)

        try:
            assert len(set(ego_id_list)) == 1, "The ego id must be same"
        except AssertionError as aeeor:
            print("assert error ego id", ego_id_list)

        return processed_data_list

    @staticmethod
    def get_pairwise_transformation(base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix across different agents.
        This is only used for v2vnet and disconet. Currently we set
        this as identity matrix as the pointcloud is projected to
        ego vehicle first.

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
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))
        # default are identity matrix
        pairwise_t_matrix[:, :] = np.identity(4)

        return pairwise_t_matrix

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
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = selected_cav_base["params"]["transformation_matrix"]

        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids = (
            self.post_processor.generate_object_center([selected_cav_base], ego_pose)
        )

        # filter lidar
        lidar_np = selected_cav_base["lidar_np"]
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        if self.proj_first:
            lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(
                lidar_np[:, :3], transformation_matrix
            )

        lidar_np = mask_points_by_range(
            lidar_np, self.params["preprocess"]["cav_lidar_range"]
        )
        # Check if filtered LiDAR points are not void
        void_lidar = True if lidar_np.shape[0] < 1 else False

        processed_lidar = self.pre_processor.preprocess(lidar_np)

        # velocity
        velocity = selected_cav_base["params"]["ego_speed"]
        # normalize veloccity by average speed 30 km/h
        velocity = velocity / 30

        selected_cav_processed.update(
            {
                "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                "object_ids": object_ids,
                "projected_lidar": lidar_np,
                "processed_features": processed_lidar,
                "velocity": velocity,
            }
        )

        return selected_cav_processed, void_lidar

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
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

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)

        return merged_feature_dict

    def collate_batch_train(self, batch):
        # Intermediate fusion is different the other two
        output_dict_list = []
        for j in range(len(batch[0])):
            output_dict = {"ego": {}}

            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            processed_lidar_list = []
            # used to record different scenario
            record_len = []
            label_dict_list = []

            # used for PriorEncoding
            velocity = []
            time_delay = []
            infra = []

            # pairwise transformation matrix
            pairwise_t_matrix_list = []

            # used for correcting the spatial transformation between delayed timestamp
            # and current timestamp
            spatial_correction_matrix_list = []

            if self.visualize:
                origin_lidar = []

            for i in range(len(batch)):
                ego_dict = batch[i][j]["ego"]
                object_bbx_center.append(ego_dict["object_bbx_center"])
                object_bbx_mask.append(ego_dict["object_bbx_mask"])
                object_ids.append(ego_dict["object_ids"])

                processed_lidar_list.append(ego_dict["processed_lidar"])
                record_len.append(ego_dict["cav_num"])
                label_dict_list.append(ego_dict["label_dict"])
                pairwise_t_matrix_list.append(ego_dict["pairwise_t_matrix"])

                velocity.append(ego_dict["velocity"])
                time_delay.append(ego_dict["time_delay"])
                infra.append(ego_dict["infra"])
                spatial_correction_matrix_list.append(
                    ego_dict["spatial_correction_matrix"]
                )

                if self.visualize:
                    origin_lidar.append(ego_dict["origin_lidar"])
            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            # example: {'voxel_features':[np.array([1,2,3]]),
            # np.array([3,5,6]), ...]}
            merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
            processed_lidar_torch_dict = self.pre_processor.collate_batch(
                merged_feature_dict
            )
            # [2, 3, 4, ..., M]
            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            label_torch_dict = self.post_processor.collate_batch(label_dict_list)

            # (B, max_cav)
            velocity = torch.from_numpy(np.array(velocity))
            time_delay = torch.from_numpy(np.array(time_delay))
            infra = torch.from_numpy(np.array(infra))
            spatial_correction_matrix_list = torch.from_numpy(
                np.array(spatial_correction_matrix_list)
            )
            # (B, max_cav, 3)
            prior_encoding = torch.stack([velocity, time_delay, infra], dim=-1).float()
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            output_dict["ego"].update(
                {
                    "object_bbx_center": object_bbx_center,
                    "object_bbx_mask": object_bbx_mask,
                    "processed_lidar": processed_lidar_torch_dict,
                    "record_len": record_len,
                    "label_dict": label_torch_dict,
                    "command": command_torch_list,
                    "object_ids": object_ids[0],
                    "prior_encoding": prior_encoding,
                    "spatial_correction_matrix": spatial_correction_matrix_list,
                    "pairwise_t_matrix": pairwise_t_matrix,
                    "lidar_pose": lidar_poses,
                    "ego_flag": ego_flag,
                }
            )

            if self.visualize:
                origin_lidar = np.array(
                    downsample_lidar_minimum(pcd_np_list=origin_lidar)
                )
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict["ego"].update({"origin_lidar": origin_lidar})
            output_dict_list.append(output_dict)

        return output_dict_list

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict_list = self.collate_batch_train(batch)

        # check if anchor box in the batch
        for i in range(len(batch[0])):
            if batch[0][i]["ego"]["anchor_box"] is not None:
                output_dict_list[i]["ego"].update(
                    {
                        "anchor_box": torch.from_numpy(
                            np.array(batch[0][i]["ego"]["anchor_box"])
                        )
                    }
                )

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()
            output_dict_list[i]["ego"].update(
                {"transformation_matrix": transformation_matrix_torch}
            )

        return output_dict_list

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
        # pred_box_tensor, pred_score = self.post_processor.post_process(data_dict, output_dict)
        preds = self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        # return pred_box_tensor, pred_score, gt_box_tensor
        return preds + (gt_box_tensor,)
