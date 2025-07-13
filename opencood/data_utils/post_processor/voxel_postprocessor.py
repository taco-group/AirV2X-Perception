# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
3D Anchor Generator for Voxel
"""

import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import sigmoid

from opencood.data_utils.post_processor.base_postprocessor import BasePostprocessor
from opencood.utils import box_utils
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.utils.common_utils import limit_period
from opencood.visualization import vis_utils


class VoxelPostprocessor(BasePostprocessor):
    def __init__(self, anchor_params, dataset, train):
        super(VoxelPostprocessor, self).__init__(anchor_params, dataset, train)
        self.anchor_num = self.params["anchor_args"].get("num", 2)
        self.num_class = self.params["anchor_args"].get("num_class", 7)
        ego_type = self.params["ego_type"]
        self.lidar_range = self.params["anchor_args"]["cav_lidar_range"]

    def generate_anchor_box(self):
        W = self.params["anchor_args"]["W"]
        H = self.params["anchor_args"]["H"]

        l = self.params["anchor_args"]["l"]
        w = self.params["anchor_args"]["w"]
        h = self.params["anchor_args"]["h"]
        r = self.params["anchor_args"]["r"]

        assert self.anchor_num == len(r)
        r = [math.radians(ele) for ele in r]

        vh = self.params["anchor_args"]["vh"]  # voxel_size
        vw = self.params["anchor_args"]["vw"]

        xrange = [
            self.lidar_range[0],
            self.lidar_range[3],
        ]
        yrange = [
            self.lidar_range[1],
            self.lidar_range[4],
        ]

        if "feature_stride" in self.params["anchor_args"]:
            feature_stride = self.params["anchor_args"]["feature_stride"]
        else:
            feature_stride = 2

        x = np.linspace(xrange[0] + vw, xrange[1] - vw, W // feature_stride)
        y = np.linspace(yrange[0] + vh, yrange[1] - vh, H // feature_stride)

        cx, cy = np.meshgrid(x, y)
        cx = np.tile(cx[..., np.newaxis], self.anchor_num)  # center
        cy = np.tile(cy[..., np.newaxis], self.anchor_num)
        cz = np.ones_like(cx) * -1.0

        w = np.ones_like(cx) * w
        l = np.ones_like(cx) * l
        h = np.ones_like(cx) * h

        r_ = np.ones_like(cx)
        for i in range(self.anchor_num):
            r_[..., i] = r[i]

        if self.params["order"] == "hwl":  # pointpillar
            anchors = np.stack([cx, cy, cz, h, w, l, r_], axis=-1)  # (50, 176, 2, 7)

        elif self.params["order"] == "lhw":
            anchors = np.stack([cx, cy, cz, l, h, w, r_], axis=-1)
        else:
            sys.exit("Unknown bbx order.")

        return anchors

    def generate_label(self, **kwargs):
        """
        Generate targets for training.

        Parameters
        ----------
        argv : list
            gt_box_center:(max_num, 7), anchor:(H, W, anchor_num, 7)

        Returns
        -------
        label_dict : dict
            Dictionary that contains all target related info.
        """
        assert self.params["order"] == "hwl", (
            "Currently Voxel only supporthwl bbx order."
        )
        # (max_num, 7)
        gt_box_center = kwargs["gt_box_center"]
        # (H, W, anchor_num, 7)
        anchors = kwargs["anchors"]
        # (max_num)
        masks = kwargs["mask"]

        # (H, W)
        feature_map_shape = anchors.shape[:2]

        # (H*W*anchor_num, 7)
        anchors = anchors.reshape(-1, 7)
        # normalization factor, (H * W * anchor_num)
        anchors_d = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2)

        # (H, W, 2)
        pos_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        neg_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        # (H, W, self.anchor_num * 7)
        targets = np.zeros((*feature_map_shape, self.anchor_num * 7))

        # (n, 7)
        gt_box_center_valid = gt_box_center[masks == 1]
        # (n, 8, 3)
        gt_box_corner_valid = box_utils.boxes_to_corners_3d(
            gt_box_center_valid, self.params["order"]
        )
        # (H*W*anchor_num, 8, 3)
        anchors_corner = box_utils.boxes_to_corners_3d(
            anchors, order=self.params["order"]
        )
        # (H*W*anchor_num, 4)
        anchors_standup_2d = box_utils.corner2d_to_standup_box(anchors_corner)
        # (n, 4)
        gt_standup_2d = box_utils.corner2d_to_standup_box(gt_box_corner_valid)

        # (H*W*anchor_n)
        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )

        # the anchor boxes has the largest iou across
        # shape: (n)
        id_highest = np.argmax(iou.T, axis=1)
        # [0, 1, 2, ..., n-1]
        id_highest_gt = np.arange(iou.T.shape[0])
        # make sure all highest iou is larger than 0
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        # find anchors iou > params['pos_iou']
        id_pos, id_pos_gt = np.where(iou > self.params["target_args"]["pos_threshold"])
        #  find anchors iou  params['neg_iou']
        id_neg = np.where(
            np.sum(iou < self.params["target_args"]["neg_threshold"], axis=1)
            == iou.shape[1]
        )[0]
        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()

        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(
            id_pos, (*feature_map_shape, self.anchor_num)
        )
        pos_equal_one[index_x, index_y, index_z] = 1

        # calculate the targets
        targets[index_x, index_y, np.array(index_z) * 7] = (
            gt_box_center[id_pos_gt, 0] - anchors[id_pos, 0]
        ) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = (
            gt_box_center[id_pos_gt, 1] - anchors[id_pos, 1]
        ) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = (
            gt_box_center[id_pos_gt, 2] - anchors[id_pos, 2]
        ) / anchors[id_pos, 3]
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
            gt_box_center[id_pos_gt, 3] / anchors[id_pos, 3]
        )
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
            gt_box_center[id_pos_gt, 4] / anchors[id_pos, 4]
        )
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            gt_box_center[id_pos_gt, 5] / anchors[id_pos, 5]
        )
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
            gt_box_center[id_pos_gt, 6] - anchors[id_pos, 6]
        )

        index_x, index_y, index_z = np.unravel_index(
            id_neg, (*feature_map_shape, self.anchor_num)
        )
        neg_equal_one[index_x, index_y, index_z] = 1

        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(
            id_highest, (*feature_map_shape, self.anchor_num)
        )
        neg_equal_one[index_x, index_y, index_z] = 0

        label_dict = {
            "pos_equal_one": pos_equal_one,
            "neg_equal_one": neg_equal_one,
            "targets": targets,
        }

        return label_dict

    def generate_label_airv2x(self, **kwargs):
        """
        Generate targets for training.

        Parameters
        ----------
        argv : list
            gt_box_center:(max_num, 7), anchor:(H, W, anchor_num, 7)

        Returns
        -------
        label_dict : dict
            Dictionary that contains all target related info.
        """
        assert self.params["order"] == "hwl", (
            "Currently Voxel only supporthwl bbx order."
        )
        # (max_num, 7)
        gt_box_center = kwargs["gt_box_center"]
        # (H, W, anchor_num, 7)
        anchors = kwargs["anchors"]
        # (max_num)
        masks = kwargs["mask"]

        # (max_num)
        class_ids_padded = kwargs["class_ids_padded"]
        # (n)
        class_ids_valid = class_ids_padded[masks == 1]

        # (H, W)
        feature_map_shape = anchors.shape[:2]

        # (H*W*anchor_num, 7)
        anchors = anchors.reshape(-1, 7)
        # normalization factor, (H * W * anchor_num)
        anchors_d = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2)

        # (H, W, 2)
        pos_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        neg_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        # (H, W, self.anchor_num * 7)
        targets = np.zeros((*feature_map_shape, self.anchor_num * 7))

        # (n, 7)
        gt_box_center_valid = gt_box_center[masks == 1]
        # (n, 8, 3)
        gt_box_corner_valid = box_utils.boxes_to_corners_3d(
            gt_box_center_valid, self.params["order"]
        )
        # (H*W*anchor_num, 8, 3)
        anchors_corner = box_utils.boxes_to_corners_3d(
            anchors, order=self.params["order"]
        )
        # (H*W*anchor_num, 4)
        anchors_standup_2d = box_utils.corner2d_to_standup_box(anchors_corner)
        # (n, 4)
        gt_standup_2d = box_utils.corner2d_to_standup_box(gt_box_corner_valid)

        # (H*W*anchor_n)
        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )

        # the anchor boxes has the largest iou across
        # shape: (n)
        id_highest = np.argmax(iou.T, axis=1)
        # [0, 1, 2, ..., n-1]
        id_highest_gt = np.arange(iou.T.shape[0])
        # make sure all highest iou is larger than 0
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        # find anchors iou > params['pos_iou']
        id_pos, id_pos_gt = np.where(iou > self.params["target_args"]["pos_threshold"])
        #  find anchors iou  params['neg_iou']
        id_neg = np.where(
            np.sum(iou < self.params["target_args"]["neg_threshold"], axis=1)
            == iou.shape[1]
        )[0]
        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()

        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(
            id_pos, (*feature_map_shape, self.anchor_num)
        )
        pos_equal_one[index_x, index_y, index_z] = 1

        # match cls to anchors
        cls_labels = np.zeros((*feature_map_shape, self.anchor_num), dtype=int)
        cls_labels[index_x, index_y, index_z] = class_ids_valid[id_pos_gt]

        # calculate the targets
        targets[index_x, index_y, np.array(index_z) * 7] = (
            gt_box_center[id_pos_gt, 0] - anchors[id_pos, 0]
        ) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = (
            gt_box_center[id_pos_gt, 1] - anchors[id_pos, 1]
        ) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = (
            gt_box_center[id_pos_gt, 2] - anchors[id_pos, 2]
        ) / anchors[id_pos, 3]
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
            gt_box_center[id_pos_gt, 3] / anchors[id_pos, 3]
        )
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
            gt_box_center[id_pos_gt, 4] / anchors[id_pos, 4]
        )
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            gt_box_center[id_pos_gt, 5] / anchors[id_pos, 5]
        )
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
            gt_box_center[id_pos_gt, 6] - anchors[id_pos, 6]
        )

        index_x, index_y, index_z = np.unravel_index(
            id_neg, (*feature_map_shape, self.anchor_num)
        )
        neg_equal_one[index_x, index_y, index_z] = 1

        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(
            id_highest, (*feature_map_shape, self.anchor_num)
        )
        neg_equal_one[index_x, index_y, index_z] = 0

        label_dict = {
            "pos_equal_one": pos_equal_one,
            "neg_equal_one": neg_equal_one,
            "targets": targets,
            "cls_labels": cls_labels,
        }

        return label_dict

    @staticmethod
    def collate_batch(label_batch_list):
        """
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several
            frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        pos_equal_one = []
        neg_equal_one = []
        targets = []

        for i in range(len(label_batch_list)):
            pos_equal_one.append(label_batch_list[i]["pos_equal_one"])
            neg_equal_one.append(label_batch_list[i]["neg_equal_one"])
            targets.append(label_batch_list[i]["targets"])

        pos_equal_one = torch.from_numpy(np.array(pos_equal_one))
        neg_equal_one = torch.from_numpy(np.array(neg_equal_one))
        targets = torch.from_numpy(np.array(targets))

        return {
            "targets": targets,
            "pos_equal_one": pos_equal_one,
            "neg_equal_one": neg_equal_one,
        }

    @staticmethod
    def collate_batch_airv2x(label_batch_list):
        """
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several
            frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        pos_equal_one = []
        neg_equal_one = []
        targets = []
        class_ids = []

        for i in range(len(label_batch_list)):
            # print(label_batch_list[i].keys())
            pos_equal_one.append(label_batch_list[i]["pos_equal_one"])
            neg_equal_one.append(label_batch_list[i]["neg_equal_one"])
            targets.append(label_batch_list[i]["targets"])
            class_ids.append(label_batch_list[i]["cls_labels"])

        pos_equal_one = torch.from_numpy(np.array(pos_equal_one))
        neg_equal_one = torch.from_numpy(np.array(neg_equal_one))
        targets = torch.from_numpy(np.array(targets))

        class_ids = torch.from_numpy(np.array(class_ids))

        return {
            "targets": targets,
            "pos_equal_one": pos_equal_one,
            "neg_equal_one": neg_equal_one,
            "class_ids": class_ids,
        }

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        For early and intermediate fusion,
            data_dict only contains ego.

        For late fusion,
            data_dcit contains all cavs, so we need transformation matrix.


        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        # the final bounding box list
        pred_box3d_list = []
        pred_box2d_list = []

        for cav_id, cav_content in data_dict.items():
            assert cav_id in output_dict
            # the transformation matrix to ego space
            transformation_matrix = cav_content["transformation_matrix"]  # no clean

            # (H, W, anchor_num, 7)
            anchor_box = cav_content["anchor_box"]

            # classification probability
            prob = output_dict[cav_id]["psm"]
            prob = F.sigmoid(prob.permute(0, 2, 3, 1))
            prob = prob.reshape(1, -1)

            # regression map
            reg = output_dict[cav_id]["rm"]

            # convert regression map back to bounding box
            batch_box3d = self.delta_to_boxes3d(reg, anchor_box)
            mask = torch.gt(prob, self.params["target_args"]["score_threshold"])
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            # during validation/testing, the batch size should be 1
            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask[0])

            # adding dir classifier
            if "dm" in output_dict[cav_id].keys() and len(boxes3d) != 0:
                dir_offset = self.params["dir_args"]["dir_offset"]
                num_bins = self.params["dir_args"]["num_bins"]

                dm = output_dict[cav_id]["dm"]  # [N, H, W, 4]
                dir_cls_preds = (
                    dm.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins)
                )  # [1, N*H*W*2, 2]
                dir_cls_preds = dir_cls_preds[mask]
                # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
                dir_labels = torch.max(
                    dir_cls_preds, dim=-1
                )[
                    1
                ]  # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0

                period = 2 * np.pi / num_bins  # pi
                dir_rot = limit_period(
                    boxes3d[..., 6] - dir_offset, 0, period
                )  # 限制在0到pi之间
                boxes3d[..., 6] = (
                    dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype)
                )  # 转化0.25pi到2.5pi
                boxes3d[..., 6] = limit_period(
                    boxes3d[..., 6], 0.5, 2 * np.pi
                )  # limit to [-pi, pi]

            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = box_utils.boxes_to_corners_3d(
                    boxes3d, order=self.params["order"]
                )

                # STEP 2
                # (N, 8, 3)
                projected_boxes3d = box_utils.project_box3d(
                    boxes3d_corner, transformation_matrix
                )
                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = box_utils.corner_to_standup_box_torch(
                    projected_boxes3d
                )
                # (N, 5)
                boxes2d_score = torch.cat(
                    (projected_boxes2d, scores.unsqueeze(1)), dim=1
                )

                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)

        if len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0:
            return None, None
        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores
        scores = pred_box2d_list[:, -1]
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        # remove large bbx
        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor, self.dataset)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]

        # STEP3
        # nms
        keep_index = box_utils.nms_rotated(
            pred_box3d_tensor, scores, self.params["nms_thresh"]
        )

        pred_box3d_tensor = pred_box3d_tensor[keep_index]

        # select cooresponding score
        scores = scores[keep_index]

        # filter out the prediction out of the range.
        # mask = box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor, self.params['gt_range'])
        mask = box_utils.get_mask_for_boxes_within_range_torch(
            pred_box3d_tensor, self.lidar_range
        )
        pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        scores = scores[mask]

        assert scores.shape[0] == pred_box3d_tensor.shape[0]

        # return pred_box3d_tensor, scores, count
        return pred_box3d_tensor, scores

    @staticmethod
    def delta_to_boxes3d(deltas, anchors):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, W, L, 14)?? should be (N, 14, H, W)
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr

        Returns
        -------
        box3d : torch.Tensor
            (N, W*L*2, 7)
        """
        # batch size
        N = deltas.shape[0]
        deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        boxes3d = torch.zeros_like(deltas)

        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        # (W*L*2, 7)
        anchors_reshaped = anchors.view(-1, 7).float()
        # the diagonal of the anchor 2d box, (W*L*2)
        anchors_d = torch.sqrt(
            anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2
        )
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        # Inv-normalize to get xyz
        boxes3d[..., [0, 1]] = (
            torch.mul(deltas[..., [0, 1]], anchors_d) + anchors_reshaped[..., [0, 1]]
        )
        boxes3d[..., [2]] = (
            torch.mul(deltas[..., [2]], anchors_reshaped[..., [3]])
            + anchors_reshaped[..., [2]]
        )
        # hwl
        boxes3d[..., [3, 4, 5]] = (
            torch.exp(deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        )
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d

    @staticmethod
    def visualize(pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None):
        """
        Visualize the prediction, ground truth with point cloud together.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        pcd : torch.Tensor
            PointCloud, (N, 4).

        show_vis : bool
            Whether to show visualization.

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        """
        vis_utils.visualize_single_sample_output_gt(
            pred_box_tensor, gt_tensor, pcd, show_vis, save_path
        )

    def post_process_airv2x(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        For early and intermediate fusion,
            data_dict only contains ego.

        For late fusion,
            data_dcit contains all cavs, so we need transformation matrix.


        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        pred_box3d_list = []
        pred_box2d_list = []
        pred_label_list = []
        boxes3d_list = [] # saving unprojected boxes3d

        C = self.num_class # num of classes
        Nanchor = self.anchor_num  # number of anchor per location
        for cav_id, cav_content in data_dict.items():
            assert cav_id in output_dict
            transformation_matrix = cav_content["transformation_matrix"]
            anchor_box = cav_content["anchor_box"]
            
            # objectness
            obj_preds = output_dict[cav_id]["obj"]  # [1, A, H, W]
            obj_preds = obj_preds.permute(0, 2, 3, 1).contiguous()  # [1, H, W, A]
            objectness = torch.sigmoid(obj_preds).view(1, -1)  # [1, N]
            
            psm = output_dict[cav_id]["psm"]  # shape: [1, A*C, H, W]
            B, AC, H, W = psm.shape
            C = getattr(self.params, "num_class", C)
            A = AC // C

            psm = psm.view(B, C, A, H, W)
            psm = psm.permute(0, 3, 4, 2, 1).contiguous()  # [1, H, W, A, C]
            prob = torch.sigmoid(psm)
            prob = prob.view(1, -1, C)  # [1, H*W*A, C]]
            prob = prob[:, :, 1:]
            
            class_scores, class_labels = torch.max(prob, dim=-1)  # [1, N] 
            class_labels = class_labels + 1

            # Apply score threshold and ignore background class (0)
            # score_thresh = self.params["target_args"]["score_threshold"]
            
            obj_thresh = self.params["target_args"]["obj_threshold"]
            # print(f"obj: {objectness}, score: {class_scores}")
            # non_bg_mask = class_labels == 3
            mask = (objectness > obj_thresh)

            if mask.sum() == 0:
                continue

            # regression map
            reg = output_dict[cav_id]["rm"]
            batch_box3d = self.delta_to_boxes3d(reg, anchor_box)  # [1, H, W, A, 7]
            
            assert batch_box3d.shape[0] == 1, (
                f"inference only has 1 batch, but got {batch_box3d.shape}"
            )
            mask_reg = mask.unsqueeze(-1).repeat(1, 1, 7)
            
            # mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)
            boxes3d = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, 7)
            scores3d = torch.masked_select(objectness[0], mask[0])
            labels3d = torch.masked_select(class_labels[0], mask[0])

            # Project boxes to ego space
            if len(boxes3d) > 0:
                boxes3d_corner = box_utils.boxes_to_corners_3d(
                    boxes3d, order=self.params["order"]
                )
                projected_boxes3d = box_utils.project_box3d(
                    boxes3d_corner, transformation_matrix
                )
                projected_boxes2d = box_utils.corner_to_standup_box_torch(
                    projected_boxes3d
                )
                boxes2d_score = torch.cat(
                    (projected_boxes2d, scores3d.unsqueeze(1)), dim=1
                )

                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)
                pred_label_list.append(labels3d)
                boxes3d_list.append(boxes3d)

            # # convert output to bounding box
            # if len(boxes3d) != 0:
            #     # (N, 8, 3)
            #     boxes3d_corner = box_utils.boxes_to_corners_3d(
            #         boxes3d, order=self.params["order"]
            #     )

            #     # STEP 2
            #     # (N, 8, 3)
            #     projected_boxes3d = box_utils.project_box3d(
            #         boxes3d_corner, transformation_matrix
            #     )
            #     # convert 3d bbx to 2d, (N,4)
            #     projected_boxes2d = box_utils.corner_to_standup_box_torch(
            #         projected_boxes3d
            #     )
            #     # (N, 5)
            #     boxes2d_score = torch.cat(
            #         (projected_boxes2d, scores.unsqueeze(1)), dim=1
            #     )

            #     pred_box2d_list.append(boxes2d_score)
            #     pred_box3d_list.append(projected_boxes3d)
            

        if len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0:
            return None, None, None, None

        # Concatenate results across CAVs
        pred_box2d_tensor = torch.vstack(pred_box2d_list)
        scores = pred_box2d_tensor[:, -1]
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        labels = torch.cat(pred_label_list)
        boxes3d = torch.cat(boxes3d_list)

        # Post-filtering: large boxes, abnormal Z, etc.
        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor, self.dataset)
        
        z_min = self.lidar_range[2]
        z_max = self.lidar_range[5]
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor, 
                                                       z_min=z_min, z_max=z_max)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]
        labels = labels[keep_index]
        boxes3d = boxes3d[keep_index]

        # Rotated NMS
        keep_index = box_utils.nms_rotated(
            pred_box3d_tensor, scores, self.params["nms_thresh"]
        )
        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]
        labels = labels[keep_index]
        boxes3d = boxes3d[keep_index]

        # Filter by range
        mask = box_utils.get_mask_for_boxes_within_range_torch(
            pred_box3d_tensor, self.lidar_range
        )
        pred_box3d_tensor = pred_box3d_tensor[mask]
        scores = scores[mask]
        labels = labels[mask]
        boxes3d = boxes3d[mask]

        assert scores.shape[0] == pred_box3d_tensor.shape[0] == labels.shape[0]
        return pred_box3d_tensor, scores, labels, boxes3d
    

    def post_process_segmentation_airv2x(self, data_dict, output_dict):
        pred_dynamic_seg_map = output_dict["ego"]["dynamic_seg"]
        pred_static_seg_map = output_dict["ego"]["static_seg"]

        pred_dynamic_seg_map = torch.sigmoid(pred_dynamic_seg_map)
        pred_static_seg_map = torch.sigmoid(pred_static_seg_map)

        pred_dynamic_seg_map = pred_dynamic_seg_map.permute(0, 2, 3, 1).contiguous()
        pred_static_seg_map = pred_static_seg_map.permute(0, 2, 3, 1).contiguous()

        # get label at each pixel

        pred_dynamic_seg_map = torch.argmax(pred_dynamic_seg_map, dim=-1) # shape: (1, H, W)
        pred_static_seg_map = torch.argmax(pred_static_seg_map, dim=-1) # shape: (1, H, W)

        # get gt
        gt_dynamic_seg_map = data_dict["ego"]["label_dict"]["dynamic_seg_label"]
        gt_static_seg_map = data_dict["ego"]["label_dict"]["static_seg_label"]

        return pred_dynamic_seg_map, pred_static_seg_map, gt_dynamic_seg_map, gt_static_seg_map