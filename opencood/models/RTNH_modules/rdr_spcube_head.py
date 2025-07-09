"""
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
"""

import nms  # pip install requirements.txt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

from opencood.utils.cuda_op import sort_vertices  # utils\Rotated_IoU\cuda_op

EPSILON = 1e-8


# from utils.Rotated_IoU.oriented_iou_loss import cal_iou
def box_intersection_th(corners1: torch.Tensor, corners2: torch.Tensor):
    """find intersection points of rectangles
    Convention: if two edges are collinear, there is no intersection point

    Args:
        corners1 (torch.Tensor): B, N, 4, 2
        corners2 (torch.Tensor): B, N, 4, 2

    Returns:
        intersectons (torch.Tensor): B, N, 4, 4, 2
        mask (torch.Tensor) : B, N, 4, 4; bool
    """
    # build edges from corners
    line1 = torch.cat(
        [corners1, corners1[:, :, [1, 2, 3, 0], :]], dim=3
    )  # B, N, 4, 4: Batch, Box, edge, point
    line2 = torch.cat([corners2, corners2[:, :, [1, 2, 3, 0], :]], dim=3)
    # duplicate data to pair each edges from the boxes
    # (B, N, 4, 4) -> (B, N, 4, 4, 4) : Batch, Box, edge1, edge2, point
    line1_ext = line1.unsqueeze(3).repeat([1, 1, 1, 4, 1])
    line2_ext = line2.unsqueeze(2).repeat([1, 1, 4, 1, 1])
    x1 = line1_ext[..., 0]
    y1 = line1_ext[..., 1]
    x2 = line1_ext[..., 2]
    y2 = line1_ext[..., 3]
    x3 = line2_ext[..., 0]
    y3 = line2_ext[..., 1]
    x4 = line2_ext[..., 2]
    y4 = line2_ext[..., 3]
    # math: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    num = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    den_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t = den_t / num
    t[num == 0.0] = -1.0
    mask_t = (t > 0) * (t < 1)  # intersection on line segment 1
    den_u = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
    u = -den_u / num
    u[num == 0.0] = -1.0
    mask_u = (u > 0) * (u < 1)  # intersection on line segment 2
    mask = mask_t * mask_u
    t = den_t / (
        num + EPSILON
    )  # overwrite with EPSILON. otherwise numerically unstable
    intersections = torch.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dim=-1)
    intersections = intersections * mask.float().unsqueeze(-1)
    return intersections, mask


def box1_in_box2(corners1: torch.Tensor, corners2: torch.Tensor):
    """check if corners of box1 lie in box2
    Convention: if a corner is exactly on the edge of the other box, it's also a valid point

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)

    Returns:
        c1_in_2: (B, N, 4) Bool
    """
    a = corners2[:, :, 0:1, :]  # (B, N, 1, 2)
    b = corners2[:, :, 1:2, :]  # (B, N, 1, 2)
    d = corners2[:, :, 3:4, :]  # (B, N, 1, 2)
    ab = b - a  # (B, N, 1, 2)
    am = corners1 - a  # (B, N, 4, 2)
    ad = d - a  # (B, N, 1, 2)
    p_ab = torch.sum(ab * am, dim=-1)  # (B, N, 4)
    norm_ab = torch.sum(ab * ab, dim=-1)  # (B, N, 1)
    p_ad = torch.sum(ad * am, dim=-1)  # (B, N, 4)
    norm_ad = torch.sum(ad * ad, dim=-1)  # (B, N, 1)
    # NOTE: the expression looks ugly but is stable if the two boxes are exactly the same
    # also stable with different scale of bboxes
    cond1 = (p_ab / norm_ab > -1e-6) * (p_ab / norm_ab < 1 + 1e-6)  # (B, N, 4)
    cond2 = (p_ad / norm_ad > -1e-6) * (p_ad / norm_ad < 1 + 1e-6)  # (B, N, 4)
    return cond1 * cond2


def box_in_box_th(corners1: torch.Tensor, corners2: torch.Tensor):
    """check if corners of two boxes lie in each other

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)

    Returns:
        c1_in_2: (B, N, 4) Bool. i-th corner of box1 in box2
        c2_in_1: (B, N, 4) Bool. i-th corner of box2 in box1
    """
    c1_in_2 = box1_in_box2(corners1, corners2)
    c2_in_1 = box1_in_box2(corners2, corners1)
    return c1_in_2, c2_in_1


def build_vertices(
    corners1: torch.Tensor,
    corners2: torch.Tensor,
    c1_in_2: torch.Tensor,
    c2_in_1: torch.Tensor,
    inters: torch.Tensor,
    mask_inter: torch.Tensor,
):
    """find vertices of intersection area

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)
        c1_in_2 (torch.Tensor): Bool, (B, N, 4)
        c2_in_1 (torch.Tensor): Bool, (B, N, 4)
        inters (torch.Tensor): (B, N, 4, 4, 2)
        mask_inter (torch.Tensor): (B, N, 4, 4)

    Returns:
        vertices (torch.Tensor): (B, N, 24, 2) vertices of intersection area. only some elements are valid
        mask (torch.Tensor): (B, N, 24) indicates valid elements in vertices
    """
    # NOTE: inter has elements equals zero and has zeros gradient (masked by multiplying with 0).
    # can be used as trick
    B = corners1.size()[0]
    N = corners1.size()[1]
    vertices = torch.cat(
        [corners1, corners2, inters.view([B, N, -1, 2])], dim=2
    )  # (B, N, 4+4+16, 2)
    mask = torch.cat(
        [c1_in_2, c2_in_1, mask_inter.view([B, N, -1])], dim=2
    )  # Bool (B, N, 4+4+16)
    return vertices, mask


def calculate_area(idx_sorted: torch.Tensor, vertices: torch.Tensor):
    """calculate area of intersection

    Args:
        idx_sorted (torch.Tensor): (B, N, 9)
        vertices (torch.Tensor): (B, N, 24, 2)

    return:
        area: (B, N), area of intersection
        selected: (B, N, 9, 2), vertices of polygon with zero padding
    """
    idx_ext = idx_sorted.unsqueeze(-1).repeat([1, 1, 1, 2])
    selected = torch.gather(vertices, 2, idx_ext)
    total = (
        selected[:, :, 0:-1, 0] * selected[:, :, 1:, 1]
        - selected[:, :, 0:-1, 1] * selected[:, :, 1:, 0]
    )
    total = torch.sum(total, dim=2)
    area = torch.abs(total) / 2
    return area, selected


class SortVertices(Function):
    @staticmethod
    def forward(ctx, vertices, mask, num_valid):
        idx = sort_vertices.sort_vertices_forward(vertices, mask, num_valid)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, gradout):
        return ()


sort_v = SortVertices.apply


def sort_indices(vertices: torch.Tensor, mask: torch.Tensor):
    """[summary]

    Args:
        vertices (torch.Tensor): float (B, N, 24, 2)
        mask (torch.Tensor): bool (B, N, 24)

    Returns:
        sorted_index: bool (B, N, 9)

    Note:
        why 9? the polygon has maximal 8 vertices. +1 to duplicate the first element.
        the index should have following structure:
            (A, B, C, ... , A, X, X, X)
        and X indicates the index of arbitary elements in the last 16 (intersections not corners) with
        value 0 and mask False. (cause they have zero value and zero gradient)
    """
    num_valid = torch.sum(mask.int(), dim=2).int()  # (B, N)
    mean = torch.sum(
        vertices * mask.float().unsqueeze(-1), dim=2, keepdim=True
    ) / num_valid.unsqueeze(-1).unsqueeze(-1)
    vertices_normalized = vertices - mean  # normalization makes sorting easier
    return sort_v(vertices_normalized, mask, num_valid).long()


def oriented_box_intersection_2d(corners1: torch.Tensor, corners2: torch.Tensor):
    """calculate intersection area of 2d rectangles

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)

    Returns:
        area: (B, N), area of intersection
        selected: (B, N, 9, 2), vertices of polygon with zero padding
    """
    inters, mask_inter = box_intersection_th(corners1, corners2)
    c12, c21 = box_in_box_th(corners1, corners2)
    vertices, mask = build_vertices(corners1, corners2, c12, c21, inters, mask_inter)
    sorted_indices = sort_indices(vertices, mask)
    return calculate_area(sorted_indices, vertices)


def box2corners_th(box: torch.Tensor) -> torch.Tensor:
    """convert box coordinate to corners

    Args:
        box (torch.Tensor): (B, N, 5) with x, y, w, h, alpha

    Returns:
        torch.Tensor: (B, N, 4, 2) corners
    """
    B = box.size()[0]
    x = box[..., 0:1]
    y = box[..., 1:2]
    w = box[..., 2:3]
    h = box[..., 3:4]
    alpha = box[..., 4:5]  # (B, N, 1)
    x4 = (
        torch.FloatTensor([0.5, -0.5, -0.5, 0.5])
        .unsqueeze(0)
        .unsqueeze(0)
        .to(box.device)
    )  # (1,1,4)
    x4 = x4 * w  # (B, N, 4)
    y4 = (
        torch.FloatTensor([0.5, 0.5, -0.5, -0.5])
        .unsqueeze(0)
        .unsqueeze(0)
        .to(box.device)
    )
    y4 = y4 * h  # (B, N, 4)
    corners = torch.stack([x4, y4], dim=-1)  # (B, N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)  # (B, N, 2)
    rot_T = torch.stack([row1, row2], dim=-2)  # (B, N, 2, 2)
    rotated = torch.bmm(corners.view([-1, 4, 2]), rot_T.view([-1, 2, 2]))
    rotated = rotated.view([B, -1, 4, 2])  # (B*N, 4, 2) -> (B, N, 4, 2)
    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated


def cal_iou(box1: torch.Tensor, box2: torch.Tensor):
    """calculate iou

    Args:
        box1 (torch.Tensor): (B, N, 5)
        box2 (torch.Tensor): (B, N, 5)

    Returns:
        iou (torch.Tensor): (B, N)
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners1 (torch.Tensor): (B, N, 4, 2)
        U (torch.Tensor): (B, N) area1 + area2 - inter_area
    """
    corners1 = box2corners_th(box1)
    corners2 = box2corners_th(box2)
    inter_area, _ = oriented_box_intersection_2d(corners1, corners2)  # (B, N)
    area1 = box1[:, :, 2] * box1[:, :, 3]
    area2 = box2[:, :, 2] * box2[:, :, 3]
    u = area1 + area2 - inter_area
    iou = inter_area / u
    return iou, corners1, corners2, u


class RdrSpcubeHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.roi = self.cfg["RDR_SP_CUBE"]["ROI"]
        self.grid_size = self.cfg["RDR_SP_CUBE"]["GRID_SIZE"]

        try:
            # self.nms_thr = self.cfg.MODEL.HEAD.NMS_OVERLAP_THRESHOLD
            self.nms_thr = self.cfg["HEAD"]["NMS_OVERLAP_THRESHOLD"]
        except:
            print("* Exception error (Head): nms threshold is set as 0.3")
            self.nms_thr = 0.3

        ### Anchors ###
        self.anchor_per_grid = []
        num_anchor_temp = 0

        self.list_anchor_classes = []
        self.list_anchor_matched_thr = []
        self.list_anchor_unmatched_thr = []
        self.list_anchor_targ_idx = []
        self.list_anchor_idx = []  # to slice tensor

        # for inference
        self.list_anc_idx_to_cls_id = []  # except bg
        # self.dict_cls_name_to_id = self.cfg.DATASET.CLASS_INFO.CLASS_ID
        self.dict_cls_name_to_id = self.cfg["CLASS_INFO"]["CLASS_ID"]
        # self.dict_cls_id_to_name = dict()
        # for k, v in self.dict_cls_name_to_id.items():
        #     if v != -1:
        #         self.dict_cls_id_to_name[v] = k

        num_prior_anchor_idx = 0
        for info_anchor in self.cfg["ANCHOR_GENERATOR_CONFIG"]:  # per class
            now_cls_name = info_anchor["class_name"]
            self.list_anchor_classes.append(now_cls_name)
            self.list_anchor_matched_thr.append(info_anchor["matched_threshold"])
            self.list_anchor_unmatched_thr.append(info_anchor["unmatched_threshold"])

            self.anchor_sizes = info_anchor["anchor_sizes"]
            self.anchor_rotations = info_anchor["anchor_rotations"]
            self.anchor_bottoms = info_anchor["anchor_bottom_heights"]

            self.list_anchor_targ_idx.append(num_prior_anchor_idx)
            num_now_anchor = int(
                len(self.anchor_sizes)
                * len(self.anchor_rotations)
                * len(self.anchor_bottoms)
            )
            num_now_anchor_idx = num_prior_anchor_idx + num_now_anchor
            self.list_anchor_idx.append(num_now_anchor_idx)
            num_prior_anchor_idx = num_now_anchor_idx

            for anchor_size in self.anchor_sizes:  # per size
                for anchor_rot in self.anchor_rotations:  # per rot
                    for (
                        anchor_bottom
                    ) in self.anchor_bottoms:  # per bottom: for predicting zc
                        temp_anchor = (
                            [anchor_bottom]
                            + anchor_size
                            + [np.cos(anchor_rot), np.sin(anchor_rot)]
                        )
                        num_anchor_temp += 1
                        self.anchor_per_grid.append(
                            temp_anchor
                        )  # [bot, xl, yl, zl, cos, sin]
                        self.list_anc_idx_to_cls_id.append(
                            self.dict_cls_name_to_id[now_cls_name]
                        )
        self.num_anchor_per_grid = num_anchor_temp
        # self.num_class = self.cfg.DATASET.CLASS_INFO.NUM_CLS
        self.num_class = self.cfg["CLASS_INFO"]["NUM_CLS"]
        self.num_box_code = len(self.cfg["HEAD"]["BOX_CODE"])
        ### Anchors ###

        ### 1x1 conv ###
        input_channels = self.cfg["HEAD"]["DIM"]
        self.conv_cls = nn.Conv2d(
            input_channels,
            cfg["anchor_num"],  # plus one for background
            kernel_size=1,
        )
        self.conv_reg = nn.Conv2d(input_channels, 7 * cfg["anchor_num"], kernel_size=1)
        ### 1x1 conv ###

        ### Loss & Logging ###
        self.bg_weight = cfg["HEAD"]["BG_WEIGHT"]
        self.categorical_focal_loss = FocalLoss()
        self.is_logging = cfg["GENERAL"]["LOGGING"]["IS_LOGGING"]
        ### Loss & Logging ###

        ### Anchor map ###
        self.anchor_map_for_batch = self.create_anchors().cuda()  # no batch
        ### Anchor map ###

    def forward(self, dict_item):
        spatial_features_2d = dict_item["bev_feat"]
        cls_pred = self.conv_cls(spatial_features_2d)  # B x num_anchor+1 x Y x X
        reg_pred = self.conv_reg(
            spatial_features_2d
        )  # B x num_anchor*num_box_code x Y x X
        dict_item.update(
            {
                "psm": cls_pred,
                "rm": reg_pred,
            }
        )

        return dict_item

    # V2
    def create_anchors(self):
        """
        * e.g., 2 anchors (a,b) per class for 3 classes (A,B,C),
        *       anchor order -> (Aa Ab Ba Bb Ca Cc)
        """
        dtype = torch.float32
        x_min, x_max = self.roi["x"]
        y_min, y_max = self.roi["y"]
        grid_size = self.grid_size
        n_x = int((x_max - x_min) / grid_size)
        n_y = int((y_max - y_min) / grid_size)

        # anchor location = center
        half_grid_size = grid_size / 2.0
        anchor_y = (
            torch.arange(y_min, y_max, grid_size, dtype=dtype) - half_grid_size
        )  # minus: checked with visualization
        anchor_x = torch.arange(x_min, x_max, grid_size, dtype=dtype) - half_grid_size
        # print(anchor_y) # 200
        # print(anchor_x) # 248

        anchor_y = anchor_y.repeat_interleave(n_x)
        anchor_x = anchor_x.repeat(n_y)
        # print(anchor_y.shape) # 49600
        # print(anchor_x.shape) # 49600

        flattened_anchor_map = (
            torch.stack((anchor_x, anchor_y), dim=1)
            .unsqueeze(0)
            .repeat(self.num_anchor_per_grid, 1, 1)
        )
        # print(flattened_anchor_map.shape) # 2 x 49600 x 2 (xc, yc)
        flattened_anchor_attr = torch.tensor(self.anchor_per_grid, dtype=dtype)
        # print(flattened_anchor_attr.shape) # 2 x 6 (bottom, xl, yl, zl, cos, sin)
        flattened_anchor_attr = flattened_anchor_attr.unsqueeze(1).repeat(
            1, flattened_anchor_map.shape[1], 1
        )
        # print(flattened_anchor_attr.shape) # 2 x 49600 x 6

        anchor_map = (
            torch.cat((flattened_anchor_map, flattened_anchor_attr), dim=-1)
            .view(self.num_anchor_per_grid, n_y, n_x, 8)
            .contiguous()
            .permute(0, 3, 1, 2)
        )
        anchor_map = anchor_map.reshape(-1, n_y, n_x).contiguous()  # 16, 200, 248
        # print(anchor_map.shape) # 2 * (2+6) x 200 x 248

        anchor_map_for_batch = anchor_map.unsqueeze(0)  # 1 x 16 x 200 x 248

        return anchor_map_for_batch

    def loss(self, dict_item):
        cls_pred = dict_item["pred"]["cls"]
        reg_pred = dict_item["pred"]["reg"]

        dtype, device = cls_pred.dtype, cls_pred.device
        B, _, n_y, n_x = cls_pred.shape
        num_grid_per_anchor = int(n_y * n_x)
        # print(num_grid_per_anchor)

        anchor_maps = self.anchor_map_for_batch.repeat(B, 1, 1, 1)

        reg_pred = anchor_maps + reg_pred  # prediction = residual

        # for iou calculation
        cls_pred = cls_pred.view(B, 1 + self.num_anchor_per_grid, n_y, n_x)
        reg_pred = reg_pred.view(B, self.num_anchor_per_grid, -1, n_y, n_x)

        # make labels
        anc_idx_targets = torch.full((B, n_y, n_x), -1, dtype=torch.long, device=device)

        pos_reg_pred = []
        pos_reg_targ = []

        is_label_contain_objs = False  # at least one
        for batch_idx, list_objs in enumerate(dict_item["label"]):
            if len(list_objs) != 0:
                is_label_contain_objs = True

            prior_anc_idx = 0
            list_anchor_per_cls = []
            for idx_anc_cls, anc_cls_name in enumerate(self.list_anchor_classes):
                now_anc_idx = self.list_anchor_idx[idx_anc_cls]
                # x,y,xl,yl,theta (sin, cos)
                temp_anc = torch.cat(
                    (
                        reg_pred[batch_idx, prior_anc_idx:now_anc_idx, :2],
                        reg_pred[batch_idx, prior_anc_idx:now_anc_idx, 3:5],
                        torch.atan(
                            reg_pred[batch_idx, prior_anc_idx:now_anc_idx, 6:7]
                            / reg_pred[batch_idx, prior_anc_idx:now_anc_idx, 5:6]
                        ),
                    ),
                    dim=1,
                )
                # print(temp_anc.shape) # n_anchor x 5 x n_y x n_x
                temp_anc = temp_anc.permute(0, 2, 3, 1).contiguous()
                temp_anc = temp_anc.view(1, -1, 5)
                # print(temp_anc.shape) # 1 x (n_anchor*n_y*n_x) x 5
                list_anchor_per_cls.append(temp_anc)
                prior_anc_idx = now_anc_idx

            for label_idx, label in enumerate(list_objs):
                cls_name, cls_id, (xc, yc, zc, rz, xl, yl, zl), _ = label

                # find anc idx
                idx_anc_cls = self.list_anchor_classes.index(cls_name)
                pred_anchors = list_anchor_per_cls[idx_anc_cls]
                cls_targ_idx = self.list_anchor_targ_idx[idx_anc_cls]
                matched_iou_thr = self.list_anchor_matched_thr[idx_anc_cls]
                unmatched_iou_thr = self.list_anchor_unmatched_thr[idx_anc_cls]
                # print(cls_targ_idx, matched_iou_thr, unmatched_iou_thr)

                label_anchor = torch.tensor(
                    [xc, yc, xl, yl, rz], dtype=dtype, device=device
                )
                # making same size as pred_anchors
                label_anchor = (
                    label_anchor.unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(1, pred_anchors.shape[1], 1)
                )

                iou, _, _, _ = cal_iou(label_anchor, pred_anchors)

                pos_iou_anc_idx = torch.where(iou > matched_iou_thr)[1]
                # make at least 1 pos box
                if len(pos_iou_anc_idx) == 0:
                    # print(torch.argmax(iou))
                    # print(torch.max(iou))
                    pos_iou_anc_idx = (torch.argmax(iou)).reshape(1)

                neg_iou_anc_idx = torch.where(iou < unmatched_iou_thr)[1]
                # print(torch.max(neg_iou_anc_idx))
                neg_iou_anc_idx = torch.remainder(neg_iou_anc_idx, num_grid_per_anchor)
                # print(torch.max(neg_iou_anc_idx))

                # print(iou.shape) # check total sum = iou
                # print(pos_iou_anc_idx.shape)
                # print(torch.where(torch.logical_and(iou<=matched_iou_thr, iou>=unmatched_iou_thr))[1].shape)
                # print(neg_iou_anc_idx.shape)

                idx_y_neg = torch.div(neg_iou_anc_idx, n_x, rounding_mode="trunc")
                idx_x_neg = torch.remainder(neg_iou_anc_idx, n_x)

                anc_idx_targets[batch_idx, idx_y_neg, idx_x_neg] = 0  # 'Background'

                pos_iou_anc_idx_offset = torch.div(
                    pos_iou_anc_idx, num_grid_per_anchor, rounding_mode="trunc"
                )  # for classification
                pos_iou_anc_idx = torch.remainder(
                    pos_iou_anc_idx, num_grid_per_anchor
                )  # for Y, X

                idx_y_pos = torch.div(pos_iou_anc_idx, n_x, rounding_mode="trunc")
                idx_x_pos = torch.remainder(pos_iou_anc_idx, n_x)

                # 1 for background, cls_targ_idx for class, offset for anc_idx
                temp_anc_idx_targets = cls_targ_idx + pos_iou_anc_idx_offset
                anc_idx_targets[batch_idx, idx_y_pos, idx_x_pos] = (
                    1 + temp_anc_idx_targets
                )  # plus 1 for background

                temp_reg_box_pred = reg_pred[
                    batch_idx, temp_anc_idx_targets, :, idx_y_pos, idx_x_pos
                ]
                temp_num_pos, _ = temp_reg_box_pred.shape
                temp_reg_box_targ = torch.tensor(
                    [[xc, yc, zc, xl, yl, zl, np.cos(rz), np.sin(rz)]],
                    dtype=dtype,
                    device=device,
                ).repeat((temp_num_pos, 1))
                # print(temp_reg_box_targ.shape)

                pos_reg_pred.append(temp_reg_box_pred)  # remain batch dim for concat
                pos_reg_targ.append(temp_reg_box_targ)

        if not is_label_contain_objs:  # All batches without objs
            loss_reg = 0.0
            loss_cls = 0.0  # focal loss
        else:
            ### Focal loss ###
            counted_anc_idx = torch.where(
                anc_idx_targets > -1
            )  # pos and neg boxes only

            # anc_idx (long)
            anc_idx_targets_counted = anc_idx_targets[counted_anc_idx]
            # print(anc_idx_targets_counted.shape)

            # logit
            anc_logit_counted = cls_pred[
                counted_anc_idx[0], :, counted_anc_idx[1], counted_anc_idx[2]
            ]
            # print(anc_logit_counted.shape)

            # weight
            anc_cls_weights = torch.ones(1 + self.num_anchor_per_grid, device=device)
            for idx_anc in range(1 + self.num_anchor_per_grid):
                len_targ_anc = float(
                    len(torch.where(anc_idx_targets_counted == idx_anc)[0])
                )
                if idx_anc == 0:  # background
                    temp_weight = self.bg_weight / len_targ_anc
                else:
                    if len_targ_anc == 0.0:  # if nothing in such class
                        temp_weight = 0.0
                    else:
                        temp_weight = 1.0 / len_targ_anc
                anc_cls_weights[idx_anc] = min(temp_weight, 1.0)

            self.categorical_focal_loss.weight = anc_cls_weights
            loss_cls = self.categorical_focal_loss(
                anc_logit_counted, anc_idx_targets_counted
            )
            ### Focal loss ###

            pos_reg_pred = torch.cat(pos_reg_pred, dim=0)
            pos_reg_targ = torch.cat(pos_reg_targ, dim=0)
            loss_reg = torch.nn.functional.smooth_l1_loss(pos_reg_pred, pos_reg_targ)
        total_loss = loss_cls + loss_reg

        if self.is_logging:
            dict_item["logging"] = dict()
            dict_item["logging"].update(
                self.logging_dict_loss(total_loss, "total_loss")
            )
            dict_item["logging"].update(self.logging_dict_loss(loss_reg, "loss_reg"))
            dict_item["logging"].update(
                self.logging_dict_loss(loss_cls, "focal_loss_cls")
            )

        return total_loss

    def logging_dict_loss(self, loss, name_key):
        try:
            log_loss = loss.cpu().detach().item()
        except:
            log_loss = loss  # for 0. loss

        return {name_key: log_loss}

    ### Validation & Inference ###
    def get_nms_pred_boxes_for_single_sample(self, dict_item, conf_thr, is_nms=True):
        """
        * This function is common function of head for validataion & inference
        * For convenience, we assume batch_size = 1
        """
        cls_pred = dict_item["pred"]["cls"][0]  # (1+n_anc) x n_y x n_x
        reg_pred = dict_item["pred"]["reg"][0]  # n_anc x n_y x n_x
        anchor_map = self.anchor_map_for_batch[0]
        reg_pred = anchor_map + reg_pred

        device = cls_pred.device

        # n_y x n_x -> (n_y*n_x)
        cls_pred = cls_pred.view(cls_pred.shape[0], -1)
        reg_pred = reg_pred.view(reg_pred.shape[0], -1)

        # bg X & more than conf_thr
        cls_pred = torch.softmax(cls_pred, dim=0)
        idx_deal = torch.where(
            (torch.argmax(cls_pred, dim=0) != 0)
            & (torch.max(cls_pred, dim=0)[0] > conf_thr)
        )

        # for finding cls (not anc idx)
        tensor_anc_idx_per_cls = torch.tensor(
            self.list_anc_idx_to_cls_id, dtype=torch.long, device=device
        )

        len_deal_anc = len(idx_deal[0])
        # print('* debug # of dealing anchor boxes == n_anc:', len_deal_anc)
        if len_deal_anc > 0:  # for only dealing grids (not bg & more than conf)
            grid_anc_cls_logit = cls_pred[
                :, idx_deal[0]
            ]  # logit x n_pred / slice 1 for bg
            grid_anc_cls_idx = torch.argmax(
                grid_anc_cls_logit, dim=0
            )  # minus 1 for bg after get conf
            grid_reg = reg_pred[:, idx_deal[0]]

            # to arange anc
            idx_range_anc = torch.arange(
                0, len_deal_anc, dtype=torch.long, device=device
            )
            anc_conf_score = grid_anc_cls_logit[
                grid_anc_cls_idx, idx_range_anc
            ].unsqueeze(0)  # 1 x n_deal
            grid_anc_cls_idx = grid_anc_cls_idx - 1  # minus 1 for bg
            # print(anc_conf_score) # check if it is larger than conf_thr

            list_sliced_reg_bbox = []
            idx_slice_start = (grid_anc_cls_idx * self.num_box_code).long()

            for idx_reg_value in range(self.num_box_code):
                list_sliced_reg_bbox.append(
                    grid_reg[idx_slice_start + idx_reg_value, idx_range_anc]
                )  # n_deal
            sliced_reg_bbox = torch.stack(list_sliced_reg_bbox)
            # print(sliced_reg_bbox.shape) # 8 x n_deal
            temp_angle = torch.atan2(
                sliced_reg_bbox[-1, :], sliced_reg_bbox[-2, :]
            ).unsqueeze(0)
            # print(temp_angle.shape)
            pred_reg_bbox_with_conf = torch.cat(
                (anc_conf_score, sliced_reg_bbox[:-2, :], temp_angle), dim=0
            )  # conf, x, y, z, xl, yl, zl, theta
            pred_reg_bbox_with_conf = pred_reg_bbox_with_conf.transpose(0, 1)
            # print(pred_reg_bbox_with_conf.shape) # n_anc x 8 (score, x, y, z, xl, yl, zl, theta)

            cls_id_per_anc = tensor_anc_idx_per_cls[grid_anc_cls_idx]
            # print(cls_id_per_anc.shape) # n_anc
            num_of_bbox = len_deal_anc
            # print('* debug before nms: ', num_of_bbox)

            ### nms ###
            try:
                if is_nms:
                    pred_reg_xy_xlyl_th = (
                        torch.cat(
                            (
                                pred_reg_bbox_with_conf[:, 1:3],
                                pred_reg_bbox_with_conf[:, 4:6],
                                pred_reg_bbox_with_conf[:, 7:8],
                            ),
                            dim=1,
                        )
                        .cpu()
                        .detach()
                        .numpy()
                    )

                    c_list = list(map(tuple, pred_reg_xy_xlyl_th[:, :2]))

                    ### Assert error (Height > 0) ###
                    dim_list = list(map(np.abs, pred_reg_xy_xlyl_th[:, 2:4]))
                    ### Assert error (Height > 0) ###

                    dim_list = list(map(tuple, pred_reg_xy_xlyl_th[:, 2:4]))
                    angle_list = list(map(float, pred_reg_xy_xlyl_th[:, 4]))

                    list_tuple_for_nms = [
                        [a, b, c] for (a, b, c) in zip(c_list, dim_list, angle_list)
                    ]
                    conf_score = pred_reg_bbox_with_conf[:, 0:1].cpu().detach().numpy()

                    indices = nms.rboxes(
                        list_tuple_for_nms, conf_score, nms_threshold=self.nms_thr
                    )
                    pred_reg_bbox_with_conf = pred_reg_bbox_with_conf[indices]
                    cls_id_per_anc = cls_id_per_anc[indices]

                    num_of_bbox = len(indices)  # after nms
            except:
                print(
                    "* Exception error (head.py): nms error, probably assert height > 0"
                )

            pred_reg_bbox_with_conf = (
                pred_reg_bbox_with_conf.cpu().detach().numpy().tolist()
            )
            cls_id_per_anc = cls_id_per_anc.cpu().detach().numpy().tolist()
            # print('* debug after nms: ', num_of_bbox)
        else:
            # empty prediction
            pred_reg_bbox_with_conf = None
            cls_id_per_anc = None
            num_of_bbox = 0  # 0

        # pp: post-processing
        dict_item["pp_bbox"] = (
            pred_reg_bbox_with_conf  # score, x, y, z, xl, yl, zl, theta
        )
        dict_item["pp_cls"] = cls_id_per_anc
        dict_item["pp_desc"] = dict_item["meta"][0]["desc"]
        dict_item["pp_num_bbox"] = num_of_bbox
        # print(dict_item['label'][0])
        # print(dict_item['pp_bbox'])

        return dict_item

    ### Validation & Inference ###


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = nn.functional.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)

        return nn.functional.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction,
        )
