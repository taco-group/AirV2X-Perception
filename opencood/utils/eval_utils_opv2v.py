# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os

import numpy as np
import torch

from opencood.hypes_yaml import yaml_utils
from opencood.utils import common_utils


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap, mrec, mpre


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend]  # from high to low
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)

        result_stat[iou_thresh]["score"] += det_score.tolist()

    result_stat[iou_thresh]["fp"] += fp
    result_stat[iou_thresh]["tp"] += tp
    result_stat[iou_thresh]["gt"] += gt


def calculate_ap(result_stat, iou, global_sort_detections):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.

    iou : float
        The threshold of iou.

    global_sort_detections : bool
        Whether to sort the detection results globally.
    """
    iou_5 = result_stat[iou]

    if global_sort_detections:
        fp = np.array(iou_5["fp"])
        tp = np.array(iou_5["tp"])
        score = np.array(iou_5["score"])

        assert len(fp) == len(tp) and len(tp) == len(score)
        sorted_index = np.argsort(-score)
        fp = fp[sorted_index].tolist()
        tp = tp[sorted_index].tolist()

    else:
        fp = iou_5["fp"]
        tp = iou_5["tp"]
        assert len(fp) == len(tp)

    gt_total = iou_5["gt"]

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(
    result_stat, save_path, global_sort_detections=False, eval_epoch=None
):
    dump_dict = {}

    ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30, global_sort_detections)
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50, global_sort_detections)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70, global_sort_detections)

    dump_dict.update(
        {
            "ap_30": ap_30,
            "ap_50": ap_50,
            "ap_70": ap_70,
            "mpre_50": mpre_50,
            "mrec_50": mrec_50,
            "mpre_70": mpre_70,
            "mrec_70": mrec_70,
        }
    )

    # output_file = 'eval.yaml' if not global_sort_detections else 'eval_global_sort.yaml'
    output_file = (
        f"eval_epoch{eval_epoch}.yaml"
        if not global_sort_detections
        else "eval_global_sort.yaml"
    )
    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, output_file))

    print(
        "The Average Precision at IOU 0.3 is %.2f, "
        "The Average Precision at IOU 0.5 is %.2f, "
        "The Average Precision at IOU 0.7 is %.2f" % (ap_30, ap_50, ap_70)
    )
    return ap_30, ap_50, ap_70

# # ==================================================================================
# # mAP for multi-class
# # ==================================================================================

# def calculate_multiclass_tp_fp(det_boxes, det_score, gt_boxes, iou_thresh, result_stat):
#     """
#     Compute TP/FP for each class separately.

#     Parameters
#     ----------
#     det_boxes_dict : dict of torch.Tensor
#         Detected boxes per class.
#     det_score_dict : dict of torch.Tensor
#         Confidence scores per class.
#     gt_boxes_dict : dict of torch.Tensor
#         Groundtruth boxes per class.
#     iou_thresh : float
#         IoU threshold.
#     class_ids : list
#         Class IDs.
#     result_stat : dict
#         Statistics for evaluation.
#     """
#     fp = {}
#     tp = {}
#     gt = gt_boxes.shape[0]
#     if det_boxes is not None:
#         det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
#         det_score = common_utils.torch_tensor_to_numpy(det_score)
#         gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

#         # sort the prediction bounding box by score
#         score_order_descend = np.argsort(-det_score)
#         det_score = det_score[score_order_descend]  # from high to low
#         det_polygon_list = list(common_utils.convert_format(det_boxes))
#         gt_polygon_list = list(common_utils.convert_format(gt_boxes))

#         # match prediction and gt bounding box


#     for cls_id in class_ids:
#         det_boxes = det_boxes_dict.get(cls_id, torch.empty((0, 4, 2)))
#         det_score = det_score_dict.get(cls_id, torch.empty((0,)))
#         gt_boxes = gt_boxes_dict.get(cls_id, torch.empty((0, 4, 2)))
#         caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat[cls_id], iou_thresh)

# def calculate_multiclass_map(result_stat, iou_thresholds, global_sort_detections):
#     """
#     Compute mean AP over all classes.

#     Parameters
#     ----------
#     result_stat : dict
#         Statistics containing TP, FP, score, and gt count.
#     iou_thresholds : list of float
#         List of IoU thresholds.
#     global_sort_detections : bool
#         Sort detection globally.
    
#     Returns
#     -------
#     dict
#         Average precisions per class and mAP.
#     """
#     ap_result = {}
#     for iou in iou_thresholds:
#         aps = []
#         for cls_id, stat in result_stat.items():
#             ap, _, _ = calculate_ap(stat, iou, global_sort_detections)
#             aps.append(ap)
#         ap_result[f"mAP@{iou}"] = np.mean(aps)
#     return ap_result

# def eval_multiclass_results(result_stat, save_path, global_sort_detections=False, eval_epoch=None):
#     """
#     Evaluate multi-class results and save to YAML.

#     Parameters
#     ----------
#     result_stat : dict
#         Statistics for evaluation.
#     save_path : str
#         Path to save the results.
#     global_sort_detections : bool, optional
#         Sort detection globally (default is False).
#     eval_epoch : int, optional
#         Evaluation epoch (default is None).
    
#     Returns
#     -------
#     dict
#         Average precisions per class and mAP.
#     """
#     dump_dict = {}

#     iou_thresholds = [0.3, 0.5, 0.7]
#     map_result = calculate_multiclass_map(result_stat, iou_thresholds, global_sort_detections)

#     for key, value in map_result.items():
#         dump_dict[key] = value

#     output_file = (
#         f"eval_epoch{eval_epoch}.yaml"
#         if not global_sort_detections
#         else "eval_global_sort.yaml"
#     )
#     yaml_utils.save_yaml(dump_dict, os.path.join(save_path, output_file))

#     print(
#         "The mean Average Precision at IOU 0.3 is %.2f, "
#         "The mean Average Precision at IOU 0.5 is %.2f, "
#         "The mean Average Precision at IOU 0.7 is %.2f" % (map_result["mAP@0.3"], map_result["mAP@0.5"], map_result["mAP@0.7"])
#     )
    
#     return map_result