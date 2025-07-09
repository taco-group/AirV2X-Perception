# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os

import numpy as np
import torch

from opencood.hypes_yaml import yaml_utils
from opencood.utils import common_utils
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt



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

    if gt_total == 0:
        return 0.0, [], []

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


# ==================================================================================
# mAP for multi-class
# ==================================================================================


def calculate_multiclass_tp_fp(
    det_boxes,
    det_score,
    det_labels,
    gt_boxes,
    gt_class_label_list,
    iou_thresh,
    result_stat,
):
    """
    Compute TP/FP for each class separately.

    Parameters
    ----------
    det_boxes_dict : dict of torch.Tensor
        Detected boxes per class.
    det_score_dict : dict of torch.Tensor
        Confidence scores per class.
    det_labels: torch.Tensor
        Detected labels per box
    gt_boxes : torch.Tensor
        Groundtruth boxes.
    gt_class_label_list : list
        List of groundtruth class labels.
    iou_thresh : float
        IoU threshold.
    class_ids : list
        Class IDs.
    result_stat : dict
        Statistics for evaluation.
    """
    device = det_boxes.device if det_boxes is not None else gt_boxes.device
    unique_classes = sorted(set(gt_class_label_list + det_labels.cpu().tolist()))

    for cls_id in unique_classes:
        # Get predicted boxes of this class
        cls_det_mask = det_labels == cls_id
        cls_det_boxes = det_boxes[cls_det_mask]
        cls_det_scores = det_score[cls_det_mask]

        # Get ground-truth boxes of this class
        cls_gt_indices = [
            i for i, label in enumerate(gt_class_label_list) if label == cls_id
        ]
        if len(cls_gt_indices) > 0:
            cls_gt_boxes = gt_boxes[cls_gt_indices]
        else:
            cls_gt_boxes = torch.empty((0, *gt_boxes.shape[1:]), device=device)

        # Initialize if needed
        if cls_id not in result_stat:
            result_stat[cls_id] = {}
        if iou_thresh not in result_stat[cls_id]:
            result_stat[cls_id][iou_thresh] = {"tp": [], "fp": [], "score": [], "gt": len(cls_gt_indices)}

        # Accumulate TP, FP, and scores
        caluclate_tp_fp(
            cls_det_boxes, cls_det_scores, cls_gt_boxes, result_stat[cls_id], iou_thresh
        )


def compute_multiclass_ap_map(result_stat, iou_thresh=0.5, global_sort_detections=True):
    """
    Compute per-class AP and overall mAP for a given IoU threshold.

    Parameters
    ----------
    result_stat : dict[int -> dict[float -> dict]]
        Evaluation results per class. Each entry contains:
            result_stat[class_id][iou_thresh] = {
                "tp": list[int], "fp": list[int], "score": list[float], "gt": int
            }

    iou_thresh : float
        IoU threshold to compute AP (e.g., 0.5 for AP@0.5).

    global_sort_detections : bool
        Whether to sort predictions across all images (global sorting).

    Returns
    -------
    ap_per_class : dict[int -> float]
        AP for each class.

    mAP : float
        Mean AP across all classes.
    """
    ap_per_class = {}
    for class_id in sorted(result_stat.keys()):
        class_stat = result_stat[class_id]

        if iou_thresh not in class_stat:
            print(
                f"[Warning] Class {class_id} does not contain IoU {iou_thresh}. Skipping."
            )
            continue

        ap, _, _ = calculate_ap(
            result_stat[class_id],
            iou=iou_thresh,
            global_sort_detections=global_sort_detections,
        )
        ap_per_class[class_id] = ap

    if len(ap_per_class) == 0:
        print("No valid AP values found. Check result_stat content.")
        mAP = 0.0
    else:
        mAP = np.mean(list(ap_per_class.values()))

    print("==== Evaluation Summary ====")
    for cls_id, ap in ap_per_class.items():
        print(f"Class {cls_id}: AP@{iou_thresh:.2f} = {ap:.4f}")
    print(f"mAP@{iou_thresh:.2f} = {mAP:.4f}")

    return ap_per_class, mAP


def eval_multiclass_results(
    all_det_boxes,
    all_det_scores,
    all_det_labels,
    all_gt_boxes,
    all_gt_class_labels,
    save_path,
    global_sort_detections=False,
    eval_epoch=None,
):
    """
    Complete evaluation for multi-class detection. Computes TP/FP, AP, and mAP at multiple IoU thresholds.

    Parameters
    ----------
    all_det_boxes : list[Tensor]
        List of predicted boxes for all samples, each tensor shape (N_i, 8, 3) or (N_i, 4, 2).

    all_det_scores : list[Tensor]
        List of confidence scores for each sample, shape (N_i,).

    all_det_labels : list[Tensor]
        List of predicted class labels for each sample, shape (N_i,).

    all_gt_boxes : list[Tensor]
        List of ground-truth boxes for each sample, shape (M_i, 8, 3) or (M_i, 4, 2).

    all_gt_class_labels : list[list[int]]
        List of class labels per ground-truth box for each sample.

    save_path : str
        Directory to save the evaluation results.

    global_sort_detections : bool
        Whether to sort all detections globally.

    eval_epoch : int, optional
        Epoch index for naming.

    Returns
    -------
    map_result : dict
        mAP values for each IoU threshold.
    """
    from collections import defaultdict

    # Initialize result statistics per class per IoU
    result_stat = defaultdict(dict)
    iou_thresholds = [0.3, 0.5, 0.7]

    # Accumulate TP/FP per sample
    # for det_boxes, det_scores, det_labels, gt_boxes, gt_class_labels in zip(
    #     all_det_boxes, all_det_scores, all_det_labels, all_gt_boxes, all_gt_class_labels
    # ):
    for iou in iou_thresholds:
        calculate_multiclass_tp_fp(
            all_det_boxes,
            all_det_scores,
            all_det_labels,
            all_gt_boxes,
            all_gt_class_labels,
            iou_thresh=iou,
            result_stat=result_stat,
        )

    # Evaluate per-class AP and mAP
    dump_dict = {}
    map_result = {}
    for iou in iou_thresholds:
        ap_per_class, mAP = compute_multiclass_ap_map(
            result_stat, iou_thresh=iou, global_sort_detections=global_sort_detections
        )

        for class_id, ap in ap_per_class.items():
            dump_dict[f"AP@{iou:.1f}/Class_{class_id}"] = round(ap, 4)
        dump_dict[f"mAP@{iou:.1f}"] = round(mAP, 4)
        map_result[f"mAP@{iou:.1f}"] = mAP

    # Save result
    output_file = (
        f"eval_epoch{eval_epoch}.yaml"
        if eval_epoch is not None and not global_sort_detections
        else "eval_global_sort.yaml"
    )
    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, output_file))

    print(
        "The mean Average Precision at IOU 0.3 is %.2f, "
        "at IOU 0.5 is %.2f, "
        "and at IOU 0.7 is %.2f"
        % (
            map_result["mAP@0.3"],
            map_result["mAP@0.5"],
            map_result["mAP@0.7"],
        )
    )

    return map_result



# ==================================================================================
# segmentation
# ==================================================================================


def to_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(to_json_serializable(i) for i in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif obj is None or isinstance(obj, (str, bool)):
        return obj
    else:
        raise TypeError(f"Type {type(obj)} not serializable")


def evaluate_segmentation(gt_dynamic, pred_dynamic, gt_static, pred_static, class_names_dynamic=None, class_names_static=None, threshold=0.5):
    """
    Evaluates segmentation performance for both dynamic and static segmentation branches.
    
    Args:
        gt_dynamic: Ground truth for dynamic segmentation (objects)
                   Shape: [B, H, W] or [H, W] with integer class labels
        pred_dynamic: Predicted dynamic segmentation after sigmoid
                     Shape: [B, C, H, W] or [C, H, W] with probabilities (0-1)
        gt_static: Ground truth for static segmentation (roads, lanes, background)
                  Shape: [B, H, W] or [H, W] with integer class labels
        pred_static: Predicted static segmentation after sigmoid
                    Shape: [B, C, H, W] or [C, H, W] with probabilities (0-1)
        class_names_dynamic: List of class names for dynamic segmentation
        class_names_static: List of class names for static segmentation
        threshold: Threshold for converting sigmoid probabilities to binary predictions (default: 0.5)
    
    Returns:
        dict: Dictionary containing evaluation metrics for both branches
    """
    results = {
        "dynamic": {},
        "static": {},
        "combined": {}
    }
    
    # Convert tensors to numpy if needed
    if isinstance(gt_dynamic, torch.Tensor):
        gt_dynamic = gt_dynamic.detach().cpu().numpy()
    if isinstance(pred_dynamic, torch.Tensor):
        pred_dynamic = pred_dynamic.detach().cpu().numpy()
    if isinstance(gt_static, torch.Tensor):
        gt_static = gt_static.detach().cpu().numpy()
    if isinstance(pred_static, torch.Tensor):
        pred_static = pred_static.detach().cpu().numpy()
    
    # For sigmoid outputs, we need to apply thresholding for multi-class segmentation
    if len(pred_dynamic.shape) == 4:  # [B, C, H, W]
        # Apply threshold to get binary masks for each class
        pred_dynamic_binary = pred_dynamic > threshold
        # Convert to class indices (take the first class where prediction exceeds threshold)
        pred_dynamic = np.zeros(pred_dynamic.shape[0:1] + pred_dynamic.shape[2:], dtype=np.int32)
        for c in range(pred_dynamic_binary.shape[1]):
            # For each position not yet assigned, assign this class if its probability exceeds the threshold
            mask = (pred_dynamic == 0) & pred_dynamic_binary[:, c]
            pred_dynamic[mask] = c + 1  # +1 because 0 is typically background/unassigned
    elif len(pred_dynamic.shape) == 3 and pred_dynamic.shape[0] > 1:  # [C, H, W]
        # Apply threshold to get binary masks for each class
        pred_dynamic_binary = pred_dynamic > threshold
        # Convert to class indices (take the first class where prediction exceeds threshold)
        pred_dynamic = np.zeros(pred_dynamic.shape[1:], dtype=np.int32)
        for c in range(pred_dynamic_binary.shape[0]):
            # For each position not yet assigned, assign this class if its probability exceeds the threshold
            mask = (pred_dynamic == 0) & pred_dynamic_binary[c]
            pred_dynamic[mask] = c + 1  # +1 because 0 is typically background/unassigned
            
    if len(pred_static.shape) == 4:  # [B, C, H, W]
        # Apply threshold to get binary masks for each class
        pred_static_binary = pred_static > threshold
        # Convert to class indices (take the first class where prediction exceeds threshold)
        pred_static = np.zeros(pred_static.shape[0:1] + pred_static.shape[2:], dtype=np.int32)
        for c in range(pred_static_binary.shape[1]):
            # For each position not yet assigned, assign this class if its probability exceeds the threshold
            mask = (pred_static == 0) & pred_static_binary[:, c]
            pred_static[mask] = c + 1  # +1 because 0 is typically background/unassigned
    elif len(pred_static.shape) == 3 and pred_static.shape[0] > 1:  # [C, H, W]
        # Apply threshold to get binary masks for each class
        pred_static_binary = pred_static > threshold
        # Convert to class indices (take the first class where prediction exceeds threshold)
        pred_static = np.zeros(pred_static.shape[1:], dtype=np.int32)
        for c in range(pred_static_binary.shape[0]):
            # For each position not yet assigned, assign this class if its probability exceeds the threshold
            mask = (pred_static == 0) & pred_static_binary[c]
            pred_static[mask] = c + 1  # +1 because 0 is typically background/unassigned
    # Evaluate dynamic segmentation
    results["dynamic"] = evaluate_branch(gt_dynamic, pred_dynamic, "dynamic", class_names_dynamic)
    
    # Evaluate static segmentation
    results["static"] = evaluate_branch(gt_static, pred_static, "static", class_names_static)
    
    # Calculate combined metrics
    results["combined"]["mean_iou"] = (results["dynamic"]["mean_iou"] + results["static"]["mean_iou"]) / 2
    results["combined"]["mean_dice"] = (results["dynamic"]["mean_dice"] + results["static"]["mean_dice"]) / 2
    results["combined"]["pixel_accuracy"] = (results["dynamic"]["pixel_accuracy"] + results["static"]["pixel_accuracy"]) / 2
    
    return to_json_serializable(results)


def evaluate_branch(gt, pred, branch_name, class_names=None):
    """
    Evaluates a single segmentation branch.
    
    Args:
        gt: Ground truth segmentation with integer class labels
        pred: Predicted segmentation with integer class labels (after thresholding for sigmoid outputs)
        branch_name: Name of the branch ("dynamic" or "static")
        class_names: List of class names for the branch
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    results = {}
    
    # Flatten the arrays for metric calculation
    if len(gt.shape) > 2:  # Handle batch dimension if present
        gt_flat = gt.reshape(-1)
        pred_flat = pred.reshape(-1)
    else:
        gt_flat = gt.flatten()
        pred_flat = pred.flatten()
    
    # Get number of classes
    num_classes = max(gt_flat.max(), pred_flat.max()) + 1
    if class_names is not None:
        num_classes = max(num_classes, len(class_names))
    
    # Calculate IoU for each class
    iou_per_class = []
    dice_per_class = []
    
    for class_idx in range(num_classes):
        # Convert to binary mask for current class
        gt_binary = (gt_flat == class_idx).astype(np.int32)
        pred_binary = (pred_flat == class_idx).astype(np.int32)
        
        # Calculate intersection and union
        intersection = np.logical_and(gt_binary, pred_binary).sum()
        union = np.logical_or(gt_binary, pred_binary).sum()
        
        # Handle division by zero
        if union == 0:
            iou = 1.0  # If both prediction and ground truth agree there's no instance of this class
        else:
            iou = intersection / union
        iou_per_class.append(iou)
        
        # Calculate Dice coefficient
        dice_denominator = gt_binary.sum() + pred_binary.sum()
        if dice_denominator == 0:
            dice = 1.0  # If both prediction and ground truth agree there's no instance of this class
        else:
            dice = 2 * intersection / dice_denominator
        dice_per_class.append(dice)
    
    # Calculate precision, recall, and F1 (handling multi-class)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_flat, pred_flat, labels=range(num_classes), average=None, zero_division=0
    )
    
    # Calculate mean metrics
    results["iou_per_class"] = np.array(iou_per_class)
    results["mean_iou"] = np.mean(iou_per_class)
    results["dice_per_class"] = np.array(dice_per_class)
    results["mean_dice"] = np.mean(dice_per_class)
    results["precision_per_class"] = precision
    results["mean_precision"] = np.mean(precision)
    results["recall_per_class"] = recall
    results["mean_recall"] = np.mean(recall)
    results["f1_per_class"] = f1
    results["mean_f1"] = np.mean(f1)
    
    # Calculate overall pixel accuracy
    results["pixel_accuracy"] = (gt_flat == pred_flat).mean()
    
    # Display class names if provided
    if class_names is not None:
        results["class_names"] = class_names
    
    return results


def visualize_segmentation_results(gt_dynamic, pred_dynamic, gt_static, pred_static, results, 
                                  class_names_dynamic=None, class_names_static=None, 
                                  sample_idx=0):
    """
    Visualizes segmentation results and metrics.
    
    Args:
        gt_dynamic: Ground truth for dynamic segmentation
        pred_dynamic: Predicted dynamic segmentation
        gt_static: Ground truth for static segmentation
        pred_static: Predicted static segmentation
        results: Dictionary of evaluation results
        class_names_dynamic: List of class names for dynamic segmentation
        class_names_static: List of class names for static segmentation
        sample_idx: Index of the sample to visualize (if batch provided)
    """
    # Handle batch dimension if present
    if len(gt_dynamic.shape) > 2:
        gt_dynamic = gt_dynamic[sample_idx]
        pred_dynamic = pred_dynamic[sample_idx]
        gt_static = gt_static[sample_idx]
        pred_static = pred_static[sample_idx]
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Visualize dynamic segmentation
    axs[0, 0].imshow(gt_dynamic, cmap='tab20')
    axs[0, 0].set_title('Ground Truth - Dynamic')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(pred_dynamic, cmap='tab20')
    axs[0, 1].set_title('Prediction - Dynamic')
    axs[0, 1].axis('off')
    
    # Visualize static segmentation
    axs[1, 0].imshow(gt_static, cmap='tab20')
    axs[1, 0].set_title('Ground Truth - Static')
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(pred_static, cmap='tab20')
    axs[1, 1].set_title('Prediction - Static')
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Plot metrics
    fig2, axs2 = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot IoU per class for dynamic segmentation
    class_indices = np.arange(len(results["dynamic"]["iou_per_class"]))
    axs2[0, 0].bar(class_indices, results["dynamic"]["iou_per_class"])
    axs2[0, 0].set_title(f'Dynamic IoU per Class (Mean: {results["dynamic"]["mean_iou"]:.4f})')
    axs2[0, 0].set_ylim(0, 1)
    if class_names_dynamic is not None:
        axs2[0, 0].set_xticks(class_indices)
        axs2[0, 0].set_xticklabels(class_names_dynamic, rotation=45, ha='right')
    
    # Plot IoU per class for static segmentation
    class_indices = np.arange(len(results["static"]["iou_per_class"]))
    axs2[0, 1].bar(class_indices, results["static"]["iou_per_class"])
    axs2[0, 1].set_title(f'Static IoU per Class (Mean: {results["static"]["mean_iou"]:.4f})')
    axs2[0, 1].set_ylim(0, 1)
    if class_names_static is not None:
        axs2[0, 1].set_xticks(class_indices)
        axs2[0, 1].set_xticklabels(class_names_static, rotation=45, ha='right')
    
    # Plot Dice per class for dynamic segmentation
    class_indices = np.arange(len(results["dynamic"]["dice_per_class"]))
    axs2[1, 0].bar(class_indices, results["dynamic"]["dice_per_class"])
    axs2[1, 0].set_title(f'Dynamic Dice per Class (Mean: {results["dynamic"]["mean_dice"]:.4f})')
    axs2[1, 0].set_ylim(0, 1)
    if class_names_dynamic is not None:
        axs2[1, 0].set_xticks(class_indices)
        axs2[1, 0].set_xticklabels(class_names_dynamic, rotation=45, ha='right')
    
    # Plot Dice per class for static segmentation
    class_indices = np.arange(len(results["static"]["dice_per_class"]))
    axs2[1, 1].bar(class_indices, results["static"]["dice_per_class"])
    axs2[1, 1].set_title(f'Static Dice per Class (Mean: {results["static"]["mean_dice"]:.4f})')
    axs2[1, 1].set_ylim(0, 1)
    if class_names_static is not None:
        axs2[1, 1].set_xticks(class_indices)
        axs2[1, 1].set_xticklabels(class_names_static, rotation=45, ha='right')
    
    plt.tight_layout()
    # TODO(YH): save to file
    plt.savefig('dummy_images/segmentation_results.png')