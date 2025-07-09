import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
import matplotlib.pyplot as plt

def evaluate_segmentation(gt_dynamic, pred_dynamic, gt_static, pred_static, class_names_dynamic=None, class_names_static=None):
    """
    Evaluates segmentation performance for both dynamic and static segmentation branches.
    
    Args:
        gt_dynamic: Ground truth for dynamic segmentation (objects)
                   Shape: [B, H, W] or [H, W] with integer class labels
        pred_dynamic: Predicted dynamic segmentation
                     Shape: [B, C, H, W] or [C, H, W] with logits/probabilities
        gt_static: Ground truth for static segmentation (roads, lanes, background)
                  Shape: [B, H, W] or [H, W] with integer class labels
        pred_static: Predicted static segmentation
                    Shape: [B, C, H, W] or [C, H, W] with logits/probabilities
        class_names_dynamic: List of class names for dynamic segmentation
        class_names_static: List of class names for static segmentation
    
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
    
    # Ensure correct shape for predictions (convert from probabilities/logits to class indices)
    if len(pred_dynamic.shape) == 4:  # [B, C, H, W]
        pred_dynamic = np.argmax(pred_dynamic, axis=1)
    elif len(pred_dynamic.shape) == 3 and pred_dynamic.shape[0] > 1:  # [C, H, W]
        pred_dynamic = np.argmax(pred_dynamic, axis=0)
        
    if len(pred_static.shape) == 4:  # [B, C, H, W]
        pred_static = np.argmax(pred_static, axis=1)
    elif len(pred_static.shape) == 3 and pred_static.shape[0] > 1:  # [C, H, W]
        pred_static = np.argmax(pred_static, axis=0)
    
    # Evaluate dynamic segmentation
    results["dynamic"] = evaluate_branch(gt_dynamic, pred_dynamic, "dynamic", class_names_dynamic)
    
    # Evaluate static segmentation
    results["static"] = evaluate_branch(gt_static, pred_static, "static", class_names_static)
    
    # Calculate combined metrics
    results["combined"]["mean_iou"] = (results["dynamic"]["mean_iou"] + results["static"]["mean_iou"]) / 2
    results["combined"]["mean_dice"] = (results["dynamic"]["mean_dice"] + results["static"]["mean_dice"]) / 2
    results["combined"]["pixel_accuracy"] = (results["dynamic"]["pixel_accuracy"] + results["static"]["pixel_accuracy"]) / 2
    
    return results


def evaluate_branch(gt, pred, branch_name, class_names=None):
    """
    Evaluates a single segmentation branch.
    
    Args:
        gt: Ground truth segmentation with integer class labels
        pred: Predicted segmentation with integer class labels
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
    # plt.show()
    plt.savefig('dummy_images/segmentation_results.png')


# Example usage
if __name__ == "__main__":
    # Define example class names
    class_names_dynamic = ["background", "car", "pedestrian", "cyclist", "truck", "bus"]
    class_names_static = ["road", "lane", "sidewalk", "terrain", "building", "vegetation"]
    
    # Create example data (just for demonstration)
    # In real usage, you'd load your actual prediction and ground truth masks
    height, width = 480, 640
    
    # Create synthetic ground truth and predictions
    # Dynamic segmentation (objects)
    gt_dynamic = np.zeros((height, width), dtype=np.int32)
    gt_dynamic[100:200, 200:300] = 1  # Car
    gt_dynamic[300:350, 400:450] = 2  # Pedestrian
    
    pred_dynamic = np.zeros((height, width), dtype=np.int32)
    pred_dynamic[110:205, 205:310] = 1  # Car (slightly offset)
    pred_dynamic[310:355, 410:455] = 2  # Pedestrian (slightly offset)
    
    # Static segmentation (roads, lanes, background)
    gt_static = np.zeros((height, width), dtype=np.int32)
    gt_static[350:, :] = 1  # Road at bottom
    gt_static[350:, 300:340] = 2  # Lane in middle of road
    
    pred_static = np.zeros((height, width), dtype=np.int32)
    pred_static[360:, :] = 1  # Road at bottom (slightly offset)
    pred_static[360:, 310:345] = 2  # Lane in middle of road (slightly offset)
    
    # Evaluate segmentation performance
    results = evaluate_segmentation(
        gt_dynamic, pred_dynamic, gt_static, pred_static,
        class_names_dynamic, class_names_static
    )
    
    # Print results
    print("Dynamic Segmentation Results:")
    print(f"Mean IoU: {results['dynamic']['mean_iou']:.4f}")
    print(f"Mean Dice: {results['dynamic']['mean_dice']:.4f}")
    print(f"Pixel Accuracy: {results['dynamic']['pixel_accuracy']:.4f}")
    
    print("\nStatic Segmentation Results:")
    print(f"Mean IoU: {results['static']['mean_iou']:.4f}")
    print(f"Mean Dice: {results['static']['mean_dice']:.4f}")
    print(f"Pixel Accuracy: {results['static']['pixel_accuracy']:.4f}")
    
    print("\nCombined Results:")
    print(f"Mean IoU: {results['combined']['mean_iou']:.4f}")
    print(f"Mean Dice: {results['combined']['mean_dice']:.4f}")
    print(f"Pixel Accuracy: {results['combined']['pixel_accuracy']:.4f}")
    
    # Visualize results
    visualize_segmentation_results(
        gt_dynamic, pred_dynamic, gt_static, pred_static, results,
        class_names_dynamic, class_names_static
    )