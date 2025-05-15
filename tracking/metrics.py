import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

def compute_iou_3d(box1, box2):
    """
    Compute IOU between two 3D boxes
    
    Args:
        box1: First box in format [x1, y1, z1, x2, y2, z2]
        box2: Second box in format [x1, y1, z1, x2, y2, z2]
    
    Returns:
        IOU value
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    z1 = max(box1[2], box2[2])
    
    x2 = min(box1[3], box2[3])
    y2 = min(box1[4], box2[4])
    z2 = min(box1[5], box2[5])
    
    # Check if boxes intersect
    if x1 >= x2 or y1 >= y2 or z1 >= z2:
        return 0.0
    
    # Calculate volumes
    intersection = (x2 - x1) * (y2 - y1) * (z2 - z1)
    
    vol1 = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    vol2 = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    
    union = vol1 + vol2 - intersection
    
    return intersection / union

def compute_metrics(gt_tracks, pred_tracks, iou_threshold=0.5):
    """
    Compute MOT metrics
    
    Args:
        gt_tracks: Ground truth tracks in format {track_id: [(frame_idx, bbox), ...]}
        pred_tracks: Predicted tracks in format {track_id: [(frame_idx, bbox), ...]}
        iou_threshold: IOU threshold for considering a match
    
    Returns:
        Dictionary of metrics
    """
    # Initialize metrics
    total_gt = 0
    total_pred = 0
    total_matches = 0
    total_fp = 0
    total_fn = 0
    total_id_switches = 0
    total_fragments = 0
    
    # For MOTP calculation
    total_iou = 0
    
    # For MT/ML calculation
    gt_track_frames = {}
    matched_gt_track_frames = {}
    
    # Process frame by frame
    all_frames = sorted(set([f for track in gt_tracks.values() for f, _ in track] + 
                          [f for track in pred_tracks.values() for f, _ in track]))
    
    prev_matches = {}  # For ID switch calculation
    
    for frame in tqdm(all_frames, total=len(all_frames), desc="Processing frames"):
        # Get ground truth objects in this frame
        gt_objects = {}
        for gt_id, gt_track in gt_tracks.items():
            for f, bbox in gt_track:
                if f == frame:
                    gt_objects[gt_id] = bbox
                    
                    # Update GT track frames count
                    if gt_id not in gt_track_frames:
                        gt_track_frames[gt_id] = 0
                    gt_track_frames[gt_id] += 1
        
        # Get predicted objects in this frame
        pred_objects = {}
        for pred_id, pred_track in pred_tracks.items():
            for f, bbox in pred_track:
                if f == frame:
                    pred_objects[pred_id] = bbox
        
        # Count objects
        n_gt = len(gt_objects)
        n_pred = len(pred_objects)
        
        total_gt += n_gt
        total_pred += n_pred
        
        # Match ground truth with predictions
        if n_gt > 0 and n_pred > 0:
            # Compute IOU matrix
            iou_matrix = np.zeros((n_gt, n_pred))
            
            for i, (gt_id, gt_bbox) in enumerate(gt_objects.items()):
                for j, (pred_id, pred_bbox) in enumerate(pred_objects.items()):
                    iou_matrix[i, j] = compute_iou_3d(gt_bbox, pred_bbox)
            
            # Get matches
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(row_ind, col_ind)))
            
            # Process matches
            for i, j in matched_indices:
                gt_id = list(gt_objects.keys())[i]
                pred_id = list(pred_objects.keys())[j]
                
                if iou_matrix[i, j] >= iou_threshold:
                    total_matches += 1
                    total_iou += iou_matrix[i, j]
                    
                    # Update matched GT track frames count
                    if gt_id not in matched_gt_track_frames:
                        matched_gt_track_frames[gt_id] = 0
                    matched_gt_track_frames[gt_id] += 1
                    
                    # Check for ID switch
                    if gt_id in prev_matches and prev_matches[gt_id] != pred_id:
                        total_id_switches += 1
                    
                    prev_matches[gt_id] = pred_id
                else:
                    total_fp += 1
                    total_fn += 1
            
            # Count unmatched objects
            n_matches = sum(1 for i, j in matched_indices if iou_matrix[i, j] >= iou_threshold)
            total_fp += n_pred - n_matches
            total_fn += n_gt - n_matches
        else:
            total_fp += n_pred
            total_fn += n_gt
    
    # Calculate fragmentation
    for gt_id, gt_track in tqdm(gt_tracks.items(), desc="Calculating fragmentation"):
        frames = sorted([f for f, _ in gt_track])
        
        # Check if track is matched in each frame
        is_matched = []
        for frame in frames:
            matched = False
            
            for pred_id, pred_track in pred_tracks.items():
                pred_frames = [f for f, _ in pred_track]
                
                if frame in pred_frames and gt_id in prev_matches and prev_matches[gt_id] == pred_id:
                    matched = True
                    break
            
            is_matched.append(matched)
        
        # Count fragments (transitions from matched to unmatched)
        fragments = 0
        for i in range(1, len(is_matched)):
            if is_matched[i-1] and not is_matched[i]:
                fragments += 1
        
        total_fragments += fragments
    
    # Calculate metrics
    mota = 1 - (total_fn + total_fp + total_id_switches) / max(1, total_gt)
    motp = total_iou / max(1, total_matches)
    
    # Calculate MT/ML
    n_gt_tracks = len(gt_track_frames)
    
    mt_tracks = 0
    ml_tracks = 0
    
    for gt_id, n_frames in gt_track_frames.items():
        n_matched = matched_gt_track_frames.get(gt_id, 0)
        
        if n_matched >= 0.8 * n_frames:
            mt_tracks += 1
        elif n_matched <= 0.2 * n_frames:
            ml_tracks += 1
    
    mt_percent = mt_tracks / max(1, n_gt_tracks)
    ml_percent = ml_tracks / max(1, n_gt_tracks)
    
    return {
        'MOTA': mota,
        'MOTP': motp,
        'MT': mt_percent,
        'ML': ml_percent,
        'ID_Switches': total_id_switches,
        'Fragmentation': total_fragments
    }
