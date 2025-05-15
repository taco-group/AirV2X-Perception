import os
import argparse
import numpy as np
from tqdm import tqdm

from mot_dataset import MOTDataset
from utils import convert_gt_to_tracks, convert_preds_to_tracks, convert_prediction_to_global, get_ego_location, visualize_tracks
from metrics import compute_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='3D Multi-Object Tracking Evaluation')
    parser.add_argument('--data_root', type=str, default='val', help='Root directory for ground truth data')
    parser.add_argument('--pred_root', type=str, default='predictions', help='Root directory for predictions')
    parser.add_argument('--max_age', type=int, default=10, help='Maximum frames to keep track alive without detection')
    parser.add_argument('--min_hits', type=int, default=5, help='Minimum hits to initialize a track')
    parser.add_argument('--iou_threshold', type=float, default=0.1, help='IOU threshold for association')
    return parser.parse_args()

def print_metrics(metrics):
    """
    Print evaluation metrics in a formatted way
    """
    print("\n=== Tracking Evaluation Results ===")
    
    for seq, seq_metrics in metrics.items():
        if seq == 'OVERALL':
            print("\n=== OVERALL ===")
        else:
            print(f"\n=== Sequence: {seq} ===")
        
        print(f"MOTA: {seq_metrics['MOTA']:.4f}")
        print(f"MOTP: {seq_metrics['MOTP']:.4f}")
        print(f"MT: {seq_metrics['MT']:.4f}")
        print(f"ML: {seq_metrics['ML']:.4f}")
        print(f"ID Switches: {seq_metrics['ID_Switches']}")
        print(f"Fragmentation: {seq_metrics['Fragmentation']}")

def evaluate(gt_tracks, pred_tracks):
    """
    Evaluate tracking performance
    
    Args:
        gt_tracks: Ground truth tracks by sequence
        pred_tracks: Predicted tracks by sequence
    
    Returns:
        Dictionary of metrics by sequence and overall
    """
    all_metrics = {}
    
    # Evaluate each sequence
    for seq in gt_tracks:
        metrics = compute_metrics(gt_tracks[seq], pred_tracks.get(seq, {}))
        all_metrics[seq] = metrics
    
    # Compute overall metrics (average across sequences)
    overall = {
        'MOTA': np.mean([m['MOTA'] for m in all_metrics.values()]),
        'MOTP': np.mean([m['MOTP'] for m in all_metrics.values()]),
        'MT': np.mean([m['MT'] for m in all_metrics.values()]),
        'ML': np.mean([m['ML'] for m in all_metrics.values()]),
        'ID_Switches': sum([m['ID_Switches'] for m in all_metrics.values()]),
        'Fragmentation': sum([m['Fragmentation'] for m in all_metrics.values()])
    }
    
    all_metrics['OVERALL'] = overall
    
    return all_metrics

def main():
    args = parse_args()
    
    print(f"Loading data from {args.data_root}")
    dataset = MOTDataset(args.data_root)
    print(f"Found {len(dataset.sequences)} sequences")
    
    print(f"Loading predictions from {args.pred_root}")
    predictions, ego_lidar_poses_seq = dataset.load_predictions(args.pred_root)
    
    # ego_location = get_ego_location(dataset, ego_id=1801)
    
    predictions_glob = convert_prediction_to_global(predictions, ego_lidar_poses_seq)
    
    print("Applying tracking to predictions")
    pred_tracks = convert_preds_to_tracks(
        predictions_glob, dataset, 
        max_age=args.max_age, 
        min_hits=args.min_hits, 
        iou_threshold=args.iou_threshold,
    )
    
    print("Converting ground truth to tracks")
    gt_tracks = convert_gt_to_tracks(dataset, ego_location=ego_lidar_poses_seq)
    
    visualize_tracks(gt_tracks, pred_tracks)
    print("Evaluating tracking performance")
    metrics = evaluate(gt_tracks, pred_tracks)
    
    print_metrics(metrics)
    
    return metrics

if __name__ == '__main__':
    main()
