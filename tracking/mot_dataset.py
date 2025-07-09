import os
import pickle
import numpy as np
from glob import glob

class MOTDataset:
    def __init__(self, data_root, sequences=None):
        """
        Initialize the dataset
        Args:
            data_root: Root directory of the dataset (e.g., 'val/')
            sequences: List of sequences to use, if None, use all sequences
        """
        self.data_root = data_root
        
        # Get all sequences
        if sequences is None:
            self.sequences = [os.path.basename(seq) for seq in glob(os.path.join(data_root, "*"))]
        else:
            self.sequences = sequences
        
        # Build an index of all timestamps
        self.index = []
        for seq in self.sequences:
            timestamps = sorted(glob(os.path.join(data_root, seq, "timestamp_*")))
            for ts in timestamps:
                ts_id = os.path.basename(ts)
                self.index.append((seq, ts_id))
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        """Get data for a specific timestamp"""
        seq, ts_id = self.index[idx]
        
        # Load ground truth
        gt_path = os.path.join(self.data_root, seq, ts_id, "objects.pkl")
        with open(gt_path, 'rb') as f:
            gt_data = pickle.load(f)
        
        return {
            'sequence': seq,
            'timestamp': ts_id,
            'ground_truth': gt_data
        }
    
    def load_predictions(self, pred_root):
        """
        Load predictions for all sequences
        
        Args:
            pred_root: Root directory of predictions, should have the same structure as data_root
        
        Returns:
            dict: Dictionary of predictions
        """
        ego_id = None  # Placeholder for ego vehicle ID, if needed
        ego_lidar_poses_seq = {}
        predictions = {}
        for seq in self.sequences:
            seq_preds = {}
            ego_lidar_poses = {}
            timestamps = sorted(glob(os.path.join(pred_root, seq, "timestamp_*", "agent_*")))
            for ts_path in timestamps:
                ts_id = ts_path.split("/")[-2]
                current_ego_id = int(ts_path.split("/")[-1].split("_")[1])
                if current_ego_id != 1785:
                    continue
                if ego_id:
                    assert ego_id == current_ego_id, f"Ego vehicle ID mismatch: {ego_id} != {current_ego_id}"
                ego_id = current_ego_id
                pred_path = os.path.join(ts_path, "predictions.pkl")
                
                if os.path.exists(pred_path):
                    with open(pred_path, 'rb') as f:
                        pred_data = pickle.load(f)
                    ego_lidar_pose = pred_data.pop('ego_lidar_pose', None)
                    ego_lidar_pose = [ego_lidar_pose[0], ego_lidar_pose[1]-3, ego_lidar_pose[2],
                                        ego_lidar_pose[3], ego_lidar_pose[5], ego_lidar_pose[4]]
                    ego_lidar_poses[ts_id] = ego_lidar_pose
                    seq_preds[ts_id] = pred_data
            ego_lidar_poses_seq[seq] = ego_lidar_poses
            predictions[seq] = seq_preds
            
        assert ego_id is not None, "Ego vehicle ID not found in predictions"
        return predictions, ego_lidar_poses_seq
