import os
import pickle
import numpy as np
import math


def get_distance(a, b):
    dis = np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
    return dis


import cv2
import numpy as np

def visualize_tracks(gt_tracks, pred_tracks, output_path='debug/tracking_visualization.png', width=1200, height=800):
    """
    Visualize tracking results focusing only on position information.
    
    Args:
        gt_tracks (dict): Ground truth tracking information.
        pred_tracks (dict): Predicted tracking information.
        output_path (str): Path to save the visualization.
        width, height (int): Dimensions of output image.
    """
    # Create a blank white image
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Define colors
    gt_color = (0, 255, 0)  # Green for ground truth
    pred_color = (0, 0, 255)  # Red for predicted
    
    # Find position ranges for scaling
    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
    
    # Helper function to update min/max from track data
    def update_ranges(tracks):
        nonlocal min_x, max_x, min_y, max_y
        for timestamp, track_dict in tracks.items():
            for track_id, track_data in track_dict.items():
                for frame_info in track_data:
                    frame_num = frame_info[0]
                    data = frame_info[1]
                    x, y = data[0], data[1]  # First two elements are x,y position
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
    
    # Update position ranges
    if gt_tracks: update_ranges(gt_tracks)
    if pred_tracks: update_ranges(pred_tracks)
    
    # Abort if no data
    if min_x == float('inf'):
        print("No tracks to visualize.")
        return None
    
    # Compute scaling factors
    x_range = max_x - min_x
    y_range = max_y - min_y
    
    # Add margin
    margin = 0.05
    min_x -= x_range * margin
    min_y -= y_range * margin
    x_range *= (1 + 2*margin)
    y_range *= (1 + 2*margin)
    
    x_scale = width / x_range if x_range > 0 else 1
    y_scale = height / y_range if y_range > 0 else 1
    
    # Transform point to image coordinates
    def transform_point(x, y):
        x_new = int((x - min_x) * x_scale)
        y_new = int(height - (y - min_y) * y_scale)  # Flip y-axis
        return (x_new, y_new)
    
    # Draw tracks function
    def draw_tracks(tracks, color):
        for timestamp, track_dict in tracks.items():
            for track_id, track_data in track_dict.items():
                points = []
                for frame_info in track_data:
                    x, y = frame_info[1][0], frame_info[1][1]
                    points.append(transform_point(x, y))
                # Draw lines between consecutive points
                for i in range(1, len(points)):
                    cv2.line(image, points[i-1], points[i], color, 2)
    
    # Draw tracks
    if gt_tracks: draw_tracks(gt_tracks, gt_color)
    if pred_tracks: draw_tracks(pred_tracks, pred_color)
    
    # Add legend
    if gt_tracks: cv2.putText(image, "Ground Truth", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 2)
    if pred_tracks: cv2.putText(image, "Predicted", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 2)
    
    # Save the image
    cv2.imwrite(output_path, image)
    
    return image


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to rotation matrix.
    Uses the z-y-x convention (yaw-pitch-roll).
    
    Args:
        roll (float): Rotation around x-axis in radians
        pitch (float): Rotation around y-axis in radians
        yaw (float): Rotation around z-axis in radians
        
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    # Convert degrees to radians if necessary
    if abs(roll) > 2*np.pi:
        roll = math.radians(roll)
    if abs(pitch) > 2*np.pi:
        pitch = math.radians(pitch)
    if abs(yaw) > 2*np.pi:
        yaw = math.radians(yaw)
    
    # Roll (X-axis rotation)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Pitch (Y-axis rotation)
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Yaw (Z-axis rotation)
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix (R = R_z * R_y * R_x)
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def ego_to_global(ego_location, object_location):
    """
    Convert object location from ego coordinates to global coordinates.
    
    Args:
        ego_location (list): Ego vehicle location [x, y, z, roll, yaw, pitch]
        object_location (list): Object location in ego coordinates [x, y, z, roll, yaw, pitch]
        
    Returns:
        list: Object location in global coordinates [x, y, z, roll, yaw, pitch]
    """
    # Extract ego position and orientation
    ego_pos = np.array(ego_location[:3])
    ego_roll, ego_yaw, ego_pitch = ego_location[3], ego_location[4], ego_location[5]
    
    # Extract object position and orientation in ego frame
    obj_pos_ego = np.array(object_location[:3])
    obj_roll_ego, obj_yaw_ego, obj_pitch_ego = object_location[3], object_location[4], object_location[5]
    
    # Get ego rotation matrix
    R_ego = euler_to_rotation_matrix(ego_roll, ego_pitch, ego_yaw)
    
    # Transform object position from ego to global coordinates
    obj_pos_global = ego_pos + np.dot(R_ego, obj_pos_ego)
    
    # Transform object orientation from ego to global coordinates
    # Convert object's ego orientation to rotation matrix
    R_obj_ego = euler_to_rotation_matrix(obj_roll_ego, obj_pitch_ego, obj_yaw_ego)
    
    # Combine rotations
    R_obj_global = np.dot(R_ego, R_obj_ego)
    
    # Convert combined rotation matrix back to Euler angles
    # Extract the Euler angles from rotation matrix using proper order
    obj_pitch_global = np.arcsin(R_obj_global[0, 2])
    
    if np.cos(obj_pitch_global) > 1e-10:
        obj_roll_global = np.arctan2(-R_obj_global[1, 2], R_obj_global[2, 2])
        obj_yaw_global = np.arctan2(-R_obj_global[0, 1], R_obj_global[0, 0])
    else:
        # Gimbal lock case
        obj_roll_global = np.arctan2(R_obj_global[2, 1], R_obj_global[1, 1])
        obj_yaw_global = 0
    
    # Assemble global location
    obj_location_global = [
        obj_pos_global[0], 
        obj_pos_global[1], 
        obj_pos_global[2],
        obj_roll_global,
        obj_yaw_global,
        obj_pitch_global
    ]
    
    return obj_location_global


def convert_prediction_to_global(preds, ego_locations):
    """
    Convert all predicted object locations from ego to global coordinates.
    
    Args:
        preds (dict): Dictionary of predictions
        ego_locations (dict): Dictionary mapping timestamps to ego locations
        
    Returns:
        dict: Dictionary with the same structure but with global coordinates
    """
    global_preds = {}
    
    for sequence_id, timestamps in preds.items():
        global_preds[sequence_id] = {}
        
        for timestamp, objects in timestamps.items():
            global_preds[sequence_id][timestamp] = {}
            
            # Get ego location for this timestamp
            ego_loc = ego_locations[sequence_id][timestamp]
            if ego_loc is None:
                print(f"Warning: No ego location for {timestamp}. Skipping conversion.")
                global_preds[sequence_id][timestamp] = objects  # Copy without conversion
                continue
            
            # Convert each object
            for obj_id, obj_data in objects.items():
                global_preds[sequence_id][timestamp][obj_id] = obj_data.copy()  # Create a copy
                
                if 'location' in obj_data:
                    pred_obj_loc = [obj_data['location'][0], 
                                    obj_data['location'][1], 
                                    obj_data['location'][2],
                                    obj_data['location'][3],
                                    obj_data['location'][5],
                                    obj_data['location'][4]]
                    # pred_obj_loc = obj_data['location']
                    # Convert location from ego to global
                    global_preds[sequence_id][timestamp][obj_id]['location'] = ego_to_global(ego_loc, pred_obj_loc)
    
    return global_preds



def get_ego_location(dataset, ego_id=0):
    """
    """
    ego_locations_by_seq = {}
    for seq in dataset.sequences:
        ego_locations = {}
        # Get timestamps for this sequence
        seq_timestamps = [ts for s, ts in dataset.index if s == seq]
        seq_timestamps.sort()  # Ensure chronological order
        
        for frame_idx, ts in enumerate(seq_timestamps):
            # Load ground truth
            gt_path = os.path.join(dataset.data_root, seq, ts, "objects.pkl")
            with open(gt_path, 'rb') as f:
                gt_data = pickle.load(f)
            
            ego_location = None
            # get ego location
            for obj_id, obj in gt_data.items():
                if obj_id == ego_id:
                    ego_location = obj['location']
                    break
            assert ego_location is not None, f"Ego vehicle not found in sequence {seq} at timestamp {ts}"
            ego_locations[ts] = ego_location
        ego_locations_by_seq[seq] = ego_locations
    return ego_locations_by_seq

def convert_gt_to_tracks(dataset, ego_location, class_grain=False):
    """
    Convert ground truth data to track format
    
    Returns:
        Dictionary of tracks by sequence: {seq_id: {track_id: [(frame_idx, bbox), ...], ...}, ...}
    """
    tracks_by_seq = {}
    
    for seq in dataset.sequences:
        tracks = {}
        # Get timestamps for this sequence
        seq_timestamps = [ts for s, ts in dataset.index if s == seq]
        seq_timestamps.sort()  # Ensure chronological order
        
        for frame_idx, ts in enumerate(seq_timestamps):
            # Load ground truth
            gt_path = os.path.join(dataset.data_root, seq, ts, "objects.pkl")
            with open(gt_path, 'rb') as f:
                gt_data = pickle.load(f)
            
            # Process each object
            for obj_id, obj in gt_data.items():
                if obj['class'] not in [0, 1, 2, 3, 4, 5, 6]:
                    continue
                # Convert to bbox format [x1, y1, z1, x2, y2, z2, class]
                location = obj['location']
                # [x,y,z,roll, yaw, pitch] to [x,y,z,roll,pith,yaw]
                location = [location[0], location[1], location[2], location[3], location[5], location[4]]
                extent = obj['extent']
                if get_distance(location, ego_location[seq][ts]) > 50:
                    continue
                
                # Create 3D bbox
                x, y, z = location[0], location[1], location[2]
                l, w, h = extent[0], extent[1], extent[2]
                # Create bbox with half-lengths
                x1, y1, z1 = x - l/2, y - w/2, z - h/2
                x2, y2, z2 = x + l/2, y + w/2, z + h/2
                
                bbox = [x1, y1, z1, x2, y2, z2, 
                        obj['class'] if class_grain else 0,
                        ]
                
                # Add to tracks
                if obj_id not in tracks:
                    tracks[obj_id] = []
                
                tracks[obj_id].append((frame_idx, bbox))
        
        tracks_by_seq[seq] = tracks
    
    return tracks_by_seq


def convert_preds_to_tracks(predictions, dataset, max_age=3, min_hits=3, iou_threshold=0.3, class_grain=False):
    """
    Convert frame-by-frame detections to consistent tracks using SORT3D tracker
    
    Args:
        predictions: Dictionary of detections by sequence and timestamp
        dataset: MOTDataset instance for reference
        max_age: Maximum age for tracks in SORT3D
        min_hits: Minimum hits for tracks in SORT3D
        iou_threshold: IOU threshold for association in SORT3D
    
    Returns:
        Dictionary of tracks by sequence: {seq_id: {track_id: [(frame_idx, bbox), ...], ...}, ...}
    """
    from sort_3d import Sort3D
    
    tracks_by_seq = {}
    
    for seq in dataset.sequences:
        # Initialize tracker
        tracker = Sort3D(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        tracks = {}
        
        # Get timestamps for this sequence
        seq_timestamps = [ts for s, ts in dataset.index if s == seq]
        seq_timestamps.sort()  # Ensure chronological order
        
        for frame_idx, ts in enumerate(seq_timestamps):
            # Get predictions for this timestamp
            if seq in predictions and ts in predictions[seq]:
                preds = predictions[seq][ts]
                
                # Convert predictions to detection format [x1, y1, z1, x2, y2, z2, score, class]
                detections = []
                
                for _, obj in preds.items():  # Note: using _ since detection IDs are not relevant
                    # Extract location and extent
                    location = obj['location']
                    extent = obj['extent']
                    confidence = obj.get('confidence', 1.0)
                    
                    # Create 3D bbox
                    x, y, z = location[0], location[1], location[2]
                    h, w, l = extent[0], extent[1], extent[2]
                    
                    # Create bbox with half-lengths
                    x1, y1, z1 = x - l/2, y - w/2, z - h/2
                    x2, y2, z2 = x + l/2, y + w/2, z + h/2
                    
                    detections.append([x1, y1, z1, x2, y2, z2, confidence, 
                                       obj['class'] if class_grain else 0
                                      ])
                
                if detections:
                    detections = np.array(detections)
                else:
                    detections = np.empty((0, 8))
            else:
                detections = np.empty((0, 8))
            
            # Update tracker - this assigns IDs to detections
            results = tracker.update(detections)
            
            # Add results to tracks
            for result in results:
                track_id = int(result[6])  # Track ID assigned by SORT
                
                if track_id not in tracks:
                    tracks[track_id] = []
                
                # Store [x1, y1, z1, x2, y2, z2, class]
                bbox = [result[0], result[1], result[2], result[3], result[4], result[5], result[7]]
                tracks[track_id].append((frame_idx, bbox))
        
        tracks_by_seq[seq] = tracks
    return tracks_by_seq
