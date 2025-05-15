import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def linear_assignment(cost_matrix):
    """
    Solve the linear assignment problem using the Hungarian algorithm
    """
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def iou_3d_batch(bb_test, bb_gt):
    """
    Compute 3D IOU between bounding boxes
    
    Args:
        bb_test: Test boxes in format [x1, y1, z1, x2, y2, z2]
        bb_gt: Ground truth boxes in format [x1, y1, z1, x2, y2, z2]
    
    Returns:
        IOU matrix
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    # Calculate intersection
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    zz1 = np.maximum(bb_test[..., 2], bb_gt[..., 2])
    
    xx2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    yy2 = np.minimum(bb_test[..., 4], bb_gt[..., 4])
    zz2 = np.minimum(bb_test[..., 5], bb_gt[..., 5])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    d = np.maximum(0., zz2 - zz1)
    
    # Calculate intersection volume
    intersection = w * h * d
    
    # Calculate volume of boxes
    vol_test = (bb_test[..., 3] - bb_test[..., 0]) * \
               (bb_test[..., 4] - bb_test[..., 1]) * \
               (bb_test[..., 5] - bb_test[..., 2])
    
    vol_gt = (bb_gt[..., 3] - bb_gt[..., 0]) * \
             (bb_gt[..., 4] - bb_gt[..., 1]) * \
             (bb_gt[..., 5] - bb_gt[..., 2])
    
    # Calculate IoU
    union = vol_test + vol_gt - intersection
    iou = intersection / union
    
    return iou

def convert_3d_bbox_to_z(bbox):
    """
    Convert 3D bounding box to state vector [x, y, z, s, r1, r2, r3]
    where:
    - x, y, z is the center of the box
    - s is the scale/volume
    - r1, r2, r3 are the aspect ratios
    
    Args:
        bbox: Bounding box in format [x1, y1, z1, x2, y2, z2]
    
    Returns:
        State vector
    """
    w = bbox[3] - bbox[0]
    h = bbox[4] - bbox[1]
    d = bbox[5] - bbox[2]
    
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    z = bbox[2] + d/2.
    
    s = w * h * d  # Volume
    r1 = w / (h + 1e-6)  # Adding small value to avoid division by zero
    r2 = w / (d + 1e-6)
    r3 = h / (d + 1e-6)
    
    return np.array([x, y, z, s, r1, r2, r3]).reshape((7, 1))

def convert_3d_x_to_bbox(x, score=None):
    """
    Convert state vector to 3D bounding box
    
    Args:
        x: State vector [x, y, z, s, r1, r2, r3]
        score: Detection score
    
    Returns:
        Bounding box in format [x1, y1, z1, x2, y2, z2] or [x1, y1, z1, x2, y2, z2, score]
    """
    r1 = x[4]  # w/h
    r2 = x[5]  # w/d
    r3 = x[6]  # h/d
    
    # Calculate dimensions
    volume = x[3]
    h = np.cbrt(volume / (r1 * r2 + 1e-6))
    w = r1 * h
    d = h / (r3 + 1e-6)
    
    # Create bbox
    x1 = x[0] - w/2
    y1 = x[1] - h/2
    z1 = x[2] - d/2
    
    x2 = x[0] + w/2
    y2 = x[1] + h/2
    z2 = x[2] + d/2
    
    if score is None:
        return np.array([x1, y1, z1, x2, y2, z2]).reshape((1, 6))
    else:
        return np.array([x1, y1, z1, x2, y2, z2, score]).reshape((1, 7))

class KalmanBoxTracker3D(object):
    """
    This class represents the internal state of individual tracked objects for 3D tracking.
    """
    count = 0
    
    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box.
        
        Args:
            bbox: Bounding box in format [x1, y1, z1, x2, y2, z2, score, class_id]
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=10, dim_z=7)  # State: [x, y, z, s, r1, r2, r3, vx, vy, vz]
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.eye(10)
        self.kf.F[0:3, 7:] = np.eye(3) * 1.0  # Add velocity component
        
        # Measurement function (we only observe position, scale, and ratios)
        self.kf.H = np.zeros((7, 10))
        self.kf.H[0:7, 0:7] = np.eye(7)
        
        # Measurement noise
        self.kf.R[2:, 2:] *= 10.
        
        # Process noise
        self.kf.P[7:, 7:] *= 1000.  # High uncertainty for velocity
        self.kf.P *= 10.
        
        # Process noise for velocity
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[7:, 7:] *= 0.01
        
        # Initialize state
        self.kf.x[:7] = convert_3d_bbox_to_z(bbox)
        
        # Tracking state
        self.time_since_update = 0
        self.id = KalmanBoxTracker3D.count
        KalmanBoxTracker3D.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.class_id = int(bbox[7]) if len(bbox) > 7 else -1  # Store class ID if available
    
    def update(self, bbox):
        """
        Update the state vector with observed bbox.
        
        Args:
            bbox: Bounding box in format [x1, y1, z1, x2, y2, z2, score, class_id]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        if len(bbox) > 7:
            self.class_id = int(bbox[7])
            
        self.kf.update(convert_3d_bbox_to_z(bbox))
    
    def predict(self):
        """
        Advance the state vector and return the predicted bounding box estimate.
        """
        if (self.kf.x[6] <= 0):
            self.kf.x[6] *= 0.0
            
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
            
        self.time_since_update += 1
        self.history.append(convert_3d_x_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_3d_x_to_bbox(self.kf.x)

def associate_detections_to_trackers_3d(detections, trackers, iou_threshold=0.3):
    """
    Associates detections to tracked objects using IOU
    
    Args:
        detections: N x 7 array of detections [x1, y1, z1, x2, y2, z2, score]
        trackers: M x 7 array of trackers [x1, y1, z1, x2, y2, z2, score]
        iou_threshold: Minimum IOU for match
    
    Returns:
        3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 7), dtype=int)
    
    # Compute IOU matrix
    iou_matrix = iou_3d_batch(detections[:, :6], trackers[:, :6])
    
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    
    # Find unmatched detections
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    
    # Find unmatched trackers
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    
    # Filter out matches with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort3D(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Initialize Sort3D for 3D multi-object tracking
        
        Args:
            max_age: Maximum number of frames to keep a track
            min_hits: Minimum number of hits to confirm a track
            iou_threshold: IOU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
    
    def update(self, dets=np.empty((0, 8))):
        """
        Update trackers with detections
        
        Args:
            dets: Detections in format [x1, y1, z1, x2, y2, z2, score, class_id]
            
        Returns:
            Tracked objects in format [x1, y1, z1, x2, y2, z2, track_id, class_id]
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 7))
        to_del = []
        ret = []
        
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t, :6] = pos[:6]
            trks[t, 6] = 0  # No confidence score for predicted boxes
            
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers_3d(
            dets[:, :7], trks, self.iou_threshold)
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
        
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker3D(dets[i, :])
            self.trackers.append(trk)
        
        # Return active trackers
        i = len(self.trackers)
        ret = []
        
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # Add track ID to bbox
                ret.append(np.concatenate((d, [trk.id + 1, trk.class_id])).reshape(1, -1))
                
            i -= 1
            
            # Remove dead tracks
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        
        return np.empty((0, 8))  # [x1, y1, z1, x2, y2, z2, track_id, class_id]
