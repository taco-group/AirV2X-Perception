import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import itertools


def roi_pooling(bev_features, boxes):
    """
    Placeholder: Pool BEV features for each box.
    Replace this with RoIAlign or a proper region pooling method.

    For now, we use avg pooled features across the entire BEV map (dummy implementation).
    """
    N = len(boxes)
    C = bev_features.size(0)
    pooled = bev_features.mean(dim=(1, 2)).repeat(N, 1)  # [N, C]
    return pooled


class TrackingHead(nn.Module):
    def __init__(self, bev_channels, embed_dim=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(bev_channels, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, bev_features, boxes):  # boxes: list of tensors, each [x, y, w, l, θ]
        pooled_feats = roi_pooling(bev_features, boxes)  # [N, C]
        embeddings = self.mlp(pooled_feats)              # [N, D]
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings


class Tracker:
    def __init__(self, embed_dim=128, sim_threshold=0.5, max_age=5):
        self.tracks = []  # list of track dicts
        self.track_id_counter = itertools.count()
        self.embed_dim = embed_dim
        self.sim_threshold = sim_threshold
        self.max_age = max_age

    def update(self, boxes, embeddings):
        """
        boxes: Tensor[N, 5] (x, y, w, l, θ)
        embeddings: Tensor[N, D]
        """
        matches, unmatched = self.associate_tracks(embeddings)

        updated_tracks = []

        # Update matched tracks
        for det_idx, track_idx in matches:
            track = self.tracks[track_idx]
            track['embedding'] = embeddings[det_idx]
            track['box'] = boxes[det_idx]
            track['age'] = 0
            updated_tracks.append(track)

        # Create new tracks for unmatched detections
        for idx in unmatched:
            new_track = {
                'id': next(self.track_id_counter),
                'embedding': embeddings[idx],
                'box': boxes[idx],
                'age': 0
            }
            updated_tracks.append(new_track)

        # Age unmatched old tracks
        for i, track in enumerate(self.tracks):
            if i not in [t[1] for t in matches]:
                track['age'] += 1
                if track['age'] <= self.max_age:
                    updated_tracks.append(track)

        # Set new track list
        self.tracks = updated_tracks

        # Return detections with assigned track ids
        track_ids = []
        for idx in range(len(boxes)):
            tid = None
            for det_idx, track_idx in matches:
                if det_idx == idx:
                    tid = self.tracks[track_idx]['id']
                    break
            if tid is None:
                tid = updated_tracks[-(len(unmatched) - unmatched.index(idx))]['id']
            track_ids.append(tid)

        return track_ids

    def associate_tracks(self, embeddings):
        """
        embeddings: Tensor[N, D]
        Return: (matches, unmatched_indices)
        """
        if len(self.tracks) == 0:
            return [], list(range(embeddings.size(0)))

        track_embeds = torch.stack([track['embedding'] for track in self.tracks], dim=0)  # [M, D]
        similarity = torch.matmul(embeddings, track_embeds.T)  # [N, M]

        cost_matrix = 1 - similarity.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched = set(range(embeddings.size(0)))

        for r, c in zip(row_ind, col_ind):
            if similarity[r, c] > self.sim_threshold:
                matches.append((r, c))
                unmatched.discard(r)

        return matches, list(unmatched)
