"""
Vehicle Tracking for Traffic Signal Optimization
DeepSORT-based tracking for vehicle trajectory analysis
Handles occlusions and ID switches common in Indian traffic
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from scipy.optimize import linear_sum_assignment

from .detector import Detection
from ..config.settings import get_settings, Direction
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Track:
    """Single vehicle track"""
    track_id: int
    class_name: str
    pcu_value: float
    direction: Direction
    
    # Trajectory
    trajectory: deque = field(default_factory=lambda: deque(maxlen=30))
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # State
    state: str = "active"  # active, lost, finished
    age: int = 0  # Frames since creation
    hits: int = 0  # Total successful updates
    time_since_update: int = 0  # Frames since last update
    
    # Timestamps
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    
    # Features
    avg_velocity: Tuple[float, float] = (0.0, 0.0)
    avg_size: Tuple[int, int] = (0, 0)
    
    @property
    def current_position(self) -> Optional[Tuple[int, int]]:
        """Get current position (center)"""
        return self.trajectory[-1] if self.trajectory else None
    
    @property
    def current_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Get current bounding box"""
        return self.bbox_history[-1] if self.bbox_history else None
    
    @property
    def lifetime(self) -> float:
        """Track lifetime in seconds"""
        return (self.last_seen - self.first_seen).total_seconds()
    
    @property
    def is_confirmed(self) -> bool:
        """Check if track is confirmed (min hits reached)"""
        min_hits = get_settings().ai_models.tracking_min_hits
        return self.hits >= min_hits
    
    def update(self, detection: Detection) -> None:
        """Update track with new detection"""
        self.trajectory.append(detection.center)
        self.bbox_history.append(detection.bbox)
        self.last_seen = detection.timestamp
        self.hits += 1
        self.time_since_update = 0
        self.age += 1
        
        # Update velocity
        if len(self.trajectory) >= 2:
            p1 = self.trajectory[-2]
            p2 = self.trajectory[-1]
            self.avg_velocity = (p2[0] - p1[0], p2[1] - p1[1])
        
        # Update average size
        if self.bbox_history:
            sizes = [(b[2] - b[0], b[3] - b[1]) for b in self.bbox_history]
            self.avg_size = (
                int(np.mean([s[0] for s in sizes])),
                int(np.mean([s[1] for s in sizes]))
            )
    
    def predict_position(self) -> Tuple[int, int]:
        """Predict next position using velocity"""
        if not self.trajectory:
            return (0, 0)
        
        current_pos = self.trajectory[-1]
        predicted = (
            int(current_pos[0] + self.avg_velocity[0]),
            int(current_pos[1] + self.avg_velocity[1])
        )
        return predicted
    
    def mark_missed(self) -> None:
        """Mark track as missed (no matching detection)"""
        self.time_since_update += 1
        self.age += 1
        
        max_age = get_settings().ai_models.tracking_max_age
        if self.time_since_update > max_age:
            self.state = "lost"


@dataclass
class TrackingResult:
    """Result of tracking on a frame"""
    frame_id: str
    timestamp: datetime
    active_tracks: List[Track]
    new_tracks: List[Track]
    lost_tracks: List[Track]
    
    @property
    def total_vehicles(self) -> int:
        """Total active vehicles"""
        return len(self.active_tracks)
    
    @property
    def confirmed_vehicles(self) -> int:
        """Confirmed vehicles only"""
        return sum(1 for t in self.active_tracks if t.is_confirmed)
    
    @property
    def total_pcu(self) -> float:
        """Total PCU of active vehicles"""
        return sum(t.pcu_value for t in self.active_tracks if t.is_confirmed)
    
    def count_by_class(self) -> Dict[str, int]:
        """Count confirmed vehicles by class"""
        counts = {}
        for track in self.active_tracks:
            if track.is_confirmed:
                counts[track.class_name] = counts.get(track.class_name, 0) + 1
        return counts


class VehicleTracker:
    """
    Multi-object tracker for vehicles
    Uses simple IoU + Kalman filter for tracking
    Optimized for Indian traffic (high density, occlusions)
    """
    
    def __init__(self, direction: Direction):
        self.direction = direction
        self.logger = get_logger(f"VehicleTracker.{direction.value}")
        
        # Tracks
        self.active_tracks: Dict[int, Track] = {}
        self.lost_tracks: Dict[int, Track] = {}
        self.finished_tracks: Dict[int, Track] = {}
        
        # Track ID counter
        self.next_track_id = 1
        
        # Statistics
        self.total_tracks_created = 0
        self.total_tracks_lost = 0
        
        self.logger.info(f"Initialized tracker for {direction.value}")
    
    def update(
        self,
        detections: List[Detection],
        frame_id: str = ""
    ) -> TrackingResult:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections from current frame
            frame_id: Optional frame identifier
        
        Returns:
            TrackingResult with current tracking state
        """
        new_tracks = []
        lost_tracks = []
        
        # Predict positions for existing tracks
        predicted_positions = {
            tid: track.predict_position()
            for tid, track in self.active_tracks.items()
        }
        
        # Match detections to tracks
        if self.active_tracks and detections:
            matches, unmatched_detections, unmatched_tracks = self._match(
                detections,
                list(self.active_tracks.values())
            )
            
            # Update matched tracks
            for det_idx, track_id in matches:
                detection = detections[det_idx]
                track = self.active_tracks[track_id]
                track.update(detection)
            
            # Handle unmatched tracks (missed detections)
            for track_id in unmatched_tracks:
                track = self.active_tracks[track_id]
                track.mark_missed()
                
                # Move to lost if exceeded max age
                if track.state == "lost":
                    lost_tracks.append(track)
                    self.lost_tracks[track_id] = track
                    del self.active_tracks[track_id]
                    self.total_tracks_lost += 1
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_detections:
                detection = detections[det_idx]
                new_track = self._create_track(detection)
                new_tracks.append(new_track)
                self.active_tracks[new_track.track_id] = new_track
        
        elif detections:
            # No existing tracks, create new ones
            for detection in detections:
                new_track = self._create_track(detection)
                new_tracks.append(new_track)
                self.active_tracks[new_track.track_id] = new_track
        
        else:
            # No detections, mark all tracks as missed
            for track in list(self.active_tracks.values()):
                track.mark_missed()
                if track.state == "lost":
                    lost_tracks.append(track)
                    self.lost_tracks[track.track_id] = track
                    del self.active_tracks[track.track_id]
                    self.total_tracks_lost += 1
        
        return TrackingResult(
            frame_id=frame_id,
            timestamp=datetime.now(),
            active_tracks=list(self.active_tracks.values()),
            new_tracks=new_tracks,
            lost_tracks=lost_tracks
        )
    
    def _create_track(self, detection: Detection) -> Track:
        """Create new track from detection"""
        track = Track(
            track_id=self.next_track_id,
            class_name=detection.class_name,
            pcu_value=detection.pcu_value,
            direction=self.direction,
            first_seen=detection.timestamp,
            last_seen=detection.timestamp
        )
        
        track.update(detection)
        
        self.next_track_id += 1
        self.total_tracks_created += 1
        
        return track
    
    def _match(
        self,
        detections: List[Detection],
        tracks: List[Track]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to tracks using IoU
        
        Returns:
            (matches, unmatched_detections, unmatched_tracks)
        """
        if not tracks or not detections:
            return [], list(range(len(detections))), [t.track_id for t in tracks]
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(tracks)))
        
        for d_idx, detection in enumerate(detections):
            for t_idx, track in enumerate(tracks):
                if track.current_bbox:
                    iou = self._compute_iou(detection.bbox, track.current_bbox)
                    iou_matrix[d_idx, t_idx] = iou
        
        # Hungarian algorithm for assignment
        det_indices, track_indices = linear_sum_assignment(-iou_matrix)
        
        # Filter matches by IoU threshold
        iou_threshold = 0.3  # Lower threshold for Indian traffic (more occlusions)
        matches = []
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(t.track_id for t in tracks)
        
        for d_idx, t_idx in zip(det_indices, track_indices):
            if iou_matrix[d_idx, t_idx] >= iou_threshold:
                matches.append((d_idx, tracks[t_idx].track_id))
                unmatched_detections.discard(d_idx)
                unmatched_tracks.discard(tracks[t_idx].track_id)
        
        return matches, list(unmatched_detections), list(unmatched_tracks)
    
    @staticmethod
    def _compute_iou(bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_statistics(self) -> Dict:
        """Get tracker statistics"""
        return {
            'direction': self.direction.value,
            'active_tracks': len(self.active_tracks),
            'lost_tracks': len(self.lost_tracks),
            'total_created': self.total_tracks_created,
            'total_lost': self.total_tracks_lost,
            'confirmed_tracks': sum(
                1 for t in self.active_tracks.values() if t.is_confirmed
            )
        }
    
    def reset(self) -> None:
        """Reset tracker state"""
        self.active_tracks.clear()
        self.lost_tracks.clear()
        self.finished_tracks.clear()
        self.next_track_id = 1
        self.logger.info(f"Reset tracker for {self.direction.value}")
