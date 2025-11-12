"""
Unified Inference Engine
Combines detection and tracking for complete scene understanding
Optimized for Indian traffic patterns
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

from .detector import VehicleDetector, DetectionResult
from .tracker import VehicleTracker, TrackingResult, Track
from ..video_processing.stream_handler import FrameData
from ..config.settings import Direction, get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SceneAnalysis:
    """Complete scene analysis for an intersection direction"""
    direction: Direction
    timestamp: datetime
    frame_id: str
    
    # Detection and tracking results
    detection_result: DetectionResult
    tracking_result: TrackingResult
    
    # Aggregated metrics
    vehicle_count: int
    confirmed_vehicle_count: int
    total_pcu: float
    vehicle_density: float  # vehicles per square meter of detection zone
    
    # Vehicle composition
    vehicle_counts_by_type: Dict[str, int]
    pcu_by_type: Dict[str, float]
    
    # Traffic characteristics
    avg_vehicle_speed: float  # pixels per frame (would need calibration for real speed)
    congestion_level: str  # "low", "medium", "high", "severe"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'direction': self.direction.value,
            'timestamp': self.timestamp.isoformat(),
            'frame_id': self.frame_id,
            'vehicle_count': self.vehicle_count,
            'confirmed_vehicle_count': self.confirmed_vehicle_count,
            'total_pcu': self.total_pcu,
            'vehicle_density': self.vehicle_density,
            'vehicle_counts_by_type': self.vehicle_counts_by_type,
            'pcu_by_type': self.pcu_by_type,
            'avg_vehicle_speed': self.avg_vehicle_speed,
            'congestion_level': self.congestion_level
        }


class InferenceEngine:
    """
    Main inference engine coordinating detection and tracking
    Processes frames from all directions
    """
    
    def __init__(self):
        self.logger = get_logger("InferenceEngine")
        self.settings = get_settings()
        
        # Initialize detector (shared across all directions)
        self.detector = VehicleDetector(self.settings.ai_models)
        
        # Initialize trackers per direction
        self.trackers: Dict[Direction, VehicleTracker] = {
            Direction.NORTH: VehicleTracker(Direction.NORTH),
            Direction.SOUTH: VehicleTracker(Direction.SOUTH),
            Direction.EAST: VehicleTracker(Direction.EAST),
            Direction.WEST: VehicleTracker(Direction.WEST)
        }
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Processing statistics
        self.frames_processed = 0
        self.total_processing_time = 0.0
        self.lock = threading.Lock()
        
        self.logger.info("InferenceEngine initialized")
    
    def process_frame(self, frame: FrameData) -> SceneAnalysis:
        """
        Process a single frame: detect + track + analyze
        
        Args:
            frame: Input frame data
        
        Returns:
            SceneAnalysis with complete scene understanding
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Detect vehicles
            detection_result = self.detector.detect(
                frame.data,
                frame_id=f"{frame.camera_id}_{frame.frame_number}"
            )
            
            # Step 2: Update tracker
            tracker = self.trackers[frame.direction]
            tracking_result = tracker.update(
                detection_result.detections,
                frame_id=f"{frame.camera_id}_{frame.frame_number}"
            )
            
            # Step 3: Analyze scene
            scene_analysis = self._analyze_scene(
                frame,
                detection_result,
                tracking_result
            )
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            with self.lock:
                self.frames_processed += 1
                self.total_processing_time += processing_time
            
            return scene_analysis
        
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}", exc_info=True)
            # Return empty analysis
            return self._empty_analysis(frame)
    
    async def process_frame_async(self, frame: FrameData) -> SceneAnalysis:
        """Async version of process_frame"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.process_frame, frame)
    
    def process_batch(self, frames: List[FrameData]) -> List[SceneAnalysis]:
        """Process multiple frames in parallel"""
        results = []
        for frame in frames:
            result = self.process_frame(frame)
            results.append(result)
        return results
    
    async def process_batch_async(self, frames: List[FrameData]) -> List[SceneAnalysis]:
        """Process multiple frames in parallel (async)"""
        tasks = [self.process_frame_async(frame) for frame in frames]
        return await asyncio.gather(*tasks)
    
    def _analyze_scene(
        self,
        frame: FrameData,
        detection_result: DetectionResult,
        tracking_result: TrackingResult
    ) -> SceneAnalysis:
        """Analyze scene from detection and tracking results"""
        
        # Get detection zone for this direction
        detection_zone = self._get_detection_zone(frame.direction)
        zone_area = self._calculate_polygon_area(detection_zone) if detection_zone else 1.0
        
        # Calculate density (PCU per square meter)
        vehicle_density = tracking_result.total_pcu / zone_area if zone_area > 0 else 0.0
        
        # Calculate average speed from tracks
        avg_speed = self._calculate_avg_speed(tracking_result.active_tracks)
        
        # Determine congestion level
        congestion_level = self._determine_congestion(
            vehicle_density,
            tracking_result.confirmed_vehicles,
            avg_speed
        )
        
        return SceneAnalysis(
            direction=frame.direction,
            timestamp=frame.timestamp,
            frame_id=f"{frame.camera_id}_{frame.frame_number}",
            detection_result=detection_result,
            tracking_result=tracking_result,
            vehicle_count=tracking_result.total_vehicles,
            confirmed_vehicle_count=tracking_result.confirmed_vehicles,
            total_pcu=tracking_result.total_pcu,
            vehicle_density=vehicle_density,
            vehicle_counts_by_type=tracking_result.count_by_class(),
            pcu_by_type=detection_result.pcu_by_class(),
            avg_vehicle_speed=avg_speed,
            congestion_level=congestion_level
        )
    
    def _get_detection_zone(self, direction: Direction) -> Optional[List[List[int]]]:
        """Get detection zone polygon for a direction"""
        if not self.settings.intersection:
            return None
        
        for zone in self.settings.intersection.detection_zones:
            if zone.direction == direction:
                return zone.polygon_points
        
        return None
    
    @staticmethod
    def _calculate_polygon_area(polygon: List[List[int]]) -> float:
        """Calculate area of polygon using shoelace formula"""
        if len(polygon) < 3:
            return 0.0
        
        area = 0.0
        for i in range(len(polygon)):
            j = (i + 1) % len(polygon)
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        
        return abs(area) / 2.0
    
    @staticmethod
    def _calculate_avg_speed(tracks: List[Track]) -> float:
        """Calculate average speed from confirmed tracks"""
        speeds = []
        
        for track in tracks:
            if track.is_confirmed and len(track.trajectory) >= 2:
                # Calculate speed from velocity
                vx, vy = track.avg_velocity
                speed = np.sqrt(vx**2 + vy**2)
                speeds.append(speed)
        
        return np.mean(speeds) if speeds else 0.0
    
    @staticmethod
    def _determine_congestion(
        density: float,
        vehicle_count: int,
        avg_speed: float
    ) -> str:
        """
        Determine congestion level
        
        Thresholds calibrated for Indian traffic:
        - Low: < 0.05 PCU/m² or < 5 vehicles
        - Medium: 0.05-0.15 PCU/m² or 5-15 vehicles
        - High: 0.15-0.30 PCU/m² or 15-30 vehicles
        - Severe: > 0.30 PCU/m² or > 30 vehicles
        """
        if density > 0.30 or vehicle_count > 30:
            return "severe"
        elif density > 0.15 or vehicle_count > 15:
            return "high"
        elif density > 0.05 or vehicle_count > 5:
            return "medium"
        else:
            return "low"
    
    def _empty_analysis(self, frame: FrameData) -> SceneAnalysis:
        """Create empty analysis on error"""
        return SceneAnalysis(
            direction=frame.direction,
            timestamp=frame.timestamp,
            frame_id=f"{frame.camera_id}_{frame.frame_number}",
            detection_result=DetectionResult(
                frame_id="",
                timestamp=datetime.now(),
                detections=[],
                inference_time_ms=0.0,
                image_shape=frame.data.shape
            ),
            tracking_result=TrackingResult(
                frame_id="",
                timestamp=datetime.now(),
                active_tracks=[],
                new_tracks=[],
                lost_tracks=[]
            ),
            vehicle_count=0,
            confirmed_vehicle_count=0,
            total_pcu=0.0,
            vehicle_density=0.0,
            vehicle_counts_by_type={},
            pcu_by_type={},
            avg_vehicle_speed=0.0,
            congestion_level="low"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        with self.lock:
            avg_processing_time = (
                self.total_processing_time / self.frames_processed
                if self.frames_processed > 0 else 0.0
            )
        
        return {
            'frames_processed': self.frames_processed,
            'avg_processing_time_sec': avg_processing_time,
            'detector_stats': self.detector.get_statistics(),
            'tracker_stats': {
                direction.value: tracker.get_statistics()
                for direction, tracker in self.trackers.items()
            }
        }
    
    def reset(self) -> None:
        """Reset all trackers"""
        for tracker in self.trackers.values():
            tracker.reset()
        
        self.logger.info("Reset all trackers")
