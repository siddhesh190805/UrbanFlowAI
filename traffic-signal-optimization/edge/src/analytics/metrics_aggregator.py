"""
Traffic Metrics Aggregator
Calculates real-time traffic metrics from detections and tracks
Generates state vectors for RL agent
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import numpy as np

from config.settings import EdgeSettings, Direction, VehicleType
from ai.tracker import Track
from storage.redis_cache import RedisCache
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DirectionMetrics:
    """Traffic metrics for a single direction"""
    direction: Direction
    
    # Vehicle counts by type
    vehicle_counts: Dict[VehicleType, int]
    
    # Traffic density (PCU per square meter)
    density: float
    
    # Queue depth (meters from stop line)
    queue_depth: float
    
    # Average waiting time (seconds)
    avg_wait_time: float
    
    # Average speed (m/s)
    avg_speed: float
    
    # Turn intentions (estimated)
    left_turn_ratio: float
    straight_ratio: float
    right_turn_ratio: float
    
    # Pedestrian metrics
    pedestrian_count: int
    pedestrian_wait_time: float
    
    # Timestamp
    timestamp: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'direction': self.direction.value,
            'vehicle_counts': {k.value: v for k, v in self.vehicle_counts.items()},
            'density': float(self.density),
            'queue_depth': float(self.queue_depth),
            'avg_wait_time': float(self.avg_wait_time),
            'avg_speed': float(self.avg_speed),
            'turn_intentions': {
                'left': float(self.left_turn_ratio),
                'straight': float(self.straight_ratio),
                'right': float(self.right_turn_ratio)
            },
            'pedestrian_count': self.pedestrian_count,
            'pedestrian_wait_time': float(self.pedestrian_wait_time),
            'timestamp': self.timestamp
        }


class MetricsAggregator:
    """
    Aggregates traffic data into meaningful metrics
    Calculates state vectors for decision making
    """
    
    def __init__(self, settings: EdgeSettings, redis_cache: Optional[RedisCache] = None):
        """
        Initialize metrics aggregator
        
        Args:
            settings: Edge device settings
            redis_cache: Redis cache instance (optional)
        """
        self.settings = settings
        self.redis = redis_cache
        
        # Detection zones (area in m²)
        self.zone_areas = {
            Direction.NORTH: 100.0,  # Default, should be configured
            Direction.SOUTH: 100.0,
            Direction.EAST: 100.0,
            Direction.WEST: 100.0
        }
        
        # Stop line positions (for queue depth calculation)
        self.stop_lines = {
            Direction.NORTH: (640, 360),  # (x, y) in pixels
            Direction.SOUTH: (640, 360),
            Direction.EAST: (640, 360),
            Direction.WEST: (640, 360)
        }
        
        # Historical data for averaging
        self.wait_time_history: Dict[Direction, deque] = {
            d: deque(maxlen=30) for d in Direction
        }
        
        self.speed_history: Dict[Direction, deque] = {
            d: deque(maxlen=30) for d in Direction
        }
        
        # PCU weights from settings
        self.pcu_weights = settings.pcu_weights
        
        logger.info("MetricsAggregator initialized")
    
    def calculate_metrics(
        self,
        direction: Direction,
        tracks: List[Track],
        frame_shape: Tuple[int, int]
    ) -> DirectionMetrics:
        """
        Calculate comprehensive traffic metrics for a direction
        
        Args:
            direction: Traffic direction
            tracks: List of vehicle tracks in this direction
            frame_shape: Frame dimensions (height, width)
        
        Returns:
            DirectionMetrics object
        """
        # Count vehicles by type
        vehicle_counts = self._count_vehicles_by_type(tracks)
        
        # Calculate density (PCU-based)
        density = self._calculate_density(tracks, direction)
        
        # Calculate queue depth
        queue_depth = self._calculate_queue_depth(tracks, direction, frame_shape)
        
        # Calculate average waiting time
        avg_wait_time = self._calculate_avg_wait_time(tracks)
        
        # Calculate average speed
        avg_speed = self._calculate_avg_speed(tracks)
        
        # Estimate turn intentions
        turn_ratios = self._estimate_turn_intentions(tracks, direction, frame_shape)
        
        # Count pedestrians
        pedestrian_count = vehicle_counts.get(VehicleType.PEDESTRIAN, 0)
        pedestrian_wait_time = self._calculate_pedestrian_wait_time(tracks)
        
        # Create metrics object
        metrics = DirectionMetrics(
            direction=direction,
            vehicle_counts=vehicle_counts,
            density=density,
            queue_depth=queue_depth,
            avg_wait_time=avg_wait_time,
            avg_speed=avg_speed,
            left_turn_ratio=turn_ratios['left'],
            straight_ratio=turn_ratios['straight'],
            right_turn_ratio=turn_ratios['right'],
            pedestrian_count=pedestrian_count,
            pedestrian_wait_time=pedestrian_wait_time,
            timestamp=time.time()
        )
        
        # Update history
        self.wait_time_history[direction].append(avg_wait_time)
        self.speed_history[direction].append(avg_speed)
        
        return metrics
    
    def _count_vehicles_by_type(self, tracks: List[Track]) -> Dict[VehicleType, int]:
        """Count vehicles by type"""
        counts = defaultdict(int)
        for track in tracks:
            counts[track.vehicle_type] += 1
        return dict(counts)
    
    def _calculate_density(self, tracks: List[Track], direction: Direction) -> float:
        """
        Calculate traffic density in PCU per square meter
        
        Args:
            tracks: Vehicle tracks
            direction: Traffic direction
        
        Returns:
            Density value
        """
        total_pcu = 0.0
        
        for track in tracks:
            vtype = track.vehicle_type.value
            pcu_weight = self.pcu_weights.get(vtype, 1.0)
            total_pcu += pcu_weight
        
        area = self.zone_areas.get(direction, 100.0)
        density = total_pcu / area
        
        return min(density, 1.0)  # Cap at 1.0 for normalization
    
    def _calculate_queue_depth(
        self,
        tracks: List[Track],
        direction: Direction,
        frame_shape: Tuple[int, int]
    ) -> float:
        """
        Calculate queue depth from stop line in meters
        
        Args:
            tracks: Vehicle tracks
            direction: Traffic direction
            frame_shape: Frame dimensions
        
        Returns:
            Queue depth in meters
        """
        if not tracks:
            return 0.0
        
        # Get stop line position
        stop_line_x, stop_line_y = self.stop_lines.get(direction, (frame_shape[1]//2, frame_shape[0]//2))
        
        # Find maximum distance from stop line
        max_distance = 0.0
        
        for track in tracks:
            # Skip fast-moving vehicles (not in queue)
            if track.speed > 1.0:  # > 1 m/s
                continue
            
            # Get vehicle center
            center_x, center_y = track.center
            
            # Calculate distance from stop line
            # This is simplified - should use actual camera calibration
            distance_pixels = np.sqrt((center_x - stop_line_x)**2 + (center_y - stop_line_y)**2)
            
            # Convert pixels to meters (approximate, needs calibration)
            # Assuming 1 meter ≈ 20 pixels (this should be calibrated)
            distance_meters = distance_pixels / 20.0
            
            max_distance = max(max_distance, distance_meters)
        
        return min(max_distance, 100.0)  # Cap at 100 meters
    
    def _calculate_avg_wait_time(self, tracks: List[Track]) -> float:
        """
        Calculate average waiting time for vehicles
        
        Args:
            tracks: Vehicle tracks
        
        Returns:
            Average wait time in seconds
        """
        if not tracks:
            return 0.0
        
        wait_times = []
        
        for track in tracks:
            # Consider vehicles with low speed as waiting
            if track.speed < 0.5:  # < 0.5 m/s
                wait_times.append(track.dwell_time)
        
        if not wait_times:
            return 0.0
        
        return float(np.mean(wait_times))
    
    def _calculate_avg_speed(self, tracks: List[Track]) -> float:
        """
        Calculate average speed of vehicles
        
        Args:
            tracks: Vehicle tracks
        
        Returns:
            Average speed in m/s
        """
        if not tracks:
            return 0.0
        
        speeds = [track.speed for track in tracks if track.speed > 0]
        
        if not speeds:
            return 0.0
        
        return float(np.mean(speeds))
    
    def _estimate_turn_intentions(
        self,
        tracks: List[Track],
        direction: Direction,
        frame_shape: Tuple[int, int]
    ) -> Dict[str, float]:
        """
        Estimate turn intentions from vehicle positions
        
        Args:
            tracks: Vehicle tracks
            direction: Traffic direction
            frame_shape: Frame dimensions
        
        Returns:
            Dictionary with left, straight, right ratios
        """
        if not tracks:
            return {'left': 0.33, 'straight': 0.34, 'right': 0.33}
        
        left_scores = []
        straight_scores = []
        right_scores = []
        
        frame_width = frame_shape[1]
        
        for track in tracks:
            center_x, _ = track.center
            
            # Lateral position (0 = left, 1 = right)
            lateral_pos = center_x / frame_width
            
            vtype = track.vehicle_type
            
            # Heuristic scoring based on vehicle type and position
            if vtype in [VehicleType.BUS, VehicleType.TRUCK]:
                # Large vehicles mostly go straight
                left_scores.append(0.05)
                straight_scores.append(0.90)
                right_scores.append(0.05)
            
            elif vtype == VehicleType.TWO_WHEELER:
                # Two-wheelers unpredictable
                left_scores.append(0.33)
                straight_scores.append(0.34)
                right_scores.append(0.33)
            
            else:
                # Use lateral position for other vehicles
                if lateral_pos < 0.35:  # Left side
                    left_scores.append(0.70)
                    straight_scores.append(0.25)
                    right_scores.append(0.05)
                elif lateral_pos > 0.65:  # Right side
                    left_scores.append(0.05)
                    straight_scores.append(0.25)
                    right_scores.append(0.70)
                else:  # Middle
                    left_scores.append(0.20)
                    straight_scores.append(0.60)
                    right_scores.append(0.20)
        
        return {
            'left': float(np.mean(left_scores)),
            'straight': float(np.mean(straight_scores)),
            'right': float(np.mean(right_scores))
        }
    
    def _calculate_pedestrian_wait_time(self, tracks: List[Track]) -> float:
        """Calculate average waiting time for pedestrians"""
        pedestrian_tracks = [
            t for t in tracks if t.vehicle_type == VehicleType.PEDESTRIAN
        ]
        
        if not pedestrian_tracks:
            return 0.0
        
        wait_times = [t.dwell_time for t in pedestrian_tracks if t.speed < 0.5]
        
        if not wait_times:
            return 0.0
        
        return float(np.mean(wait_times))
    
    def build_state_vector(
        self,
        all_metrics: Dict[Direction, DirectionMetrics],
        current_phase: int,
        time_in_phase: float
    ) -> np.ndarray:
        """
        Build state vector for RL agent
        
        Args:
            all_metrics: Metrics for all directions
            current_phase: Current signal phase
            time_in_phase: Time spent in current phase
        
        Returns:
            State vector as numpy array
        """
        state_components = []
        
        # For each direction
        for direction in [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]:
            metrics = all_metrics.get(direction)
            
            if metrics is None:
                # Use zero values if no data
                state_components.extend([0.0] * 15)
                continue
            
            # Add direction-specific features
            state_components.append(metrics.density)
            state_components.append(metrics.queue_depth / 100.0)  # Normalize
            state_components.append(metrics.avg_wait_time / 300.0)  # Normalize
            state_components.append(metrics.avg_speed / 15.0)  # Normalize (max ~15 m/s)
            
            # Vehicle counts (normalized)
            for vtype in self.settings.vehicle_classes:
                count = metrics.vehicle_counts.get(vtype, 0)
                state_components.append(count / 50.0)  # Normalize
            
            # Turn intentions
            state_components.append(metrics.left_turn_ratio)
            state_components.append(metrics.straight_ratio)
            state_components.append(metrics.right_turn_ratio)
            
            # Pedestrian metrics
            state_components.append(metrics.pedestrian_count / 30.0)  # Normalize
            state_components.append(metrics.pedestrian_wait_time / 120.0)  # Normalize
        
        # Signal state
        phase_onehot = [0.0] * len(self.settings.signal_phases)
        if 0 <= current_phase < len(phase_onehot):
            phase_onehot[current_phase] = 1.0
        state_components.extend(phase_onehot)
        
        # Time in phase (normalized)
        state_components.append(time_in_phase / 120.0)
        
        # Convert to numpy array
        state = np.array(state_components, dtype=np.float32)
        
        return state
    
    async def store_metrics(self, metrics: DirectionMetrics) -> None:
        """
        Store metrics in Redis cache
        
        Args:
            metrics: Direction metrics to store
        """
        if self.redis:
            await self.redis.set_traffic_state(
                metrics.direction.value,
                metrics.to_dict()
            )
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics across all directions"""
        avg_wait_times = {}
        avg_speeds = {}
        
        for direction in Direction:
            if len(self.wait_time_history[direction]) > 0:
                avg_wait_times[direction.value] = float(
                    np.mean(list(self.wait_time_history[direction]))
                )
            
            if len(self.speed_history[direction]) > 0:
                avg_speeds[direction.value] = float(
                    np.mean(list(self.speed_history[direction]))
                )
        
        return {
            'avg_wait_times': avg_wait_times,
            'avg_speeds': avg_speeds
        }