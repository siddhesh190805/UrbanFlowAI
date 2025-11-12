"""
Vehicle Detection Engine using YOLOv8
Optimized for Indian traffic scenarios with vehicle classification
Supports TensorRT acceleration, zone filtering, and PCU calculations
"""

import time
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from config.settings import EdgeSettings, VehicleType
from utils.logger import get_logger

logger = get_logger(__name__)


# PCU (Passenger Car Unit) values for Indian traffic
PCU_VALUES = {
    VehicleType.PEDESTRIAN: 0.2,
    VehicleType.BICYCLE: 0.5,
    VehicleType.TWO_WHEELER: 0.5,
    VehicleType.THREE_WHEELER: 0.75,
    VehicleType.CAR: 1.0,
    VehicleType.SUV: 1.2,
    VehicleType.BUS: 3.0,
    VehicleType.TRUCK: 2.5,
    VehicleType.OTHER: 1.0
}


@dataclass
class Detection:
    """Single vehicle detection result with PCU"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    vehicle_type: VehicleType
    area: float = field(init=False)
    center: Tuple[float, float] = field(init=False)
    pcu_value: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived properties"""
        x1, y1, x2, y2 = self.bbox
        self.area = (x2 - x1) * (y2 - y1)
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        self.pcu_value = PCU_VALUES.get(self.vehicle_type, 1.0)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'bbox': self.bbox,
            'confidence': float(self.confidence),
            'class_id': int(self.class_id),
            'class_name': self.class_name,
            'vehicle_type': self.vehicle_type.value,
            'area': float(self.area),
            'center': self.center,
            'pcu_value': float(self.pcu_value)
        }


@dataclass
class DetectionResult:
    """Complete detection result for a frame"""
    detections: List[Detection]
    frame_shape: Tuple[int, int, int]
    inference_time_ms: float
    timestamp: float
    camera_id: str
    
    def __len__(self) -> int:
        return len(self.detections)
    
    def filter_by_type(self, vehicle_type: VehicleType) -> List[Detection]:
        """Filter detections by vehicle type"""
        return [d for d in self.detections if d.vehicle_type == vehicle_type]
    
    def filter_by_confidence(self, min_confidence: float) -> List[Detection]:
        """Filter detections by minimum confidence"""
        return [d for d in self.detections if d.confidence >= min_confidence]
    
    def filter_by_zone(self, zone_polygon: List[List[int]]) -> List[Detection]:
        """Filter detections inside polygon zone"""
        return [
            d for d in self.detections
            if self._point_in_polygon(d.center, zone_polygon)
        ]
    
    @staticmethod
    def _point_in_polygon(point: Tuple[float, float], polygon: List[List[int]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def count_by_type(self) -> Dict[VehicleType, int]:
        """Count detections by vehicle type"""
        counts = {vtype: 0 for vtype in VehicleType}
        for detection in self.detections:
            counts[detection.vehicle_type] += 1
        return counts
    
    def total_pcu(self) -> float:
        """Calculate total PCU value"""
        return sum(d.pcu_value for d in self.detections)
    
    def pcu_by_type(self) -> Dict[VehicleType, float]:
        """Calculate PCU value by vehicle type"""
        pcu = {vtype: 0.0 for vtype in VehicleType}
        for detection in self.detections:
            pcu[detection.vehicle_type] += detection.pcu_value
        return pcu
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'detections': [d.to_dict() for d in self.detections],
            'frame_shape': self.frame_shape,
            'inference_time_ms': float(self.inference_time_ms),
            'timestamp': float(self.timestamp),
            'camera_id': self.camera_id,
            'vehicle_counts': {k.value: v for k, v in self.count_by_type().items()},
            'total_pcu': float(self.total_pcu()),
            'pcu_by_type': {k.value: float(v) for k, v in self.pcu_by_type().items()}
        }


class VehicleDetector:
    """
    YOLOv8-based vehicle detector optimized for Indian traffic surveillance
    Handles multiple vehicle types with zone filtering and PCU calculations
    """
    
    # Mapping from COCO classes to our vehicle types
    CLASS_MAPPING = {
        0: VehicleType.PEDESTRIAN,      # person
        1: VehicleType.BICYCLE,         # bicycle
        2: VehicleType.CAR,             # car
        3: VehicleType.TWO_WHEELER,     # motorcycle
        5: VehicleType.BUS,             # bus
        6: VehicleType.TRUCK,           # truck
        7: VehicleType.TRUCK,           # truck (alternative)
    }
    
    # Indian-specific mappings (for custom trained models)
    INDIAN_CLASS_MAPPING = {
        0: VehicleType.PEDESTRIAN,
        1: VehicleType.BICYCLE,
        2: VehicleType.CAR,
        3: VehicleType.TWO_WHEELER,
        4: VehicleType.THREE_WHEELER,   # auto-rickshaw
        5: VehicleType.BUS,
        6: VehicleType.TRUCK,
        7: VehicleType.SUV,
    }
    
    def __init__(self, settings: EdgeSettings):
        """
        Initialize vehicle detector
        
        Args:
            settings: Edge device settings
        """
        self.settings = settings
        self.model: Optional[YOLO] = None
        self.device = settings.device
        self.confidence_threshold = settings.detection_confidence_threshold
        self.use_tensorrt = settings.use_tensorrt
        
        # Performance tracking
        self.total_detections = 0
        self.total_inference_time = 0.0
        self.frame_count = 0
        
        # Class mapping based on traffic mode
        if settings.traffic_mode.value == "indian_direction_based":
            self.class_mapping = self.INDIAN_CLASS_MAPPING
        else:
            self.class_mapping = self.CLASS_MAPPING
        
        logger.info(f"VehicleDetector initialized with device: {self.device}")
    
    def load_model(self, model_path: Optional[Path] = None) -> bool:
        """
        Load YOLO model
        
        Args:
            model_path: Path to model file (uses settings if None)
        
        Returns:
            True if model loaded successfully
        """
        try:
            model_path = model_path or self.settings.detector_model_path
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            logger.info(f"Loading model from: {model_path}")
            
            # Load model
            self.model = YOLO(str(model_path))
            
            # Move to appropriate device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')
                logger.info(f"Model loaded on CUDA (GPU: {torch.cuda.get_device_name(0)})")
            elif self.device == 'mps' and torch.backends.mps.is_available():
                self.model.to('mps')
                logger.info("Model loaded on Apple Silicon (MPS)")
            else:
                self.model.to('cpu')
                logger.info("Model loaded on CPU")
            
            # Warm-up inference
            logger.info("Running warm-up inference...")
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_frame, verbose=False)
            logger.success("Model loaded and ready")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def detect(
        self,
        frame: np.ndarray,
        camera_id: str,
        conf_threshold: Optional[float] = None
    ) -> DetectionResult:
        """
        Perform vehicle detection on a frame
        
        Args:
            frame: Input frame (BGR format)
            camera_id: Camera identifier
            conf_threshold: Confidence threshold (uses default if None)
        
        Returns:
            DetectionResult containing all detections
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        conf_threshold = conf_threshold or self.confidence_threshold
        
        # Start timing
        start_time = time.time()
        
        # Run inference
        results = self.model(
            frame,
            conf=conf_threshold,
            verbose=False,
            device=self.device
        )
        
        # Calculate inference time
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Parse results
        detections = []
        if len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for bbox, conf, cls_id in zip(boxes, confidences, class_ids):
                    # Map to vehicle type
                    vehicle_type = self.class_mapping.get(cls_id, VehicleType.OTHER)
                    
                    # Skip if not a vehicle we care about
                    if vehicle_type not in self.settings.vehicle_classes:
                        continue
                    
                    # Get class name
                    class_name = result.names[cls_id] if cls_id in result.names else "unknown"
                    
                    detection = Detection(
                        bbox=tuple(bbox.tolist()),
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=class_name,
                        vehicle_type=vehicle_type
                    )
                    detections.append(detection)
        
        # Update statistics
        self.frame_count += 1
        self.total_detections += len(detections)
        self.total_inference_time += inference_time_ms
        
        # Create result
        detection_result = DetectionResult(
            detections=detections,
            frame_shape=frame.shape,
            inference_time_ms=inference_time_ms,
            timestamp=time.time(),
            camera_id=camera_id
        )
        
        return detection_result
    
    def detect_in_zone(
        self,
        frame: np.ndarray,
        camera_id: str,
        detection_zone: List[List[int]],
        conf_threshold: Optional[float] = None
    ) -> DetectionResult:
        """
        Perform detection only within specified polygon zone (for Indian direction-based traffic)
        
        Args:
            frame: Input frame
            camera_id: Camera identifier
            detection_zone: Polygon points defining detection zone
            conf_threshold: Confidence threshold
        
        Returns:
            DetectionResult with detections filtered to zone
        """
        # Perform full detection
        result = self.detect(frame, camera_id, conf_threshold)
        
        # Filter to zone
        result.detections = result.filter_by_zone(detection_zone)
        
        return result
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        camera_ids: List[str],
        conf_threshold: Optional[float] = None
    ) -> List[DetectionResult]:
        """
        Perform batch detection on multiple frames
        
        Args:
            frames: List of input frames
            camera_ids: List of camera identifiers
            conf_threshold: Confidence threshold
        
        Returns:
            List of DetectionResult objects
        """
        if len(frames) != len(camera_ids):
            raise ValueError("Number of frames must match number of camera_ids")
        
        results = []
        for frame, camera_id in zip(frames, camera_ids):
            result = self.detect(frame, camera_id, conf_threshold)
            results.append(result)
        
        return results
    
    async def detect_batch_async(
        self,
        frames: List[np.ndarray],
        camera_ids: List[str],
        conf_threshold: Optional[float] = None
    ) -> List[DetectionResult]:
        """
        Async batch detection for better performance
        
        Args:
            frames: List of input frames
            camera_ids: List of camera identifiers
            conf_threshold: Confidence threshold
        
        Returns:
            List of DetectionResult objects
        """
        if len(frames) != len(camera_ids):
            raise ValueError("Number of frames must match number of camera_ids")
        
        loop = asyncio.get_event_loop()
        tasks = []
        
        for frame, camera_id in zip(frames, camera_ids):
            task = loop.run_in_executor(
                None,
                self.detect,
                frame,
                camera_id,
                conf_threshold
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    def visualize(
        self,
        frame: np.ndarray,
        detection_result: DetectionResult,
        show_conf: bool = True,
        show_class: bool = True,
        show_pcu: bool = True,
        detection_zone: Optional[List[List[int]]] = None
    ) -> np.ndarray:
        """
        Visualize detections on frame
        
        Args:
            frame: Input frame
            detection_result: Detection results
            show_conf: Show confidence scores
            show_class: Show class names
            show_pcu: Show PCU values
            detection_zone: Optional zone polygon to draw
        
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        # Draw detection zone if provided
        if detection_zone is not None:
            zone_array = np.array(detection_zone, dtype=np.int32)
            cv2.polylines(vis_frame, [zone_array], True, (255, 255, 0), 2)
            cv2.fillPoly(vis_frame, [zone_array], (255, 255, 0), lineType=cv2.LINE_AA)
            cv2.addWeighted(frame, 0.7, vis_frame, 0.3, 0, vis_frame)
        
        # Color mapping for vehicle types
        colors = {
            VehicleType.TWO_WHEELER: (0, 255, 255),     # Yellow
            VehicleType.THREE_WHEELER: (255, 165, 0),   # Orange
            VehicleType.CAR: (0, 255, 0),               # Green
            VehicleType.SUV: (0, 200, 0),               # Dark Green
            VehicleType.BUS: (255, 0, 0),               # Blue
            VehicleType.TRUCK: (0, 0, 255),             # Red
            VehicleType.BICYCLE: (255, 255, 0),         # Cyan
            VehicleType.PEDESTRIAN: (255, 0, 255),      # Magenta
            VehicleType.OTHER: (128, 128, 128)          # Gray
        }
        
        for detection in detection_result.detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            color = colors.get(detection.vehicle_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cx, cy = map(int, detection.center)
            cv2.circle(vis_frame, (cx, cy), 4, color, -1)
            
            # Prepare label
            label_parts = []
            if show_class:
                label_parts.append(detection.vehicle_type.value)
            if show_conf:
                label_parts.append(f"{detection.confidence:.2f}")
            if show_pcu:
                label_parts.append(f"PCU:{detection.pcu_value:.1f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Draw label background
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    vis_frame,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    vis_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
        
        # Draw statistics
        stats_text = [
            f"Detections: {len(detection_result.detections)}",
            f"Total PCU: {detection_result.total_pcu():.1f}",
            f"Inference: {detection_result.inference_time_ms:.1f}ms",
            f"FPS: {1000.0 / detection_result.inference_time_ms:.1f}"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(
                vis_frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            y_offset += 30
        
        return vis_frame
    
    @property
    def average_inference_time(self) -> float:
        """Get average inference time in milliseconds"""
        if self.frame_count == 0:
            return 0.0
        return self.total_inference_time / self.frame_count
    
    @property
    def average_detections_per_frame(self) -> float:
        """Get average number of detections per frame"""
        if self.frame_count == 0:
            return 0.0
        return self.total_detections / self.frame_count
    
    @property
    def fps(self) -> float:
        """Get effective FPS (frames per second)"""
        if self.average_inference_time == 0:
            return 0.0
        return 1000.0 / self.average_inference_time
    
    def get_stats(self) -> Dict:
        """Get detector statistics"""
        return {
            'frame_count': self.frame_count,
            'total_detections': self.total_detections,
            'avg_inference_time_ms': round(self.average_inference_time, 2),
            'avg_detections_per_frame': round(self.average_detections_per_frame, 2),
            'fps': round(self.fps, 2),
            'device': self.device,
            'model_loaded': self.model is not None,
            'confidence_threshold': self.confidence_threshold
        }
    
    def reset_stats(self) -> None:
        """Reset performance statistics"""
        self.total_detections = 0
        self.total_inference_time = 0.0
        self.frame_count = 0
        logger.info("Detector statistics reset")
