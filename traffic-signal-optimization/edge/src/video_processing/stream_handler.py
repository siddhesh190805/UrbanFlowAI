"""
Enhanced Video Stream Handler for Indian Traffic Signal Optimization
Handles multiple camera streams with reconnection, Indian traffic adaptation
Fully async-compatible with existing architecture
"""

import asyncio
import cv2
import numpy as np
from typing import Optional, Callable, Dict, List, Tuple, Any
from queue import Queue, Full, Empty
from threading import Thread, Event, Lock
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

from ..config.settings import CameraConfig, Direction, StreamType, get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FrameData:
    """
    Container for video frame with metadata
    Aligned with your existing architecture
    """
    camera_id: str
    direction: Direction
    timestamp: datetime
    frame_number: int
    data: np.ndarray
    width: int
    height: int
    
    # Additional metadata for analytics
    fps: float = 0.0
    latency_ms: float = 0.0
    
    def __post_init__(self):
        """Validate frame data"""
        if self.data is None or self.data.size == 0:
            raise ValueError("Frame data cannot be None or empty")
        if len(self.data.shape) != 3:
            raise ValueError(f"Expected 3D array (H, W, C), got shape {self.data.shape}")


class StreamHealth(str, Enum):
    """Stream health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class StreamStatistics:
    """Statistics for a video stream"""
    camera_id: str
    direction: Direction
    status: StreamHealth
    
    # Counters
    total_frames: int = 0
    dropped_frames: int = 0
    reconnect_attempts: int = 0
    successful_reconnects: int = 0
    
    # Performance metrics
    current_fps: float = 0.0
    average_latency_ms: float = 0.0
    buffer_utilization: float = 0.0
    
    # Timestamps
    started_at: Optional[datetime] = None
    last_frame_at: Optional[datetime] = None
    last_reconnect_at: Optional[datetime] = None
    
    # Health indicators
    consecutive_failures: int = 0
    uptime_seconds: float = 0.0


class VideoStream:
    """
    Handles a single video stream with reconnection logic
    Async-compatible, thread-safe implementation for Indian traffic
    """
    
    def __init__(
        self,
        camera_config: CameraConfig,
        frame_callback: Optional[Callable[[FrameData], None]] = None,
        use_gstreamer: bool = True
    ):
        self.config = camera_config
        self.frame_callback = frame_callback
        self.use_gstreamer = use_gstreamer
        
        self.logger = get_logger(f"VideoStream.{camera_config.id}")
        
        # Stream state
        self._is_running = False
        self._stop_event = Event()
        self._capture_thread: Optional[Thread] = None
        
        # Frame buffer (thread-safe queue)
        self._frame_buffer: Queue[FrameData] = Queue(maxsize=camera_config.buffer_size)
        self._buffer_lock = Lock()
        
        # Video capture
        self._cap: Optional[cv2.VideoCapture] = None
        
        # Statistics
        self._stats = StreamStatistics(
            camera_id=camera_config.id,
            direction=camera_config.direction,
            status=StreamHealth.STOPPED
        )
        self._stats_lock = Lock()
        
        # Performance tracking
        self._frame_times: List[float] = []
        self._last_frame_time: Optional[float] = None
        
        self.logger.info(
            f"Initialized stream for {camera_config.id} "
            f"({camera_config.direction.value}) - {camera_config.url}"
        )
    
    @property
    def is_running(self) -> bool:
        """Check if stream is running"""
        return self._is_running
    
    @property
    def camera_id(self) -> str:
        """Get camera ID"""
        return self.config.id
    
    @property
    def direction(self) -> Direction:
        """Get traffic direction"""
        return self.config.direction
    
    def start(self) -> bool:
        """Start the video stream"""
        if self._is_running:
            self.logger.warning(f"Stream {self.config.id} is already running")
            return False
        
        if not self.config.enabled:
            self.logger.warning(f"Stream {self.config.id} is disabled in config")
            return False
        
        self.logger.info(f"Starting stream: {self.config.id}")
        
        self._stop_event.clear()
        self._is_running = True
        
        # Update statistics
        with self._stats_lock:
            self._stats.status = StreamHealth.HEALTHY
            self._stats.started_at = datetime.now()
        
        # Start capture thread
        self._capture_thread = Thread(
            target=self._capture_loop,
            name=f"Capture-{self.config.id}",
            daemon=True
        )
        self._capture_thread.start()
        
        self.logger.info(f"Started stream: {self.config.id}")
        return True
    
    def stop(self) -> None:
        """Stop the video stream gracefully"""
        if not self._is_running:
            return
        
        self.logger.info(f"Stopping stream: {self.config.id}")
        
        self._is_running = False
        self._stop_event.set()
        
        # Wait for thread to finish
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5.0)
            
            if self._capture_thread.is_alive():
                self.logger.warning(
                    f"Capture thread for {self.config.id} did not stop gracefully"
                )
        
        # Release resources
        self._release_capture()
        
        # Update statistics
        with self._stats_lock:
            self._stats.status = StreamHealth.STOPPED
            if self._stats.started_at:
                self._stats.uptime_seconds = (
                    datetime.now() - self._stats.started_at
                ).total_seconds()
        
        self.logger.info(f"Stopped stream: {self.config.id}")
    
    def _capture_loop(self) -> None:
        """Main capture loop with reconnection logic"""
        reconnect_attempt = 0
        consecutive_failures = 0
        
        while self._is_running:
            try:
                # Initialize capture
                if not self._initialize_capture():
                    reconnect_attempt += 1
                    consecutive_failures += 1
                    
                    with self._stats_lock:
                        self._stats.status = StreamHealth.RECONNECTING
                        self._stats.reconnect_attempts += 1
                        self._stats.consecutive_failures = consecutive_failures
                        self._stats.last_reconnect_at = datetime.now()
                    
                    self.logger.error(
                        f"Failed to initialize {self.config.id}. "
                        f"Attempt {reconnect_attempt}/{self.config.reconnect_attempts}"
                    )
                    
                    if reconnect_attempt >= self.config.reconnect_attempts:
                        self.logger.critical(
                            f"Max reconnection attempts reached for {self.config.id}"
                        )
                        with self._stats_lock:
                            self._stats.status = StreamHealth.FAILED
                        break
                    
                    time.sleep(self.config.reconnect_delay)
                    continue
                
                # Successfully connected - reset counters
                reconnect_attempt = 0
                consecutive_failures = 0
                
                with self._stats_lock:
                    self._stats.status = StreamHealth.HEALTHY
                    self._stats.successful_reconnects += 1
                
                self.logger.info(f"Successfully connected: {self.config.id}")
                
                # Main frame capture loop
                while self._is_running and not self._stop_event.is_set():
                    loop_start = time.time()
                    
                    # Read frame
                    success, frame_data = self._read_frame()
                    
                    if not success:
                        consecutive_failures += 1
                        
                        if consecutive_failures > 10:
                            self.logger.warning(
                                f"Multiple consecutive failures for {self.config.id}. "
                                "Attempting reconnection..."
                            )
                            break
                        
                        # Brief wait before retry
                        time.sleep(0.1)
                        continue
                    
                    # Reset failure counter on success
                    consecutive_failures = 0
                    
                    # Calculate performance metrics
                    current_time = time.time()
                    if self._last_frame_time:
                        frame_interval = current_time - self._last_frame_time
                        self._frame_times.append(frame_interval)
                        
                        # Keep only last 30 frames for FPS calculation
                        if len(self._frame_times) > 30:
                            self._frame_times.pop(0)
                    
                    self._last_frame_time = current_time
                    
                    # Calculate current FPS
                    if len(self._frame_times) >= 2:
                        avg_interval = np.mean(self._frame_times)
                        current_fps = 1.0 / avg_interval if avg_interval > 0 else 0.0
                    else:
                        current_fps = 0.0
                    
                    # Create frame object
                    with self._stats_lock:
                        frame_number = self._stats.total_frames
                    
                    frame = FrameData(
                        camera_id=self.config.id,
                        direction=self.config.direction,
                        timestamp=datetime.now(),
                        frame_number=frame_number,
                        data=frame_data,
                        width=frame_data.shape[1],
                        height=frame_data.shape[0],
                        fps=current_fps,
                        latency_ms=(time.time() - loop_start) * 1000
                    )
                    
                    # Update statistics
                    with self._stats_lock:
                        self._stats.total_frames += 1
                        self._stats.last_frame_at = frame.timestamp
                        self._stats.current_fps = current_fps
                    
                    # Add to buffer
                    try:
                        self._frame_buffer.put(frame, block=False)
                    except Full:
                        # Buffer full - drop oldest and add new
                        try:
                            self._frame_buffer.get_nowait()
                            self._frame_buffer.put(frame, block=False)
                            
                            with self._stats_lock:
                                self._stats.dropped_frames += 1
                        except (Empty, Full):
                            pass
                    
                    # Update buffer utilization
                    with self._stats_lock:
                        self._stats.buffer_utilization = (
                            self._frame_buffer.qsize() / self.config.buffer_size
                        )
                    
                    # Call frame callback
                    if self.frame_callback:
                        try:
                            self.frame_callback(frame)
                        except Exception as e:
                            self.logger.error(f"Error in frame callback: {e}", exc_info=True)
                    
                    # Adaptive sleep to maintain target FPS
                    processing_time = time.time() - loop_start
                    target_interval = 1.0 / self.config.fps
                    sleep_time = max(0, target_interval - processing_time)
                    
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                # Connection lost
                if self._is_running:
                    self.logger.warning(f"Connection lost for {self.config.id}")
                    self._release_capture()
                    time.sleep(self.config.reconnect_delay)
            
            except Exception as e:
                self.logger.error(
                    f"Unexpected error in capture loop for {self.config.id}: {e}",
                    exc_info=True
                )
                reconnect_attempt += 1
                time.sleep(self.config.reconnect_delay)
        
        # Cleanup
        self._release_capture()
        self.logger.info(f"Capture loop ended for {self.config.id}")
    
    def _initialize_capture(self) -> bool:
        """Initialize video capture based on stream type"""
        try:
            if self.config.stream_type == StreamType.RTSP:
                return self._init_rtsp()
            elif self.config.stream_type == StreamType.USB:
                return self._init_usb()
            elif self.config.stream_type == StreamType.FILE:
                return self._init_file()
            elif self.config.stream_type == StreamType.TEST:
                return self._init_test()
            else:
                self.logger.error(f"Unsupported stream type: {self.config.stream_type}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error initializing capture: {e}", exc_info=True)
            return False
    
    def _init_rtsp(self) -> bool:
        """Initialize RTSP stream with GStreamer or OpenCV"""
        if self.use_gstreamer:
            # Try GStreamer pipeline first (better performance on Jetson)
            width, height = self.config.resolution
            
            pipeline = (
                f"rtspsrc location={self.config.url} latency=0 ! "
                f"rtph264depay ! h264parse ! "
                f"nvv4l2decoder ! nvvidconv ! "
                f"video/x-raw,format=BGRx,width={width},height={height} ! "
                f"videoconvert ! video/x-raw,format=BGR ! appsink"
            )
            
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if self._cap.isOpened():
                self.logger.info(f"Initialized GStreamer RTSP: {self.config.id}")
                return True
            
            self.logger.warning(
                f"GStreamer failed for {self.config.id}, falling back to OpenCV"
            )
        
        # Fallback to OpenCV RTSP
        self._cap = cv2.VideoCapture(self.config.url, cv2.CAP_FFMPEG)
        
        if not self._cap.isOpened():
            return False
        
        # Configure capture
        width, height = self.config.resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Low latency
        
        self.logger.info(f"Initialized OpenCV RTSP: {self.config.id}")
        return True
    
    def _init_usb(self) -> bool:
        """Initialize USB camera"""
        device_id = int(self.config.url) if self.config.url.isdigit() else self.config.url
        
        self._cap = cv2.VideoCapture(device_id)
        
        if not self._cap.isOpened():
            return False
        
        # Configure camera
        width, height = self.config.resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        self.logger.info(f"Initialized USB camera: {self.config.id}")
        return True
    
    def _init_file(self) -> bool:
        """Initialize video file"""
        if not Path(self.config.url).exists():
            self.logger.error(f"Video file not found: {self.config.url}")
            return False
        
        self._cap = cv2.VideoCapture(self.config.url)
        
        if not self._cap.isOpened():
            return False
        
        self.logger.info(f"Initialized video file: {self.config.id}")
        return True
    
    def _init_test(self) -> bool:
        """Initialize test pattern (synthetic video)"""
        self._cap = None  # Will generate synthetic frames
        self.logger.info(f"Initialized test pattern: {self.config.id}")
        return True
    
    def _read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from capture source"""
        if self.config.stream_type == StreamType.TEST:
            # Generate test frame
            width, height = self.config.resolution
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Add overlay
            cv2.putText(
                frame,
                f"{self.config.id} - {self.config.direction.value}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            with self._stats_lock:
                frame_num = self._stats.total_frames
            
            cv2.putText(
                frame,
                f"Frame: {frame_num}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            return True, frame
        
        if self._cap is None:
            return False, None
        
        ret, frame = self._cap.read()
        
        if not ret or frame is None:
            return False, None
        
        # Resize if needed
        width, height = self.config.resolution
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        
        return True, frame
    
    def _release_capture(self) -> None:
        """Release capture resources"""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                self.logger.error(f"Error releasing capture: {e}")
            finally:
                self._cap = None
    
    def get_frame(self, timeout: Optional[float] = None) -> Optional[FrameData]:
        """Get a frame from buffer (thread-safe)"""
        try:
            return self._frame_buffer.get(timeout=timeout)
        except Empty:
            return None
    
    def get_statistics(self) -> StreamStatistics:
        """Get current stream statistics"""
        with self._stats_lock:
            # Create a copy to avoid lock issues
            return StreamStatistics(
                camera_id=self._stats.camera_id,
                direction=self._stats.direction,
                status=self._stats.status,
                total_frames=self._stats.total_frames,
                dropped_frames=self._stats.dropped_frames,
                reconnect_attempts=self._stats.reconnect_attempts,
                successful_reconnects=self._stats.successful_reconnects,
                current_fps=self._stats.current_fps,
                average_latency_ms=self._stats.average_latency_ms,
                buffer_utilization=self._stats.buffer_utilization,
                started_at=self._stats.started_at,
                last_frame_at=self._stats.last_frame_at,
                last_reconnect_at=self._stats.last_reconnect_at,
                consecutive_failures=self._stats.consecutive_failures,
                uptime_seconds=self._stats.uptime_seconds
            )


class StreamManager:
    """
    Manages multiple video streams
    Thread-safe, async-compatible
    """
    
    def __init__(self):
        self.logger = get_logger("StreamManager")
        self._streams: Dict[str, VideoStream] = {}
        self._streams_lock = Lock()
        self._is_running = False
        
        self.logger.info("StreamManager initialized")
    
    def add_stream(
        self,
        camera_config: CameraConfig,
        frame_callback: Optional[Callable[[FrameData], None]] = None
    ) -> VideoStream:
        """Add a new video stream"""
        with self._streams_lock:
            if camera_config.id in self._streams:
                self.logger.warning(f"Stream {camera_config.id} already exists")
                return self._streams[camera_config.id]
            
            stream = VideoStream(camera_config, frame_callback)
            self._streams[camera_config.id] = stream
            
            # Auto-start if manager is running
            if self._is_running and camera_config.enabled:
                stream.start()
            
            self.logger.info(
                f"Added stream: {camera_config.id} ({camera_config.direction.value})"
            )
            return stream
    
    def remove_stream(self, camera_id: str) -> bool:
        """Remove and stop a video stream"""
        with self._streams_lock:
            if camera_id not in self._streams:
                self.logger.warning(f"Stream {camera_id} not found")
                return False
            
            stream = self._streams[camera_id]
            stream.stop()
            del self._streams[camera_id]
            
            self.logger.info(f"Removed stream: {camera_id}")
            return True
    
    def get_stream(self, camera_id: str) -> Optional[VideoStream]:
        """Get a specific stream"""
        with self._streams_lock:
            return self._streams.get(camera_id)
    
    def start_all(self) -> int:
        """Start all video streams"""
        self._is_running = True
        started_count = 0
        
        with self._streams_lock:
            for stream in self._streams.values():
                if stream.start():
                    started_count += 1
        
        self.logger.info(f"Started {started_count}/{len(self._streams)} video streams")
        return started_count
    
    def stop_all(self) -> None:
        """Stop all video streams"""
        self._is_running = False
        
        with self._streams_lock:
            for stream in self._streams.values():
                stream.stop()
        
        self.logger.info("Stopped all video streams")
    
    def get_all_statistics(self) -> List[StreamStatistics]:
        """Get statistics for all streams"""
        with self._streams_lock:
            return [stream.get_statistics() for stream in self._streams.values()]
    
    def get_latest_frames(self) -> Dict[str, FrameData]:
        """Get latest frame from each stream"""
        frames = {}
        
        with self._streams_lock:
            stream_list = list(self._streams.items())
        
        for camera_id, stream in stream_list:
            frame = stream.get_frame(timeout=0.1)
            if frame:
                frames[camera_id] = frame
        
        return frames
    
    def get_frames_by_direction(self, direction: Direction) -> List[FrameData]:
        """Get latest frames for a specific direction"""
        frames = []
        
        with self._streams_lock:
            streams = [
                s for s in self._streams.values()
                if s.direction == direction
            ]
        
        for stream in streams:
            frame = stream.get_frame(timeout=0.1)
            if frame:
                frames.append(frame)
        
        return frames
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        stats = self.get_all_statistics()
        
        total_streams = len(stats)
        healthy = sum(1 for s in stats if s.status == StreamHealth.HEALTHY)
        degraded = sum(1 for s in stats if s.status == StreamHealth.DEGRADED)
        failed = sum(1 for s in stats if s.status == StreamHealth.FAILED)
        
        return {
            'total_streams': total_streams,
            'healthy': healthy,
            'degraded': degraded,
            'failed': failed,
            'overall_health': 'healthy' if failed == 0 and degraded == 0 else 'degraded',
            'streams': [
                {
                    'camera_id': s.camera_id,
                    'direction': s.direction.value,
                    'status': s.status.value,
                    'fps': s.current_fps,
                    'uptime': s.uptime_seconds
                }
                for s in stats
            ]
        }
