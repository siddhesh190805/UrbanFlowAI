"""
InfluxDB Client for Time-Series Traffic Data
Stores metrics, detections, and performance data with batching and retry
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import deque
from threading import Lock, Thread

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import ASYNCHRONOUS
from influxdb_client.client.exceptions import InfluxDBError

from config.settings import EdgeSettings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point"""
    measurement: str
    tags: Dict[str, str]
    fields: Dict[str, Any]
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_point(self) -> Point:
        """Convert to InfluxDB Point"""
        point = Point(self.measurement)
        
        # Add tags
        for key, value in self.tags.items():
            point.tag(key, str(value))
        
        # Add fields
        for key, value in self.fields.items():
            point.field(key, value)
        
        # Set timestamp
        point.time(int(self.timestamp * 1e9), WritePrecision.NS)
        
        return point


class InfluxDBStorage:
    """
    InfluxDB client with batching, retry logic, and automatic flushing
    """
    
    def __init__(self, settings: EdgeSettings):
        """Initialize InfluxDB storage"""
        self.settings = settings
        self.client: Optional[InfluxDBClient] = None
        self.write_api = None
        
        # Batch buffer
        self.batch_buffer: deque[MetricPoint] = deque(maxlen=10000)
        self.buffer_lock = Lock()
        
        # Configuration
        self.batch_size = settings.influxdb_batch_size
        self.flush_interval = settings.influxdb_flush_interval
        self.max_age_seconds = 300  # Drop points older than 5 minutes
        
        # Statistics
        self.points_written = 0
        self.points_dropped = 0
        self.write_errors = 0
        self.last_write_time: Optional[float] = None
        
        # Background flush thread
        self.flush_thread: Optional[Thread] = None
        self.flush_thread_running = False
        
        logger.info(
            f"InfluxDB storage initialized (batch_size: {self.batch_size}, "
            f"flush_interval: {self.flush_interval}s)"
        )
    
    async def connect(self, max_retries: int = 3) -> bool:
        """
        Connect to InfluxDB with retry logic
        
        Args:
            max_retries: Maximum number of connection attempts
        
        Returns:
            True if connection successful
        """
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Connecting to InfluxDB at {self.settings.influxdb_url} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                
                # Create client
                self.client = InfluxDBClient(
                    url=self.settings.influxdb_url,
                    token=self.settings.influxdb_token,
                    org=self.settings.influxdb_org,
                    timeout=10000  # 10 second timeout
                )
                
                # Test connection
                health = self.client.health()
                if health.status != "pass":
                    raise ConnectionError(f"InfluxDB health check failed: {health.status}")
                
                # Create write API
                self.write_api = self.client.write_api(write_options=ASYNCHRONOUS)
                
                logger.success(f"Connected to InfluxDB successfully")
                
                # Start background flush thread
                self._start_flush_thread()
                
                return True
            
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("All connection attempts failed")
                    return False
    
    async def disconnect(self) -> None:
        """Disconnect from InfluxDB"""
        logger.info("Disconnecting from InfluxDB...")
        
        # Stop flush thread
        self._stop_flush_thread()
        
        # Flush remaining data
        await self.flush()
        
        # Close client
        if self.write_api:
            self.write_api.close()
        
        if self.client:
            self.client.close()
        
        logger.info("Disconnected from InfluxDB")
    
    def write_point(self, point: MetricPoint) -> bool:
        """
        Add point to batch buffer
        
        Args:
            point: Metric point to write
        
        Returns:
            True if added successfully
        """
        # Check if point is too old
        age = time.time() - point.timestamp
        if age > self.max_age_seconds:
            logger.warning(f"Dropping stale point (age: {age:.1f}s)")
            self.points_dropped += 1
            return False
        
        # Add to buffer
        with self.buffer_lock:
            self.batch_buffer.append(point)
        
        # Auto-flush if buffer is full
        if len(self.batch_buffer) >= self.batch_size:
            asyncio.create_task(self.flush())
        
        return True
    
    async def write_traffic_metrics(
        self,
        direction: str,
        metrics: Dict[str, Any]
    ) -> bool:
        """Write traffic metrics for a direction"""
        point = MetricPoint(
            measurement="traffic_metrics",
            tags={
                "intersection_id": self.settings.intersection_id,
                "direction": direction,
                "city": self.settings.city
            },
            fields=metrics
        )
        
        return self.write_point(point)
    
    async def write_detection_count(
        self,
        camera_id: str,
        vehicle_type: str,
        count: int
    ) -> bool:
        """Write vehicle detection count"""
        point = MetricPoint(
            measurement="vehicle_detections",
            tags={
                "intersection_id": self.settings.intersection_id,
                "camera_id": camera_id,
                "vehicle_type": vehicle_type
            },
            fields={
                "count": count
            }
        )
        
        return self.write_point(point)
    
    async def write_signal_state(
        self,
        phase: int,
        state: str,
        time_in_phase: float
    ) -> bool:
        """Write signal controller state"""
        point = MetricPoint(
            measurement="signal_state",
            tags={
                "intersection_id": self.settings.intersection_id,
                "phase": str(phase)
            },
            fields={
                "state": state,
                "time_in_phase": time_in_phase
            }
        )
        
        return self.write_point(point)
    
    async def write_performance_metrics(
        self,
        component: str,
        metrics: Dict[str, float]
    ) -> bool:
        """Write system performance metrics"""
        point = MetricPoint(
            measurement="system_performance",
            tags={
                "intersection_id": self.settings.intersection_id,
                "component": component
            },
            fields=metrics
        )
        
        return self.write_point(point)
    
    async def flush(self) -> bool:
        """
        Flush batch buffer to InfluxDB
        
        Returns:
            True if flush successful
        """
        if not self.write_api:
            logger.warning("Write API not initialized")
            return False
        
        # Get points from buffer
        with self.buffer_lock:
            if len(self.batch_buffer) == 0:
                return True
            
            points = list(self.batch_buffer)
            self.batch_buffer.clear()
        
        try:
            # Convert to InfluxDB points
            influx_points = [p.to_point() for p in points]
            
            # Write to InfluxDB
            self.write_api.write(
                bucket=self.settings.influxdb_bucket,
                record=influx_points
            )
            
            self.points_written += len(points)
            self.last_write_time = time.time()
            
            logger.debug(f"Flushed {len(points)} points to InfluxDB")
            return True
        
        except InfluxDBError as e:
            logger.error(f"InfluxDB write error: {e}")
            self.write_errors += 1
            
            # Re-add points to buffer for retry
            with self.buffer_lock:
                self.batch_buffer.extend(points)
            
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error during flush: {e}")
            self.write_errors += 1
            return False
    
    def _start_flush_thread(self) -> None:
        """Start background flush thread"""
        if self.flush_thread_running:
            return
        
        self.flush_thread_running = True
        self.flush_thread = Thread(target=self._flush_loop, daemon=True)
        self.flush_thread.start()
        
        logger.info("Flush thread started")
    
    def _stop_flush_thread(self) -> None:
        """Stop flush thread"""
        if not self.flush_thread_running:
            return
        
        self.flush_thread_running = False
        
        if self.flush_thread:
            self.flush_thread.join(timeout=5.0)
        
        logger.info("Flush thread stopped")
    
    def _flush_loop(self) -> None:
        """Background flush loop"""
        logger.info("Flush loop started")
        
        while self.flush_thread_running:
            try:
                # Create event loop for async flush
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.flush())
                loop.close()
                
                # Sleep until next flush
                time.sleep(self.flush_interval)
            
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")
                time.sleep(1.0)
        
        logger.info("Flush loop stopped")
    
    @property
    def buffer_size(self) -> int:
        """Get current buffer size"""
        with self.buffer_lock:
            return len(self.batch_buffer)
    
    def get_stats(self) -> Dict:
        """Get storage statistics"""
        return {
            'connected': self.client is not None,
            'points_written': self.points_written,
            'points_dropped': self.points_dropped,
            'write_errors': self.write_errors,
            'buffer_size': self.buffer_size,
            'buffer_capacity': self.batch_buffer.maxlen,
            'last_write_time': self.last_write_time
        }
    
    async def health_check(self) -> Dict:
        """Perform health check"""
        if not self.client:
            return {
                'status': 'unhealthy',
                'message': 'Not connected'
            }
        
        try:
            health = self.client.health()
            
            return {
                'status': 'healthy' if health.status == 'pass' else 'degraded',
                'influxdb_status': health.status,
                'buffer_utilization': self.buffer_size / self.batch_buffer.maxlen
            }
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': str(e)
            }
