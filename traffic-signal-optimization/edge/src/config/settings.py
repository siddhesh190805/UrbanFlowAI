"""
Configuration management for edge device
Combines environment variables, YAML files, and validation
"""

import os
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class Direction(str, Enum):
    """Traffic direction enum for type safety"""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


class StreamType(str, Enum):
    """Video stream type"""
    RTSP = "rtsp"
    USB = "usb"
    FILE = "file"
    TEST = "test"


class CameraConfig(BaseModel):
    """Configuration for a single camera"""
    id: str = Field(..., description="Unique camera identifier")
    name: str = Field(..., description="Human-readable camera name")
    direction: Direction = Field(..., description="Traffic direction")
    stream_type: StreamType = Field(default=StreamType.RTSP)
    url: str = Field(..., description="Stream URL or device path")
    
    # Camera properties
    resolution: Tuple[int, int] = Field(default=(1920, 1080))
    fps: int = Field(default=30, ge=10, le=60)
    
    # Physical mounting
    fov: float = Field(default=90.0, description="Field of view in degrees")
    height: float = Field(default=5.5, description="Mounting height in meters")
    angle: float = Field(default=45.0, description="Mounting angle in degrees")
    
    # Connection settings
    enabled: bool = Field(default=True)
    reconnect_attempts: int = Field(default=10, ge=1)
    reconnect_delay: float = Field(default=5.0, ge=1.0)
    buffer_size: int = Field(default=30, ge=10, le=200)
    
    class Config:
        use_enum_values = True


class DetectionZone(BaseModel):
    """Polygonal detection zone for direction-based counting (Indian traffic)"""
    direction: Direction
    polygon_points: List[List[int]] = Field(
        ..., 
        description="List of [x, y] coordinates defining the polygon"
    )
    pcu_equivalent: float = Field(
        default=2.0, 
        description="PCU (Passenger Car Unit) equivalent for this zone"
    )


class SignalPhaseConfig(BaseModel):
    """Configuration for signal phases"""
    id: int
    name: str
    directions: List[Direction]  # Directions that get green light
    movements: List[str]  # e.g., ['THROUGH', 'LEFT', 'RIGHT']
    
    # Timing constraints (Indian traffic adapted)
    min_green: int = Field(default=10, ge=7, description="Minimum green time (seconds)")
    max_green: int = Field(default=120, le=180, description="Maximum green time (seconds)")
    yellow_time: int = Field(default=4, ge=3, le=5, description="Yellow clearance time")
    all_red_time: int = Field(default=2, ge=1, le=4, description="All-red clearance time")
    
    # Pedestrian priority (important for India)
    pedestrian_phase: bool = Field(default=False)
    pedestrian_min_time: int = Field(default=15, description="Minimum pedestrian crossing time")


class IntersectionConfig(BaseModel):
    """Configuration for the intersection"""
    intersection_id: str
    name: str
    city: str
    region: str
    location: Dict[str, float] = Field(
        ..., 
        description="Location coordinates: {lat, lon}"
    )
    
    cameras: List[CameraConfig]
    detection_zones: List[DetectionZone]
    signal_phases: List[SignalPhaseConfig]
    
    # Traffic characteristics (Indian specific)
    mixed_traffic: bool = Field(default=True, description="Handle mixed vehicle types")
    vehicle_types: List[str] = Field(
        default=["car", "bus", "truck", "motorcycle", "auto", "bicycle", "pedestrian"]
    )


class AIModelConfig(BaseModel):
    """AI model configuration"""
    detector_model_path: str = Field(default="models/yolov8_indian_traffic.engine")
    classifier_model_path: str = Field(default="models/vehicle_classifier_india.engine")
    rl_model_path: str = Field(default="models/rl_agent_india.pth")
    
    # Inference settings
    device: str = Field(default="cuda", description="'cuda' or 'cpu'")
    confidence_threshold: float = Field(default=0.5, ge=0.1, le=1.0)
    nms_threshold: float = Field(default=0.4, ge=0.1, le=1.0)
    batch_size: int = Field(default=1, ge=1, le=8)
    use_tensorrt: bool = Field(default=True)
    
    # Tracking settings
    tracking_max_age: int = Field(default=30, description="Max frames without detection")
    tracking_min_hits: int = Field(default=3, description="Min detections to start track")


class SignalControllerConfig(BaseModel):
    """Signal controller configuration"""
    controller_ip: str = Field(default="192.168.1.100")
    controller_protocol: str = Field(default="NTCIP", description="NTCIP, MODBUS, or RELAY")
    controller_port: int = Field(default=161)
    
    # Safety settings
    fail_safe_mode: bool = Field(default=True, description="Enable fail-safe fallback")
    emergency_override: bool = Field(default=True, description="Allow emergency preemption")


class CommunicationConfig(BaseModel):
    """MQTT and communication settings"""
    mqtt_broker: str = Field(default="localhost")
    mqtt_port: int = Field(default=1883)
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    mqtt_client_id: Optional[str] = None
    mqtt_qos: int = Field(default=1, ge=0, le=2)
    use_tls: bool = Field(default=False)
    tls_ca_cert: Optional[str] = None
    
    # Topic structure will be formatted at runtime
    topic_metrics: str = "traffic/{city}/{region}/{intersection}/metrics"
    topic_events: str = "traffic/{city}/{region}/{intersection}/events"
    topic_commands: str = "traffic/{city}/{region}/{intersection}/commands"
    topic_status: str = "traffic/{city}/{region}/{intersection}/status"


class StorageConfig(BaseModel):
    """Storage configuration"""
    # InfluxDB
    influxdb_url: str = Field(default="http://localhost:8086")
    influxdb_token: Optional[str] = Field(default="my-super-secret-auth-token")
    influxdb_org: str = Field(default="traffic-org")
    influxdb_bucket: str = Field(default="traffic-metrics")
    
    # Redis
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: Optional[str] = None
    
    # Video buffer
    video_buffer_path: str = Field(default="/var/traffic/video_buffer")
    video_retention_hours: int = Field(default=2, ge=1)


class EdgeSettings(BaseSettings):
    """
    Main configuration for edge device
    Combines environment variables with YAML configuration
    """
    
    # Device identity
    device_id: str = Field(default="edge_001", env="DEVICE_ID")
    
    # Components
    intersection: Optional[IntersectionConfig] = None
    ai_models: AIModelConfig = Field(default_factory=AIModelConfig)
    signal_controller: SignalControllerConfig = Field(default_factory=SignalControllerConfig)
    communication: CommunicationConfig = Field(default_factory=CommunicationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    # Processing settings
    target_fps: int = Field(default=30, env="TARGET_FPS")
    rl_update_frequency: int = Field(default=5, description="Update decision every N seconds")
    
    # Monitoring
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default="logs/edge.log")
    log_rotation: str = Field(default="500 MB")
    log_retention: str = Field(default="30 days")
    
    enable_monitoring: bool = Field(default=True)
    prometheus_port: int = Field(default=9091, env="PROMETHEUS_PORT")
    metrics_export_interval: int = Field(default=10)
    
    # Development & Testing
    simulation_mode: bool = Field(default=False, env="SIMULATION_MODE")
    debug_visualization: bool = Field(default=False)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = ""
    
    def load_from_yaml(self, config_path: str):
        """Load configuration from YAML file"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Parse intersection config
        if 'intersection' in config_data:
            self.intersection = IntersectionConfig(**config_data['intersection'])
        
        # Update other configs
        if 'ai_models' in config_data:
            self.ai_models = AIModelConfig(**config_data['ai_models'])
        
        if 'signal_controller' in config_data:
            self.signal_controller = SignalControllerConfig(**config_data['signal_controller'])
        
        if 'communication' in config_data:
            self.communication = CommunicationConfig(**config_data['communication'])
        
        if 'storage' in config_data:
            self.storage = StorageConfig(**config_data['storage'])
        
        # Update top-level settings
        for key in ['device_id', 'target_fps', 'log_level', 'simulation_mode']:
            if key in config_data:
                setattr(self, key, config_data[key])
    
    def get_mqtt_topics(self) -> Dict[str, str]:
        """Get formatted MQTT topics"""
        if not self.intersection:
            raise ValueError("Intersection configuration not loaded")
        
        return {
            'metrics': self.communication.topic_metrics.format(
                city=self.intersection.city,
                region=self.intersection.region,
                intersection=self.intersection.intersection_id
            ),
            'events': self.communication.topic_events.format(
                city=self.intersection.city,
                region=self.intersection.region,
                intersection=self.intersection.intersection_id
            ),
            'commands': self.communication.topic_commands.format(
                city=self.intersection.city,
                region=self.intersection.region,
                intersection=self.intersection.intersection_id
            ),
            'status': self.communication.topic_status.format(
                city=self.intersection.city,
                region=self.intersection.region,
                intersection=self.intersection.intersection_id
            )
        }


# Global settings instance
_settings: Optional[EdgeSettings] = None

def get_settings(config_path: Optional[str] = None) -> EdgeSettings:
    """Get or create settings instance (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = EdgeSettings()
        if config_path:
            _settings.load_from_yaml(config_path)
    return _settings
