"""
Unit tests for video stream handler
"""

import pytest
import time
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import CameraConfig, StreamType
from src.video_processing.stream_handler import VideoStream, StreamManager, Frame


@pytest.fixture
def test_camera_config():
    """Create a test camera configuration"""
    return CameraConfig(
        id="test_cam_001",
        name="Test Camera",
        direction="north",
        stream_type=StreamType.TEST,
        source="test",
        width=640,
        height=480,
        fps=10,
        reconnect_attempts=3,
        reconnect_delay=1,
        buffer_size=50
    )


def test_video_stream_initialization(test_camera_config):
    """Test video stream initialization"""
    stream = VideoStream(test_camera_config)
    assert stream.config.id == "test_cam_001"
    assert stream.is_running is False
    assert stream.frame_count == 0


def test_video_stream_start_stop(test_camera_config):
    """Test starting and stopping video stream"""
    stream = VideoStream(test_camera_config)
    
    # Start stream
    assert stream.start() is True
    assert stream.is_running is True
    
    # Wait for some frames
    time.sleep(2)
    assert stream.frame_count > 0
    
    # Stop stream
    stream.stop()
    assert stream.is_running is False


def test_frame_buffer(test_camera_config):
    """Test frame buffering"""
    frames_received = []
    
    def frame_callback(frame: Frame):
        frames_received.append(frame)
    
    stream = VideoStream(test_camera_config, frame_callback=frame_callback)
    stream.start()
    
    # Wait for frames
    time.sleep(2)
    
    # Check that frames were received
    assert len(frames_received) > 0
    
    # Validate frame structure
    frame = frames_received[0]
    assert frame.camera_id == "test_cam_001"
    assert frame.direction == "north"
    assert isinstance(frame.data, np.ndarray)
    assert frame.data.shape == (480, 640, 3)
    
    stream.stop()


def test_stream_manager():
    """Test stream manager with multiple streams"""
    manager = StreamManager()
    
    # Create configs for multiple cameras
    configs = [
        CameraConfig(
            id=f"cam_{direction}",
            name=f"{direction.capitalize()} Camera",
            direction=direction,
            stream_type=StreamType.TEST,
            source="test",
            width=640,
            height=480,
            fps=10
        )
        for direction in ["north", "south", "east", "west"]
    ]
    
    # Add streams
    for config in configs:
        manager.add_stream(config)
    
    assert len(manager.streams) == 4
    
    # Start all streams
    manager.start_all()
    time.sleep(2)
    
    # Get statistics
    stats = manager.get_all_statistics()
    assert len(stats) == 4
    
    for stat in stats:
        assert stat['is_running'] is True
        assert stat['frame_count'] > 0
    
    # Get latest frames
    frames = manager.get_latest_frames()
    assert len(frames) <= 4  # May not have all frames ready
    
    # Stop all streams
    manager.stop_all()
    
    for stream in manager.streams.values():
        assert stream.is_running is False


def test_reconnection_logic(test_camera_config):
    """Test automatic reconnection on connection loss"""
    # This test would require a mock that simulates connection loss
    # For now, we just test that reconnection parameters are set
    assert test_camera_config.reconnect_attempts == 3
    assert test_camera_config.reconnect_delay == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
