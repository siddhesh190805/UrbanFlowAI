"""
Test script for video stream handler
Run this to verify your camera setup
"""

import sys
from pathlib import Path
import time
import cv2
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import ConfigManager
from src.video_processing.stream_handler import StreamManager, Frame

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def frame_callback(frame: Frame):
    """Callback function to process each frame"""
    logger.info(
        f"Received frame from {frame.camera_id} ({frame.direction}): "
        f"Frame #{frame.frame_number}, Size: {frame.width}x{frame.height}"
    )


def main():
    """Main test function"""
    logger.info("Starting video stream test...")
    
    # Load configuration
    config_manager = ConfigManager('config/intersection_config.yaml')
    config = config_manager.load_config()
    
    logger.info(f"Loaded configuration for intersection: {config.intersection.name}")
    logger.info(f"Number of cameras: {len(config.intersection.cameras)}")
    
    # Create stream manager
    manager = StreamManager()
    
    # Add all cameras
    for camera_config in config.intersection.cameras:
        logger.info(f"Adding camera: {camera_config.id} ({camera_config.direction})")
        manager.add_stream(camera_config, frame_callback)
    
    # Start all streams
    logger.info("Starting all video streams...")
    manager.start_all()
    
    try:
        # Run for 30 seconds
        logger.info("Streaming for 30 seconds... Press Ctrl+C to stop")
        
        for i in range(30):
            time.sleep(1)
            
            # Print statistics every 5 seconds
            if (i + 1) % 5 == 0:
                logger.info("\n=== Stream Statistics ===")
                stats = manager.get_all_statistics()
                for stat in stats:
                    logger.info(
                        f"{stat['camera_id']} ({stat['direction']}): "
                        f"Frames: {stat['frame_count']}, "
                        f"Dropped: {stat['dropped_frames']}, "
                        f"Buffer: {stat['buffer_size']}/{stat['buffer_capacity']}"
                    )
        
        # Display latest frames
        logger.info("\nDisplaying latest frames (press 'q' to quit)...")
        while True:
            frames = manager.get_latest_frames()
            
            if frames:
                # Create a grid to display all camera feeds
                display_frames = []
                for camera_id in sorted(frames.keys()):
                    frame = frames[camera_id]
                    img = frame.data.copy()
                    
                    # Add text overlay
                    cv2.putText(
                        img, 
                        f"{camera_id} - {frame.direction}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), 
                        2
                    )
                    cv2.putText(
                        img, 
                        f"Frame: {frame.frame_number}", 
                        (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 255, 0), 
                        1
                    )
                    
                    display_frames.append(img)
                
                # Create 2x2 grid
                if len(display_frames) >= 4:
                    top_row = cv2.hconcat([display_frames[0], display_frames[1]])
                    bottom_row = cv2.hconcat([display_frames[2], display_frames[3]])
                    grid = cv2.vconcat([top_row, bottom_row])
                    
                    # Resize for display
                    grid = cv2.resize(grid, (1280, 720))
                    
                    cv2.imshow('Traffic Cameras', grid)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.033)  # ~30 FPS display
    
    except KeyboardInterrupt:
        logger.info("\nStopping streams...")
    
    finally:
        # Stop all streams
        manager.stop_all()
        cv2.destroyAllWindows()
        logger.info("Test completed.")


if __name__ == "__main__":
    main()
