"""
Main Edge Application for Traffic Signal Optimization
Orchestrates all components: video, AI, analytics, decision, control
Optimized for Indian traffic conditions
"""

import asyncio
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from config.settings import EdgeSettings, Direction, VehicleType
from utils.logger import get_logger, setup_logging

# Import all components
from video_processing.stream_handler import StreamManager, FrameData
from ai.detector import VehicleDetector
from ai.tracker import VehicleTracker
from analytics.metrics_aggregator import MetricsAggregator, DirectionMetrics
from decision.rl_agent import RLAgent
from signal_control.controller_interface import SignalController
from communication.mqtt_client import MQTTClient
from storage.influxdb_client import InfluxDBStorage
from storage.redis_cache import RedisCache

logger = get_logger(__name__)


class TrafficOptimizationSystem:
    """
    Main system orchestrating all components for traffic signal optimization
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize traffic optimization system
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("=" * 70)
        logger.info("🚦 TRAFFIC SIGNAL OPTIMIZATION SYSTEM - EDGE DEVICE")
        logger.info("=" * 70)
        
        # Load configuration
        self.settings = EdgeSettings()
        if config_path:
            self.settings.load_from_yaml(str(config_path))
        
        # Setup logging
        setup_logging(
            level=self.settings.log_level,
            log_file=self.settings.log_file,
            rotation=self.settings.log_rotation,
            retention=self.settings.log_retention
        )
        
        logger.info(f"Device ID: {self.settings.device_id}")
        logger.info(f"Intersection: {self.settings.intersection_name}")
        logger.info(f"City: {self.settings.city}, Region: {self.settings.region}")
        logger.info(f"Traffic Mode: {self.settings.traffic_mode.value}")
        logger.info("=" * 70)
        
        # Initialize components
        self.stream_manager: Optional[StreamManager] = None
        self.detector: Optional[VehicleDetector] = None
        self.trackers: Dict[Direction, VehicleTracker] = {}
        self.metrics_aggregator: Optional[MetricsAggregator] = None
        self.rl_agent: Optional[RLAgent] = None
        self.signal_controller: Optional[SignalController] = None
        self.mqtt_client: Optional[MQTTClient] = None
        self.influxdb: Optional[InfluxDBStorage] = None
        self.redis_cache: Optional[RedisCache] = None
        
        # System state
        self.running = False
        self.initialization_complete = False
        self.last_decision_time = time.time()
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.frames_processed = 0
        self.decisions_made = 0
        self.errors = 0
    
    async def initialize(self) -> bool:
        """
        Initialize all system components
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Starting system initialization...")
            
            # 1. Initialize Redis Cache
            logger.info("[1/9] Initializing Redis cache...")
            self.redis_cache = RedisCache(self.settings)
            if not await self.redis_cache.connect():
                logger.warning("Redis connection failed, continuing without cache")
            else:
                logger.success("✓ Redis cache ready")
            
            # 2. Initialize InfluxDB
            logger.info("[2/9] Initializing InfluxDB storage...")
            self.influxdb = InfluxDBStorage(self.settings)
            if not await self.influxdb.connect():
                logger.warning("InfluxDB connection failed, continuing without storage")
            else:
                logger.success("✓ InfluxDB storage ready")
            
            # 3. Initialize MQTT Client
            logger.info("[3/9] Initializing MQTT communication...")
            self.mqtt_client = MQTTClient(self.settings)
            if not await self.mqtt_client.connect():
                logger.warning("MQTT connection failed, continuing in offline mode")
            else:
                # Subscribe to commands
                self.mqtt_client.subscribe_to_commands(self._handle_command)
                logger.success("✓ MQTT communication ready")
            
            # 4. Initialize Video Stream Manager
            logger.info("[4/9] Initializing video streams...")
            self.stream_manager = StreamManager()
            
            for camera_config in self.settings.camera_configs:
                self.stream_manager.add_stream(camera_config)
            
            logger.success(f"✓ {len(self.settings.camera_configs)} video streams configured")
            
            # 5. Initialize Vehicle Detector
            logger.info("[5/9] Loading vehicle detection model...")
            self.detector = VehicleDetector(self.settings)
            
            if not self.detector.load_model():
                logger.error("Failed to load detection model")
                return False
            
            logger.success("✓ Vehicle detector ready")
            
            # 6. Initialize Trackers (one per direction)
            logger.info("[6/9] Initializing vehicle trackers...")
            for direction in Direction:
                self.trackers[direction] = VehicleTracker(
                    direction=direction,
                    max_age=self.settings.tracking_max_age,
                    min_hits=self.settings.tracking_min_hits
                )
            
            logger.success(f"✓ {len(self.trackers)} vehicle trackers ready")
            
            # 7. Initialize Metrics Aggregator
            logger.info("[7/9] Initializing metrics aggregator...")
            self.metrics_aggregator = MetricsAggregator(
                self.settings,
                self.redis_cache
            )
            logger.success("✓ Metrics aggregator ready")
            
            # 8. Initialize RL Agent
            logger.info("[8/9] Loading RL agent...")
            self.rl_agent = RLAgent(self.settings)
            
            if not await self.rl_agent.load_model():
                logger.warning("RL model not loaded, using rule-based control")
            else:
                logger.success("✓ RL agent ready")
            
            # 9. Initialize Signal Controller
            logger.info("[9/9] Initializing signal controller...")
            self.signal_controller = SignalController(self.settings)
            
            if not await self.signal_controller.connect():
                if not self.settings.simulation_mode:
                    logger.error("Failed to connect to signal controller")
                    return False
                else:
                    logger.warning("Signal controller connection failed (simulation mode)")
            
            logger.success("✓ Signal controller ready")
            
            # Mark initialization complete
            self.initialization_complete = True
            logger.success("=" * 70)
            logger.success("✅ SYSTEM INITIALIZATION COMPLETE")
            logger.success("=" * 70)
            
            return True
        
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            return False
    
    async def start(self) -> None:
        """Start the traffic optimization system"""
        if not self.initialization_complete:
            logger.error("System not initialized. Call initialize() first.")
            return
        
        try:
            logger.info("Starting traffic optimization system...")
            self.start_time = datetime.now()
            self.running = True
            
            # Start video streams
            self.stream_manager.start_all()
            
            # Publish online status
            if self.mqtt_client and self.mqtt_client.is_connected:
                await self.mqtt_client.publish_status({
                    'state': 'online',
                    'mode': 'rl' if self.rl_agent.model_loaded else 'rule_based',
                    'capabilities': self._get_capabilities()
                })
            
            # Start main processing loop
            await self._main_loop()
        
        except Exception as e:
            logger.error(f"Error in start: {e}", exc_info=True)
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the traffic optimization system gracefully"""
        logger.info("Stopping traffic optimization system...")
        self.running = False
        
        try:
            # Stop video streams
            if self.stream_manager:
                self.stream_manager.stop_all()
            
            # Disconnect signal controller
            if self.signal_controller:
                await self.signal_controller.disconnect()
            
            # Publish offline status
            if self.mqtt_client and self.mqtt_client.is_connected:
                await self.mqtt_client.publish_status({
                    'state': 'offline',
                    'reason': 'graceful_shutdown'
                })
            
            # Disconnect MQTT
            if self.mqtt_client:
                await self.mqtt_client.disconnect()
            
            # Flush and disconnect storage
            if self.influxdb:
                await self.influxdb.disconnect()
            
            if self.redis_cache:
                await self.redis_cache.disconnect()
            
            logger.success("System stopped successfully")
        
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
    
    async def _main_loop(self) -> None:
        """Main processing loop"""
        logger.info("Main processing loop started")
        
        loop_count = 0
        last_stats_time = time.time()
        
        while self.running:
            try:
                loop_start = time.time()
                
                # Get latest frames from all cameras
                frames = self.stream_manager.get_latest_frames()
                
                if not frames:
                    await asyncio.sleep(0.01)
                    continue
                
                # Process each frame
                all_metrics: Dict[Direction, DirectionMetrics] = {}
                
                for camera_id, frame in frames.items():
                    direction = frame.direction
                    
                    # Detect vehicles
                    detection_result = self.detector.detect(
                        frame.data,
                        camera_id
                    )
                    
                    # Track vehicles
                    tracking_result = self.trackers[direction].update(
                        detection_result.detections,
                        frame.frame_number
                    )
                    
                    # Calculate metrics
                    metrics = self.metrics_aggregator.calculate_metrics(
                        direction=direction,
                        tracking_result=tracking_result,
                        detection_result=detection_result,
                        timestamp=frame.timestamp
                    )
                    
                    all_metrics[direction] = metrics
                    
                    # Cache in Redis
                    if self.redis_cache:
                        await self.redis_cache.set_traffic_state(
                            direction.value,
                            metrics.to_dict()
                        )
                    
                    # Store in InfluxDB
                    if self.influxdb:
                        await self.influxdb.write_traffic_metrics(
                            direction.value,
                            metrics.to_dict()
                        )
                    
                    self.frames_processed += 1
                
                # Make signal control decision
                time_since_decision = time.time() - self.last_decision_time
                
                if time_since_decision >= self.settings.decision_interval:
                    await self._make_signal_decision(all_metrics)
                    self.last_decision_time = time.time()
                
                # Publish metrics via MQTT
                if self.mqtt_client and loop_count % 10 == 0:  # Every 10 iterations
                    await self._publish_metrics(all_metrics)
                
                # Log statistics periodically
                if time.time() - last_stats_time >= 30:  # Every 30 seconds
                    self._log_statistics(all_metrics)
                    last_stats_time = time.time()
                
                loop_count += 1
                
                # Adaptive sleep to maintain target FPS
                loop_time = time.time() - loop_start
                target_interval = 1.0 / self.settings.target_fps
                sleep_time = max(0, target_interval - loop_time)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self.errors += 1
                await asyncio.sleep(1.0)
    
    async def _make_signal_decision(
        self,
        all_metrics: Dict[Direction, DirectionMetrics]
    ) -> None:
        """
        Make signal control decision using RL agent or rule-based fallback
        
        Args:
            all_metrics: Metrics for all directions
        """
        try:
            if self.rl_agent and self.rl_agent.model_loaded:
                # Use RL agent
                action = await self._rl_decision(all_metrics)
            else:
                # Use rule-based fallback
                action = await self._rule_based_decision(all_metrics)
            
            # Execute action
            success = await self.signal_controller.execute_action(action)
            
            if success:
                self.decisions_made += 1
                logger.info(
                    f"Decision #{self.decisions_made}: {action['type']} "
                    f"(Phase: {action.get('phase_id', 'N/A')})"
                )
        
        except Exception as e:
            logger.error(f"Error making signal decision: {e}", exc_info=True)
    
    async def _rl_decision(
        self,
        all_metrics: Dict[Direction, DirectionMetrics]
    ) -> Dict:
        """Make decision using RL agent"""
        # Build state vector
        state_vector = self.metrics_aggregator.build_state_vector(
            all_metrics,
            self.signal_controller.current_phase,
            self.signal_controller.time_in_phase
        )
        
        # Get action from RL agent
        action, confidence = await self.rl_agent.predict(state_vector)
        
        logger.debug(f"RL action: {action} (confidence: {confidence:.2f})")
        
        return action
    
    async def _rule_based_decision(
        self,
        all_metrics: Dict[Direction, DirectionMetrics]
    ) -> Dict:
        """Fallback rule-based decision"""
        # Simple rule: switch to direction with highest queue or wait time
        max_queue_direction = max(
            all_metrics.keys(),
            key=lambda d: all_metrics[d].queue_length
        )
        
        max_wait_direction = max(
            all_metrics.keys(),
            key=lambda d: all_metrics[d].avg_wait_time
        )
        
        # Check current phase timing
        current_state = await self.signal_controller.get_state()
        
        if current_state.time_in_phase < self.settings.min_green_time:
            return {'type': 'extend', 'duration': 5.0}
        
        if current_state.time_in_phase >= self.settings.max_green_time:
            # Must switch
            target_phase = self._find_phase_for_direction(max_queue_direction)
            return {'type': 'change_phase', 'phase_id': target_phase}
        
        # Check if should switch
        current_direction = self._get_phase_direction(current_state.current_phase)
        current_queue = all_metrics[current_direction].queue_length
        max_queue = all_metrics[max_queue_direction].queue_length
        
        if max_queue > current_queue * 1.5:  # 50% higher queue
            target_phase = self._find_phase_for_direction(max_queue_direction)
            return {'type': 'change_phase', 'phase_id': target_phase}
        
        return {'type': 'extend', 'duration': 5.0}
    
    def _find_phase_for_direction(self, direction: Direction) -> int:
        """Find phase ID that serves a direction"""
        for phase in self.settings.signal_phases:
            if direction in phase.directions:
                return phase.phase_id
        return 0
    
    def _get_phase_direction(self, phase_id: int) -> Direction:
        """Get primary direction for a phase"""
        for phase in self.settings.signal_phases:
            if phase.phase_id == phase_id:
                return phase.directions[0]  # Return first direction
        return Direction.NORTH
    
    async def _publish_metrics(
        self,
        all_metrics: Dict[Direction, DirectionMetrics]
    ) -> None:
        """Publish metrics via MQTT"""
        if not self.mqtt_client or not self.mqtt_client.is_connected:
            return
        
        metrics_payload = {
            'intersection_id': self.settings.intersection_id,
            'timestamp': time.time(),
            'directions': {
                d.value: m.to_dict()
                for d, m in all_metrics.items()
            }
        }
        
        await self.mqtt_client.publish_metrics(metrics_payload)
    
    def _log_statistics(self, all_metrics: Dict[Direction, DirectionMetrics]) -> None:
        """Log system statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        logger.info("=" * 70)
        logger.info(f"📊 SYSTEM STATISTICS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        logger.info(f"Uptime: {uptime:.0f}s | Frames: {self.frames_processed} | Decisions: {self.decisions_made}")
        logger.info(f"Errors: {self.errors} | FPS: {self.frames_processed / uptime:.1f}")
        
        # Traffic metrics
        total_vehicles = sum(m.vehicle_count for m in all_metrics.values())
        total_pcu = sum(m.total_pcu for m in all_metrics.values())
        
        logger.info(f"Total Vehicles: {total_vehicles} | Total PCU: {total_pcu:.1f}")
        
        for direction, metrics in all_metrics.items():
            logger.info(
                f"  {direction.value.upper()}: "
                f"Vehicles={metrics.vehicle_count}, "
                f"Queue={metrics.queue_length}, "
                f"Wait={metrics.avg_wait_time:.1f}s, "
                f"Congestion={metrics.congestion_level}"
            )
        
        # Component stats
        if self.detector:
            detector_stats = self.detector.get_stats()
            logger.info(f"Detector: {detector_stats['avg_inference_time_ms']:.1f}ms avg")
        
        if self.mqtt_client:
            mqtt_stats = self.mqtt_client.get_stats()
            logger.info(
                f"MQTT: Sent={mqtt_stats['messages_sent']}, "
                f"Queue={mqtt_stats['current_queue_size']}"
            )
        
        logger.info("=" * 70)
    
    def _handle_command(self, command: Dict) -> None:
        """Handle command received via MQTT"""
        try:
            command_type = command.get('type')
            
            logger.info(f"Received command: {command_type}")
            
            if command_type == 'emergency_override':
                asyncio.create_task(self.signal_controller.emergency_override())
            
            elif command_type == 'change_phase':
                phase_id = command.get('phase_id')
                asyncio.create_task(self.signal_controller.change_phase(phase_id))
            
            elif command_type == 'enable_simulation':
                self.settings.simulation_mode = True
            
            elif command_type == 'disable_simulation':
                self.settings.simulation_mode = False
            
            else:
                logger.warning(f"Unknown command type: {command_type}")
        
        except Exception as e:
            logger.error(f"Error handling command: {e}", exc_info=True)
    
    def _get_capabilities(self) -> List[str]:
        """Get system capabilities"""
        capabilities = ['detection', 'tracking', 'metrics']
        
        if self.rl_agent and self.rl_agent.model_loaded:
            capabilities.append('rl_control')
        
        capabilities.append('rule_based_control')
        
        if self.mqtt_client and self.mqtt_client.is_connected:
            capabilities.append('mqtt_communication')
        
        return capabilities


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Traffic Signal Optimization System - Edge Device"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/intersection_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--simulation',
        action='store_true',
        help='Run in simulation mode'
    )
    
    args = parser.parse_args()
    
    # Create system
    config_path = Path(args.config) if args.config else None
    system = TrafficOptimizationSystem(config_path)
    
    # Override simulation mode if specified
    if args.simulation:
        system.settings.simulation_mode = True
        logger.info("Running in SIMULATION MODE")
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        asyncio.create_task(system.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize and start
    try:
        if await system.initialize():
            await system.start()
        else:
            logger.error("Initialization failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        await system.stop()
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        await system.stop()
        sys.exit(1)


if __name__ == "__main__":
    asyncio
