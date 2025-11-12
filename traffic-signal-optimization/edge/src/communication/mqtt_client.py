"""
MQTT Communication Module
Handles pub/sub communication with regional server
Implements TLS, automatic reconnection, last will, and message queuing
"""

import asyncio
import json
import ssl
import time
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, asdict
from collections import deque
import paho.mqtt.client as mqtt
from threading import Thread, Lock

from config.settings import EdgeSettings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MQTTMessage:
    """MQTT message container with age tracking"""
    topic: str
    payload: Dict[str, Any]
    qos: int = 1
    retain: bool = False
    timestamp: float = None
    max_age_seconds: float = 300.0  # 5 minutes default
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_json(self) -> str:
        """Convert payload to JSON string"""
        return json.dumps(self.payload)
    
    @property
    def age_seconds(self) -> float:
        """Get message age in seconds"""
        return time.time() - self.timestamp
    
    @property
    def is_stale(self) -> bool:
        """Check if message is too old"""
        return self.age_seconds > self.max_age_seconds


class MQTTClient:
    """
    Production-grade MQTT client with TLS, reconnection, and message queuing
    """
    
    def __init__(self, settings: EdgeSettings):
        """
        Initialize MQTT client
        
        Args:
            settings: Edge device settings
        """
        self.settings = settings
        
        # MQTT client
        self.client: Optional[mqtt.Client] = None
        self.connected = False
        self.connecting = False
        
        # Message queue for offline buffering
        self.message_queue: deque[MQTTMessage] = deque(maxlen=1000)
        self.queue_lock = Lock()
        
        # Callbacks
        self.message_callbacks: Dict[str, Callable] = {}
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_queued = 0
        self.messages_dropped_stale = 0
        self.connection_attempts = 0
        self.last_connection_time: Optional[float] = None
        
        # Background thread for processing queue
        self.queue_thread: Optional[Thread] = None
        self.queue_thread_running = False
        
        # Retry settings with exponential backoff
        self.retry_delay = 0.1  # Start with 100ms
        self.max_retry_delay = 5.0  # Max 5 seconds
        
        logger.info(
            f"MQTT Client initialized for {settings.intersection_id} "
            f"(broker: {settings.mqtt_broker}:{settings.mqtt_port})"
        )
    
    async def connect(self) -> bool:
        """
        Connect to MQTT broker with TLS support
        
        Returns:
            True if connection successful
        """
        if self.connected:
            logger.warning("Already connected to MQTT broker")
            return True
        
        if self.connecting:
            logger.warning("Connection attempt already in progress")
            return False
        
        try:
            self.connecting = True
            self.connection_attempts += 1
            
            logger.info(
                f"Connecting to MQTT broker {self.settings.mqtt_broker}:"
                f"{self.settings.mqtt_port} (attempt {self.connection_attempts})"
            )
            
            # Create client
            client_id = f"{self.settings.intersection_id}_{int(time.time())}"
            self.client = mqtt.Client(client_id=client_id)
            
            # Set credentials if provided
            if self.settings.mqtt_username and self.settings.mqtt_password:
                self.client.username_pw_set(
                    self.settings.mqtt_username,
                    self.settings.mqtt_password
                )
                logger.info("MQTT authentication configured")
            
            # Configure TLS if enabled
            if self.settings.mqtt_use_tls:
                self._configure_tls()
            
            # Set last will and testament
            self._configure_last_will()
            
            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            
            # Connect
            self.client.connect(
                self.settings.mqtt_broker,
                self.settings.mqtt_port,
                keepalive=self.settings.mqtt_keepalive
            )
            
            # Start network loop in background
            self.client.loop_start()
            
            # Wait for connection (with timeout)
            timeout = 10.0
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)
            
            if self.connected:
                logger.success(f"Connected to MQTT broker")
                self.last_connection_time = time.time()
                
                # Publish online status
                await self._publish_online_status()
                
                # Start queue processing thread
                self._start_queue_processor()
                
                return True
            else:
                logger.error("Connection timeout")
                return False
            
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
        
        finally:
            self.connecting = False
    
    def _configure_tls(self):
        """Configure TLS/SSL for MQTT connection"""
        try:
            tls_context = ssl.create_default_context()
            
            # Load CA certificate if provided
            if self.settings.mqtt_ca_cert:
                tls_context.load_verify_locations(self.settings.mqtt_ca_cert)
                logger.info(f"Loaded CA certificate: {self.settings.mqtt_ca_cert}")
            
            # Load client certificate and key if provided
            if self.settings.mqtt_client_cert and self.settings.mqtt_client_key:
                tls_context.load_cert_chain(
                    certfile=self.settings.mqtt_client_cert,
                    keyfile=self.settings.mqtt_client_key
                )
                logger.info("Loaded client certificate and key")
            
            # Set verification mode
            tls_context.check_hostname = True
            tls_context.verify_mode = ssl.CERT_REQUIRED
            
            # Apply TLS settings to client
            self.client.tls_set_context(tls_context)
            self.client.tls_insecure_set(False)
            
            logger.success("TLS/SSL configured for MQTT connection")
        
        except Exception as e:
            logger.error(f"Failed to configure TLS: {e}")
            raise
    
    def _configure_last_will(self):
        """Configure last will and testament for unexpected disconnections"""
        status_topic = self.settings.mqtt_topics['status']
        
        last_will = {
            'intersection_id': self.settings.intersection_id,
            'timestamp': time.time(),
            'status': 'offline',
            'reason': 'unexpected_disconnect'
        }
        
        self.client.will_set(
            status_topic,
            json.dumps(last_will),
            qos=1,
            retain=True
        )
        
        logger.info(f"Last will configured on topic: {status_topic}")
    
    async def disconnect(self) -> None:
        """Disconnect from MQTT broker gracefully"""
        if not self.connected:
            return
        
        logger.info("Disconnecting from MQTT broker")
        
        # Publish offline status
        await self._publish_offline_status()
        
        # Stop queue processor
        self._stop_queue_processor()
        
        # Disconnect
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()
        
        self.connected = False
        logger.info("Disconnected from MQTT broker")
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to broker"""
        if rc == 0:
            self.connected = True
            logger.success("MQTT connection established")
            
            # Subscribe to command topic
            command_topic = self.settings.mqtt_topics['commands']
            self.client.subscribe(command_topic, qos=self.settings.mqtt_qos)
            logger.info(f"Subscribed to {command_topic}")
            
            # Reset retry delay on successful connection
            self.retry_delay = 0.1
            
        else:
            error_messages = {
                1: "Connection refused - incorrect protocol version",
                2: "Connection refused - invalid client identifier",
                3: "Connection refused - server unavailable",
                4: "Connection refused - bad username or password",
                5: "Connection refused - not authorized"
            }
            error_msg = error_messages.get(rc, f"Unknown error code: {rc}")
            logger.error(f"MQTT connection failed: {error_msg}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from broker"""
        self.connected = False
        
        if rc == 0:
            logger.info("MQTT disconnected cleanly")
        else:
            logger.warning(f"MQTT disconnected unexpectedly (code: {rc})")
            # Automatic reconnection handled by paho-mqtt
    
    def _on_message(self, client, userdata, msg):
        """Callback when message received"""
        try:
            self.messages_received += 1
            
            # Decode payload
            payload = json.loads(msg.payload.decode('utf-8'))
            
            logger.debug(f"Received message on {msg.topic}: {payload}")
            
            # Call registered callback if exists
            if msg.topic in self.message_callbacks:
                callback = self.message_callbacks[msg.topic]
                # Run callback in thread pool to avoid blocking
                asyncio.get_event_loop().run_in_executor(
                    None, callback, payload
                )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message payload: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        qos: Optional[int] = None,
        retain: bool = False,
        max_age_seconds: float = 300.0
    ) -> bool:
        """
        Publish message to topic with age limit
        
        Args:
            topic: MQTT topic
            payload: Message payload (will be JSON encoded)
            qos: Quality of Service (0, 1, or 2)
            retain: Retain message flag
            max_age_seconds: Maximum message age before dropping
        
        Returns:
            True if published successfully
        """
        qos = qos if qos is not None else self.settings.mqtt_qos
        
        message = MQTTMessage(
            topic=topic,
            payload=payload,
            qos=qos,
            retain=retain,
            max_age_seconds=max_age_seconds
        )
        
        # Check if message is already stale
        if message.is_stale:
            logger.warning(f"Message for {topic} is stale, dropping")
            self.messages_dropped_stale += 1
            return False
        
        if not self.connected:
            # Queue message for later
            with self.queue_lock:
                self.message_queue.append(message)
                self.messages_queued += 1
            
            logger.warning(
                f"Not connected, queued message for {topic} "
                f"(queue size: {len(self.message_queue)})"
            )
            return False
        
        try:
            # Publish message
            result = self.client.publish(
                topic,
                message.to_json(),
                qos=qos,
                retain=retain
            )
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.messages_sent += 1
                logger.debug(f"Published to {topic}")
                return True
            else:
                logger.error(f"Failed to publish to {topic}: {result.rc}")
                # Queue for retry
                with self.queue_lock:
                    self.message_queue.append(message)
                return False
            
        except Exception as e:
            logger.error(f"Error publishing to {topic}: {e}")
            # Queue for retry
            with self.queue_lock:
                self.message_queue.append(message)
            return False
    
    async def publish_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Publish traffic metrics
        
        Args:
            metrics: Metrics dictionary
        
        Returns:
            True if published successfully
        """
        topic = self.settings.mqtt_topics['metrics']
        
        # Add metadata
        payload = {
            'intersection_id': self.settings.intersection_id,
            'timestamp': time.time(),
            'metrics': metrics
        }
        
        # Metrics should be published quickly
        return await self.publish(topic, payload, max_age_seconds=30.0)
    
    async def publish_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """
        Publish traffic event
        
        Args:
            event_type: Type of event
            event_data: Event data
        
        Returns:
            True if published successfully
        """
        topic = self.settings.mqtt_topics['events']
        
        payload = {
            'intersection_id': self.settings.intersection_id,
            'timestamp': time.time(),
            'event_type': event_type,
            'data': event_data
        }
        
        return await self.publish(topic, payload, max_age_seconds=60.0)
    
    async def publish_status(self, status: Dict[str, Any]) -> bool:
        """
        Publish system status
        
        Args:
            status: Status dictionary
        
        Returns:
            True if published successfully
        """
        topic = self.settings.mqtt_topics['status']
        
        payload = {
            'intersection_id': self.settings.intersection_id,
            'timestamp': time.time(),
            'status': status
        }
        
        return await self.publish(topic, payload, retain=True, max_age_seconds=120.0)
    
    async def _publish_online_status(self):
        """Publish online status after connection"""
        await self.publish_status({
            'state': 'online',
            'version': self.settings.version if hasattr(self.settings, 'version') else '1.0.0',
            'capabilities': ['detection', 'tracking', 'rl_control']
        })
    
    async def _publish_offline_status(self):
        """Publish offline status before disconnection"""
        await self.publish_status({
            'state': 'offline',
            'reason': 'graceful_shutdown'
        })
    
    def subscribe_to_commands(self, callback: Callable[[Dict], None]) -> None:
        """
        Subscribe to command topic with callback
        
        Args:
            callback: Function to call when command received
        """
        topic = self.settings.mqtt_topics['commands']
        self.message_callbacks[topic] = callback
        logger.info(f"Registered callback for {topic}")
    
    def _start_queue_processor(self) -> None:
        """Start background thread to process message queue"""
        if self.queue_thread_running:
            return
        
        self.queue_thread_running = True
        self.queue_thread = Thread(target=self._process_queue, daemon=True)
        self.queue_thread.start()
        logger.info("Queue processor thread started")
    
    def _stop_queue_processor(self) -> None:
        """Stop queue processor thread"""
        if not self.queue_thread_running:
            return
        
        self.queue_thread_running = False
        if self.queue_thread is not None:
            self.queue_thread.join(timeout=5.0)
        logger.info("Queue processor thread stopped")
    
    def _process_queue(self) -> None:
        """Process queued messages with exponential backoff"""
        logger.info("Queue processor started")
        
        while self.queue_thread_running:
            try:
                # Process messages if connected
                if self.connected and len(self.message_queue) > 0:
                    with self.queue_lock:
                        if len(self.message_queue) > 0:
                            message = self.message_queue.popleft()
                    
                    # Check if message is stale
                    if message.is_stale:
                        logger.warning(
                            f"Dropping stale message for {message.topic} "
                            f"(age: {message.age_seconds:.1f}s)"
                        )
                        self.messages_dropped_stale += 1
                        continue
                    
                    # Try to publish
                    try:
                        result = self.client.publish(
                            message.topic,
                            message.to_json(),
                            qos=message.qos,
                            retain=message.retain
                        )
                        
                        if result.rc == mqtt.MQTT_ERR_SUCCESS:
                            self.messages_sent += 1
                            logger.debug(f"Sent queued message to {message.topic}")
                            # Reset retry delay on success
                            self.retry_delay = 0.1
                        else:
                            # Re-queue if failed
                            with self.queue_lock:
                                self.message_queue.append(message)
                            # Increase retry delay
                            self.retry_delay = min(self.retry_delay * 2, self.max_retry_delay)
                    
                    except Exception as e:
                        logger.error(f"Error sending queued message: {e}")
                        # Re-queue
                        with self.queue_lock:
                            self.message_queue.append(message)
                        # Increase retry delay
                        self.retry_delay = min(self.retry_delay * 2, self.max_retry_delay)
                
                # Sleep with current retry delay
                time.sleep(self.retry_delay)
            
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                time.sleep(1.0)
        
        logger.info("Queue processor stopped")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to broker"""
        return self.connected
    
    @property
    def queue_size(self) -> int:
        """Get current queue size"""
        with self.queue_lock:
            return len(self.message_queue)
    
    @property
    def uptime(self) -> float:
        """Get connection uptime in seconds"""
        if self.last_connection_time is None:
            return 0.0
        return time.time() - self.last_connection_time
    
    def get_stats(self) -> Dict:
        """Get MQTT client statistics"""
        return {
            'connected': self.connected,
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'messages_queued': self.messages_queued,
            'messages_dropped_stale': self.messages_dropped_stale,
            'current_queue_size': self.queue_size,
            'connection_attempts': self.connection_attempts,
            'last_connection': self.last_connection_time,
            'uptime_seconds': self.uptime,
            'retry_delay_ms': self.retry_delay * 1000
        }
    
    async def health_check(self) -> Dict:
        """Perform health check"""
        return {
            'status': 'healthy' if self.connected else 'unhealthy',
            'connected': self.connected,
            'queue_size': self.queue_size,
            'queue_capacity': self.message_queue.maxlen,
            'uptime_seconds': self.uptime
        }
