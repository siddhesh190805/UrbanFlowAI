"""
Redis Cache for Real-Time Traffic State
Stores current state, temporary data, and inter-service communication
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from datetime import timedelta

import redis.asyncio as redis
from redis.asyncio import ConnectionPool
from redis.exceptions import RedisError

from config.settings import EdgeSettings
from utils.logger import get_logger

logger = get_logger(__name__)


class RedisCache:
    """
    Async Redis client for real-time state management
    """
    
    def __init__(self, settings: EdgeSettings):
        """Initialize Redis cache"""
        self.settings = settings
        self.client: Optional[redis.Redis] = None
        self.pool: Optional[ConnectionPool] = None
        
        # Key prefixes
        self.prefix = f"traffic:{settings.intersection_id}"
        
        # Statistics
        self.reads = 0
        self.writes = 0
        self.errors = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(
            f"Redis cache initialized for {settings.intersection_id} "
            f"({settings.redis_host}:{settings.redis_port})"
        )
    
    async def connect(self, max_retries: int = 3) -> bool:
        """
        Connect to Redis with retry logic
        
        Args:
            max_retries: Maximum connection attempts
        
        Returns:
            True if connected successfully
        """
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Connecting to Redis at {self.settings.redis_host}:{self.settings.redis_port} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                
                # Create connection pool
                self.pool = ConnectionPool(
                    host=self.settings.redis_host,
                    port=self.settings.redis_port,
                    db=self.settings.redis_db,
                    password=self.settings.redis_password,
                    max_connections=self.settings.redis_max_connections,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_keepalive=True
                )
                
                # Create client
                self.client = redis.Redis(connection_pool=self.pool)
                
                # Test connection
                await self.client.ping()
                
                logger.success("Connected to Redis successfully")
                return True
            
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("All Redis connection attempts failed")
                    return False
    
    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        if self.client:
            await self.client.close()
        
        if self.pool:
            await self.pool.disconnect()
        
        logger.info("Disconnected from Redis")
    
    # Traffic State Methods
    
    async def set_traffic_state(
        self,
        direction: str,
        state: Dict[str, Any],
        ttl: int = 60
    ) -> bool:
        """
        Set current traffic state for a direction
        
        Args:
            direction: Traffic direction
            state: State dictionary
            ttl: Time to live in seconds
        
        Returns:
            True if successful
        """
        if not self.client:
            return False
        
        try:
            key = f"{self.prefix}:state:{direction}"
            await self.client.setex(
                key,
                ttl,
                json.dumps(state)
            )
            self.writes += 1
            return True
        
        except RedisError as e:
            logger.error(f"Error setting traffic state: {e}")
            self.errors += 1
            return False
    
    async def get_traffic_state(self, direction: str) -> Optional[Dict[str, Any]]:
        """Get current traffic state for a direction"""
        if not self.client:
            return None
        
        try:
            key = f"{self.prefix}:state:{direction}"
            data = await self.client.get(key)
            self.reads += 1
            
            if data:
                self.cache_hits += 1
                return json.loads(data)
            
            self.cache_misses += 1
            return None
        
        except RedisError as e:
            logger.error(f"Error getting traffic state: {e}")
            self.errors += 1
            return None
    
    # Signal State Methods
    
    async def set_signal_state(self, state: Dict[str, Any]) -> bool:
        """Set current signal state"""
        if not self.client:
            return False
        
        try:
            key = f"{self.prefix}:signal_state"
            await self.client.setex(
                key,
                30,  # 30 second TTL
                json.dumps(state)
            )
            self.writes += 1
            return True
        
        except RedisError as e:
            logger.error(f"Error setting signal state: {e}")
            self.errors += 1
            return False
    
    async def get_signal_state(self) -> Optional[Dict[str, Any]]:
        """Get current signal state"""
        if not self.client:
            return None
        
        try:
            key = f"{self.prefix}:signal_state"
            data = await self.client.get(key)
            
            if data:
                return json.loads(data)
            return None
        
        except RedisError as e:
            logger.error(f"Error getting signal state: {e}")
            return None
    
    # Detection Caching
    
    async def cache_detection(
        self,
        camera_id: str,
        detection_data: Dict[str, Any],
        ttl: int = 5
    ) -> bool:
        """Cache detection result temporarily"""
        if not self.client:
            return False
        
        try:
            key = f"{self.prefix}:detection:{camera_id}"
            await self.client.setex(key, ttl, json.dumps(detection_data))
            self.writes += 1
            return True
        
        except RedisError as e:
            logger.error(f"Error caching detection: {e}")
            self.errors += 1
            return False
    
    async def get_cached_detection(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get cached detection"""
        if not self.client:
            return None
        
        try:
            key = f"{self.prefix}:detection:{camera_id}"
            data = await self.client.get(key)
            
            if data:
                self.cache_hits += 1
                return json.loads(data)
            
            self.cache_misses += 1
            return None
        
        except RedisError as e:
            logger.error(f"Error getting cached detection: {e}")
            return None
    
    # Counter Methods
    
    async def increment_counter(self, counter_name: str, amount: int = 1) -> int:
        """Increment a counter"""
        if not self.client:
            return 0
        
        try:
            key = f"{self.prefix}:counter:{counter_name}"
            result = await self.client.incrby(key, amount)
            self.writes += 1
            return result
        
        except RedisError as e:
            logger.error(f"Error incrementing counter: {e}")
            self.errors += 1
            return 0
    
    async def get_counter(self, counter_name: str) -> int:
        """Get counter value"""
        if not self.client:
            return 0
        
        try:
            key = f"{self.prefix}:counter:{counter_name}"
            value = await self.client.get(key)
            self.reads += 1
            return int(value) if value else 0
        
        except RedisError as e:
            logger.error(f"Error getting counter: {e}")
            return 0
    
    async def reset_counter(self, counter_name: str) -> bool:
        """Reset counter to zero"""
        if not self.client:
            return False
        
        try:
            key = f"{self.prefix}:counter:{counter_name}"
            await self.client.set(key, 0)
            return True
        
        except RedisError as e:
            logger.error(f"Error resetting counter: {e}")
            return False
    
    # List Operations (for queues)
    
    async def push_to_list(self, list_name: str, value: Any) -> bool:
        """Push value to list (right push)"""
        if not self.client:
            return False
        
        try:
            key = f"{self.prefix}:list:{list_name}"
            await self.client.rpush(key, json.dumps(value))
            self.writes += 1
            return True
        
        except RedisError as e:
            logger.error(f"Error pushing to list: {e}")
            return False
    
    async def pop_from_list(self, list_name: str) -> Optional[Any]:
        """Pop value from list (left pop)"""
        if not self.client:
            return None
        
        try:
            key = f"{self.prefix}:list:{list_name}"
            data = await self.client.lpop(key)
            self.reads += 1
            
            if data:
                return json.loads(data)
            return None
        
        except RedisError as e:
            logger.error(f"Error popping from list: {e}")
            return None
    
    # Pub/Sub Methods
    
    async def publish(self, channel: str, message: Dict[str, Any]) -> bool:
        """Publish message to channel"""
        if not self.client:
            return False
        
        try:
            full_channel = f"{self.prefix}:channel:{channel}"
            await self.client.publish(full_channel, json.dumps(message))
            return True
        
        except RedisError as e:
            logger.error(f"Error publishing message: {e}")
            return False
    
    # Health and Stats
    
    async def health_check(self) -> Dict:
        """Perform health check"""
        if not self.client:
            return {
                'status': 'unhealthy',
                'message': 'Not connected'
            }
        
        try:
            await self.client.ping()
            info = await self.client.info()
            
            return {
                'status': 'healthy',
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', 'unknown'),
                'uptime_seconds': info.get('uptime_in_seconds', 0),
                'hit_rate': self.hit_rate
            }
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': str(e)
            }
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'connected': self.client is not None,
            'reads': self.reads,
            'writes': self.writes,
            'errors': self.errors,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.hit_rate
        }
