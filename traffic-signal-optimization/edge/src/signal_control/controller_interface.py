"""
Signal Controller Interface
Communicates with traffic signal controller via NTCIP protocol
Implements safety checks and fail-safe mechanisms
"""

import asyncio
import time
from typing import Optional, Dict
from enum import Enum
from dataclasses import dataclass

from config.settings import EdgeSettings, SignalPhaseConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class PhaseState(str, Enum):
    """Signal phase states"""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    ALL_RED = "all_red"


@dataclass
class SignalState:
    """Current signal state"""
    current_phase: int
    phase_state: PhaseState
    time_in_state: float
    time_in_phase: float
    cycle_start_time: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'current_phase': self.current_phase,
            'phase_state': self.phase_state.value,
            'time_in_state': round(self.time_in_state, 2),
            'time_in_phase': round(self.time_in_phase, 2),
            'cycle_start_time': self.cycle_start_time
        }


class SignalController:
    """
    Traffic signal controller interface
    Handles communication with physical signal controller
    Implements safety checks and timing constraints
    """
    
    def __init__(self, settings: EdgeSettings):
        """
        Initialize signal controller
        
        Args:
            settings: Edge device settings
        """
        self.settings = settings
        
        # Current state
        self.current_state = SignalState(
            current_phase=0,
            phase_state=PhaseState.RED,
            time_in_state=0.0,
            time_in_phase=0.0,
            cycle_start_time=time.time()
        )
        
        # Safety tracking
        self.last_phase_change = time.time()
        self.phase_change_count = 0
        self.emergency_mode = False
        
        # Connection state
        self.connected = False
        self.controller_type = settings.controller_protocol
        
        # Statistics
        self.phase_changes = 0
        self.emergency_activations = 0
        self.safety_violations = 0
        
        logger.info(
            f"SignalController initialized "
            f"(protocol={self.controller_type}, ip={settings.controller_ip})"
        )
    
    async def connect(self) -> bool:
        """
        Connect to signal controller
        
        Returns:
            True if connection successful
        """
        try:
            logger.info(f"Connecting to signal controller at {self.settings.controller_ip}")
            
            if self.controller_type == "MOCK":
                # Mock controller for testing
                logger.warning("Using MOCK signal controller (testing mode)")
                self.connected = True
                return True
            
            elif self.controller_type == "NTCIP":
                # NTCIP protocol implementation
                # This would use SNMP to communicate with controller
                logger.info("NTCIP controller connection not yet implemented")
                # TODO: Implement actual NTCIP/SNMP communication
                self.connected = False
                return False
            
            elif self.controller_type == "MODBUS":
                # Modbus protocol implementation
                logger.info("MODBUS controller connection not yet implemented")
                # TODO: Implement Modbus communication
                self.connected = False
                return False
            
            else:
                logger.error(f"Unknown controller protocol: {self.controller_type}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to connect to signal controller: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from signal controller"""
        if self.connected:
            logger.info("Disconnecting from signal controller")
            # Return to safe state before disconnecting
            await self._activate_failsafe()
            self.connected = False
            logger.info("Disconnected from signal controller")
    
    async def change_phase(self, target_phase: int, duration: Optional[float] = None) -> bool:
        """
        Change to target phase with safety checks
        
        Args:
            target_phase: Target phase ID
            duration: Green duration in seconds (None for phase defaults)
        
        Returns:
            True if phase change initiated successfully
        """
        # Validate phase
        phase_config = self.settings.get_phase_by_id(target_phase)
        if not phase_config:
            logger.error(f"Invalid phase ID: {target_phase}")
            return False
        
        # Safety check: minimum green time
        if self.current_state.time_in_phase < self._get_min_green():
            logger.warning(
                f"Cannot change phase: minimum green time not satisfied "
                f"({self.current_state.time_in_phase:.1f}s < {self._get_min_green()}s)"
            )
            self.safety_violations += 1
            return False
        
        # Safety check: prevent rapid phase changes
        time_since_last_change = time.time() - self.last_phase_change
        if time_since_last_change < 10.0:  # Minimum 10 seconds between changes
            logger.warning(
                f"Cannot change phase: too soon since last change "
                f"({time_since_last_change:.1f}s < 10.0s)"
            )
            self.safety_violations += 1
            return False
        
        # Already in target phase
        if self.current_state.current_phase == target_phase:
            logger.debug(f"Already in phase {target_phase}")
            return True
        
        logger.info(
            f"Changing phase from {self.current_state.current_phase} to {target_phase} "
            f"(duration={duration}s)"
        )
        
        # Execute phase transition sequence
        await self._execute_phase_transition(target_phase, duration or phase_config.max_green)
        
        return True
    
    async def _execute_phase_transition(self, target_phase: int, duration: float) -> None:
        """
        Execute complete phase transition with yellow and all-red
        
        Args:
            target_phase: Target phase ID
            duration: Green duration for target phase
        """
        current_config = self.settings.get_phase_by_id(self.current_state.current_phase)
        target_config = self.settings.get_phase_by_id(target_phase)
        
        if not current_config or not target_config:
            logger.error("Invalid phase configuration")
            return
        
        # 1. Yellow clearance
        logger.debug(f"Yellow clearance: {current_config.yellow_time}s")
        await self._set_phase_state(PhaseState.YELLOW)
        await asyncio.sleep(current_config.yellow_time)
        
        # 2. All-red clearance
        logger.debug(f"All-red clearance: {current_config.all_red_time}s")
        await self._set_phase_state(PhaseState.ALL_RED)
        await asyncio.sleep(current_config.all_red_time)
        
        # 3. Activate target phase
        logger.debug(f"Green for phase {target_phase}: {duration}s")
        self.current_state.current_phase = target_phase
        self.current_state.time_in_phase = 0.0
        await self._set_phase_state(PhaseState.GREEN)
        
        # Update tracking
        self.last_phase_change = time.time()
        self.phase_changes += 1
        self.phase_change_count += 1
        
        logger.success(f"Phase transition to {target_phase} complete")
    
    async def _set_phase_state(self, state: PhaseState) -> None:
        """
        Set phase state (RED, YELLOW, GREEN, ALL_RED)
        
        Args:
            state: Target phase state
        """
        self.current_state.phase_state = state
        self.current_state.time_in_state = 0.0
        
        if self.controller_type == "MOCK":
            # Mock implementation - just update state
            logger.debug(f"Set phase state to {state.value}")
        else:
            # Real controller communication
            # TODO: Send command to actual controller via NTCIP/MODBUS
            pass
    
    async def extend_green(self, extension_seconds: float) -> bool:
        """
        Extend current green phase
        
        Args:
            extension_seconds: Seconds to extend green
        
        Returns:
            True if extension successful
        """
        if self.current_state.phase_state != PhaseState.GREEN:
            logger.warning("Cannot extend: not in green phase")
            return False
        
        phase_config = self.settings.get_phase_by_id(self.current_state.current_phase)
        if not phase_config:
            return False
        
        # Check if extension would exceed max green
        new_duration = self.current_state.time_in_phase + extension_seconds
        if new_duration > phase_config.max_green:
            logger.warning(
                f"Cannot extend: would exceed max green "
                f"({new_duration}s > {phase_config.max_green}s)"
            )
            return False
        
        logger.info(f"Extending green by {extension_seconds}s")
        
        # In real implementation, send extension command to controller
        # For now, just track it
        
        return True
    
    async def emergency_preemption(self, direction: str) -> bool:
        """
        Activate emergency vehicle preemption
        
        Args:
            direction: Direction of emergency vehicle
        
        Returns:
            True if preemption activated
        """
        if not self.settings.emergency_vehicle_priority:
            logger.warning("Emergency vehicle priority disabled")
            return False
        
        logger.warning(f"EMERGENCY PREEMPTION ACTIVATED for {direction}")
        
        self.emergency_mode = True
        self.emergency_activations += 1
        
        # Determine appropriate phase for emergency vehicle
        # This is simplified - should consider actual emergency vehicle routing
        emergency_phase = self._get_emergency_phase(direction)
        
        # Force immediate phase change
        await self.change_phase(emergency_phase)
        
        return True
    
    def _get_emergency_phase(self, direction: str) -> int:
        """Get appropriate phase for emergency vehicle direction"""
        direction_phase_map = {
            'north': 1,
            'south': 1,
            'east': 3,
            'west': 3
        }
        return direction_phase_map.get(direction.lower(), 1)
    
    async def _activate_failsafe(self) -> None:
        """Activate fail-safe mode (all red or fixed timing)"""
        logger.warning("Activating FAIL-SAFE mode")
        
        # Set all red
        await self._set_phase_state(PhaseState.ALL_RED)
        
        # If fail-safe pattern configured, activate it
        if self.settings.fail_safe_pattern_id:
            await self.change_phase(self.settings.fail_safe_pattern_id)
    
    def _get_min_green(self) -> float:
        """Get minimum green time for current phase"""
        phase_config = self.settings.get_phase_by_id(self.current_state.current_phase)
        if phase_config:
            return float(phase_config.min_green)
        return 7.0  # Default minimum
    
    async def update(self, delta_time: float) -> None:
        """
        Update signal state (call every frame)
        
        Args:
            delta_time: Time since last update in seconds
        """
        self.current_state.time_in_state += delta_time
        self.current_state.time_in_phase += delta_time
        
        # Check for max green exceeded
        phase_config = self.settings.get_phase_by_id(self.current_state.current_phase)
        if phase_config and self.current_state.time_in_phase > phase_config.max_green:
            logger.warning(
                f"Phase {self.current_state.current_phase} exceeded max green "
                f"({self.current_state.time_in_phase:.1f}s > {phase_config.max_green}s)"
            )
    
    def get_state(self) -> SignalState:
        """Get current signal state"""
        return self.current_state
    
    def get_stats(self) -> Dict:
        """Get controller statistics"""
        return {
            'connected': self.connected,
            'current_phase': self.current_state.current_phase,
            'phase_state': self.current_state.phase_state.value,
            'time_in_phase': round(self.current_state.time_in_phase, 2),
            'phase_changes': self.phase_changes,
            'emergency_activations': self.emergency_activations,
            'safety_violations': self.safety_violations,
            'emergency_mode': self.emergency_mode
        }
    
    async def health_check(self) -> Dict:
        """Perform health check"""
        return {
            'status': 'healthy' if self.connected else 'unhealthy',
            'connected': self.connected,
            'emergency_mode': self.emergency_mode,
            'safety_violations': self.safety_violations
        }