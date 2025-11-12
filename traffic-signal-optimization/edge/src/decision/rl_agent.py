"""
Reinforcement Learning Agent for Traffic Signal Control
Implements PPO (Proximal Policy Optimization) for Indian traffic
Includes model loading, inference, and fallback logic
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import EdgeSettings, Direction
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class State:
    """RL state representation for traffic signal control"""
    # Per-direction metrics (4 directions)
    queue_lengths: Dict[Direction, int]
    wait_times: Dict[Direction, float]
    vehicle_counts: Dict[Direction, int]
    pcu_values: Dict[Direction, float]
    densities: Dict[Direction, float]
    speeds: Dict[Direction, float]
    
    # Signal state
    current_phase: int
    time_in_phase: float
    
    # Temporal context
    time_of_day: float  # 0-24 hours
    day_of_week: int    # 0-6
    
    def to_vector(self) -> np.ndarray:
        """
        Convert state to feature vector for neural network
        
        Returns:
            Feature vector (28 dimensions)
        """
        features = []
        
        # Per-direction features (6 features × 4 directions = 24)
        for direction in [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]:
            features.extend([
                self.queue_lengths.get(direction, 0) / 30.0,       # Normalized queue
                self.wait_times.get(direction, 0.0) / 180.0,       # Normalized wait (max 3 min)
                self.vehicle_counts.get(direction, 0) / 20.0,      # Normalized count
                self.pcu_values.get(direction, 0.0) / 40.0,        # Normalized PCU
                self.densities.get(direction, 0.0),                # Already 0-1
                self.speeds.get(direction, 0.0) / 50.0             # Normalized speed
            ])
        
        # Signal state (2 features)
        features.extend([
            self.current_phase / 8.0,           # Normalized phase
            self.time_in_phase / 120.0          # Normalized time (max 2 min)
        ])
        
        # Temporal features (2 features)
        features.extend([
            self.time_of_day / 24.0,            # Normalized hour
            self.day_of_week / 6.0              # Normalized day
        ])
        
        return np.array(features, dtype=np.float32)


@dataclass
class Action:
    """RL action representation"""
    action_type: str  # 'extend', 'change_phase', 'emergency'
    phase_id: Optional[int] = None
    duration: float = 5.0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'type': self.action_type,
            'phase_id': self.phase_id,
            'duration': self.duration,
            'confidence': self.confidence
        }


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic neural network for PPO
    Optimized for Indian traffic signal control
    """
    
    def __init__(
        self,
        state_dim: int = 28,
        action_dim: int = 8,
        hidden_dim: int = 256
    ):
        """
        Initialize network
        
        Args:
            state_dim: State vector dimension
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: State tensor [batch_size, state_dim]
        
        Returns:
            (action_logits, state_value)
        """
        features = self.shared_layers(state)
        action_logits = self.actor(features)
        state_value = self.critic(features)
        return action_logits, state_value
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Args:
            state: State tensor
            deterministic: If True, take argmax instead of sampling
        
        Returns:
            (action_index, log_probability, state_value)
        """
        action_logits, state_value = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
        
        log_prob = F.log_softmax(action_logits, dim=-1)[0, action]
        
        return action.item(), log_prob, state_value


class RLAgent:
    """
    Reinforcement Learning agent for traffic signal control
    Uses PPO algorithm trained on Indian traffic patterns
    """
    
    # Action space definition
    ACTIONS = {
        0: {'type': 'extend', 'duration': 5.0},          # Extend current phase 5s
        1: {'type': 'extend', 'duration': 10.0},         # Extend current phase 10s
        2: {'type': 'change_phase', 'offset': 1},        # Switch to next phase
        3: {'type': 'change_phase', 'offset': -1},       # Switch to previous phase
        4: {'type': 'change_phase', 'max_queue': True},  # Switch to max queue direction
        5: {'type': 'change_phase', 'max_wait': True},   # Switch to max wait direction
        6: {'type': 'change_phase', 'max_pcu': True},    # Switch to max PCU direction
        7: {'type': 'emergency', 'duration': 30.0}       # Emergency override
    }
    
    def __init__(self, settings: EdgeSettings):
        """
        Initialize RL agent
        
        Args:
            settings: Edge device settings
        """
        self.settings = settings
        
        # Neural network
        self.device = torch.device(
            settings.device if torch.cuda.is_available() else 'cpu'
        )
        
        self.network = ActorCriticNetwork(
            state_dim=28,
            action_dim=len(self.ACTIONS),
            hidden_dim=256
        ).to(self.device)
        
        self.network.eval()  # Inference mode
        
        # Model state
        self.model_loaded = False
        self.model_path = Path(settings.rl_model_path)
        
        # Decision history for analysis
        self.decision_history: deque = deque(maxlen=1000)
        
        # Statistics
        self.total_decisions = 0
        self.action_counts = {i: 0 for i in range(len(self.ACTIONS))}
        
        logger.info(f"RL Agent initialized (device: {self.device})")
    
    async def load_model(self, model_path: Optional[Path] = None) -> bool:
        """
        Load trained model
        
        Args:
            model_path: Path to model checkpoint (uses settings if None)
        
        Returns:
            True if model loaded successfully
        """
        model_path = model_path or self.model_path
        
        if not model_path.exists():
            logger.warning(f"RL model not found at {model_path}")
            return False
        
        try:
            logger.info(f"Loading RL model from {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load model weights
            self.network.load_state_dict(checkpoint['model_state_dict'])
            
            # Load training metadata if available
            if 'training_stats' in checkpoint:
                stats = checkpoint['training_stats']
                logger.info(
                    f"Model trained for {stats.get('episodes', 'unknown')} episodes, "
                    f"Average reward: {stats.get('avg_reward', 'unknown'):.2f}"
                )
            
            self.model_loaded = True
            logger.success(f"RL model loaded successfully")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to load RL model: {e}", exc_info=True)
            return False
    
    async def predict(
        self,
        state_vector: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[Dict[str, Any], float]:
        """
        Predict action for given state
        
        Args:
            state_vector: State feature vector
            deterministic: Use deterministic policy (argmax)
        
        Returns:
            (action_dict, confidence)
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Convert to tensor
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            
            # Get action from network
            with torch.no_grad():
                action_idx, log_prob, state_value = self.network.get_action(
                    state_tensor,
                    deterministic=deterministic
                )
            
            # Convert to action dictionary
            action_template = self.ACTIONS[action_idx]
            action_dict = action_template.copy()
            
            # Calculate confidence from action probabilities
            action_logits, _ = self.network.forward(state_tensor)
            action_probs = F.softmax(action_logits, dim=-1)
            confidence = action_probs[0, action_idx].item()
            
            action_dict['confidence'] = confidence
            
            # Update statistics
            self.total_decisions += 1
            self.action_counts[action_idx] += 1
            
            # Record decision
            self.decision_history.append({
                'timestamp': time.time(),
                'state': state_vector.tolist(),
                'action_idx': action_idx,
                'action': action_dict,
                'confidence': confidence,
                'value_estimate': state_value.item()
            })
            
            logger.debug(
                f"RL decision: action={action_idx} ({action_dict['type']}), "
                f"confidence={confidence:.2f}"
            )
            
            return action_dict, confidence
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            # Return safe fallback action
            return {'type': 'extend', 'duration': 5.0, 'confidence': 0.0}, 0.0
    
    def interpret_action(
        self,
        action_dict: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Interpret raw action into concrete signal command
        
        Args:
            action_dict: Raw action from network
            current_state: Current traffic state
        
        Returns:
            Concrete action with phase_id
        """
        action_type = action_dict['type']
        
        if action_type == 'extend':
            return {
                'type': 'extend',
                'duration': action_dict.get('duration', 5.0)
            }
        
        elif action_type == 'change_phase':
            # Determine target phase
            if 'offset' in action_dict:
                # Relative phase change
                current_phase = current_state.get('current_phase', 0)
                num_phases = len(self.settings.signal_phases)
                target_phase = (current_phase + action_dict['offset']) % num_phases
            
            elif action_dict.get('max_queue'):
                # Switch to phase with maximum queue
                target_phase = self._find_phase_for_max_metric(
                    current_state,
                    'queue_length'
                )
            
            elif action_dict.get('max_wait'):
                # Switch to phase with maximum wait time
                target_phase = self._find_phase_for_max_metric(
                    current_state,
                    'avg_wait_time'
                )
            
            elif action_dict.get('max_pcu'):
                # Switch to phase with maximum PCU
                target_phase = self._find_phase_for_max_metric(
                    current_state,
                    'total_pcu'
                )
            
            else:
                target_phase = current_state.get('current_phase', 0)
            
            return {
                'type': 'change_phase',
                'phase_id': target_phase
            }
        
        elif action_type == 'emergency':
            return {
                'type': 'emergency',
                'duration': action_dict.get('duration', 30.0)
            }
        
        else:
            # Unknown action type, return safe default
            logger.warning(f"Unknown action type: {action_type}")
            return {'type': 'extend', 'duration': 5.0}
    
    def _find_phase_for_max_metric(
        self,
        current_state: Dict[str, Any],
        metric_name: str
    ) -> int:
        """
        Find phase that serves direction with maximum metric value
        
        Args:
            current_state: Current state with direction metrics
            metric_name: Name of metric to maximize
        
        Returns:
            Phase ID
        """
        direction_metrics = current_state.get('direction_metrics', {})
        
        if not direction_metrics:
            return current_state.get('current_phase', 0)
        
        # Find direction with max metric
        max_direction = None
        max_value = -float('inf')
        
        for direction_str, metrics in direction_metrics.items():
            value = metrics.get(metric_name, 0)
            if value > max_value:
                max_value = value
                max_direction = Direction(direction_str)
        
        if max_direction is None:
            return current_state.get('current_phase', 0)
        
        # Find phase that serves this direction
        for phase in self.settings.signal_phases:
            if max_direction in phase.directions:
                return phase.phase_id
        
        return current_state.get('current_phase', 0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        action_distribution = {
            f"action_{i}": count / max(self.total_decisions, 1)
            for i, count in self.action_counts.items()
        }
        
        return {
            'model_loaded': self.model_loaded,
            'model_path': str(self.model_path),
            'device': str(self.device),
            'total_decisions': self.total_decisions,
            'action_distribution': action_distribution,
            'history_size': len(self.decision_history)
        }
    
    def get_recent_decisions(self, n: int = 10) -> List[Dict]:
        """Get recent decision history"""
        return list(self.decision_history)[-n:]
    
    async def explain_decision(self, state_vector: np.ndarray) -> Dict[str, Any]:
        """
        Explain why agent made a particular decision
        
        Args:
            state_vector: State vector
        
        Returns:
            Explanation dictionary
        """
        if not self.model_loaded:
            return {'error': 'Model not loaded'}
        
        try:
            # Get action probabilities
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action_logits, state_value = self.network.forward(state_tensor)
                action_probs = F.softmax(action_logits, dim=-1)
            
            # Get top 3 actions
            top_probs, top_indices = torch.topk(action_probs[0], k=3)
            
            top_actions = []
            for prob, idx in zip(top_probs, top_indices):
                action = self.ACTIONS[idx.item()].copy()
                action['probability'] = prob.item()
                top_actions.append(action)
            
            return {
                'state_value': state_value.item(),
                'top_actions': top_actions,
                'action_probabilities': action_probs[0].tolist()
            }
        
        except Exception as e:
            logger.error(f"Error explaining decision: {e}")
            return {'error': str(e)}


# Fallback rule-based agent (used when RL model unavailable)
class RuleBasedAgent:
    """
    Simple rule-based agent as fallback
    Implements Webster's method adapted for Indian traffic
    """
    
    def __init__(self, settings: EdgeSettings):
        self.settings = settings
        logger.info("Rule-based agent initialized (fallback mode)")
    
    async def predict(
        self,
        state_vector: np.ndarray
    ) -> Tuple[Dict[str, Any], float]:
        """
        Make decision using simple rules
        
        Returns:
            (action_dict, confidence=1.0)
        """
        # Decode state vector (simplified)
        # This is a placeholder - in practice, pass structured state
        
        action = {
            'type': 'extend',
            'duration': 5.0,
            'confidence': 1.0
        }
        
        return action, 1.0
