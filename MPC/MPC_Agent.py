import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import cvxpy as cp
from scipy.optimize import minimize
from collections import deque
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class MPCConfig:
    """
    MPC hyperparameters and settings
    """

    horizon = 20
    dt = 0.1
    num_iterations = 3

    goal_weights = 10.0
    collision_weights = 100.0
    control_weights = 0.1
    velocity_weights = 1.0
    smoothness_weights = 0.5

    max_velocity = 5.0
    max_acceleration = 10.0
    min_separation = 2.0
    obstacle_buffer = 0.5

    state_dim = 92
    action_dim = 6
    hidden_dim = 256

    dynamics_lr = 1e-3
    dynamics_buffer_size = 100000
    dynamics_batch_size = 256
    dynamics_update_freq = 100

    use_gpu_optimization = True
    warm_start = True
    adaptive_horizon = True


class LearnedDynamicsModel(nn.Module):
    """
    Neural network to learn the dynamics model of the agents
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Dynamics network with residual connections
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Residual prediction (predict change in state)
        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next state and uncertainty.
        
        Args:
            state: Current state (batch_size, state_dim)
            action: Action to take (batch_size, action_dim)
        
        Returns:
            next_state: Predicted next state
            uncertainty: Prediction uncertainty (for robust MPC)
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        
        # Encode
        features = self.encoder(x)
        
        # Predict state change (residual)
        state_delta = self.dynamics_head(features)
        
        # Next state = current state + predicted change
        next_state = state + state_delta
        
        # Predict uncertainty
        log_uncertainty = self.uncertainty_head(features)
        uncertainty = torch.exp(log_uncertainty)
        
        return next_state, uncertainty
    
    def rollout(self, initial_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Rollout dynamics for multiple steps.
        
        Args:
            initial_state: Initial state (batch_size, state_dim)
            actions: Sequence of actions (batch_size, horizon, action_dim)
        
        Returns:
            states: Predicted state trajectory (batch_size, horizon+1, state_dim)
        """
        batch_size, horizon, _ = actions.shape
        states = [initial_state]
        
        state = initial_state
        for t in range(horizon):
            action = actions[:, t, :]
            next_state, _ = self.forward(state, action)
            states.append(next_state)
            state = next_state
        
        return torch.stack(states, dim=1)
    

class DynamicsBuffer:
    """
    Experience buffer for training dynamics model.
    Stores (state, action, next_state) transitions.
    """
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray):
        """Add transition to buffer"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample random batch"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'states': torch.FloatTensor(self.states[indices]).to(device),
            'actions': torch.FloatTensor(self.actions[indices]).to(device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(device),
        }
    
class MPCController:
    """
    Model Predictive Controller for single drone.
    Uses learned dynamics model and handles constraints.
    """
    
    def __init__(self, agent_id: int, config: MPCConfig):
        self.agent_id = agent_id
        self.config = config
        
        # Dynamics model
        self.dynamics = LearnedDynamicsModel(
            config.state_dim,
            config.action_dim,
            config.hidden_dim
        ).to(device)
        
        # Dynamics optimizer
        self.dynamics_optimizer = optim.Adam(
            self.dynamics.parameters(),
            lr=config.dynamics_lr
        )
        
        # Experience buffer
        self.dynamics_buffer = DynamicsBuffer(
            config.dynamics_buffer_size,
            config.state_dim,
            config.action_dim
        )
        
        # Warm start solution
        self.prev_solution = None
        
        # Statistics
        self.solve_times = deque(maxlen=100)
        self.constraint_violations = deque(maxlen=100)
    
    def extract_state(self, camera_obs: np.ndarray, vector_obs: np.ndarray) -> np.ndarray:
        """
        Extract state representation from observations.
        
        For MPC, we need a compact state representation:
        - Position (3D)
        - Velocity (3D)
        - Orientation (3D euler or quaternion)
        - Angular velocity (3D)
        """
        # Extract relevant state from vector observations
        # Assuming vector_obs contains: [position, velocity, orientation, angular_vel, ...]
        state = vector_obs[:self.config.state_dim]
        
        # Could also incorporate visual features if needed
        # visual_features = self.encode_vision(camera_obs)
        # state = np.concatenate([state, visual_features])
        
        return state
    
    def solve_mpc(
        self,
        current_state: np.ndarray,
        goal_state: np.ndarray,
        other_agents_predictions: Optional[Dict[int, np.ndarray]] = None,
        obstacles: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Solve MPC optimization problem.
        
        Args:
            current_state: Current state of the drone
            goal_state: Target state
            other_agents_predictions: Predicted trajectories of other drones
            obstacles: Static obstacles in environment
        
        Returns:
            optimal_action: First action from optimal control sequence
        """
        start_time = time.time()
        
        # Convert to torch tensors
        state_tensor = torch.FloatTensor(current_state).to(device)
        goal_tensor = torch.FloatTensor(goal_state).to(device)
        
        # Initialize control sequence
        if self.config.warm_start and self.prev_solution is not None:
            # Shift previous solution
            u_init = np.vstack([self.prev_solution[1:], self.prev_solution[-1:]])
        else:
            # Random initialization
            u_init = np.random.randn(self.config.horizon, self.config.action_dim) * 0.1
        
        # Define optimization problem
        def objective(u_flat):
            """Compute total cost"""
            u = u_flat.reshape(self.config.horizon, self.config.action_dim)
            u_tensor = torch.FloatTensor(u).unsqueeze(0).to(device)
            
            # Rollout dynamics
            with torch.no_grad():
                states = self.dynamics.rollout(state_tensor.unsqueeze(0), u_tensor)
                states = states.squeeze(0)
            
            # Goal reaching cost
            goal_cost = torch.sum((states - goal_tensor.unsqueeze(0))**2).item()
            
            # Control effort cost
            control_cost = torch.sum(u_tensor**2).item()
            
            # Control smoothness cost
            if self.config.horizon > 1:
                u_diff = u_tensor[:, 1:, :] - u_tensor[:, :-1, :]
                smooth_cost = torch.sum(u_diff**2).item()
            else:
                smooth_cost = 0
            
            # Total cost
            total_cost = (
                self.config.goal_weight * goal_cost +
                self.config.control_weight * control_cost +
                self.config.smoothness_weight * smooth_cost
            )
            
            return total_cost
        
        def constraints(u_flat):
            """Compute constraint violations"""
            u = u_flat.reshape(self.config.horizon, self.config.action_dim)
            u_tensor = torch.FloatTensor(u).unsqueeze(0).to(device)
            
            constraints_list = []
            
            # Rollout dynamics
            with torch.no_grad():
                states = self.dynamics.rollout(state_tensor.unsqueeze(0), u_tensor)
                states = states.squeeze(0).cpu().numpy()
            
            # Velocity constraints
            velocities = states[:, 3:6]  # Assuming velocity is in indices 3-6
            for t in range(self.config.horizon + 1):
                vel_magnitude = np.linalg.norm(velocities[t])
                constraints_list.append(self.config.max_velocity - vel_magnitude)
            
            # Collision avoidance with other agents
            if other_agents_predictions is not None:
                for t in range(self.config.horizon):
                    my_pos = states[t, :3]  # Assuming position is in indices 0-3
                    for agent_id, other_traj in other_agents_predictions.items():
                        if agent_id != self.agent_id and t < len(other_traj):
                            other_pos = other_traj[t, :3]
                            distance = np.linalg.norm(my_pos - other_pos)
                            constraints_list.append(distance - self.config.min_separation)
            
            # Obstacle avoidance
            if obstacles is not None:
                for t in range(self.config.horizon):
                    my_pos = states[t, :3]
                    for obs in obstacles:
                        obs_pos = obs[:3]
                        obs_radius = obs[3] if len(obs) > 3 else 1.0
                        distance = np.linalg.norm(my_pos - obs_pos)
                        constraints_list.append(distance - obs_radius - self.config.obstacle_buffer)
            
            return np.array(constraints_list)
        
        # Optimization bounds
        bounds = [(-1, 1)] * (self.config.horizon * self.config.action_dim)
        
        # Constraint definition
        constraint_dict = {'type': 'ineq', 'fun': constraints}
        
        # Solve optimization
        result = minimize(
            objective,
            u_init.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_dict,
            options={
                'maxiter': self.config.num_iterations * 10,
                'ftol': 1e-4,
                'disp': False
            }
        )
        
        # Extract solution
        if result.success:
            u_optimal = result.x.reshape(self.config.horizon, self.config.action_dim)
            self.prev_solution = u_optimal
        else:
            # Fall back to simple controller if optimization fails
            u_optimal = self.fallback_controller(current_state, goal_state)
            self.prev_solution = u_optimal
        
        # Record statistics
        solve_time = time.time() - start_time
        self.solve_times.append(solve_time)
        
        # Return first action
        return u_optimal[0]
    
    def fallback_controller(self, current_state: np.ndarray, goal_state: np.ndarray) -> np.ndarray:
        """
        Simple PD controller as fallback when MPC fails.
        """
        # Extract positions and velocities
        current_pos = current_state[:3]
        current_vel = current_state[3:6]
        goal_pos = goal_state[:3]
        
        # PD control
        kp = 1.0
        kd = 0.5
        
        pos_error = goal_pos - current_pos
        vel_error = -current_vel
        
        # Generate simple trajectory
        actions = []
        for t in range(self.config.horizon):
            action = kp * pos_error + kd * vel_error
            action = np.clip(action, -1, 1)
            # Pad to action dimension
            if len(action) < self.config.action_dim:
                action = np.pad(action, (0, self.config.action_dim - len(action)))
            actions.append(action[:self.config.action_dim])
        
        return np.array(actions)
    
    def update_dynamics(self, batch_size: Optional[int] = None):
        """
        Update dynamics model from experience buffer.
        """
        if self.dynamics_buffer.size < self.config.dynamics_batch_size:
            return
        
        batch_size = batch_size or self.config.dynamics_batch_size
        batch = self.dynamics_buffer.sample(batch_size)
        
        # Forward pass
        pred_next_states, uncertainties = self.dynamics(
            batch['states'],
            batch['actions']
        )
        
        # Compute loss
        state_loss = F.mse_loss(pred_next_states, batch['next_states'])
        
        # Add uncertainty regularization
        uncertainty_loss = torch.mean(uncertainties)
        
        total_loss = state_loss + 0.01 * uncertainty_loss
        
        # Backward pass
        self.dynamics_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1.0)
        self.dynamics_optimizer.step()
        
        return {
            'dynamics_loss': state_loss.item(),
            'uncertainty': uncertainty_loss.item()
        }

class MultiAgentMPC:
    """
    Coordinates multiple MPC controllers for multi-agent system.
    Handles communication and collision avoidance between agents.
    """
    
    def __init__(
        self,
        num_agents: int,
        camera_shape: Tuple[int, ...],
        vector_shape: Tuple[int, ...],
        action_dim: int,
        config: MPCConfig,
        feature_extractor: Optional[nn.Module] = None
    ):
        self.num_agents = num_agents
        self.camera_shape = camera_shape
        self.vector_shape = vector_shape
        self.action_dim = action_dim
        self.config = config
        
        # Create MPC controller for each agent
        self.controllers = [
            MPCController(i, config)
            for i in range(num_agents)
        ]
        
        # Visual feature extractor (reuse from MAPPO if available)
        if feature_extractor is not None:
            self.vision_encoder = feature_extractor.to(device)
        else:
            self.vision_encoder = self._create_simple_cnn(camera_shape).to(device)
        
        # Communication protocol
        self.communication_rounds = 3  # Number of rounds for trajectory negotiation
        
        # Trajectory predictions for each agent
        self.trajectory_predictions = {}
        
        # Performance tracking
        self.solve_times = deque(maxlen=100)
        self.success_count = 0
        self.total_count = 0
    
    def _create_simple_cnn(self, camera_shape):
        """Create a simple CNN for vision encoding"""
        return nn.Sequential(
            nn.Conv2d(camera_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )
    
    def encode_observations(self, camera_obs: np.ndarray, vector_obs: np.ndarray) -> np.ndarray:
        """
        Encode camera and vector observations.
        
        Args:
            camera_obs: (num_agents, C, H, W)
            vector_obs: (num_agents, vector_dim)
        
        Returns:
            encoded_obs: (num_agents, encoded_dim)
        """
        # Process camera observations
        if camera_obs.dtype == np.uint8:
            camera_obs = camera_obs.astype(np.float32) / 255.0
        
        camera_tensor = torch.FloatTensor(camera_obs).to(device)
        
        # Extract visual features
        with torch.no_grad():
            visual_features = self.vision_encoder(camera_tensor)
        
        # Combine with vector observations
        vector_tensor = torch.FloatTensor(vector_obs).to(device)
        
        # For MPC, we primarily use vector observations
        # Visual features could be used for obstacle detection
        return vector_obs  # Return raw vector obs for state extraction
    
    def get_action(
        self,
        camera_obs: np.ndarray,
        vector_obs: np.ndarray,
        goals: Optional[np.ndarray] = None,
        obstacles: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get actions for all agents using distributed MPC with coordination.
        
        Args:
            camera_obs: Camera observations (num_agents, C, H, W)
            vector_obs: Vector observations (num_agents, vector_dim)
            goals: Goal states for each agent (num_agents, state_dim)
            obstacles: Static obstacles in environment
        
        Returns:
            actions: Actions for all agents (num_agents, action_dim)
            info: Additional information (solve times, etc.)
        """
        start_time = time.time()
        
        # Encode observations
        encoded_obs = self.encode_observations(camera_obs, vector_obs)
        
        # Extract states for each agent
        states = []
        for i in range(self.num_agents):
            state = self.controllers[i].extract_state(camera_obs[i], vector_obs[i])
            states.append(state)
        
        # Set default goals if not provided
        if goals is None:
            # Simple goal: maintain position or move forward
            goals = np.copy(states)
            goals[:, 0] += 5.0  # Move 5 units forward
        
        # Iterative trajectory negotiation
        for comm_round in range(self.communication_rounds):
            new_predictions = {}
            
            # Each agent solves MPC with current trajectory predictions
            for i in range(self.num_agents):
                # Get other agents' predictions
                other_predictions = {
                    j: self.trajectory_predictions.get(j, None)
                    for j in range(self.num_agents)
                    if j != i
                }
                
                # Solve MPC
                action = self.controllers[i].solve_mpc(
                    states[i],
                    goals[i],
                    other_predictions,
                    obstacles
                )
                
                # Update trajectory prediction
                # For now, just store the planned trajectory
                # In practice, would rollout the full trajectory
                new_predictions[i] = self._predict_trajectory(states[i], action)
            
            # Update predictions
            self.trajectory_predictions = new_predictions
        
        # Extract final actions
        actions = []
        for i in range(self.num_agents):
            action = self.controllers[i].solve_mpc(
                states[i],
                goals[i],
                self.trajectory_predictions,
                obstacles
            )
            actions.append(action)
        
        actions = np.array(actions)
        
        # Record statistics
        total_solve_time = time.time() - start_time
        self.solve_times.append(total_solve_time)
        
        info = {
            'solve_time': total_solve_time,
            'avg_solve_time': np.mean(self.solve_times),
            'trajectory_predictions': self.trajectory_predictions,
        }
        
        return actions, info
    
    def _predict_trajectory(self, state: np.ndarray, first_action: np.ndarray) -> np.ndarray:
        """
        Predict agent trajectory based on first action.
        Simple linear prediction for now.
        """
        trajectory = []
        current_state = np.copy(state)
        
        for t in range(self.config.horizon):
            # Simple dynamics: integrate velocity
            if t == 0:
                # Use provided action for first step
                current_state[:3] += first_action[:3] * self.config.dt
            else:
                # Continue with constant velocity
                current_state[:3] += current_state[3:6] * self.config.dt
            
            trajectory.append(np.copy(current_state))
        
        return np.array(trajectory)
    
    def update_dynamics_models(self, transitions: List[Dict]):
        """
        Update all agents' dynamics models from collected transitions.
        
        Args:
            transitions: List of transition dictionaries with
                        'state', 'action', 'next_state' for each agent
        """
        for transition in transitions:
            for i in range(self.num_agents):
                if i in transition:
                    self.controllers[i].dynamics_buffer.add(
                        transition[i]['state'],
                        transition[i]['action'],
                        transition[i]['next_state']
                    )
        
        # Update each controller's dynamics model
        update_stats = []
        for i, controller in enumerate(self.controllers):
            stats = controller.update_dynamics()
            if stats:
                update_stats.append(stats)
        
        return update_stats
    
    def save(self, path: str):
        """Save MPC models"""
        checkpoint = {
            'config': self.config.__dict__,
            'dynamics_models': [
                controller.dynamics.state_dict()
                for controller in self.controllers
            ],
            'vision_encoder': self.vision_encoder.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"MPC models saved to {path}")
    
    def load(self, path: str):
        """Load MPC models"""
        checkpoint = torch.load(path, map_location=device)
        
        for i, controller in enumerate(self.controllers):
            controller.dynamics.load_state_dict(checkpoint['dynamics_models'][i])
        
        self.vision_encoder.load_state_dict(checkpoint['vision_encoder'])
        print(f"MPC models loaded from {path}")