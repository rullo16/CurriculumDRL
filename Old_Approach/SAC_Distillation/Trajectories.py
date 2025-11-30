"""
Fixed Experience Replay Buffer for Multi-Agent SAC
===================================================

This module provides a fixed experience replay buffer with:
1. Proper n-step return calculation with episode boundary masking
2. Consistent reward normalization (no double normalization)
3. Correct prioritized experience replay for multi-agent
4. Running statistics using Welford's algorithm

Critical Fixes Applied:
- N-step returns properly mask rewards after episode termination
- Reward normalization happens once in buffer, not again in training
- PER priorities correctly handle multi-agent joint sampling
- Statistics are numerically stable with Welford's algorithm

Author: Fixed Implementation
Date: November 2025
"""

import numpy as np


# ============================================================================
# REWARD NORMALIZATION UTILITY
# ============================================================================

def _norm_reward(r, d, mean, std):
    """
    Normalize rewards using running statistics.
    Does NOT normalize terminal rewards (where done=1).
    
    Args:
        r: Rewards array
        d: Done flags array
        mean: Running mean
        std: Running std
    
    Returns:
        normalized_rewards: Normalized reward array
    """
    r = r.copy()
    idx = (d == 0.0)  # Only normalize non-terminal rewards
    r[idx] = (r[idx] - mean) / (std + 1e-8)
    return r


# ============================================================================
# RUNNING STATISTICS
# ============================================================================

class RunningStat:
    """
    Maintains running mean and variance using Welford's online algorithm.
    Numerically stable and works for scalars, vectors, and images.
    """
    def __init__(self, shape, eps=1e-4):
        self._mean = np.zeros(shape, dtype=np.float32)
        self._var = np.ones(shape, dtype=np.float32)
        self._count = eps

    def update(self, x):
        """
        Update statistics with new data.
        
        Accepts either:
        - Single sample with shape == self._mean.shape
        - Batch with shape (B, *self._mean.shape)
        
        Uses Welford's online algorithm for numerical stability.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return

        target_shape = self._mean.shape
        
        # Handle single sample vs batch
        if x.shape == target_shape:
            x = x.reshape((1, *target_shape))
        elif x.shape[1:] != target_shape:
            raise ValueError(
                f"RunningStat update shape mismatch: got {x.shape}, "
                f"expected (*, {target_shape})."
            )

        # Welford's online algorithm
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self._mean
        tot_count = self._count + batch_count

        # Update mean
        self._mean += delta * batch_count / tot_count
        
        # Update variance
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self._count * batch_count / tot_count
        self._var = m2 / tot_count
        self._count = tot_count
    
    @property
    def mean(self):
        return self._mean
    
    @property
    def std(self):
        return np.sqrt(self._var + 1e-8)


# ============================================================================
# FIXED EXPERIENCE BUFFER
# ============================================================================

class SAC_ExperienceBuffer:
    """
    Experience replay buffer for multi-agent SAC with critical bug fixes.
    
    Key Features:
    - Proper n-step returns with episode boundary masking
    - Consistent reward normalization (only once)
    - Prioritized experience replay support
    - Multi-agent joint sampling
    - Running statistics for normalization
    
    Critical Fixes:
    1. N-step returns stop accumulating after done=True
    2. Rewards normalized only during sampling, not again in training
    3. PER priorities correctly updated for all agents in joint mode
    """
    
    def __init__(self, camera_obs_dim, vector_obs_dim, action_dim, params):
        """
        Initialize experience buffer.
        
        Args:
            camera_obs_dim: Shape of camera observations (C, H, W)
            vector_obs_dim: Shape of vector observations (dim,)
            action_dim: Shape of action space (dim,)
            params: Dictionary with:
                - buffer_size: Maximum number of transitions
                - gamma: Discount factor
                - lambda_: GAE lambda (if using GAE)
        """
        self.buffer_size = params.get('buffer_size', 1_000_000)
        self.gamma = params.get('gamma', 0.99)
        self.lambda_ = params.get('lambda_', 0.95)
        
        # Running statistics for normalization
        self.camera_stat = RunningStat(camera_obs_dim)
        self.vector_stat = RunningStat(vector_obs_dim)
        self.reward_stat = RunningStat((1,))

        # Storage arrays
        self.camera_obs = np.zeros((self.buffer_size, *camera_obs_dim), dtype=np.uint8)
        self.next_camera_obs = np.zeros((self.buffer_size, *camera_obs_dim), dtype=np.uint8)
        self.vector_obs = np.zeros((self.buffer_size, *vector_obs_dim), dtype=np.float32)
        self.next_vector_obs = np.zeros((self.buffer_size, *vector_obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, action_dim[0]), dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.priorities = np.ones(self.buffer_size, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def store(self, camera_obs, vector_obs, action, reward, next_camera_obs, next_vector_obs, done, priority=1.0):
        """
        Store a single transition.
        
        Args:
            camera_obs: Current camera observation
            vector_obs: Current vector observation
            action: Action taken
            reward: Reward received
            next_camera_obs: Next camera observation
            next_vector_obs: Next vector observation
            done: Episode termination flag
            priority: Priority for PER (default 1.0)
        """
        idx = self.ptr % self.buffer_size
        
        # Store transition
        self.camera_obs[idx] = camera_obs
        self.vector_obs[idx] = vector_obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_camera_obs[idx] = next_camera_obs
        self.next_vector_obs[idx] = next_vector_obs
        self.dones[idx] = done
        self.priorities[idx] = priority

        # Update statistics
        self.camera_stat.update(camera_obs)
        self.vector_stat.update(vector_obs)
        self.reward_stat.update(reward)

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def store_joint(self, camera_obs, vector_obs, action, reward, next_camera_obs, next_vector_obs, done, priority=1.0, num_agents=4):
        """
        Store transitions for all agents in a multi-agent environment.
        
        Args:
            camera_obs: (N, C, H, W) observations for N agents
            vector_obs: (N, vec_dim) observations for N agents
            action: (N, act_dim) actions for N agents
            reward: (N,) rewards for N agents
            next_camera_obs: (N, C, H, W) next observations
            next_vector_obs: (N, vec_dim) next observations
            done: (N,) done flags for N agents
            priority: Priority for PER
            num_agents: Number of agents
        """
        for i in range(num_agents):
            self.store(
                camera_obs[i],
                vector_obs[i],
                action[i],
                reward[i],
                next_camera_obs[i],
                next_vector_obs[i],
                done[i],
                priority
            )

    def _compute_nstep_returns(self, batch_idxs, n_step):
        """
        Compute n-step returns with PROPER episode boundary masking.
        
        CRITICAL FIX: Stops accumulating rewards after episode termination.
        
        Args:
            batch_idxs: Indices of sampled transitions
            n_step: Number of steps to look ahead
        
        Returns:
            nstep_returns: (batch_size, 1) n-step discounted returns
            idx_n: (batch_size,) indices of n-step next states
        """
        k = np.arange(n_step, dtype=np.int32)
        idxs_k = np.minimum(batch_idxs[:, None] + k, self.size - 1)
        
        # Get rewards and dones for n future steps
        r_k = self.rewards[idxs_k]
        d_k = self.dones[idxs_k]
        
        # Compute discount factors
        discounts = (self.gamma ** k).astype(np.float32)
        
        # CRITICAL FIX: Proper masking for episode boundaries
        # mask[i, j] = 0 if episode ended at any step < j
        mask = np.ones_like(d_k, dtype=np.float32)
        for i in range(1, n_step):
            mask[:, i] = mask[:, i-1] * (1.0 - d_k[:, i-1])
        
        # Compute n-step returns with proper masking
        # This ensures we don't include rewards after episode termination
        nstep_returns = (r_k * discounts * mask).sum(axis=1, keepdims=True)
        
        # Index of n-step next state
        idx_n = np.minimum(batch_idxs + n_step, self.size - 1)
        
        return nstep_returns, idx_n

    def sample(self, batch_size, alpha=0.6, beta=0.4, n_step=1):
        """
        Sample a batch of transitions with prioritized experience replay.
        
        Args:
            batch_size: Number of transitions to sample
            alpha: PER exponent for priority sampling
            beta: PER exponent for importance sampling weights
            n_step: Number of steps for n-step returns
        
        Returns:
            batch: Dictionary containing:
                - camera_obs, vector_obs, actions, rewards, dones
                - next_camera_obs, next_vector_obs
                - indices, weights (for PER)
                - nstep_returns, nstep_next_idxs
        """
        if self.size == 0:
            return None

        # Compute sampling probabilities based on priorities
        scaled_priorities = self.priorities[:self.size] ** alpha
        prob_sum = scaled_priorities.sum()
        
        if prob_sum == 0 or np.isnan(scaled_priorities).any():
            probs = np.full(self.size, 1.0 / self.size, dtype=np.float32)
        else:
            probs = scaled_priorities / prob_sum

        # Sample indices
        batch_idxs = np.random.choice(
            self.size,
            batch_size,
            replace=self.size < batch_size,
            p=probs
        )
        
        # Compute n-step returns with proper masking
        nstep_returns, idx_n = self._compute_nstep_returns(batch_idxs, n_step)
        
        # Compute importance sampling weights for PER
        weights = (self.size * probs[batch_idxs]) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)[:, None]

        # Normalization functions
        norm_cam = lambda x: (x - self.camera_stat.mean) / (self.camera_stat.std + 1e-8)
        norm_vec = lambda x: (x - self.vector_stat.mean) / (self.vector_stat.std + 1e-8)

        # CRITICAL FIX: Reward normalization happens HERE, not in training loop
        # This prevents double normalization
        rewards_normalized = _norm_reward(
            self.rewards[batch_idxs, None],
            self.dones[batch_idxs, None],
            self.reward_stat.mean,
            self.reward_stat.std
        )

        return dict(
            camera_obs=norm_cam(self.camera_obs[batch_idxs]),
            vector_obs=norm_vec(self.vector_obs[batch_idxs]),
            actions=self.actions[batch_idxs],
            rewards=rewards_normalized,  # Already normalized!
            next_camera_obs=norm_cam(self.next_camera_obs[batch_idxs]),
            next_vector_obs=norm_vec(self.next_vector_obs[batch_idxs]),
            dones=self.dones[batch_idxs, None],
            indices=batch_idxs,
            weights=weights,
            nstep_returns=nstep_returns,  # Already computed with proper masking
            nstep_next_idxs=idx_n
        )
    
    def sample_joint(self, batch_size, alpha=0.6, beta=0.4, n_step=1, num_agents=4):
        """
        Sample joint transitions for multi-agent learning.
        
        Samples complete environment steps (all agents together).
        
        Args:
            batch_size: Number of environment steps to sample
            alpha: PER exponent for priority sampling
            beta: PER exponent for importance sampling weights
            n_step: Number of steps for n-step returns
            num_agents: Number of agents
        
        Returns:
            batch: Dictionary with all data reshaped as (batch_size, num_agents, ...)
        """
        assert self.size % num_agents == 0, \
            "Buffer size must be divisible by number of agents."
        
        E = self.size // num_agents  # Number of environment steps

        # Compute environment-level priorities (max over agents)
        env_priorities = self.priorities[:E * num_agents].reshape(E, num_agents).max(axis=1)
        probs = env_priorities ** alpha
        probs_sum = probs.sum()
        
        if probs_sum == 0 or np.isnan(probs).any():
            probs = np.ones_like(env_priorities) / len(env_priorities)
        else:
            probs /= probs.sum()

        # Sample environment indices
        env_idxs = np.random.choice(E, batch_size, replace=E < batch_size, p=probs)

        # Convert to flat indices for all agents
        idx_matrix = env_idxs[:, None] * num_agents + np.arange(num_agents)
        idx_flat = idx_matrix.reshape(-1)
        
        # Compute importance sampling weights
        weights = (E * probs[env_idxs]) ** (-beta)
        weights /= weights.max()
        weights = np.repeat(weights, num_agents).astype(np.float32).reshape(batch_size, num_agents, 1)

        # CRITICAL FIX: Proper n-step returns with episode masking
        k = np.arange(n_step, dtype=np.int32)
        idxs_k = np.minimum(idx_flat[:, None] + k, self.size - 1)
        
        r_k = self.rewards[idxs_k]
        d_k = self.dones[idxs_k]
        
        discounts = (self.gamma ** k).astype(np.float32)
        
        # Proper mask: accumulate only until episode ends
        mask = np.ones_like(d_k, dtype=np.float32)
        for i in range(1, n_step):
            mask[:, i] = mask[:, i-1] * (1.0 - d_k[:, i-1])
        
        n_step_ret = (r_k * discounts * mask).sum(axis=1, keepdims=True)
        
        idx_n_flat = np.minimum(idx_flat + n_step, self.size - 1)

        # Normalization functions
        def norm_cam(x):
            return (x - self.camera_stat.mean) / (self.camera_stat.std + 1e-8)
        
        def norm_vec(x):
            return (x - self.vector_stat.mean) / (self.vector_stat.std + 1e-8)
        
        # CRITICAL FIX: Normalize rewards once here, not in training loop
        rewards_normalized = _norm_reward(
            self.rewards[idx_flat],
            self.dones[idx_flat],
            self.reward_stat.mean,
            self.reward_stat.std
        ).reshape(batch_size, num_agents, 1)
        
        batch = dict(
            camera_obs=norm_cam(self.camera_obs[idx_flat]).reshape(
                batch_size, num_agents, *self.camera_obs.shape[1:]
            ),
            vector_obs=norm_vec(self.vector_obs[idx_flat]).reshape(
                batch_size, num_agents, -1
            ),
            actions=self.actions[idx_flat].reshape(batch_size, num_agents, -1),
            rewards=rewards_normalized,  # Already normalized!
            next_camera_obs=norm_cam(self.next_camera_obs[idx_flat]).reshape(
                batch_size, num_agents, *self.camera_obs.shape[1:]
            ),
            next_vector_obs=norm_vec(self.next_vector_obs[idx_flat]).reshape(
                batch_size, num_agents, -1
            ),
            dones=self.dones[idx_flat].reshape(batch_size, num_agents, 1),
            indices=idx_flat.reshape(batch_size, num_agents),
            weights=weights.astype(np.float32),
            nstep_returns=n_step_ret.reshape(batch_size, num_agents, 1),
            nstep_next_idxs=idx_n_flat.reshape(batch_size, num_agents),
        )

        return batch
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled transitions (for PER).
        
        Args:
            indices: Indices of transitions to update
            priorities: New priority values
        """
        assert indices.shape[0] == priorities.shape[0], \
            "Indices and priorities must have the same length."
        
        # Ensure priorities are valid
        priorities = np.maximum(priorities, 1e-6)  # Avoid zero priorities
        self.priorities[indices] = priorities.astype(np.float32)

    def __len__(self):
        """Return current buffer size."""
        return self.size

    def save(self, path):
        """
        Save buffer to disk.
        
        Args:
            path: File path for saving
        """
        np.savez_compressed(
            path,
            camera_obs=self.camera_obs[:self.size],
            vector_obs=self.vector_obs[:self.size],
            actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            next_camera_obs=self.next_camera_obs[:self.size],
            next_vector_obs=self.next_vector_obs[:self.size],
            dones=self.dones[:self.size],
            priorities=self.priorities[:self.size],
            ptr=self.ptr,
            size=self.size
        )

    def load(self, path):
        """
        Load buffer from disk.
        
        Args:
            path: File path for loading
        """
        data = np.load(path)
        
        # Load arrays
        for k in ('camera_obs', 'vector_obs', 'actions', 'rewards',
                  'next_camera_obs', 'next_vector_obs', 'dones', 'priorities'):
            loaded_data = data[k]
            getattr(self, k)[:len(loaded_data)] = loaded_data
        
        # Load metadata
        self.size = int(data['size'])
        self.ptr = int(data['ptr'])

        # Update statistics
        self.camera_stat.update(self.camera_obs[:self.size])
        self.vector_stat.update(self.vector_obs[:self.size])
        self.reward_stat.update(self.rewards[:self.size])


