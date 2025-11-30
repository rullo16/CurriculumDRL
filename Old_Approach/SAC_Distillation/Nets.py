"""
Fixed Neural Network Architectures for Multi-Agent SAC
======================================================

This module contains all neural network architectures with critical bug fixes applied:
1. CentralizedCriticNet: Uses MEAN aggregation (not SUM)
2. RND: Proper statistics update timing (after loss computation)
3. All other components maintained with improvements

Critical Fixes Applied:
- Q-value aggregation using mean prevents value explosion
- RND stats update only via explicit method call, never in forward()
- Proper observation normalization with clamping
- Spectral normalization applied consistently (or removed where unnecessary)

Author: Fixed Implementation
Date: November 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as nn_utils
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# INITIALIZATION UTILITIES
# ============================================================================

def safe_xavier_initialization(module, gain=1.0):
    """
    Safely initialize module weights with Xavier uniform initialization.
    Handles uninitialized parameters gracefully.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        if module.weight is not None:
            if not isinstance(module.weight, nn.parameter.UninitializedParameter):
                nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            if not isinstance(module.bias, nn.parameter.UninitializedParameter):
                nn.init.constant_(module.bias, 0)
    return module


def safe_orthogonal_initialization(module, gain=nn.init.calculate_gain('relu')):
    """
    Safely initialize module weights with orthogonal initialization.
    Handles uninitialized parameters gracefully.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        if isinstance(module.weight, nn.parameter.UninitializedParameter):
            return module
        if module.weight is not None:
            nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    return module


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class FeatureExtractionNet(nn.Module):
    """
    Convolutional feature extractor for visual observations.
    
    Supports distillation from teacher models via optional bottleneck.
    """
    def __init__(self, input_shape, distilled_dim=2048):
        super(FeatureExtractionNet, self).__init__()
        self.expected_hw = (input_shape[1], input_shape[2])
        
        # Standard CNN pipeline for visual features
        self.convolutional_pipeline = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate convolutional output dimension
        dummy = torch.zeros(1, *input_shape)
        conv_out = self.convolutional_pipeline(dummy).view(1, -1).shape[1]
        
        # Optional distillation bottleneck
        self.distilled_converter = nn.Linear(conv_out, distilled_dim)
        self.dropout = nn.Dropout(0.5)

        self.convolutional_pipeline.apply(safe_xavier_initialization)
        self._distillation_done = False

    def forward(self, x, distill=False):
        """
        Forward pass with optional distillation mode.
        
        Args:
            x: Image tensor (B, C, H, W), either uint8 [0,255] or float [0,1]
            distill: If True, apply dropout and return distilled features
        
        Returns:
            Features tensor (B, feature_dim)
        """
        # Normalize uint8 images to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()

        # Resize if needed
        if x.shape[-2:] != self.expected_hw:
            x = F.interpolate(x, size=self.expected_hw, mode='bilinear', align_corners=False)

        # Extract convolutional features
        conv_out = self.convolutional_pipeline(x).view(x.size(0), -1)

        # Return distilled features if distillation is complete
        if hasattr(self, '_distillation_done') and self._distillation_done:
            return self.distilled_converter(conv_out)
        
        # Apply dropout during distillation training
        if distill:
            conv_out = self.dropout(conv_out)
            return self.distilled_converter(conv_out)
        
        return conv_out


# ============================================================================
# RUNNING STATISTICS (for normalization)
# ============================================================================

class RunningStat:
    """
    Maintains running mean and variance using Welford's online algorithm.
    Numerically stable and memory efficient.
    """
    def __init__(self, shape, eps=1e-4):
        self._mean = torch.zeros(shape, dtype=torch.float32).to(device)
        self._var = torch.ones(shape, dtype=torch.float32).to(device)
        self._count = eps

    def update(self, x):
        """
        Update statistics with new batch of data.
        
        Uses Welford's online algorithm for numerical stability.
        """
        if x.device != device:
            x = x.to(device)
        
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        if batch_count == 0:
            return
        
        delta = batch_mean - self._mean
        tot_count = self._count + batch_count

        # Update mean
        self._mean += delta * batch_count / tot_count
        
        # Update variance using Welford's algorithm
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta.pow(2) * self._count * batch_count / tot_count)
        self._var = m2 / tot_count
        self._count = tot_count

    @property
    def mean(self):
        return self._mean
    
    @property
    def std(self):
        return torch.sqrt(self._var + 1e-8)


# ============================================================================
# RANDOM NETWORK DISTILLATION (FIXED)
# ============================================================================

class RND(nn.Module):
    """
    Random Network Distillation for Exploration Bonuses
    
    CRITICAL FIX: Statistics are updated ONLY via explicit update_obs_stats() call,
    NEVER during forward pass. This prevents distribution shift within training batches.
    
    Usage:
        rnd = RND(input_dim=256)
        
        # During training:
        intrinsic_reward = rnd(obs)  # Forward pass - NO stats update
        loss = rnd.compute_loss(obs)  # Compute loss - NO stats update
        optimizer.step()
        rnd.update_obs_stats(obs)  # Explicitly update stats AFTER training
    """
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, eps=1e-8):
        super().__init__()
        
        # Fixed random target network (never trained)
        self.target_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Trainable predictor network
        self.predictor_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Freeze target network permanently
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.target_net.eval()

        # Running statistics for observation normalization
        self.obs_normalizer = RunningStat(shape=(input_dim,))
        self.rew_normaliser = RunningStat(shape=(1,))
        self._eps = eps

    def _to_stat_tensor(self, arr, device, dtype):
        """Convert RunningStat arrays to tensors on correct device/dtype."""
        if isinstance(arr, torch.Tensor):
            return arr.to(device=device, dtype=dtype)
        return torch.as_tensor(arr, device=device, dtype=dtype)
    
    def _normalise_obs(self, x):
        """
        Z-score normalize observations using running statistics.
        Clamped to [-5, 5] for stability.
        """
        m = self._to_stat_tensor(self.obs_normalizer.mean, x.device, x.dtype)
        s = self._to_stat_tensor(self.obs_normalizer.std, x.device, x.dtype)
        x_norm = (x - m) / (s + self._eps)
        return x_norm.clamp(-5.0, 5.0)
    
    @torch.no_grad()
    def update_obs_stats(self, obs):
        """
        Update observation statistics.
        
        CRITICAL: This should ONLY be called AFTER computing loss/rewards,
        never during forward() or compute_loss()!
        
        Args:
            obs: Observations tensor
        """
        if obs.is_cuda:
            self.obs_normalizer.update(obs.detach().cpu())
        else:
            self.obs_normalizer.update(obs.detach())

    @torch.no_grad()
    def update_reward_stats(self, rewards):
        """Update reward normalization statistics."""
        if rewards.is_cuda:
            self.rew_normaliser.update(rewards.detach().cpu())
        else:
            self.rew_normaliser.update(rewards.detach())

    @torch.no_grad()
    def forward(self, x, normalise_reward=False):
        """
        Compute intrinsic reward based on prediction error.
        
        CRITICAL FIX: This method does NOT update statistics!
        
        Args:
            x: Observation tensor (B, input_dim)
            normalise_reward: If True, normalize reward using running stats
        
        Returns:
            intrinsic_reward: (B, 1) - prediction error as curiosity bonus
        """
        # Normalize observations using CURRENT statistics (no update)
        x_n = self._normalise_obs(x)
        
        # Compute target and predicted features
        t = self.target_net(x_n)
        p = self.predictor_net(x_n)
        
        # Intrinsic reward = MSE between predictor and target
        intr = (p - t).pow(2).mean(dim=1, keepdim=True)
        
        # Optional reward normalization (uses existing stats, no update)
        if normalise_reward:
            rm = self._to_stat_tensor(self.rew_normaliser.mean, intr.device, intr.dtype)
            rs = self._to_stat_tensor(self.rew_normaliser.std, intr.device, intr.dtype)
            intr = (intr - rm) / (rs + self._eps)
        
        return intr.clamp_min(1e-3)

    def compute_loss(self, obs):
        """
        Compute prediction loss for training the predictor network.
        
        CRITICAL FIX: This method does NOT update statistics!
        
        Args:
            obs: Observation tensor (B, input_dim)
        
        Returns:
            loss: MSE between predictor and target
        """
        if obs.numel() == 0:
            return obs.sum() * 0.0
        
        # Normalize using current stats (no update)
        x_n = self._normalise_obs(obs)
        
        # Compute target (no grad) and prediction
        with torch.no_grad():
            t = self.target_net(x_n)
        p = self.predictor_net(x_n)
        
        return F.mse_loss(p, t)


# ============================================================================
# CENTRALIZED CRITIC (FIXED)
# ============================================================================

class CentralizedCriticNet(nn.Module):
    """
    Centralized Critic for Multi-Agent SAC
    
    CRITICAL FIX: Uses MEAN aggregation instead of SUM for Q-values.
    
    Why this matters:
    - Summing Q-values creates arbitrary value scales that grow with num_agents
    - Breaks the Bellman equation target computation
    - Causes critic loss to explode during training
    - Mean aggregation keeps values in reasonable range
    
    Architecture:
    - Takes all agents' features and actions as input (centralized)
    - Processes through shared backbone
    - Outputs per-agent Q-values from separate heads
    - Aggregates using MEAN (not sum)
    """
    
    def __init__(self, per_agent_dim, action_dim, num_agents):
        super().__init__()
        self.num_agents = num_agents
        centralized_input_dim = num_agents * (per_agent_dim + action_dim)
        
        # Shared backbone for all agents
        self.backbone = nn.Sequential(
            nn.Linear(centralized_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        
        # Per-agent Q-heads for dual Q-learning
        # Each agent gets its own Q-value estimate
        self.q1_heads = nn.ModuleList([
            nn.Linear(256, 1) for _ in range(num_agents)
        ])
        self.q2_heads = nn.ModuleList([
            nn.Linear(256, 1) for _ in range(num_agents)
        ])
        
        self.backbone.apply(safe_xavier_initialization)
    
    def forward(self, feats, actions, return_per_agent=False):
        """
        Forward pass with correct aggregation.
        
        Args:
            feats: (B*N, feat_dim) or (B, N, feat_dim) - agent features
            actions: (B*N, act_dim) or (B, N, act_dim) - agent actions
            return_per_agent: If True, return (B, N, 1), else (B, 1)
        
        Returns:
            q1, q2: Q-value estimates
                - If return_per_agent: (B, N, 1) per-agent values
                - Else: (B, 1) aggregated values using MEAN
        """
        # Handle both batched and flat inputs
        if feats.dim() == 3:  # (B, N, feat_dim)
            batch_size = feats.shape[0]
            feats = feats.view(batch_size * self.num_agents, -1)
            actions = actions.view(batch_size * self.num_agents, -1)
        else:  # (B*N, feat_dim)
            batch_size = feats.shape[0] // self.num_agents
        
        # Centralize input: concatenate all agents' features and actions
        x_per_agent = torch.cat([feats, actions], dim=-1)
        x_centralized = x_per_agent.view(batch_size, -1)
        
        # Shared representation
        h = self.backbone(x_centralized)  # (B, 256)
        
        # Per-agent Q-values from separate heads
        q1_list = [head(h) for head in self.q1_heads]  # List of (B, 1)
        q2_list = [head(h) for head in self.q2_heads]
        
        # Stack to (B, N, 1)
        q1 = torch.cat(q1_list, dim=1).view(batch_size, self.num_agents, 1)
        q2 = torch.cat(q2_list, dim=1).view(batch_size, self.num_agents, 1)
        
        if return_per_agent:
            return q1, q2
        
        # CRITICAL FIX: Use MEAN instead of SUM
        # This prevents value explosion and maintains proper Bellman equation
        q1_mean = q1.mean(dim=1)  # (B, 1)
        q2_mean = q2.mean(dim=1)  # (B, 1)
        
        return q1_mean, q2_mean


# ============================================================================
# SINGLE-AGENT CRITIC (for comparison)
# ============================================================================

class CriticNet(nn.Module):
    """
    Standard single-agent critic network.
    Dual Q-learning with spectral normalization on output layers.
    """
    def __init__(self, input_dim, hidden_dim=256):
        super(CriticNet, self).__init__()
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        mid = hidden_dim // 2
        
        # Q1 head with spectral normalization on final layer
        self.q1 = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.LayerNorm(mid),
            nn.ReLU(inplace=True),
            nn_utils.spectral_norm(nn.Linear(mid, 1))
        )

        # Q2 head with spectral normalization on final layer
        self.q2 = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.LayerNorm(mid),
            nn.ReLU(inplace=True),
            nn_utils.spectral_norm(nn.Linear(mid, 1))
        )

        self.q1.apply(safe_xavier_initialization)
        self.q2.apply(safe_xavier_initialization)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: State-action features (B, input_dim)
        
        Returns:
            q1, q2: Dual Q-value estimates (B, 1)
        """
        x = self.backbone(x)
        return self.q1(x), self.q2(x)


# ============================================================================
# POLICY NETWORK
# ============================================================================

class GaussianPolicy(nn.Module):
    """
    Gaussian policy for continuous action spaces.
    Outputs mean and log_std for action distribution.
    """
    def __init__(self, input_dim, act_dim):
        super().__init__()
        self.mu = nn.Linear(input_dim, act_dim)
        self.log_std = nn.Linear(input_dim, act_dim)
        
        # Initialize with small weights for stable initial policy
        self.apply(safe_xavier_initialization)

    def forward(self, x):
        """
        Compute action distribution parameters.
        
        Args:
            x: State features (B, input_dim)
        
        Returns:
            dist: torch.distributions.Normal object
        """
        x = x.float()
        
        # Compute mean and log_std
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), min=-20, max=2)
        std = torch.exp(log_std).clamp(min=1e-6)
        
        # Create Gaussian distribution
        dist = torch.distributions.Normal(mu, std)
        return dist


# ============================================================================
# COMPLETE SAC NETWORK
# ============================================================================

class SACNet(nn.Module):
    """
    Complete SAC network combining visual and vector observations.
    
    Architecture:
    - Visual features: CNN feature extractor
    - Vector features: MLP processor
    - Combined features: Shared backbone
    - Output: Gaussian policy distribution
    """
    
    def __init__(self, camera_obs_dim, vector_obs_dim, n_actions):
        super(SACNet, self).__init__()
        self.camera_obs_dim = camera_obs_dim
        self.vector_obs_dim = vector_obs_dim
        self.feat_dim = 256
        self.n_actions = n_actions

        # Visual feature extractor
        self.convolution_pipeline = FeatureExtractionNet(
            camera_obs_dim,
            distilled_dim=2048
        )
        self.conv_out_size = self._get_conv_out(camera_obs_dim)

        # Vector observation processor
        self.vector_processor = nn.Sequential(
            nn.LayerNorm(vector_obs_dim[0]),
            nn.Linear(vector_obs_dim[0], 128),
            nn.ReLU(inplace=True),
        )

        # Combined feature backbone
        combined_input_dim = self.conv_out_size + 128
        self.backbone = nn.Sequential(
            nn.Linear(combined_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
            nn.ReLU(inplace=True),
        )

        # Policy head
        self.policy_head = GaussianPolicy(self.feat_dim, self.n_actions)

        # Initialize weights
        self.vector_processor.apply(safe_xavier_initialization)
        self.backbone.apply(safe_xavier_initialization)
        self.policy_head.apply(safe_xavier_initialization)

    def _get_conv_out(self, shape):
        """Calculate convolutional output dimension."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output = self.convolution_pipeline(dummy_input)
        return int(np.prod(output.size()[1:]))
    
    def encode(self, camera_obs, vector_obs):
        """
        Encode camera and vector observations into features.
        
        Args:
            camera_obs: Visual observations (B, C, H, W)
            vector_obs: Vector observations (B, vec_dim)
        
        Returns:
            feats: Combined features (B, feat_dim)
        """
        # Extract visual features
        cam = self.convolution_pipeline(camera_obs)
        
        # Process vector observations
        vec_feat = self.vector_processor(vector_obs)
        
        # Combine and process through backbone
        feats = self.backbone(torch.cat([cam, vec_feat], dim=1))
        
        return feats
    
    def dist_from_feats(self, feats):
        """
        Get action distribution from features.
        
        Args:
            feats: Features (B, feat_dim)
        
        Returns:
            dist: Gaussian distribution over actions
        """
        return self.policy_head(feats)
    
    def forward(self, camera_obs, vector_obs):
        """
        Forward pass from observations to action distribution.
        
        Args:
            camera_obs: Visual observations (B, C, H, W)
            vector_obs: Vector observations (B, vec_dim)
        
        Returns:
            dist: Gaussian distribution over actions
        """
        feats = self.encode(camera_obs, vector_obs)
        return self.dist_from_feats(feats)


# ============================================================================
# ENTROPY TARGET NETWORK (OPTIONAL)
# ============================================================================

class EntropyTargetNet(nn.Module):
    """
    Learned entropy target for SAC temperature tuning.
    Maps episode progress to target entropy value.
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Compute target entropy based on input signal.
        
        Args:
            x: Input signal (e.g., episode progress)
        
        Returns:
            target_entropy: Scaled entropy target
        """
        return -2.0 * self.fc(x)

