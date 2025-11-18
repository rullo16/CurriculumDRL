"""
Key Components:
1. RolloutBuffer - Stores trajectories and computes advatages using GAE
2. MAPPOActor - Decentralized policy
3. MAPPOCritic - Centralized value function
4. MAPPOAgent - Main agent class
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import copy
from .models import MAPPOActor, MAPPOCritic
from .rollout_buffer import RolloutBuffer
from .vision_encoders import EfficientVisionEncoder
from .curiosity import CuriosityModule, RunningMeanStd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KLDivergenceEntropyScheduler:
    def __init__ (self):
        self.target_kl = 0.02
        self.entropy_coef = 0.1
        self.min_entropy_coef = 0.001

    def update(self, current_kl):
        if current_kl > self.target_kl:
            self.entropy_coef = min(self.entropy_coef * 1.5, 0.10)
        elif current_kl < self.target_kl * 0.5:
            self.entropy_coef = max(self.entropy_coef * 0.9, self.min_entropy_coef)
        
        return self.entropy_coef
    
    def switch_env(self):
        self.entropy_coef = 0.1

class MAPPOAgent:
    """
    MAPPO agent

    Uses featureextraction for vision processing, handles both camera and vector observations, 
    implements full MAPPO algorithm with PPO Loss and supports multi-agent training
    """

    def __init__(self, camera_shape, vector_shape, action_dim, num_agents, config, feature_extractor=None):
        """
        camera_shape: Shape of camera observations (C, H, W)
        vector_shape: Shape of vector observations
        action_dim: Dimension of action space
        num_agents: Number of agents
        config: Dictionary with hyperparameters
        feature_extractor: Optional pre-trained feature extractor
        """
        
        
        self.device = device
        self.camera_shape = camera_shape
        self.vector_shape = vector_shape
        self.action_dim = action_dim
        self.num_agents = num_agents

        self.training = False
        
        
        #Extract Hyperparams
        self.lr = config.get('learning_rate', 1e-4)
        self.clip_param = config.get('clip_param', 0.2)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.001)
        self.max_grad_norm = config.get('max_grad_norm',0.5)
        self.gamma = config.get('gamma',0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.num_steps = config.get('num_steps', 2048)
        self.num_mini_batches = config.get('num_mini_batches', 8)
        self.ppo_epochs = config.get('ppo_epochs', 4)

        self.curiosity_coef = config.get('curiosity_coef', 0.01)
        self.reward_clip = config.get('reward_clip', 10.0)


        #Import existing feature extractor
        # if feature_extractor is None:
        #     from SAC_Distillation.Nets import FeatureExtractionNet
        #     self.vision_encoder = FeatureExtractionNet(
        #         camera_shape,
        #         distilled_dim = 2048
        #     ).to(device)

        # else:
        #     self.vision_encoder = feature_extractor.to(device)

        self.vision_encoder = EfficientVisionEncoder(
            input_shape=camera_shape,
            output_dim=512
        ).to(device)

        self.vision_encoder._distillation_done=True
        #Vector observation processor
        self.vector_processor = nn.Sequential(
            nn.LayerNorm(vector_shape[0]),
            nn.Linear(vector_shape[0], 128),
            nn.Tanh()
        ).to(device)

        #Calculate encoded observation dim, vision features + vector features
        self.encoded_obs_dim = 256+128

        #Init actor and critic

        self.actor = MAPPOActor(self.encoded_obs_dim, action_dim).to(device)
        self.critic = MAPPOCritic(self.encoded_obs_dim*num_agents, num_agents=num_agents).to(device)
        
        #Curiosity module
        self.curiosity_module = CuriosityModule(
            obs_dim=self.encoded_obs_dim,
            action_dim=action_dim,
            hidden_dim=512
        ).to(device)

        self.intrinsic_reward_normalizer = RunningMeanStd()

        #Single optimizer for all networks, simpler than having separate optimizers
        self.optimizer = optim.Adam([
            {'params': self.vision_encoder.parameters(), 'lr': self.lr*0.5}, #Slower for pretrained
            {'params': self.vector_processor.parameters(), 'lr': self.lr},
            {'params': self.actor.parameters(), 'lr': self.lr},
            {'params': self.critic.parameters(), 'lr': self.lr},
            {'params': self.curiosity_module.parameters(), 'lr': self.lr},
        ], eps=1e-5)

        #Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max = config.get('max_steps', 3000000) // self.num_steps,
            eta_min=self.lr*0.1
        )

        self.entropy_scheduler = KLDivergenceEntropyScheduler()

        #Statistics tracking
        self.episode_rewards = []
        self.episode_length = []

        #Init weights
        self._init_weights()

    def _init_weights(self):
        for m in self.vector_processor.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode_observations(self, camera_obs, vector_obs):
        """
        Encode raw observations using vision encoder + vector processor

        camera_obs: (num_agents, C, H, W) - raw camera observations
        vector_obs: (num_agents, vector_dim) - vector observations

        returns encoded_obs: (num_agents, encoded_dim) - encoded observations
        """

        if isinstance(camera_obs, np.ndarray):
            camera_obs = torch.from_numpy(camera_obs).float().to(self.device)
        
        #Normalize
        if camera_obs.dtype == torch.uint8:
            camera_obs = camera_obs.float() / 255.0
        elif camera_obs.max() > 1.5:
            camera_obs = camera_obs / 255.0

        #Extract visual features
        with torch.no_grad() if not self.training else torch.enable_grad():
            visual_features = self.vision_encoder(camera_obs)

        #Process vector obs
        if isinstance(vector_obs, np.ndarray):
            vector_obs = torch.from_numpy(vector_obs).float().to(self.device)

        vector_features = self.vector_processor(vector_obs)

        #Concatenate
        encoded_obs = torch.cat([visual_features, vector_features], dim=-1)

        return encoded_obs
    
    def compute_intrinsic_rewards(self, obs, next_obs, actions):
        """
        Compute intrinsic rewards using curiosity module

        obs: (num_agents, obs_dim) - current encoded observations
        next_obs: (num_agents, obs_dim) - next encoded observations
        actions: (num_agents, action_dim) - actions taken

        returns intrinsic_rewards: (num_agents,) - intrinsic rewards
        """
        with torch.no_grad():
            intrinsic_rewards, _ = self.curiosity_module(obs, next_obs, actions)
            intrinsic_rewards = intrinsic_rewards.detach().cpu().numpy()

            self.intrinsic_reward_normalizer.update(intrinsic_rewards)
            intrinsic_rewards = intrinsic_rewards / (self.intrinsic_reward_normalizer.std + 1e-8)

            intrinsic_rewards = np.clip(intrinsic_rewards, -self.reward_clip, self.reward_clip)

        return intrinsic_rewards
        

    @torch.no_grad()
    def get_action(self, camera_obs, vector_obs, deterministic=False):
        """
        Get action for environment interaction.

        camera_obs: (num_agents, C, H, W) - camera observations
        vector_obs: (num_agents, vector_dim) - vector observations
        deterministic: If True, return mean action (for evaluation)

        returns:
        actions: (num_agents, action_dim) - actions to take
        log_probs: (num_agents,) - log probabilities
        values: (num_agents,) - value estimates
        """
        self.actor.eval()

        encoded_obs = self.encode_observations(camera_obs, vector_obs)

        #get actions from actor
        actions, log_probs = self.actor.get_action(encoded_obs, deterministic)

        #Get values from critic (use global state), during execution, batch all agents' observations.
        global_obs = encoded_obs.unsqueeze(0)
        values = self.critic(global_obs).squeeze(0)

        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy()
        )
    
    def train(self, rollout_buffer: RolloutBuffer):
        """
        Update policy using PPO

        Sample mini-batches from rollout, compute PPO loss, update all nets, repeat for multiple epochs

        rollout_buffer: Buffer containing collected trajectories

        returns stats: Dictionary with training statistics
        """

        self.training = True

        data = rollout_buffer.get()

        # ================================================================
        # STEP 1: Load data from buffer
        # ================================================================
        obs = torch.tensor(data['observations'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(data['actions'], dtype=torch.float32).to(self.device)
        returns = torch.tensor(data['returns'], dtype=torch.float32).to(self.device)
        advantages = torch.tensor(data['advantages'], dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(data['log_probs'], dtype=torch.float32).to(self.device)
        
        # ================================================================
        # STEP 2: Debug - Print shapes to verify data structure
        # ================================================================
        print(f"\n{'='*70}")
        print(f"DEBUG: Buffer returned shapes:")
        print(f"  obs:           {obs.shape}")
        print(f"  actions:       {actions.shape}")
        print(f"  returns:       {returns.shape}")
        print(f"  advantages:    {advantages.shape}")
        print(f"  old_log_probs: {old_log_probs.shape}")
        print(f"\nExpected shapes:")
        print(f"  obs:           ({self.num_steps}, {self.num_agents}, {self.encoded_obs_dim})")
        print(f"  actions:       ({self.num_steps}, {self.num_agents}, {self.action_dim})")
        print(f"  returns:       ({self.num_steps}, {self.num_agents})")
        print(f"{'='*70}\n")
        
        # ================================================================
        # STEP 3: CRITICAL FIX - Reshape flattened buffer data
        # ================================================================
        # Buffer returns: (total_samples, feature_dim) = (2048, 384)
        # We need: (num_steps, num_agents, feature_dim) = (512, 4, 384)
        
        if len(obs.shape) == 2:  # Flattened format detected
            total_samples = obs.shape
            expected_total = self.num_steps * self.num_agents
            
            if total_samples != expected_total:
                raise ValueError(
                    f"Buffer size mismatch! Got {total_samples} samples, "
                    f"expected {expected_total} (num_steps={self.num_steps} × num_agents={self.num_agents})"
                )
            
            print("⚠️  Reshaping flattened buffer data...")
            obs = obs.reshape(self.num_steps, self.num_agents, self.encoded_obs_dim)
            actions = actions.reshape(self.num_steps, self.num_agents, self.action_dim)
            returns = returns.reshape(self.num_steps, self.num_agents)
            advantages = advantages.reshape(self.num_steps, self.num_agents)
            old_log_probs = old_log_probs.reshape(self.num_steps, self.num_agents)
            print(f"✓ Reshaped obs to: {obs.shape}\n")
        
        # ================================================================
        # STEP 4: Validate data integrity
        # ================================================================
        if torch.isnan(obs).any():
            raise ValueError(f"NaN detected in observations! Count: {torch.isnan(obs).sum()}")
        if torch.isinf(obs).any():
            raise ValueError(f"Inf detected in observations!")
        if torch.isnan(advantages).any():
            raise ValueError(f"NaN detected in advantages!")
        
        # ================================================================
        # STEP 5: Normalize advantages ONCE
        # ================================================================
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # ================================================================
        # STEP 6: Setup mini-batch training
        # ================================================================
        num_steps = obs.shape[0]  # NOW correct: 512 (not 2048!)
        timesteps_per_minibatch = num_steps // self.num_mini_batches

        #Training stats
        stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl':[],
            'clip_fraction': [],
            'explained_variance': []
        }

        #Multiple epochs over the same data (PPO reuse data a few times)
        for epoch in range(self.ppo_epochs):
            #Shuffle data
            indices = torch.randperm(num_steps, device=self.device)

            if epoch > 0 and len(stats['approx_kl'])>0:
                mean_kl = np.mean(stats['approx_kl'])
                if mean_kl > 0.03:
                    print(f"Early stopping at epoch {epoch} due to reaching max KL.")
                    break

            #Mini-batch updates
            for start in range(0, num_steps, timesteps_per_minibatch):

                end = min(start + timesteps_per_minibatch, num_steps)
                mb_indices = indices[start:end]

                #Get mini-batch
                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]

                #Policy Loss

                mb_obs_flat = mb_obs.reshape(-1, self.encoded_obs_dim)
                mb_actions_flat = mb_actions.reshape(-1, self.action_dim)
                mb_advantages_flat = mb_advantages.reshape(-1)
                mb_old_log_probs_flat = mb_old_log_probs.reshape(-1)
                mb_returns_flat = mb_returns.reshape(-1)

                if torch.isnan(mb_advantages_flat).any():
                    raise ValueError("NaN detected in advantages!")
                if torch.isnan(mb_obs_flat).any():
                    raise ValueError("NaN detected in observations!")

                #eval actions under current policy
                log_probs, entropy = self.actor.evaluate_actions(mb_obs_flat, mb_actions_flat)

                #Compute probability ration: π_new / π_old
                ratio = torch.exp(log_probs - mb_old_log_probs_flat)

                #PPO objective
                # Maximize J = E[min(r*A, clip(r)*A)]
                #where r = π_new/π_old,
                surr1 = ratio * mb_advantages_flat
                surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * mb_advantages_flat

                #Take min
                policy_loss = -torch.min(surr1,surr2).mean()

                #Value Loss (MSE)

                # mb_obs_global = mb_obs

                values_pred = self.critic(mb_obs).reshape(-1)

                value_loss = F.mse_loss(values_pred, mb_returns_flat)

                #Entropy Loss (Exploration bonus)
                #Encourage exploration by maximising entropy
                #High entropy = more rnd actions = more exploration
                entropy_loss = -entropy.mean()

                curiosity_batch_size = min(512, len(mb_obs_flat))
                curiosity_indices = torch.randperm(len(mb_obs_flat), device=self.device)[:curiosity_batch_size]

                curiosity_obs = mb_obs_flat[curiosity_indices]
                curiosity_actions = mb_actions_flat[curiosity_indices]

                next_indices = (curiosity_indices + 1) % len(mb_obs_flat)
                curiosity_next_obs = mb_obs_flat[next_indices]

                _, curiosity_loss = self.curiosity_module(
                    curiosity_obs,
                    curiosity_next_obs,
                    curiosity_actions
                )

                #Total loss
                loss = (
                    policy_loss +
                    self.value_loss_coef * value_loss +
                    self.entropy_coef * entropy_loss +
                    self.curiosity_coef * curiosity_loss
                )

                #Optimization
                self.optimizer.zero_grad()
                loss.backward()

                #Clip grads to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters())+
                                               list(self.critic.parameters())+
                                               list(self.vision_encoder.parameters())+
                                               list(self.vector_processor.parameters()),
                                               self.max_grad_norm)
                
                self.optimizer.step()

                #Logs
                with torch.no_grad():
                    #Approximate KL divergence (for monitoring)
                    approx_kl = ((log_probs - mb_old_log_probs_flat).pow(2)*0.5).mean()

                    #Clip fraction (how often we clip)
                    clip_fraction = ((ratio - 1.0).abs()> self.clip_param).float().mean()

                    #Explained variance (how well value function fits returns)
                    y_pred = values_pred.cpu().numpy()
                    y_true = mb_returns_flat.cpu().numpy()
                    var_y = np.var(y_true)
                    explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
                    explained_var = np.clip(explained_var, -1.0, 1.0)

                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['entropy'].append(entropy.mean().item())
                stats['approx_kl'].append(approx_kl.item())
                stats['clip_fraction'].append(clip_fraction.item())
                stats['explained_variance'].append(explained_var)

        self.scheduler.step()
        self.entropy_coef = self.entropy_scheduler.update(np.mean(stats['approx_kl']))
        
        #Return averaged statistics
        return {k: np.mean(v) for k,v in stats.items()}
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'vision_encoder': self.vision_encoder.state_dict(),
            'vector_processor': self.vector_processor.state_dict(),
            'curiosity_module': self.curiosity_module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.vision_encoder.load_state_dict(checkpoint['vision_encoder'])
        self.vector_processor.load_state_dict(checkpoint['vector_processor'])
        self.curiosity_module.load_state_dict(checkpoint['curiosity_module'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"Model loaded from {path}")



