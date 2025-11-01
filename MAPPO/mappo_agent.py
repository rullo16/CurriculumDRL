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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.lr = config.get('learning_rate', 3e-4)
        self.clip_param = config.get('clip_param', 0.2)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm',0.5)
        self.gamma = config.get('gamma',0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.num_steps = config.get('num_steps', 2048)
        self.num_mini_batches = config.get('num_mini_batches', 8)
        self.ppo_epochs = config.get('ppo_epochs', 4)

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
            output_dim=256
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

        #Single optimizer for all networks, simpler than having separate optimizers
        self.optimizer = optim.Adam([
            {'params': self.vision_encoder.parameters(), 'lr': self.lr*0.5}, #Slower for pretrained
            {'params': self.vector_processor.parameters(), 'lr': self.lr},
            {'params': self.actor.parameters(), 'lr': self.lr},
            {'params': self.critic.parameters(), 'lr': self.lr},
        ], eps=1e-5)

        #Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max = config.get('max_steps', 3000000) // self.num_steps,
            eta_min=self.lr*0.1
        )

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

        #Convert to tensors
        obs = torch.tensor(data['observations']).to(self.device)
        actions = torch.tensor(data['actions']).to(self.device)
        returns = torch.tensor(data['returns']).to(self.device)
        advantages = torch.tensor(data['advantages']).to(self.device)
        old_log_probs = torch.tensor(data['log_probs']).to(self.device)

        #Flatten time and agents dims for easier processing
        num_steps = obs.shape[0]
        obs_flat = obs.reshape(-1, self.encoded_obs_dim) # numsteps * num_Agents, obs_dim
        actions_flat = actions.reshape(-1, self.action_dim)
        returns_flat = returns.reshape(-1)
        advantages_flat = advantages.reshape(-1)
        old_log_probs_flat = old_log_probs.reshape(-1)

        #Calculate total samples and mini-batch size
        total_samples = num_steps*self.num_agents
        mini_batch_size = total_samples // self.num_mini_batches

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
            indices = torch.randperm(total_samples, device=self.device)

            #Mini-batch updates
            for start in range(0, total_samples, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]

                #Get mini-batch
                mb_obs = obs_flat[mb_indices]
                mb_actions = actions_flat[mb_indices]
                mb_returns = returns_flat[mb_indices]
                mb_advantages = advantages_flat[mb_indices]
                mb_old_log_probs = old_log_probs_flat[mb_indices]

                #Policy Loss

                #eval actions under current policy
                log_probs, entropy = self.actor.evaluate_actions(mb_obs, mb_actions)

                #Compute probability ration: π_new / π_old
                ratio = torch.exp(log_probs - mb_old_log_probs)

                #PPO objective
                # Maximize J = E[min(r*A, clip(r)*A)]
                #where r = π_new/π_old,
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * mb_advantages

                #Take min
                policy_loss = -torch.min(surr1,surr2).mean()

                #Value Loss (MSE)

                mb_obs_reshaped = mb_obs.view(-1, self.num_agents, self.encoded_obs_dim)
                mb_obs_global = mb_obs_reshaped

                values_pred = self.critic(mb_obs_global).reshape(-1)

                value_loss = F.mse_loss(values_pred, mb_returns)

                #Entropy Loss (Exploration bonus)
                #Encourage exploration by maximising entropy
                #High entropy = more rnd actions = more exploration
                entropy_loss = -entropy.mean()

                #Total loss
                loss = (
                    policy_loss +
                    self.value_loss_coef * value_loss +
                    self.entropy_coef * entropy_loss
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
                    approx_kl = (log_probs - mb_old_log_probs).mean()

                    #Clip fraction (how often we clip)
                    clip_fraction = ((ratio - 1.0).abs()> self.clip_param).float().mean()

                    #Explained variance (how well value function fits returns)
                    y_pred = values_pred.cpu().numpy()
                    y_true = mb_returns.cpu().numpy()
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
        
        #Return averaged statistics
        return {k: np.mean(v) for k,v in stats.items()}
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'vision_encoder': self.vision_encoder.state_dict(),
            'vector_processor': self.vector_processor.state_dict(),
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
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"Model loaded from {path}")



