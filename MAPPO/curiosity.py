import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CuriosityModule(nn.Module):
    """
    Rewards agents for encountering novel states.
    Helps preventing getting stuck in local optima and encourages exploration.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=512):
        """
        obs_dim: Dimension of encoded observations
        action_dim: Dimension of action space
        hidden_dim: Hidden layer size
        """
        super().__init__()

        #Forward model predicts next state given current state and action
        self.forward_model = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        #Inverse model predicts action taken given current and next state
        self.inverse_model = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, obs, next_obs, actions):
        """
        Compute intrinsic rewards and curiosity loss.

        obs: (batch_size, obs_dim) - current encoded observations
        next_obs: (batch_size, obs_dim) - next encoded observations
        actions: (batch_size, action_dim) - actions taken
        returns intrinsic_rewards, curiosity_loss
        """

        obs_actions = torch.cat([obs, actions], dim=-1)
        next_obs_pred = self.forward_model(obs_actions)

        forward_loss = F.mse_loss(next_obs_pred, next_obs, reduction='none').mean(dim=-1)
        intrinsic_rewards = forward_loss.detach()

        obs_pair = torch.cat([obs, next_obs], dim=-1)
        actions_pred = self.inverse_model(obs_pair)

        inverse_loss = F.mse_loss(actions_pred, actions)

        curiosity_loss = forward_loss.mean() + inverse_loss

        return intrinsic_rewards, curiosity_loss
    

class RunningMeanStd:
    """
    Running Mean and std for normalizing intrinsic rewards
    """
    
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    @property
    def std(self):
        return np.sqrt(self.var)
    
    