import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F





class MAPPOActor(nn.Module):
    """
    Decentralized policy net.

    "Decentralized" because each agent makes decisions based on its own observation only, there is no communication between agetns during execution 
    and is critical for real-world deployement.

    All agents share the same network parameters (params sharing), which yields faster learning, better generalization and smaller models.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        """
        obs_dim: Dimension of encoded observations (vision + vector)
        action_dim: Dimension of action space
        hidden_dim: Hidden layer size
        """
        super().__init__()

        #Feature extraction layer, process observations into features
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

        #Policy head, use Gaussian (Normal) distribution for continuous control
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        #Init weights properly, small weights for policy head prevent large initial policy changes
        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal init - better for DNN.
        Small init for policy head, prevents large updates
        """

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        #Policy head gets special small init
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0)

    def forward(self, obs):
        """
        Compute action distribution parameters
        obs: (batch_size, obs_dim) or (batch_size, num_agents, obs_dim)
        
        returns Distribution object that can sample actions
        """

        features = self.feature_net(obs)

        #MEan of Gauss dist
        action_mean = self.mean_layer(features)

        #Log std
        #Clamp to prevent issues
        action_log_std = torch.clamp(self.log_std_layer(features), -20,2)
        action_std = torch.exp(action_log_std)

        #Gauss dist, agents sample from this for actions
        dist = torch.distributions.Normal(action_mean, action_std)

        return dist
    
    def get_action(self, obs, deterministic=False):
        """
        Sample action from policy

        obs: Observations
        deterministic: If True, return mean (for evaluation)

        returns 
        action: Sampled action
        log_prob: Log probability of the action
        """

        dist = self.forward(obs)

        if deterministic:
            #for eval, use mean action, no exploration
            action_raw = dist.mean
        else:
            #for train, sample from dist, exploration
            action_raw = dist.sample()

        #squash action to [-1,1] using tanh, make sure actions are in valid range
        action = torch.tanh(action_raw)

        #compute log prob of action, for PPO loss
        log_prob = dist.log_prob(action_raw)

        #correct tahn squashing
        log_prob -= torch.log(1-action.pow(2)+1e-6)
        log_prob = log_prob.sum(dim=-1) # sum over action dims

        return action, log_prob
    
    def evaluate_actions(self, obs, actions):
        """
        Eval log prob and entropy of given actions. Used during trainig to compute policy loss.

        obs: Observations that led to actions
        actions: Actions that were taken

        returns
        log_probs: Log probability of actions under current policy
        entropy: Entropy of the action distribution
        """

        dist = self.forward(obs)

        #Inverse tanh to unsquash actions, needed because we stored squashed actions
        actions_unsquashed = torch.atanh(actions.clamp(-0.9999, 0.9999))

        #compute log prob
        log_prob = dist.log_prob(actions_unsquashed)

        #Correct tanh squashing
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        #Compute entropy (measure of exploration), higer entropy = more exploration
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy
    

class MAPPOCritic(nn.Module):
    """
    Centralized value network.

    "Centralized" because it sees all agents' observations during training, learn better value estimates, and helps with credit assignment in multi-agents
    
    Only used during training, key from CTDE (Centralized Trainig, Decentralized Execution)
    """

    def __init__(self, global_obs_dim, hidden_dim=512, num_agents=4):
        """
        global_obs_dim: Total observation dimension (num_agents * obs_dim)
        hidden_dim: Hidden layer size
        num_agents: Number of agents
        """

        super().__init__()
        self.num_agents = num_agents

        #Value network, used only during traianing so speed does not matter
        self.value_net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_agents)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, global_obs):
        """
        Compute value estimates.
        global_obs: (batch_size, num_agents, obs_dim) - all agents' observations
        
        returns values: (batch_size, num_agents) - per-agent value estimates
        """
        #Flatten observations from all agents
        if global_obs.dim()==3:
            batch_size, num_agents, obs_dim = global_obs.shape
            global_obs_flat = global_obs.reshape(batch_size, -1)
        else:
            batch_size = global_obs.shape[0]
            global_obs_flat = global_obs

        #Compute values
        values = self.value_net(global_obs_flat)

        return values