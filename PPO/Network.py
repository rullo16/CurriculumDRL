import torch
from torch import nn

class Network(nn.Module):

    def __init__(self, obs_space, action_space):
        super(Network, self).__init__()
        self.obs_space = obs_space[0]
        self.inventory = obs_space[1]
        self.action_space = action_space

        self.obs_features = nn.Sequential(
            nn.Linear(self.obs_space, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        self.inventory_features = nn.Sequential(
            nn.Linear(self.inventory, 128),
            nn.ReLU(),
        )

        self.q_net=nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )


    def forward(self, x):

        obs_features = self.obs_features(x[0])
        inventory_features = self.inventory_features(x[1])
        x = torch.cat([obs_features, inventory_features.unsqueeze(0)], dim=1)
        x = self.q_net(x)

        return x
    
class PPONet(nn.Module):

    def __init__(self, obs_space, action_space):
        super(PPONet, self).__init__()
        self.obs_space = obs_space[0]
        self.inventory = obs_space[1]
        self.action_space = action_space

        self.obs_features = nn.Sequential(
            nn.Linear(self.obs_space, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.inventory_features = nn.Sequential(
            nn.Linear(self.inventory, 128),
            nn.ReLU(),
        )

        self.policy_net=nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        obs_features = self.obs_features(x[0])
        inventory_features = self.inventory_features(x[1])
        x = torch.cat([obs_features, inventory_features], dim=1)
        policy = self.policy_net(x)
        value = self.critic(x)
        return policy, value