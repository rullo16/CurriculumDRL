import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def xavier_initialization(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module

def orthogonal_initialization(module, gain=nn.init.calculate_gain('relu')):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module

class SkipConnectionBlock(nn.Module):
    def __init__(self, input_channels):
        super(SkipConnectionBlock, self).__init__()
        self.convolutional_block = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return x+self.convolutional_block(x)
    
class FeatureExtrationBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FeatureExtrationBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            SkipConnectionBlock(output_channels),
            nn.ReLU(),
            SkipConnectionBlock(output_channels)
        )

    def forward(self, x):
        return self.layers(x)
    
class KnowledgeDistillationNetwork(nn.Module):
    def __init__(self, input_channels):
        super(KnowledgeDistillationNetwork, self).__init__()
        self.convolutional_pipeline = nn.Sequential(
            FeatureExtrationBlock(input_channels[0], 32),
            FeatureExtrationBlock(32, 64),
            FeatureExtrationBlock(64, 128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.5),
        )

        self.convolutional_pipeline.apply(xavier_initialization)

    def forward(self, x):
        normalized_x = x/255.0
        conv_out = self.convolutional_pipeline(normalized_x).view(normalized_x.size(0), -1)
        return conv_out
    
class PPONetWithDistillation(nn.Module):
    def __init__(self, convolution_pipeline, camera_obs_dim, vector_obs_dims, n_actions):
        super(PPONetWithDistillation, self).__init__()
        self.camera_obs_dim = camera_obs_dim
        self.vector_obs_dims = vector_obs_dims
        self.n_actions = n_actions
        self.convolution_pipeline = convolution_pipeline

        self.conv_out_size = self._get_conv_out(self.camera_obs_dim)

        self.fully_connected_pipeline = nn.Sequential(
            nn.Linear(self.conv_out_size + vector_obs_dims[0], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(128, n_actions[0]),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(128, 1)
        )

    def _get_conv_out(self, shape):
        cnn = KnowledgeDistillationNetwork(shape)
        o = cnn(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, camera_obs, vector_obs):
        normalized_camera_obs = camera_obs/255.0
        conv_out = self.convolution_pipeline(normalized_camera_obs)
        fc_input = torch.cat([conv_out, vector_obs], dim=1)
        fc_out = self.fully_connected_pipeline(fc_input)
        return self.actor(fc_out), self.critic(fc_out)


        