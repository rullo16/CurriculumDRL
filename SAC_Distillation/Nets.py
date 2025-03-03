import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_xavier_initialization(module, gain=1.0):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        if module.weight is not None:
            nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    return module

def safe_orthogonal_initialization(module, gain=nn.init.calculate_gain('relu')):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        if module.weight is not None:
            nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    return module

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def xavier_initialization(module, gain=1.0):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.xavier_uniform_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, 0)
    return module

def orthogonal_initialization(module, gain=nn.init.calculate_gain('relu')):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.orthogonal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, 0)
    return module

class SkipConnectionBlock(nn.Module):
    """
    Residual block that applies two convolutional layers with ReLU activations,
    then adds the input back to the output.
    """
    def __init__(self, channels):
        super(SkipConnectionBlock, self).__init__()
        self.convolutional_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),  # modified to preserve dimensions
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x):
        return x + self.convolutional_block(x)

class FeatureExtractionBlock(nn.Module):
    """
    Convolutional block for extracting features from images.
    Consists of a convolution, pooling, ReLU, and two residual blocks.
    """
    def __init__(self, in_channels, out_channels):
        super(FeatureExtractionBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),  # modified kernel & stride
            nn.MaxPool2d(kernel_size=2, stride=2),  # modified pooling
            nn.ReLU(inplace=True),
            SkipConnectionBlock(out_channels),
            nn.ReLU(inplace=True),
            SkipConnectionBlock(out_channels)
        )
    
    def forward(self, x):
        return self.layers(x)

class FeatureExtractionNet(nn.Module):
    """
    Network for feature extraction that supports knowledge distillation.
    When 'distill' is True, the output is passed through the distilled converter.
    """
    def __init__(self, input_shape):
        """
        input_shape: tuple in the format (channels, height, width)
        """
        super(FeatureExtractionNet, self).__init__()
        # Build convolutional pipeline with feature extraction blocks
        self.convolutional_pipeline = nn.Sequential(
            FeatureExtractionBlock(input_shape[0], 32),
            FeatureExtractionBlock(32, 64),
            FeatureExtractionBlock(64, 128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(0.5)
        )
        # Apply initialization to the conv pipeline and converter
        self.distilled_converter = nn.Linear(12800, 768) #HARDCODED MODIFY TO MAKE DYANMIC

        self.convolutional_pipeline.apply(xavier_initialization)
    
    def forward(self, x, distill=False):
        """
        Forward pass.
        Normalizes input and processes through the conv pipeline.
        If 'distill' is True, converts conv features to a distilled representation.
        """
        # Ensure data is float, normalized if coming in as bytes, and sent to the same device as the model
        normalized_x = (x.float() / 255.0)
        conv_out = self.convolutional_pipeline(normalized_x).view(normalized_x.size(0), -1)
        if distill:
            return self.distilled_converter(conv_out)
        return conv_out
    
class AttentionMudule(nn.Module):
    def __init__(self,input_shape):
        super(AttentionMudule, self).__init__()
        self.query = nn.Linear(input_shape, 258)
        self.key = nn.Linear(input_shape, 258)
        self.value = nn.Linear(input_shape, 258)
        self.scale = torch.sqrt(torch.tensor([258])).to(device)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, values)
        return attended_values
    
class SACNet(nn.Module):
    def __init__(self, camera_obs_dim, vector_obs_dim, n_actions, num_agents):
        super(SACNet, self).__init__()
        self.camera_obs_dim = camera_obs_dim
        self.vector_obs_dim = vector_obs_dim
        self.n_actions = n_actions if isinstance(n_actions, (list, tuple)) else [n_actions]
        self.convolution_pipeline = FeatureExtractionNet(camera_obs_dim)
        self.conv_out_size = self._get_conv_out(camera_obs_dim)
        self.fully_connected_pipeline = nn.Sequential(
            nn.Linear(self.conv_out_size + self.vector_obs_dim[0], 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )
        
        self.actor_mean = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.n_actions[0]),
            nn.Tanh()
        )

        self.actor_std = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.n_actions[0]),
            nn.Softplus()
        )

        self.attention = AttentionMudule((256+n_actions[0]))

        self.critic_1 = nn.Sequential(
            nn.Linear(258, 128),  # modified input features from 258*num_agents to 258
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        self.critic_2 = nn.Sequential(
            nn.Linear(258, 128),  # modified input features from 258*num_agents to 258
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        self.fully_connected_pipeline.apply(xavier_initialization)
        self.actor_mean.apply(xavier_initialization)
        self.actor_std.apply(xavier_initialization)
        self.critic_1.apply(xavier_initialization)
        self.critic_2.apply(xavier_initialization)

    def _get_conv_out(self, shape):
        self.convolution_pipeline.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output = self.convolution_pipeline(dummy_input)
        self.convolution_pipeline.train()
        return int(np.prod(output.size()[1:]))
    
    def forward(self, camera_obs, vector_obs):
        if vector_obs.ndim == 1:
            vector_obs = vector_obs.unsqueeze(0)
        fc_input = torch.cat([self.convolution_pipeline(camera_obs), vector_obs], dim=1)  # modified to ensure same dimensions
        fc_out = self.fully_connected_pipeline(fc_input)
        action_mean = self.actor_mean(fc_out)
        action_log_std = self.actor_std(fc_out)
        action_log_std = torch.clamp(action_log_std, -20, 2)
        return action_mean, action_log_std
    
    def get_values(self, camera_obs, vector_obs, action_mean):
        if vector_obs.dim() == 1:
            vector_obs = vector_obs.unsqueeze(0)
        if action_mean.dim() == 1:
            action_mean = action_mean.unsqueeze(0)

        conv_out = self.convolution_pipeline(camera_obs)
        fc_input = torch.cat([conv_out, vector_obs], dim=1).to(device)
        fc_out = self.fully_connected_pipeline(fc_input)
        critic_input = torch.cat([fc_out, action_mean], dim=1).to(device)
        critic_input = self.attention(critic_input)
        value_1 = self.critic_1(critic_input)
        value_2 = self.critic_2(critic_input)
        return value_1, value_2