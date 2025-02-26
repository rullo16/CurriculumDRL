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
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(inplace=True),
            SkipConnectionBlock(out_channels),
            nn.ReLU(inplace=True),
            SkipConnectionBlock(out_channels)
        )
    
    def forward(self, x):
        return self.layers(x)

class KnowledgeDistillationNetwork(nn.Module):
    """
    Network for feature extraction that supports knowledge distillation.
    When 'distill' is True, the output is passed through the distilled converter.
    """
    def __init__(self, input_shape):
        """
        input_shape: tuple in the format (channels, height, width)
        """
        super(KnowledgeDistillationNetwork, self).__init__()
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
        self.distilled_converter = nn.Linear(18432, 768)

        self.convolutional_pipeline.apply(xavier_initialization)
    
    def forward(self, x, distill=False):
        """
        Forward pass.
        Normalizes input and processes through the conv pipeline.
        If 'distill' is True, converts conv features to a distilled representation.
        """
        # Ensure data is float, normalized if coming in as bytes, and sent to the same device as the model
        normalized_x = (x.float() / 255.0).to(next(self.parameters()).device)
        conv_out = self.convolutional_pipeline(normalized_x).view(normalized_x.size(0), -1)
        if distill:
            return self.distilled_converter(conv_out)
        return conv_out

class SACNetWithDistillation(nn.Module):
    """
    SAC network that integrates a convolutional feature extractor with additional
    fully-connected layers for both actor and critic networks.
    """
    def __init__(self, camera_obs_shape, vector_obs_dim, n_actions, sac_distilled):
        """
        camera_obs_shape: tuple representing the shape of camera observations.
        vector_obs_dim: integer or tuple representing the dimension of vector observations.
        n_actions: integer or a tuple/list with one element representing action dimension.
        sac_distilled: additional parameters for distillation (unused in this simple example).
        """
        super(SACNetWithDistillation, self).__init__()
        self.camera_obs_shape = camera_obs_shape
        # Convert vector_obs_dim to an integer in case a tuple is provided.
        if isinstance(vector_obs_dim, tuple):
            self.vector_obs_dim = int(np.prod(vector_obs_dim))
        else:
            self.vector_obs_dim = vector_obs_dim
        self.n_actions = n_actions if isinstance(n_actions, (list, tuple)) else [n_actions]
        
        # Build convolutional pipeline using the KnowledgeDistillationNetwork.
        self.convolution_pipeline = KnowledgeDistillationNetwork(camera_obs_shape)
        
        # Compute convolutional output size dynamically.
        self.conv_out_size = self._get_conv_out(camera_obs_shape)
        
        self.fully_connected_pipeline = nn.Sequential(
            nn.Linear(self.conv_out_size + self.vector_obs_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(128, self.n_actions[0]),
            nn.Tanh()
        )
        
        self.actor_log_std = nn.Linear(128, self.n_actions[0])
        
        self.critic_1 = nn.Sequential(
            nn.Linear(128 + self.n_actions[0], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        
        self.critic_2 = nn.Sequential(
            nn.Linear(128 + self.n_actions[0], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        
        # Initialize weights for fully connected parts.
        self.fully_connected_pipeline.apply(xavier_initialization)
        self.actor.apply(xavier_initialization)
        self.actor_log_std.apply(xavier_initialization)
        self.critic_1.apply(xavier_initialization)
        self.critic_2.apply(xavier_initialization)

        self.critic_2 = nn.Sequential(
            nn.Linear(128 + self.n_actions[0], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

        # Initialize weights for fully connected parts
        self.fully_connected_pipeline.apply(xavier_initialization)
        self.actor.apply(xavier_initialization)

    def _get_conv_out(self, shape):
        # Pass a dummy input through the convolution_pipeline to get the output size.
        self.convolution_pipeline.eval()
        with torch.no_grad():
            # Create dummy input on the same device as the model parameters.
            model_device = next(self.parameters()).device
            dummy_input = torch.zeros(1, *shape, device=model_device)
            output = self.convolution_pipeline(dummy_input)
        self.convolution_pipeline.train()
        return int(np.prod(output.size()[1:]))

    def forward(self, camera_obs, vector_obs):
        """
        Process image and vector observations to produce actor distribution parameters.
        Returns mean and standard deviation for action sampling.
        """
        # Ensure camera_obs has a batch dimension.
        if camera_obs.dim() == 3:
            camera_obs = camera_obs.unsqueeze(0)
        conv_out = self.convolution_pipeline(camera_obs)
        if vector_obs.dim() == 1:
            vector_obs = vector_obs.unsqueeze(0)
        vector_obs = vector_obs.to(conv_out.device)  # ensure same device
        fc_input = torch.cat([conv_out, vector_obs], dim=1)
        fc_out = self.fully_connected_pipeline(fc_input)
        action_mean = self.actor(fc_out)
        action_log_std = self.actor_log_std(fc_out)
        action_std = torch.exp(action_log_std)
        return action_mean, action_std

    def get_critics(self, camera_obs, vector_obs, actions):
        """
        Given observations and actions, compute the two critic outputs.
        """
        # Ensure camera_obs has a batch dimension.
        if camera_obs.dim() == 3:
            camera_obs = camera_obs.unsqueeze(0)
        if vector_obs.dim() == 1:
            vector_obs = vector_obs.unsqueeze(0).to(device)
        conv_out = self.convolution_pipeline(camera_obs).to(device)
        fc_input = torch.cat([conv_out, vector_obs], dim=1)
        fc_out = self.fully_connected_pipeline(fc_input)
        critic_input = torch.cat([fc_out, actions], dim=1)
        return self.critic_1(critic_input), self.critic_2(critic_input)