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
        self.distilled_converter = nn.Linear(12800, 256) #HARDCODED MODIFY TO MAKE DYANMIC

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
    
# class AttentionMudule(nn.Module):
    # def __init__(self,input_shape):
    #     super(AttentionMudule, self).__init__()
    #     self.query = nn.Linear(input_shape, 258)
    #     self.key = nn.Linear(input_shape, 258)
    #     self.value = nn.Linear(input_shape, 258)
    #     self.scale = torch.sqrt(torch.tensor([258])).to(device)

    # def forward(self, x):
    #     queries = self.query(x)
    #     keys = self.key(x)
    #     values = self.value(x)

    #     attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
    #     attention_weights = F.softmax(attention_scores, dim=-1)
    #     attended_values = torch.matmul(attention_weights, values)
    #     return attended_values

class SparseAttention(nn.Module):
    def __init__(self, input_dim, head_dim=64):
        super(SparseAttention, self).__init__()
        self.query = nn.Linear(input_dim, head_dim)
        self.key = nn.Linear(input_dim, head_dim)
        self.value = nn.Linear(input_dim, head_dim)
        self.scale = np.sqrt(head_dim)

    def forward(self, x):
        Q = self.query(x)  # [batch_size, head_dim]
        K = self.key(x)    # [batch_size, head_dim]
        V = self.value(x)  # [batch_size, head_dim]

        attn_scores = (Q @ K.T) / self.scale  # [batch_size, batch_size]
        attention_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, batch_size]

        # explicitly correct operation to get correct shape:
        attended_output = attention_weights @ V  # [batch_size, head_dim]

        return attended_output  # explicitly correct shape


class AdaptiveAttention(nn.Module):
    def __init__(self, input_dim, max_heads=4, head_dim=64):
        super(AdaptiveAttention, self).__init__()
        self.max_heads = max_heads
        self.head_selector = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, max_heads),
            nn.Softmax(dim=-1)
        )
        self.attention_heads = nn.ModuleList(
            [SparseAttention(input_dim, head_dim) for _ in range(max_heads)]
        )
        self.output_layer = nn.Linear(head_dim, input_dim)

    def forward(self, x, step_fraction):
        """
        x: Tensor [batch_size, input_dim]
        step_fraction: Scalar tensor indicating training progress [0,1]
        """
        # Determine number of heads dynamically
        step_tensor = torch.tensor([[step_fraction]], device=device)
        head_logits = self.head_selector(step_tensor).view(self.max_heads, 1,1)  # [1, max_heads]
        # Collect outputs from all heads explicitly
        attention_outputs = torch.stack(
            [head(x) for head in self.attention_heads], dim=0  # [max_heads, batch_size, head_dim]
        )

        # Explicitly compute weighted sum (smooth adaptive attention)
        combined_attention = torch.sum(attention_outputs * head_logits, dim=0)  # [batch_size, head_dim]

        # Project back to input_dim explicitly
        final_attention = self.output_layer(combined_attention)  # [batch_size, input_dim]

        return final_attention  # [batch_size, input_dim]

class EntropyTargetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Tanh()
        )

    def forward(self, x):
        return -2.0 * self.fc(x)

class CriticNet(nn.Module):
    def __init__(self, action_dim, hidden_dim=256):
        super(CriticNet, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(action_dim + hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        self.q1.apply(xavier_initialization)

        self.q2 = nn.Sequential(
            nn.Linear(action_dim + hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        self.q2.apply(xavier_initialization)
    
    def forward(self, x):
        return self.q1(x), self.q2(x)
    
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

        self.attention = AdaptiveAttention(256 + self.n_actions[0])

        self.critic = CriticNet(self.n_actions[0], 256)

        self.target_critic = CriticNet(self.n_actions[0], 256)

        self.fully_connected_pipeline.apply(xavier_initialization)
        self.actor_mean.apply(xavier_initialization)
        self.actor_std.apply(xavier_initialization)

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
    
    def get_values(self, camera_obs, vector_obs, actions, step_fraction, target=False):
        if vector_obs.dim() == 1:
            vector_obs = vector_obs.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        conv_out = self.convolution_pipeline(camera_obs)  # [batch_size, conv_dim]
        fc_input = torch.cat([conv_out, vector_obs], dim=1).to(device)
        fc_out = self.fully_connected_pipeline(fc_input)

        critic_input = torch.cat([fc_out, actions], dim=1).to(device)

        # explicitly correct attention layer output shape
        critic_input = self.attention(critic_input, step_fraction)  # explicitly [batch_size, hidden_dim]

        if critic_input.dim() != 2:
            critic_input = critic_input.view(critic_input.size(0), -1)

        # clearly defined critics returning [batch_size, 1]
        if target:
            return self.target_critic(critic_input)
        return self.critic(critic_input)