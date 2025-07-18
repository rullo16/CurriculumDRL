import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as nn_utils
import math

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
        if isinstance(module.weight, nn.parameter.UninitializedParameter):
            return
        if module.weight is not None:
            nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    return module

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class FeatureExtractionNet(nn.Module):
    def __init__(self, input_shape, distilled_dim=2048):
        super(FeatureExtractionNet, self).__init__()
        self.convolutional_pipeline = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten(),
        )

        dummy = torch.zeros(1, *input_shape)
        conv_out = self.convolutional_pipeline(dummy).view(1, -1).shape[1]
        self.distilled_converter = nn.Linear(conv_out, distilled_dim)
        self.dropout = nn.Dropout(0.5)

        self.convolutional_pipeline.apply(safe_xavier_initialization)

    def forward(self, x, distill=False):
        x = x.float()
        if x.max() > 1.01:
            x = x / 255.0  # Normalize input to [0, 1]
        conv_out = self.convolutional_pipeline(x).view(x.size(0), -1)
        if distill:
            conv_out = self.dropout(conv_out)
            return self.distilled_converter(conv_out)
        return conv_out


class SparseAttention(nn.Module):
    def __init__(self, input_dim, head_dim=64):
        super(SparseAttention, self).__init__()
        self.query = nn.Linear(input_dim, head_dim)
        self.key = nn.Linear(input_dim, head_dim)
        self.value = nn.Linear(input_dim, head_dim)
        self.scale = np.sqrt(head_dim)

    def forward(self, x):
        x_seq = x.unsqueeze(1)  
        Q = self.query(x_seq)  # [batch_size, head_dim]
        K = self.key(x_seq)    # [batch_size, head_dim]
        V = self.value(x_seq)  # [batch_size, head_dim]

        scores = torch.matmul(Q, K.transpose(-2,-1))/self.scale
        weights = F.softmax(scores, dim=-1)  
        attended = torch.matmul(weights, V)
        return attended.squeeze(1)  


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
    
class AdaptiveGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1,32), nn.ReLU(),
            nn.Linear(32, dim), nn.Sigmoid()
        )
    def forward(self, x, step_fraction):
        step_fraction = torch.as_tensor(step_fraction, device=x.device, dtype=x.dtype)
        gate = self.fc(step_fraction.expand(1,1))
        return x * gate  # Element-wise multiplication

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
    def __init__(self, input_dim,hidden_dim=256):
        super(CriticNet, self).__init__()
        

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        mid = hidden_dim // 2
        self.q1 = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.LayerNorm(mid),
            nn.ReLU(inplace=True),
            nn_utils.spectral_norm(nn.Linear(mid, 1))
        )

        self.q2 = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.LayerNorm(mid),
            nn.ReLU(inplace=True),
            nn_utils.spectral_norm(nn.Linear(mid, 1))
        )


        self.q1.apply(safe_xavier_initialization)
        self.q2.apply(safe_xavier_initialization)
        
    def forward(self, x):
        x = self.backbone(x)
        return self.q1(x), self.q2(x)
    
class RunningStat:
    def __init__(self, shape, eps=1e-4):
        self._mean = torch.zeros(shape, dtype=torch.float32).to(device)
        self._var = torch.ones(shape, dtype=torch.float32).to(device)
        self._count = eps

    def update(self, x):
        if x.device != device:
            x = x.to(device)

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        if batch_count == 0:
            return
        

        delta = batch_mean - self._mean
        tot_count = self._count + batch_count

        self._mean += delta * batch_count / tot_count
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta.pow(2) * self._count * batch_count / tot_count)
        self._var = m2 / tot_count
        self._count = tot_count

    @property
    def mean(self): return self._mean
    @property
    def std(self): return torch.sqrt(self._var + 1e-8)
    

class RND(nn.Module):
    """
    Random Network Distillation (RND) module.
    Generates intrinsic rewards based on the difference between a target network and a predictor network.
    """

    def __init__(self, input_dim, hidden_dim=256, output_dim=512):
        super().__init__()
        self.target_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.predictor_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        for param in self.target_net.parameters():
            param.requires_grad = False

        self.obs_normalizer = RunningStat(shape=(input_dim,))

    def forward(self, x):
        
        self.obs_normalizer.update(x)
        normalized_obs = (x - self.obs_normalizer.mean) / self.obs_normalizer.std
        normalized_obs = torch.clamp(normalized_obs, -5.0, 5.0)

        with torch.no_grad():
            target_features = self.target_net(normalized_obs)
        
        predicted_features = self.predictor_net(normalized_obs)
        intrinsic_reward = torch.mean((predicted_features - target_features).pow(2), dim=1, keepdim=True)

        return intrinsic_reward
    
    def compute_loss(self, obs):

        self.obs_normalizer.update(obs)
        normalized_obs = (obs - self.obs_normalizer.mean) / self.obs_normalizer.std
        normalized_obs = torch.clamp(normalized_obs, -5.0, 5.0)

        with torch.no_grad():
            target_features = self.target_net(normalized_obs.detach())
        predicted_features = self.predictor_net(normalized_obs.detach())
        return F.mse_loss(predicted_features, target_features)


    
class CentralizedCriticNet(nn.Module):
    def __init__(self, per_agent_dim, action_dim, num_agents):
        super().__init__()
        self.num_agents = num_agents
        self.per_agent_dim = per_agent_dim
        self.action_dim = action_dim
        centralized_input_dim = num_agents * (per_agent_dim + action_dim)
        

        self.backbone = nn.Sequential(
            nn.Linear(centralized_input_dim,512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )

        self.q_head_1 = nn.Linear(256,1)
        self.q_head_2 = nn.Linear(256,1)

        self.apply(safe_orthogonal_initialization)

    def forward(self, feats, actions):
        batch_size = feats.shape[0] // self.num_agents
        x_per_agent = torch.cat([feats, actions], dim=-1)
        x_centralized = x_per_agent.view(batch_size, -1)

        h = self.backbone(x_centralized)  # [batch_size, 256]

        q1s = self.q_head_1(h).repeat(1, self.num_agents)
        q2s = self.q_head_2(h).repeat(1, self.num_agents)
        return q1s, q2s
    
class GaussianPolicy(nn.Module):
    def __init__(self, input_dim, act_dim):
        super().__init__()
        self.mu = nn.Linear(input_dim, act_dim)
        self.log_std = nn.Linear(input_dim, act_dim)

    def forward(self, x):
        x = x.float()
        if x.max() > 1.01:
            x = x / 255.0  # Normalize input to [0, 1]
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), min=-20, max=2)
        std = torch.exp(log_std).clamp(min=1e-6)
        dist = torch.distributions.Normal(mu, std)
        return dist
    
class SACNet(nn.Module):
    def __init__(self, camera_obs_dim, vector_obs_dim, n_actions):
        super(SACNet, self).__init__()
        self.camera_obs_dim = camera_obs_dim
        self.vector_obs_dim = vector_obs_dim
        self.convolution_pipeline = FeatureExtractionNet(camera_obs_dim, distilled_dim=12800)
        self.conv_out_size = self._get_conv_out(camera_obs_dim)
        self.feat_dim = 256
        self.n_actions = n_actions

        self.vector_processor = nn.Sequential(
            nn.LayerNorm(vector_obs_dim[0]),
            nn.Linear(vector_obs_dim[0], 128),
            nn.ReLU(inplace=True),
        )

        combined_input_dim = self.conv_out_size + 128  # 128 from vector_obs processing
        
        self.backbone = nn.Sequential(
            nn.Linear(combined_input_dim,512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
            nn.ReLU(inplace=True),
        )

        input_dim = 128 + self.n_actions

        self.policy_head = GaussianPolicy(256, self.n_actions)

        self.vector_processor.apply(safe_xavier_initialization)
        self.backbone.apply(safe_xavier_initialization)
        self.policy_head.apply(safe_xavier_initialization)

    def dist_from_feats(self, feats):
        return self.policy_head(feats)

    def _get_conv_out(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output = self.convolution_pipeline(dummy_input)
        return int(np.prod(output.size()[1:]))
    
    def forward(self, camera_obs, vector_obs):
        feats = self.encode(camera_obs, vector_obs)
        logits = self.policy_head(feats)
        return logits
    
    def encode(self, camera_obs, vector_obs):
        cam = self.convolution_pipeline(camera_obs)  # [batch_size, conv_dim]
        vec_feat = self.vector_processor(vector_obs)  # [batch_size, 128]
        feats = self.backbone(torch.cat([cam, vec_feat], dim=1))  # [batch_size, 512]
        return feats