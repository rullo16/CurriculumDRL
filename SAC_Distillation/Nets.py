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
        self.expected_hw = (input_shape[1], input_shape[2])
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
        # scale only if raw bytes
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()

        if x.shape[-2:] != self.expected_hw:
            x = F.interpolate(x, size=self.expected_hw, mode='bilinear', align_corners=False)

        conv_out = self.convolutional_pipeline(x).view(x.size(0), -1)

        if hasattr(self, '_distillation_done') and self._distillation_done:
            return self.distilled_converter(conv_out)
        
        if distill:
            conv_out = self.dropout(conv_out)
            return self.distilled_converter(conv_out)
        return conv_out

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

    def __init__(self, input_dim, hidden_dim=256, output_dim=512, eps = 1e-8):
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
        self.target_net.eval()

        self.obs_normalizer = RunningStat(shape=(input_dim,))
        self.rew_normaliser = RunningStat(shape=(1,))
        self._eps = eps

    def _to_stat_tensor(self, arr, device, dtype):
        """Return arr as a torch.Tensor on (device, dtype) with no grad."""
        if isinstance(arr, torch.Tensor):
            return arr.to(device=device, dtype=dtype)
        # RunningStat may store numpy arrays
        return torch.as_tensor(arr, device=device, dtype=dtype)
    
    def _normalise_obs(self, x):
        """Z-score using running stats (eps-stable) and clamp."""
        m = self._to_stat_tensor(self.obs_normalizer.mean, x.device, x.dtype)
        s = self._to_stat_tensor(self.obs_normalizer.std,  x.device, x.dtype)
        x = (x - m) / (s + self._eps)
        return x.clamp_(-5.0, 5.0)
    
    @torch.no_grad()
    def update_obs_stats(self, obs):
        if obs.is_cuda:
            self.obs_normalizer.update(obs.detach().cpu())
        else:
            self.obs_normalizer.update(obs.detach())

    @torch.no_grad()
    def reset_episode(self):
        pass

    @torch.no_grad()
    def forward(self, x, normalise_reward=False):
        # NEVER update stats in forward
        # Only compute reward
        x_n = self._normalise_obs(x)
        t = self.target_net(x_n)
        p = self.predictor_net(x_n)
        intr = (p - t).pow(2).mean(dim=1, keepdim=True)
        
        if normalise_reward:
            # Use existing stats, don't update
            rm = self._to_stat_tensor(self.rew_normaliser.mean, intr.device, intr.dtype)
            rs = self._to_stat_tensor(self.rew_normaliser.std, intr.device, intr.dtype)
            intr = (intr - rm) / (rs + self._eps)
        
        return intr.clamp_min(1e-3)

    def update_reward_stats(self, rewards):
        self.rew_normaliser.update(rewards.detach().cpu())

    def compute_loss(self, obs):
       
        if obs.numel() == 0:
            return obs.sum() * 0.0
        x_n = self._normalise_obs(obs)
        with torch.no_grad():
            t = self.target_net(x_n)
        p = self.predictor_net(x_n)
        return F.mse_loss(p, t)


    
class CentralizedCriticNet(nn.Module):
    def __init__(self, per_agent_dim, action_dim, num_agents):
        super().__init__()
        self.num_agents = num_agents
        centralized_input_dim = num_agents * (per_agent_dim + action_dim)
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(centralized_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        
        # PER-AGENT Q-heads (not joint!)
        self.q1_heads = nn.ModuleList([
            nn.Linear(256, 1) for _ in range(num_agents)
        ])
        self.q2_heads = nn.ModuleList([
            nn.Linear(256, 1) for _ in range(num_agents)
        ])
    
    def forward(self, feats, actions):
        # feats: (B*N, feat_dim), actions: (B*N, act_dim)
        batch_size = feats.shape[0] // self.num_agents
        
        # Centralize input
        x_per_agent = torch.cat([feats, actions], dim=-1)
        x_centralized = x_per_agent.view(batch_size, -1)
        
        # Shared representation
        h = self.backbone(x_centralized)  # (B, 256)
        
        # Per-agent Q-values
        q1_list = [head(h) for head in self.q1_heads]  # List of (B, 1)
        q2_list = [head(h) for head in self.q2_heads]
        
        q1 = torch.cat(q1_list, dim=1).view(batch_size * self.num_agents, 1)
        q2 = torch.cat(q2_list, dim=1).view(batch_size * self.num_agents, 1)
        
        return q1, q2  # (B*N, 1) - per-agent values!
    
class GaussianPolicy(nn.Module):
    def __init__(self, input_dim, act_dim):
        super().__init__()
        self.mu = nn.Linear(input_dim, act_dim)
        self.log_std = nn.Linear(input_dim, act_dim)

    def forward(self, x):
        x = x.float()
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
        self.convolution_pipeline = FeatureExtractionNet(camera_obs_dim, distilled_dim=2048)
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