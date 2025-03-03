import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from .Trajectories import SAC_ExperienceBuffer
from .Nets import SACNet, FeatureExtractionNet
from transformers import AutoImageProcessor, AutoModel, ViTModel
import random
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Explanation of key math formulas used in DistilledSAC:
# 1. Actor loss:
#    L_actor = -E[log π(a|s) * d] - λ * H[π(a|s)]
# 2. Critic loss:
#    L_critic = E[(Q(s, a) - (r + γ * V(s')))^2]
# 3. Knowledge distillation loss:
#    L_KD = KL( softmax(teacher) || log_softmax(student) )
#

class DistilledSAC:

    def __init__(self, camera_obs_dim, vector_obs_dim, action_dims,num_agents, params):
        self.device = device
        self.params = params
        self.camera_obs_dim = camera_obs_dim
        self.vector_obs_dim = vector_obs_dim
        self.action_dims = action_dims

        # Initialize Networks
        self.model = SACNet(camera_obs_dim, vector_obs_dim, action_dims, num_agents).to(self.device)
        self.actor_optimizer = optim.Adam(self.model.parameters(), lr=params.actor_lr)
        self.critic_1_optimizer = optim.Adam(self.model.critic_1.parameters(), lr=params.critic_lr)
        self.model.critic_2.load_state_dict(self.model.critic_1.state_dict())
        self.model.critic_2.eval()
        
        if self.params.target_entropy is None:
            self.params.target_entropy = -float(action_dims[0])
        self.target_entropy = self.params.target_entropy
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=params.alpha_lr)

        self.alpha = self.log_alpha.exp().item()

        #Teacher and student networks
        self.model.convolution_pipeline.eval()
        self.teacher_processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k", use_fast=True
        )
        self.teacher = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(self.device)
        self.teacher.eval()
        self.distill_optimizer = optim.Adam(self.model.convolution_pipeline.parameters(), lr=params.distill_lr)
        
        #dynamically adjust entropy_beta parameters for entropy regularization
        self.entropy_beta = torch.tensor(1.0).to(self.device).requires_grad_()
        self.entropy_optimizer = optim.Adam([self.entropy_beta], lr=params.entropy_lr)

    def _process_vector(self, vector):
        if not isinstance(vector, torch.Tensor):
            vector = torch.as_tensor(vector, dtype=torch.float32, device=self.device)
        return vector
    
    def get_action(self, camera_obs, vector_obs, train=False):
        camera_obs = self._process_vector(camera_obs)
        vector_obs = self._process_vector(vector_obs)

        mean, log_std = self.model(camera_obs, vector_obs)
        std = log_std.exp()
        action_distribution = distributions.Normal(mean, std)
        z = action_distribution.rsample()
        action = torch.tanh(z)
        action = torch.clamp(action, -1, 1)
        log_prob = action_distribution.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        if train: 
            return action, log_prob
        
        return action.cpu().detach().numpy()[0], log_prob.cpu().detach().numpy()
    
    def get_values(self, camera_obs, vector_obs, action):
        camera_obs = self._process_vector(camera_obs)
        vector_obs = self._process_vector(vector_obs)
        action = self._process_vector(action)

        value_1, value_2 = self.model.get_values(camera_obs, vector_obs, action)
        return value_1, value_2
    
    def adjust_lr(self, optimizer, lr, timesteps, max_timesteps):
        new_lr = lr * (1-timesteps/max_timesteps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return optimizer
    def _get_teacher_output(self, camera_obs):
        camera_obs_rescaled = (camera_obs * 255).byte()
        teacher_input = self.teacher_processor(images=camera_obs_rescaled, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            teacher_output = self.teacher(teacher_input)
        return teacher_output
    
    def train(self, trajectories):
        if len(trajectories) < self.params.batch_size:
            return
        
        for _ in range(self.params.train_epochs):

            sample = trajectories.sample(self.params.batch_size)
            if sample is None:
                return
            camera_obs = self._process_vector(sample["camera_obs"])
            vector_obs = self._process_vector(sample["vector_obs"])
            actions = self._process_vector(sample["actions"])
            rewards = self._process_vector(sample["rewards"])
            done_flags = self._process_vector(sample["dones"])
            values = self._process_vector(sample["values"])
            # advantages = self._process_vector(sample["advantages"])
            next_camera_obs = self._process_vector(sample["next_camera_obs"])
            next_vector_obs = self._process_vector(sample["next_vector_obs"])

        # Train Critic
        with torch.no_grad():
            next_mean, next_log_std = self.model(next_camera_obs, next_vector_obs)
            q1_target, q2_target = self.model.get_values(next_camera_obs, next_vector_obs, next_mean)
            q_target = torch.min(q1_target, q2_target)
            
            next_log_std = next_log_std.sum(1, keepdim=True)
            y = rewards + (1-done_flags) * self.params.gamma * (q_target - self.alpha * next_log_std)
            target_q = 0.5 * (y+values)

        q1, _  = self.model.get_values(camera_obs, vector_obs, actions)
        critic_loss = F.mse_loss(q1, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()

        # Train Actor
        new_actions, log_pi = self.get_action(camera_obs, vector_obs, train=True)
        
        q_pi, _ = self.model.get_values(camera_obs, vector_obs, new_actions)
        actor_loss = (self.alpha * log_pi - q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #Entropy Regularization
        entropy_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.entropy_optimizer.zero_grad()
        entropy_loss.backward()
        self.entropy_optimizer.step()
        self.alpha = float(self.log_alpha.exp().detach().cpu().numpy())

        #Update critic target
        for target_param, param in zip(self.model.critic_2.parameters(), self.model.critic_1.parameters()):
            target_param.data.copy_(self.params.tau * param.data + (1 - self.params.tau) * target_param.data)

        #Distillation
        teacher_features = self._get_teacher_output(camera_obs).last_hidden_state[:, 0, :]
        student_features = self.model.convolution_pipeline(camera_obs, distill=True)
        distill_loss = F.kl_div(F.log_softmax(student_features, dim=-1), F.softmax(teacher_features, dim=-1),reduction='batchmean')
        self.distill_optimizer.zero_grad()
        distill_loss.backward()
        self.distill_optimizer.step()