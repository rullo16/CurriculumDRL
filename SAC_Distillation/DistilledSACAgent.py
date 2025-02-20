import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .Trajectories import ExperienceBuffer
from .Nets import SACNetWithDistillation, KnowledgeDistillationNetwork
from transformers import AutoImageProcessor, ViTModel
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Explanation of key math formulas used in DistilledSAC:
# 1. Actor loss:
#    L_actor = -E[log π(a|s) * d] - λ * H[π(a|s)]
#    where:
#      • π(a|s) is the Gaussian action distribution produced by the actor network.
#      • d represents the policy's weight (or advantage) on the action.
#      • H[π(a|s)] is the entropy of the action distribution, encouraging exploration.
#
# 2. Critic loss:
#    L_critic = E[(Q(s, a) - (r + γ * V(s')))^2]
#    where:
#      • Q(s, a) is the estimated action-value from the critic networks.
#      • r is the received reward.
#      • γ is the discount factor.
#      • V(s') is the target value estimated from the next state.
#
# 3. Knowledge distillation loss:
#    L_KD = KL( softmax(teacher) || log_softmax(student) )
#    where:
#      • The teacher's output (after softmax) represents a target distribution.
#      • The student's output (after log_softmax) is trained to match the teacher.
#      • KL(·||·) refers to the Kullback-Leibler divergence between the two distributions.
class DistilledSAC:
    def __init__(self, camera_obs_dim, vector_obs_dim, action_dims, params):
        self.params = params
        self.camera_obs_dim = camera_obs_dim
        self.vector_obs_dim = vector_obs_dim
        self.action_dims = action_dims
        
        self.distill_student = KnowledgeDistillationNetwork(camera_obs_dim).to(device)
        self.distill_student.eval()
        self.processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k", use_fast=True
        )
        self.distilled_teacher = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
        self.distilled_teacher.eval()
        
        self.actor_net = SACNetWithDistillation(
            self.distill_student, camera_obs_dim, vector_obs_dim, action_dims
        ).to(device)
        self.critic_1_net = SACNetWithDistillation(
            self.distill_student, camera_obs_dim, vector_obs_dim, action_dims
        ).to(device)
        self.critic_2_net = SACNetWithDistillation(
            self.distill_student, camera_obs_dim, vector_obs_dim, action_dims
        ).to(device)

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=params.lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1_net.parameters(), lr=params.lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2_net.parameters(), lr=params.lr)
        self.optimizer_distill = optim.Adam(self.distill_student.parameters(), lr=params.lr)

        self.mini_batch_size = params.mini_batch_size
        self.sac_epochs = params.train_epochs

    def get_action(self, camera_obs, vector_obs, train=False):
        if not isinstance(camera_obs, torch.Tensor):
            camera_obs = torch.tensor(camera_obs, dtype=torch.float32, device=device)
        if not isinstance(vector_obs, torch.Tensor):
            vector_obs = torch.tensor(vector_obs, dtype=torch.float32, device=device)

        mu_v, values = self.actor_net(camera_obs, vector_obs)
        values = values.squeeze(-1)
        std = torch.full_like(mu_v, self.params.action_std)

        # Replace any NaN in mu_v with zeros
        if torch.isnan(mu_v).any():
            mu_v = torch.zeros_like(mu_v)

        action_distribution = torch.distributions.Normal(mu_v, std)
        action = action_distribution.sample()
        action = torch.round(action).clamp(-1, 1)
        if not train:
            return (
                action.cpu().numpy()[0],
                action_distribution.log_prob(action).cpu().detach().numpy()[0],
                values.cpu().detach().numpy()[0],
            )
        return action.cpu().numpy()[0], action_distribution, values

    def improv_lr(self, optimizer, lr, timesteps, num_timesteps):
        lr = lr * (1 - timesteps / num_timesteps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return optimizer

    def fine_tune_teacher(self, trajectories):
        # Set teacher and student to training mode for distillation
        self.distilled_teacher.train()
        self.distill_student.train()
        for _ in range(self.params.train_epochs):
            for sample in trajectories.sample():
                camera_obs, *_ = sample
                camera_obs_rescaled = (camera_obs * 255).byte()
                teacher_in = self.processor(
                    images=camera_obs_rescaled, return_tensors="pt"
                ).pixel_values.to(device)

                teacher_out = self.distilled_teacher(teacher_in).last_hidden_state[:, 0, :]
                soft_teacher = F.softmax(teacher_out, dim=-1)
                student_out = self.distill_student(camera_obs, distill=True)
                log_soft_student = F.log_softmax(student_out, dim=-1)
                distill_loss = F.kl_div(log_soft_student, soft_teacher, reduction="batchmean")

                self.optimizer_distill.zero_grad()
                distill_loss.backward()
                self.optimizer_distill.step()
        self.distilled_teacher.eval()
        self.distill_student.eval()

    def train(self, steps, trajectories):
        # Put networks into training mode
        self.actor_net.train()
        self.critic_1_net.train()
        self.critic_2_net.train()
        self.distill_student.train()

        samples_processed = 0

        for epoch in range(self.sac_epochs):
            for sample in trajectories.sample():
                if samples_processed >= steps:
                    break

                (camera_obs, vector_obs, actions, rewards,
                 next_camera_obs, next_vector_obs, dones) = sample

                camera_obs = camera_obs.to(device)
                vector_obs = vector_obs.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_camera_obs = next_camera_obs.to(device)
                next_vector_obs = next_vector_obs.to(device)
                dones = dones.to(device)

                # --- Actor update ---
                mu_v, dist_batch, _ = self.actor_net(camera_obs, vector_obs)
                std = torch.full_like(mu_v, self.params.action_std)
                action_distribution = torch.distributions.Normal(mu_v, std)
                log_probs = action_distribution.log_prob(actions).sum(dim=-1, keepdim=True)
                entropy = action_distribution.entropy().mean()
                actor_loss = -(log_probs * dist_batch).mean() - self.params.entropy * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                # --- Critic update ---
                with torch.no_grad():
                    next_mu_v, _, _ = self.actor_net(next_camera_obs, next_vector_obs)
                    next_values = torch.min(
                        self.critic_1_net(next_camera_obs, next_vector_obs)[1],
                        self.critic_2_net(next_camera_obs, next_vector_obs)[1]
                    )
                    target_q = rewards + self.params.gamma * (1 - dones) * next_values

                q1 = self.critic_1_net(camera_obs, vector_obs)[1]
                critic_1_loss = F.mse_loss(q1, target_q)
                q2 = self.critic_2_net(camera_obs, vector_obs)[1]
                critic_2_loss = F.mse_loss(q2, target_q)

                self.critic_1_optimizer.zero_grad()
                critic_1_loss.backward(retain_graph=True)
                self.critic_1_optimizer.step()

                self.critic_2_optimizer.zero_grad()
                critic_2_loss.backward(retain_graph=True)
                self.critic_2_optimizer.step()

                # --- Distillation update ---
                camera_obs_rescaled = (camera_obs * 255).byte()
                teacher_in = self.processor(
                    images=camera_obs_rescaled, return_tensors="pt"
                ).pixel_values.to(device)
                teacher_out = self.distilled_teacher(teacher_in).last_hidden_state[:, 0, :]
                soft_teacher = F.softmax(teacher_out, dim=-1)
                student_out = self.distill_student(camera_obs, distill=True)
                log_soft_student = F.log_softmax(student_out, dim=-1)
                distill_loss = F.kl_div(
                    log_soft_student, soft_teacher, reduction="batchmean"
                )

                self.optimizer_distill.zero_grad()
                distill_loss.backward()
                nn.utils.clip_grad_norm_(self.distill_student.parameters(), 0.5)
                self.optimizer_distill.step()

                samples_processed += 1
            if samples_processed >= steps:
                break

        # Set networks back to evaluation mode
        self.actor_net.eval()
        self.critic_1_net.eval()
        self.critic_2_net.eval()
        self.distill_student.eval()

    def save(self, path):
        torch.save(
            {
                "actor": self.actor_net.state_dict(),
                "critic_1": self.critic_1_net.state_dict(),
                "critic_2": self.critic_2_net.state_dict(),
                "distill_student": self.distill_student.state_dict(),
                "optimizer_distill": self.optimizer_distill.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
                "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor_net.load_state_dict(checkpoint["actor"])
        self.critic_1_net.load_state_dict(checkpoint["critic_1"])
        self.critic_2_net.load_state_dict(checkpoint["critic_2"])
        self.distill_student.load_state_dict(checkpoint["distill_student"])
        self.optimizer_distill.load_state_dict(checkpoint["optimizer_distill"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(checkpoint["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(checkpoint["critic_2_optimizer"])
        self.actor_net.eval()
        self.critic_1_net.eval()
        self.critic_2_net.eval()
            
