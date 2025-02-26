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
# 2. Critic loss:
#    L_critic = E[(Q(s, a) - (r + γ * V(s')))^2]
# 3. Knowledge distillation loss:
#    L_KD = KL( softmax(teacher) || log_softmax(student) )
#
class DistilledSAC:
    """
    SAC agent with feature distillation. Uses a teacher ViT network and a student network for convolutional feature extraction.
    """
    def __init__(self, camera_obs_dim, vector_obs_dim, action_dims, params):
        self.device = device  # fix: use instance attribute for device
        self.params = params
        self.camera_obs_dim = camera_obs_dim
        self.vector_obs_dim = vector_obs_dim
        self.action_dims = action_dims

        # Initialize teacher and student networks
        self.distill_student = KnowledgeDistillationNetwork(camera_obs_dim).to(self.device)
        self.distill_student.eval()
        self.processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k", use_fast=True
        )
        self.distilled_teacher = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(self.device)
        self.distilled_teacher.eval()

        self.net = SACNetWithDistillation(
            camera_obs_dim, vector_obs_dim, action_dims, sac_distilled=self.distill_student
        ).to(self.device)

        self.actor_optimizer = optim.Adam(self.net.actor.parameters(), lr=params.actor_lr)
        self.critic_1_optimizer = optim.Adam(self.net.critic_1.parameters(), lr=params.critic_lr)
        self.critic_2_optimizer = optim.Adam(self.net.critic_2.parameters(), lr=params.critic_lr)
        self.optimizer_distill = optim.Adam(self.distill_student.parameters(), lr=params.distill_lr)
        self.alpha = torch.tensor(1.0, device=self.device, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.alpha], lr=params.alpha_lr)

        self.mini_batch_size = params.mini_batch_size
        self.sac_epochs = params.train_epochs

    def _process_camera_obs(self, camera_obs):
        if not isinstance(camera_obs, torch.Tensor):
            camera_obs = torch.as_tensor(camera_obs, dtype=torch.float32, device=self.device)
        return camera_obs

    def _process_vector_obs(self, vector_obs):
        if not isinstance(vector_obs, torch.Tensor):
            vector_obs = torch.as_tensor(vector_obs, dtype=torch.float32, device=self.device)
        return vector_obs

    def get_action(self, camera_obs, vector_obs, train=False):
        camera_obs = self._process_camera_obs(camera_obs)
        vector_obs = self._process_vector_obs(vector_obs)

        mu_v, std = self.net(camera_obs, vector_obs)
        # Handle potential nan values in mu_v
        if torch.isnan(mu_v).any():
            mu_v = torch.zeros_like(mu_v)

        action_distribution = torch.distributions.Normal(mu_v, std)
        action = action_distribution.sample()
        action = torch.round(action).clamp(-1, 1)
        log_prob = action_distribution.log_prob(action).sum(dim=-1, keepdim=True)

        # Evaluate critic values once action distribution is computed
        q1, q2 = self.net.get_critics(camera_obs, vector_obs, mu_v)
        values = torch.min(q1, q2).squeeze(-1)

        if not train:
            return (
                action[0].cpu().numpy(),
                log_prob[0].cpu().detach().numpy(),
                values[0].cpu().detach().numpy(),
            )
        return action[0].cpu().numpy(), action_distribution, values

    def adjust_lr(self, optimizer, lr, timesteps, num_timesteps):
        """Adjust the learning rate linearly over timesteps."""
        new_lr = lr * (1 - timesteps / num_timesteps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return optimizer

    def _get_teacher_output(self, camera_obs):
        # Rescale [0,1] images to [0,255] and convert to byte format
        camera_obs_rescaled = (camera_obs * 255).byte()
        # Add channel dimension if missing (assuming grayscale image)
        teacher_input = self.processor(
            images=camera_obs_rescaled, return_tensors="pt"
        ).pixel_values.to(self.device)
        # Do teacher inference without gradient calculation
        with torch.no_grad():
            teacher_features = self.distilled_teacher(teacher_input).last_hidden_state[:, 0, :]
        return teacher_features

    def fine_tune_teacher(self, trajectories):
        """
        Fine-tune the student network using teacher features.
        For each epoch, sample batches from trajectories and optimize the distillation loss.
        """
        self.distilled_teacher.train()
        self.distill_student.train()
        for epoch in range(self.params.train_epochs):
            for sample in trajectories.sample():
                # Unpack batch and process camera observations
                camera_obs, *_ = sample
                camera_obs = self._process_camera_obs(camera_obs)
                # Get teacher and student outputs
                teacher_out = self._get_teacher_output(camera_obs)
                soft_teacher = F.softmax(teacher_out, dim=-1)
                student_out = self.distill_student(camera_obs, distill=True)
                log_soft_student = F.log_softmax(student_out, dim=-1)
                # Compute the distillation loss (KL divergence)
                distill_loss = F.kl_div(log_soft_student, soft_teacher, reduction="batchmean")

                self.optimizer_distill.zero_grad()
                distill_loss.backward()
                nn.utils.clip_grad_norm_(self.distill_student.parameters(), max_norm=0.5)
                self.optimizer_distill.step()
        self.distilled_teacher.eval()
        self.distill_student.eval()

    def train(self, steps, trajectories):

        self.net.train()
        self.distill_student.train()
        target_entropy = -float(self.action_dims[0])
        samples_processed = 0

        for epoch in range(self.sac_epochs):
            for sample in trajectories.sample():
                if samples_processed > steps:
                    break

                # Unpack mini-batch and send tensors to device
                camera_obs, vector_obs, actions, action_logs, old_values, advantages, returns, rewards, dones = sample
                camera_obs = self._process_camera_obs(camera_obs)
                vector_obs = self._process_vector_obs(vector_obs)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                returns = returns.to(self.device)
                # Ensure action_logs, old_values, advantages are float tensors on device
                action_logs = action_logs.to(self.device).float()
                old_values = old_values.to(self.device).float()
                advantages = advantages.to(self.device).float()

                # --- Actor update ---
                mu_v, std = self.net(camera_obs, vector_obs)
                action_dist = torch.distributions.Normal(mu_v, std)
                new_log_probs = action_dist.log_prob(mu_v).sum(dim=-1, keepdim=True)
                # Difference: Uses importance sampling ratio between new and stored log_probs
                ratio = torch.exp(new_log_probs - action_logs)
                # Actor objective modified to reweight advantage estimates by the ratio and include entropy term
                actor_loss = (-ratio * advantages + self.alpha * new_log_probs).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # --- Alpha update ---
                # Difference: Alpha (temperature) update using target entropy
                alpha_loss = -(self.alpha * (new_log_probs.detach() + target_entropy)).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                # --- Critic update ---
                # Compute target Q using next actions from the actor (differentiable sampling)
                with torch.no_grad():
                    next_action = action_dist.rsample()  # Differentiable sampling for target Q
                    next_q1, next_q2 = self.net.get_critics(camera_obs, vector_obs, next_action)
                    next_values = torch.min(next_q1, next_q2)
                    bootstrapped_q = rewards + self.params.gamma * (1 - dones) * next_values
                    # Integrate returns from the trajectory with the bootstrapped target
                    target_q = 0.5 * (bootstrapped_q + returns)

                # Get current Q estimates for the taken actions
                q1, q2 = self.net.get_critics(camera_obs, vector_obs, actions)
                
                # Difference: Compute advantage prediction from current critic estimates using actor's output
                q1_pi, q2_pi = self.net.get_critics(camera_obs, vector_obs, mu_v)
                min_q_pi = torch.min(q1_pi, q2_pi)
                advantage_pred = min_q_pi - old_values
                advantage_loss = F.mse_loss(advantage_pred, advantages)

                critic_1_loss = F.mse_loss(q1, target_q) + advantage_loss
                critic_2_loss = F.mse_loss(q2, target_q) + advantage_loss

                self.critic_1_optimizer.zero_grad()
                critic_1_loss.backward()
                self.critic_1_optimizer.step()

                self.critic_2_optimizer.zero_grad()
                critic_2_loss.backward()
                self.critic_2_optimizer.step()

                # --- Distillation update ---
                # Difference: Added knowledge distillation loss to align student network with teacher features
                teacher_out = self._get_teacher_output(camera_obs)
                soft_teacher = F.softmax(teacher_out, dim=-1)
                student_out = self.distill_student(camera_obs, distill=True)
                log_soft_student = F.log_softmax(student_out, dim=-1)
                distill_loss = F.kl_div(log_soft_student, soft_teacher, reduction="batchmean")

                self.optimizer_distill.zero_grad()
                distill_loss.backward()
                nn.utils.clip_grad_norm_(self.distill_student.parameters(), max_norm=0.5)
                self.optimizer_distill.step()

                samples_processed += 1

            if samples_processed >= steps:
                break

        self.net.eval()
        self.distill_student.eval()
        # Difference: Update the convolution pipeline to the fine-tuned student network after training
        self.net.convolution_pipeline = self.distill_student

    def save(self, path):
        torch.save(
            {
                "actor": self.net.actor.state_dict(),
                "critic_1": self.net.critic_1.state_dict(),
                "critic_2": self.net.critic_2.state_dict(),
                "distill_student": self.distill_student.state_dict(),
                "optimizer_distill": self.optimizer_distill.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
                "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
                "alpha": self.alpha,
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.net.actor.load_state_dict(checkpoint["actor"])
        self.net.critic_1.load_state_dict(checkpoint["critic_1"])
        self.net.critic_2.load_state_dict(checkpoint["critic_2"])
        self.distill_student.load_state_dict(checkpoint["distill_student"])
        self.optimizer_distill.load_state_dict(checkpoint["optimizer_distill"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(checkpoint["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(checkpoint["critic_2_optimizer"])
        self.alpha = checkpoint.get("alpha", self.alpha)
        # Ensure alpha remains a learnable parameter
        self.alpha.requires_grad_(True)
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        self.net.eval()
        self.distill_student.eval()

