import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .Trajectories import ExperienceBuffer
from .Nets import PPONetWithDistillation, KnowledgeDistillationNetwork
from transformers import AutoImageProcessor, ViTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DistilledPPO:
    """
    Implements a PPO agent with knowledge distillation.
    """
    def __init__(self, camera_obs_dim, vector_obs_dim, action_dim, params):
        self.params = params
        self.camera_obs_dim = camera_obs_dim
        self.vector_obs_dim = vector_obs_dim
        self.action_dim = action_dim
        self.distill_student = KnowledgeDistillationNetwork(camera_obs_dim).to(device)
        self.distill_student.train(False)
        self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)
        self.distilled_teacher = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
        self.distilled_teacher.eval()
        self.net = PPONetWithDistillation(self.distill_student, camera_obs_dim, vector_obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=params.lr)
        self.optimizer_distill = optim.Adam(self.distill_student.parameters(), lr=params.lr)
        self.mini_batch_size = params.mini_batch_size
        self.ppo_epochs = params.train_epochs

    def get_action(self, camera_obs, vector_obs, train=False):
        if not isinstance(camera_obs, torch.Tensor):
            camera_obs = torch.tensor(camera_obs, dtype=torch.float32, device=device)
        if not isinstance(vector_obs, torch.Tensor):
            vector_obs = torch.tensor(vector_obs, dtype=torch.float32, device=device)

        mu_v, values = self.net(camera_obs, vector_obs)
        values = values.squeeze(-1)
        std = torch.full_like(mu_v, self.params.action_std)

        # Check for NaN values in mu_v
        if torch.isnan(mu_v).any():
            mu_v = torch.zeros_like(mu_v)

        action_probs = torch.distributions.normal.Normal(mu_v, std)
        action = action_probs.sample()
        action = torch.round(action).clamp(-1, 1)  # Ensure action values are -1, 0, or 1
        if not train:
            return action.cpu().numpy()[0], action_probs.log_prob(action).cpu().detach().numpy()[0], values.cpu().detach().numpy()[0]
        return action.cpu().numpy()[0], action_probs, values
    
    def improv_lr(self, optimizer, lr, timesteps, num_timesteps):
        lr = lr * (1 - timesteps / num_timesteps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer
    
    def train(self, steps, trajectories):
        batch_size = self.params.n_steps // self.mini_batch_size
        if batch_size < self.mini_batch_size:
            batch_size = self.mini_batch_size
        gradient_accumulation_steps = batch_size // self.mini_batch_size
        grad_step = 1

        self.net.train()

        for _ in range(self.params.train_epochs):
            generator = trajectories.sample()
            for sample in generator:
                camera_obs, vector_obs, actions, returns, advantages, action_log_probs, old_values = sample

                camera_obs = camera_obs.to(device)
                vector_obs = vector_obs.to(device)
                actions = actions.to(device)
                returns = returns.to(device)
                advantages = advantages.to(device)
                action_log_probs = action_log_probs.to(device)
                old_values = old_values.to(device)
                
                _, dist_batch, values_batch = self.get_action(camera_obs, vector_obs,train=True)

                log_prob_actions_batch = dist_batch.log_prob(actions).sum(dim=-1, keepdim=True)
                ratio = torch.exp(log_prob_actions_batch - action_log_probs).sum(dim=-1)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.params.epsilon, 1.0 + self.params.epsilon) * advantages
                actions_loss = -torch.min(surr1, surr2).mean()

                #Clipped Bellman error
                clipped_value = old_values + torch.clamp(old_values - values_batch, -self.params.epsilon, self.params.epsilon)
                v_surr1 = (values_batch - returns).pow(2)
                v_surr2 = (clipped_value - returns).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                # Distillation Training
                self.optimizer_distill.zero_grad()
                camera_obs_rescaled = (camera_obs * 255).byte()
                teacher_in = self.processor(images=camera_obs_rescaled, return_tensors="pt").pixel_values.to(device)
                teacher_out = self.distilled_teacher(teacher_in).last_hidden_state[:, 0, :]  # Extract the [CLS] token representation
                soft_teacher = F.softmax(teacher_out, dim=-1)
                student_out = self.distill_student(camera_obs, distill=True)
                log_soft_student = F.log_softmax(student_out, dim=-1)
                distill_loss = F.kl_div(log_soft_student, soft_teacher, reduction='batchmean')
                distill_loss.backward()

                #Policy entropy
                entropy_beta = max(0.01, self.params.entropy_coef * (1 - steps / self.params.n_steps))
                entropy_loss = dist_batch.entropy()
                if entropy_loss.ndim > 1:
                    entropy_loss = entropy_loss.sum(dim=-1)
                entropy_loss = entropy_loss.mean()

                loss = actions_loss + self.params.value_loss_coef * value_loss - entropy_beta * entropy_loss
                loss.backward()

                if grad_step % gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(self.distill_student.parameters(), 0.5)
                    self.optimizer_distill.step()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_step += 1

    def load_model(self, model_path):
        self.net.load_state_dict(torch.load(model_path, weights_only=True))
