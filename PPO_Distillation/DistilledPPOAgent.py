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
    def __init__(self, camera_obs_dim, vector_obs_dim, action_dim, params):
        self.params = params
        self.camera_obs_dim = camera_obs_dim
        self.vector_obs_dim = vector_obs_dim
        self.action_dim = action_dim
        self.distill_student = KnowledgeDistillationNetwork(camera_obs_dim).to(device)
        self.distill_student.train(False)
        self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.distilled_teacher = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.net = PPONetWithDistillation(self.distill_student, camera_obs_dim, vector_obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=params.lr)
        self.optimizer_distill = optim.Adam(self.distill_student.parameters(), lr=params.lr)
        self.mini_batch_size = params.mini_batch_size
        self.ppo_epochs = params.train_epochs

    def get_action(self, camera_obs, vector_obs):
        if not isinstance(camera_obs, torch.Tensor):
            camera_obs = torch.tensor(camera_obs, dtype=torch.float32).to(device)
        if not isinstance(vector_obs, torch.Tensor):
            vector_obs = torch.tensor(vector_obs, dtype=torch.float32).to(device)

        mu_v, values = self.net(camera_obs, vector_obs)
        values = values.squeeze(-1)
        action_probs = F.log_softmax(mu_v, dim=1)
        # action_probs = torch.clip(action_probs, -20, 20)
        action_probs = torch.distributions.Categorical(logits=(action_probs))
        action = action_probs.sample()
        return action.cpu().numpy(), action_probs, values
    
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
                camera_obs, vector_obs, actions, returns, advantages, action_log_probs, ext_values, int_values = sample

                camera_obs = camera_obs.to(device)
                vector_obs = vector_obs.to(device)
                actions = actions.to(device)
                returns = returns.to(device)
                advantages = advantages.to(device)
                action_log_probs = action_log_probs.to(device)
                ext_values = ext_values.to(device)
                int_values = int_values.to(device)
                
                _, dist_batch, values = self.get_action(camera_obs, vector_obs)

                log_prob_act_batch = dist_batch.log_prob(actions)
                ratio = torch.exp(log_prob_act_batch - action_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.params.epsilon, 1.0 + self.params.epsilon) * advantages
                actions_loss = -torch.min(surr1, surr2).mean()

                #Clipped Bellman error
                clipped_value_ext = ext_values + torch.clamp(values - ext_values, -self.params.epsilon, self.params.epsilon)
                clipped_value_int = int_values + torch.clamp(values - int_values, -self.params.epsilon, self.params.epsilon)
                v_surr1 = (values - returns).pow(2) + (values - returns).pow(2)
                v_surr2 = (clipped_value_ext - returns).pow(2) + (clipped_value_int - returns).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                #distillation Training
                self.optimizer_distill.zero_grad()
                print(camera_obs[0])
                teacher_in = self.processor(images=camera_obs, return_tesors="pt", do_rescale=False)
                teacher_out = self.distilled_teacher(**camera_obs).last_hidden_state
                student_out = self.distill_student(camera_obs, vector_obs)
                distill_loss = F.mse_loss(student_out, teacher_out)
                distill_loss.backward()

                #Policy entropy
                entropy_beta = max(0.01, self.params.entropy_coef * (1 - steps / self.params.n_steps))
                entropy_loss = dist_batch.entropy().mean()
                loss = actions_loss + self.params.value_loss_coef * value_loss - entropy_beta * entropy_loss
                loss.backward()

                if grad_step % gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(self.distill_student.parameters(), 0.5)
                    self.optimizer_distill.step()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_step += 1
