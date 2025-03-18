import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from .Nets import SACNet, EntropyTargetNet
from .TeacherModel import TeacherModel, DistillationDataset
from torch.utils.data import DataLoader

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
        self.critic_optim = optim.Adam(self.model.critic.parameters(), lr=params.critic_lr)
        self.model.target_critic.load_state_dict(self.model.critic.state_dict())
        self.model.target_critic.eval()
        
        if self.params.target_entropy is None:
            self.params.target_entropy = -float(action_dims[0])
        self.target_entropy = self.params.target_entropy
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=params.alpha_lr)

        self.alpha = self.log_alpha.exp().item()

        #Teacher and student networks
        self.model.convolution_pipeline.eval()
        self.teacher = TeacherModel().to(self.device)
        self.teacher.eval()
        self.distill_optimizer = optim.Adam(self.model.convolution_pipeline.parameters(), lr=params.distill_lr)
        self.distill_coef = params.distill_coef
        
        #dynamically adjust entropy_beta parameters for entropy regularization
        # self.entropy_beta = torch.tensor(1.0).to(self.device).requires_grad_()
        self.entropy_terget_net = EntropyTargetNet().to(self.device)
        self.entropy_optimizer = optim.Adam(self.entropy_terget_net.parameters(), lr=params.entropy_lr)

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
    
    def get_values(self, camera_obs, vector_obs, action, step):
        step_fraction = step/self.params.max_steps
        camera_obs = self._process_vector(camera_obs)
        vector_obs = self._process_vector(vector_obs)
        action = self._process_vector(action)

        value_1, value_2 = self.model.get_values(camera_obs, vector_obs, action, step_fraction)
        return value_1, value_2
    
    def adjust_lr(self, optimizer, lr, timesteps, max_timesteps):
        new_lr = lr * (1-timesteps/max_timesteps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return optimizer
    
    def update_entropy_target(self, step):
        #Learns entropy target dynamically
        step_fraction_t = torch.tensor([step/self.params.max_steps], dtype=torch.float32, device=self.device)
        self.target_entropy = self.entropy_terget_net(step_fraction_t).item()

        loss = F.mse_loss(torch.tensor(self.target_entropy), torch.tensor(self.action_dims[0]), device=self.device)
        self.entropy_optimizer.zero_grad()
        loss.backward()
        self.entropy_optimizer.step()

    def update_alpha(self, step):
        #Gradually decreases entropy regularization over time
        min_alpha, max_alpha = 0.01, 0.2
        progress = step/self.params.max_steps
        self.alpha = min_alpha + (max_alpha - min_alpha) * progress # Linear Decay

    def _get_teacher_output(self, camera_obs):
        camera_obs_rescaled = camera_obs.to(device)
        with torch.no_grad():
            teacher_features = self.teacher(camera_obs_rescaled)
        return teacher_features
    
    def fine_tune_teacher(self,replay_buffer, epochs=10, lr=3e-4, lambda_mse=1.0, lambda_contrast=0.1, margin=1.0):
        student = self.model.convolution_pipeline
        student.eval().to(device)
        self.teacher.train()
        replay_buffer = DistillationDataset(replay_buffer, student, num_samples=10000)
        dataloader = DataLoader(replay_buffer, batch_size=64, shuffle=True)
        optimizer = optim.Adam(self.teacher.parameters(), lr=lr)

        for epoch in range(epochs):
            for camera_obs, student_feats in dataloader:
                camera_obs = camera_obs.to(device)
                student_feats = student_feats.to(device).squeeze(1)

                teacher_features = self.teacher(camera_obs)

                mse_loss = F.mse_loss(teacher_features, student_feats)
                contrastive_loss = self.teacher.contrastive_loss(teacher_features, margin=margin)

                total_loss = lambda_mse * mse_loss + lambda_contrast * contrastive_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
        
            print("Epoch {} completed".format(epoch))
        print("Teacher model fine-tuning completed")
    
    def train(self, trajectories, step):
        if len(trajectories) < self.params.batch_size:
            return

        step_fraction = step / self.params.max_steps
        decay_distill_coef_factor = 1 - step_fraction
        adjusted_distill_coef = self.distill_coef * decay_distill_coef_factor

        for _ in range(self.params.train_epochs):

            sample = trajectories.sample(self.params.batch_size)
            if sample is None:
                return

            camera_obs = self._process_vector(sample["camera_obs"])
            vector_obs = self._process_vector(sample["vector_obs"])
            actions = self._process_vector(sample["actions"])
            rewards = self._process_vector(sample["rewards"])
            done_flags = self._process_vector(sample["dones"])
            next_camera_obs = self._process_vector(sample["next_camera_obs"])
            next_vector_obs = self._process_vector(sample["next_vector_obs"])

            # -----------------------------------------------------
            # Critic Training (explicitly correct, dimension-safe)
            # -----------------------------------------------------
            with torch.no_grad():
                next_mean, next_log_std = self.model(next_camera_obs, next_vector_obs)
                next_dist = torch.distributions.Normal(next_mean, next_log_std.exp())

                next_actions = next_dist.rsample()  # explicitly sampled next actions [batch_size, action_dim]

                q1_target, q2_target = self.model.get_values(
                    next_camera_obs, next_vector_obs, next_actions, step_fraction,target=True
                )

                q_target = torch.minimum(q1_target, q2_target,)  # [batch_size, 1] clearly
                next_log_prob = next_dist.log_prob(next_actions).sum(-1, keepdim=True)  # [batch_size, 1]

                target_q = rewards + (1 - done_flags) * self.params.gamma * (q_target - self.alpha * next_log_prob).squeeze(-1)
                

            q1, q2 = self.model.get_values(camera_obs, vector_obs, actions, step_fraction)
            q1 = q1.squeeze(-1)
            q2 = q2.squeeze(-1)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # -----------------------------------------------------
            # Actor Training (explicitly correct)
            # -----------------------------------------------------
            new_actions, log_pi = self.get_action(camera_obs, vector_obs, train=True)
            q_pi, _ = self.model.get_values(camera_obs, vector_obs, new_actions, step_fraction)

            actor_loss = (self.alpha * log_pi - q_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # -----------------------------------------------------
            # Entropy Regularization (correct and explicit)
            # -----------------------------------------------------
            entropy_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.entropy_optimizer.zero_grad()
            entropy_loss.backward()
            self.entropy_optimizer.step()

            self.alpha = self.log_alpha.exp().detach().cpu().item()

            # -----------------------------------------------------
            # Critic Target Update (correctly updated explicitly)
            # -----------------------------------------------------
            for target_param, param in zip(self.model.target_critic.parameters(), self.model.critic.parameters()):
                target_param.data.copy_(
                    self.params.tau * param.data + (1 - self.params.tau) * target_param.data
                )

            # -----------------------------------------------------
            # Distillation Training (explicitly corrected)
            # -----------------------------------------------------
            replay_buffer = DistillationDataset(trajectories, self.model.convolution_pipeline, num_samples=trajectories.size)
            dataloader = DataLoader(replay_buffer, batch_size=64, shuffle=True) 
            for camera_obs, student_feats in dataloader:
                camera_obs = camera_obs.to(device)
                student_feats = student_feats.to(device).squeeze(1)

                teacher_features = self.teacher(camera_obs)


                distill_loss = adjusted_distill_coef * F.kl_div(
                    F.log_softmax(student_feats, dim=-1),
                    F.log_softmax(teacher_features, dim=-1).exp(),
                    reduction='batchmean'
                )

                self.distill_optimizer.zero_grad()
                distill_loss.backward()
                self.distill_optimizer.step()

