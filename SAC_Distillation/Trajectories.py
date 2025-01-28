import torch
import numpy as np
from collections import deque
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class ExperienceBuffer:

    def __init__(self, camera_obs_dim, vector_obs_dim, action_dim, params):

        self.camera_obs_dims = camera_obs_dim
        self.vector_obs_dims = vector_obs_dim
        self.action_dim = action_dim
        self.max_steps = params.max_steps
        self.discount_factor = params.gamma
        self.lambda_factor = params.lam
        self.batch_size = params.batch_size
        self.mini_batch_size = params.mini_batch_size

        self.camera_obs = torch.zeros(self.max_steps+1, *self.camera_obs_dims)
        self.vector_obs = torch.zeros(self.max_steps+1, *self.vector_obs_dims)
        self.action_memory = torch.zeros(self.max_steps, *self.action_dim)
        self.rewards = torch.zeros(self.max_steps+1,)
        self.accumulated_rewards = torch.zeros(self.max_steps+1,)
        self.done_flags = torch.zeros(self.max_steps+1,)
        self.action_log_probs = torch.zeros(self.max_steps, *self.action_dim)
        self.values_estimations = torch.zeros(self.max_steps+1,)
        self.return_estimations = torch.zeros(self.max_steps+1,)
        self.adv_estimations = torch.zeros(self.max_steps,)
        self.ref_values = torch.zeros(self.max_steps+1,)

        self.step = 0
        

    def add(self, camera_obs, vector_obs, action, rewards, done, log_prob, value):

        self.camera_obs[self.step].copy_(torch.as_tensor(camera_obs, dtype=torch.float32))
        self.vector_obs[self.step].copy_(torch.as_tensor(vector_obs, dtype=torch.float32))
        self.action_memory[self.step] = torch.as_tensor(action, dtype=torch.float32)
        self.rewards[self.step] = torch.as_tensor(rewards, dtype=torch.float32)
        self.done_flags[self.step] = torch.as_tensor(done, dtype=torch.float32)
        self.action_log_probs[self.step] = torch.as_tensor(log_prob, dtype=torch.float32)
        self.values_estimations[self.step] = torch.as_tensor(value, dtype=torch.float32)

        self.step = (self.step + 1) % self.max_steps

    def add_final_state(self, camera_obs, vector_obs, value):

        self.camera_obs[-1].copy_(torch.as_tensor(camera_obs, dtype=torch.float32))
        self.vector_obs[-1].copy_(torch.as_tensor(vector_obs, dtype=torch.float32))
        self.values_estimations[-1] = torch.as_tensor(value, dtype=torch.float32)

    def compute_advantages_and_returns(self, normalize_advantages=True):
        rewards = self.rewards

        for step in reversed(range(self.max_steps - 1)):
            combined_values = self.values_estimations[step]
            next_combined_values = self.values_estimations[step + 1]
            delta = rewards[step] + self.discount_factor * next_combined_values * (1-self.done_flags[step]) - combined_values
            self.adv_estimations[step] = delta + self.discount_factor * self.lambda_factor * (1-self.done_flags[step]) * self.adv_estimations[step + 1]

        self.return_estimations = self.adv_estimations + (self.values_estimations[:-1])

        if normalize_advantages:
            self.adv_estimations = (self.adv_estimations - self.adv_estimations.mean()) / (self.adv_estimations.std() + 1e-8)

    def sample(self):
        indices = BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, drop_last=True)
        for index in indices:
            camera_obs = self.camera_obs[:-1].reshape(-1, *self.camera_obs_dims)[index]
            vector_obs = self.vector_obs[:-1].reshape(-1, *self.vector_obs_dims)[index]
            actions = self.action_memory[index]
            returns = self.return_estimations.reshape(-1)[index]
            advantages = self.adv_estimations[index]
            action_log_probs = self.action_log_probs[index]
            values = self.values_estimations[:-1].reshape(-1)[index]

            yield camera_obs, vector_obs, actions, returns, advantages, action_log_probs, values

