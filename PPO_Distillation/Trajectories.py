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
        self.rewards_split = torch.zeros(self.max_steps+1,2)
        self.accumulated_rewards = torch.zeros(self.max_steps+1,)
        self.done_flags = torch.zeros(self.max_steps+1,)
        self.action_log_probs = torch.zeros(self.max_steps,)
        self.ext_values_estimations = torch.zeros(self.max_steps+1,)
        self.int_values_estimations = torch.zeros(self.max_steps+1,)
        self.return_estimations = torch.zeros(self.max_steps+1,)
        self.adv_estimations = torch.zeros(self.max_steps,)
        self.int_ref_values = torch.zeros(self.max_steps+1,)
        self.ext_ref_values = torch.zeros(self.max_steps+1,)

        self.step = 0
        

    def add(self, camera_obs, vector_obs, action, rewards, done, log_prob, ext_value, int_value):

        self.camera_obs[self.step].copy_(torch.as_tensor(camera_obs, dtype=torch.float32))
        self.vector_obs[self.step].copy_(torch.as_tensor(vector_obs, dtype=torch.float32))
        self.action_memory[self.step] = torch.as_tensor(action, dtype=torch.float32)
        self.rewards_split[self.step] = torch.as_tensor(rewards, dtype=torch.float32)
        self.done_flags[self.step] = torch.as_tensor(done, dtype=torch.float32)
        self.action_log_probs[self.step] = torch.as_tensor(log_prob, dtype=torch.float32)
        self.ext_values_estimations[self.step] = torch.as_tensor(ext_value, dtype=torch.float32)
        self.int_values_estimations[self.step] = torch.as_tensor(int_value, dtype=torch.float32)

        self.step = (self.step + 1) % self.max_steps

    def add_final_state(self, camera_obs, vector_obs, ext_value, int_value):
        self.camera_obs[-1].copy_(camera_obs)
        self.vector_obs[-1].copy_(vector_obs)
        self.ext_values_estimations[-1] = ext_value
        self.int_values_estimations[-1] = int_value

    def compute_advantages_and_returns(self, normalize_advantages=True):
        rewards = self.rewards_split

        for step in reversed(range(self.max_steps - 1)):
            combined_values = self.int_values_estimations[step] + self.ext_values_estimations[step]
            next_combined_values = self.int_values_estimations[step + 1] + self.ext_values_estimations[step + 1]
            delta = rewards[step] + self.discount_factor * next_combined_values * (1-self.done_flags[step]) - combined_values
            self.adv_estimations[step] = delta + self.discount_factor * self.lambda_factor * (1-self.done_flags[step]) * self.adv_estimations[step + 1]

        self.return_estimations = self.adv_estimations + (self.ext_values_estimations[:-1] + self.int_values_estimations[:-1])

        if normalize_advantages:
            self.adv_estimations = (self.adv_estimations - self.adv_estimations.mean()) / (self.adv_estimations.std() + 1e-8)

    def compute_reference_values(self):
        ext_rewards = self.rewards_split[:,0]
        int_rewards = self.rewards_split[:,1]
        extrinsic_buffer = torch.zeros_like(ext_rewards)
        intrinsic_buffer = torch.zeros_like(int_rewards)

        for step in reversed(range(self.total_steps-1)):
            delta_ext = ext_rewards[step] + self.discount_factor * self.ext_ref_values[step + 1] * (1-self.done_flags[step])-self.ext_values_estimations[step]
            delta_int = int_rewards[step] + self.discount_factor * self.int_ref_values[step + 1] * (1-self.done_flags[step])-self.int_values_estimations[step]

            extrinsic_buffer[step] = delta_ext + self.discount_factor * self.lambda_factor * (1-self.done_flags[step]) * extrinsic_buffer[step + 1]
            intrinsic_buffer[step] = delta_int + self.discount_factor * self.lambda_factor * (1-self.done_flags[step]) * intrinsic_buffer[step + 1]

        self.ext_ref_values = extrinsic_buffer + self.ext_values_estimations[:-1]
        self.int_ref_values = intrinsic_buffer + self.int_values_estimations[:-1]

    def sample(self):
        indices = BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, drop_last=True)
        for index in indices:
            camera_obs = self.camera_obs[:-1].reshape(-1, *self.camera_obs_dims)[index]
            vector_obs = self.vector_obs[:-1].reshape(-1, *self.vector_obs_dims)[index]
            actions = self.action_memory.reshape(-1)[index]
            returns = self.return_estimations.reshape(-1)[index]
            advantages = self.adv_estimations[index].reshape(-1)[index]
            action_log_probs = self.action_log_probs.reshape(-1)[index]
            ext_values = self.ext_values_estimations[:-1].reshape(-1)[index]
            int_values = self.int_values_estimations[:-1].reshape(-1)[index]

            yield camera_obs, vector_obs, actions, returns, advantages, action_log_probs, ext_values, int_values
        

