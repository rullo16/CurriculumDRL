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
        self.batch_size = params.batch_size
        self.mini_batch_size = params.mini_batch_size

        self.camera_obs = torch.zeros(self.max_steps, *self.camera_obs_dims)
        self.vector_obs = torch.zeros(self.max_steps, *self.vector_obs_dims)
        self.next_camera_obs = torch.zeros(self.max_steps, *self.camera_obs_dims)
        self.next_vector_obs = torch.zeros(self.max_steps, *self.vector_obs_dims)
        self.action_memory = torch.zeros(self.max_steps, *self.action_dim)
        self.rewards = torch.zeros(self.max_steps,)
        self.done_flags = torch.zeros(self.max_steps,)

        self.step = 0

    def add(self, camera_obs, vector_obs, action, reward, next_camera_obs, next_vector_obs, done):

        self.camera_obs[self.step].copy_(torch.as_tensor(camera_obs, dtype=torch.float32))
        self.vector_obs[self.step].copy_(torch.as_tensor(vector_obs, dtype=torch.float32))
        self.action_memory[self.step] = torch.as_tensor(action, dtype=torch.float32)
        self.rewards[self.step] = torch.as_tensor(reward, dtype=torch.float32)
        self.next_camera_obs[self.step].copy_(torch.as_tensor(next_camera_obs, dtype=torch.float32))
        self.next_vector_obs[self.step].copy_(torch.as_tensor(next_vector_obs, dtype=torch.float32))
        self.done_flags[self.step] = torch.as_tensor(done, dtype=torch.float32)

        self.step = (self.step + 1) % self.max_steps

    def sample(self):
        indices = BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, drop_last=True)
        for index in indices:
            camera_obs = self.camera_obs[index]
            vector_obs = self.vector_obs[index]
            actions = self.action_memory[index]
            rewards = self.rewards[index]
            next_camera_obs = self.next_camera_obs[index]
            next_vector_obs = self.next_vector_obs[index]
            dones = self.done_flags[index]

            yield camera_obs, vector_obs, actions, rewards, next_camera_obs, next_vector_obs, dones

