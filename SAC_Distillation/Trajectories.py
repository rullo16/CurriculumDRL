import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class SAC_ExperienceBuffer:
    """
    Buffer for storing and sampling experience transitions.
    Supports calculating discounted returns and advantages.
    """
    def __init__(self, camera_obs_dim, vector_obs_dim, action_dim, params):
        self.buffer_size = params.buffer_size
        self.gamma = params.gamma
        self.lambda_ = params.lambda_

        self.camera_obs = np.zeros((self.buffer_size, *camera_obs_dim), dtype=np.float32)
        self.next_camera_obs = np.zeros((self.buffer_size, *camera_obs_dim), dtype=np.float32)
        self.vector_obs = np.zeros((self.buffer_size, *vector_obs_dim), dtype=np.float32)
        self.next_vector_obs = np.zeros((self.buffer_size, *vector_obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, action_dim[0]), dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)

        self.ptr, self.size = 0,0

    def store(self, camera_obs, vector_obs, action, reward, next_camera_obs, next_vector_obs, done, value):
        idx = self.ptr
        self.camera_obs[idx] = camera_obs
        self.vector_obs[idx] = vector_obs
        self.actions[idx] = action
        self.dones[idx] = done
        self.rewards[idx] = reward
        self.next_camera_obs[idx] = next_camera_obs
        self.next_vector_obs[idx] = next_vector_obs
        self.values[idx] = value

        self.ptr = (idx + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

        if self.size >= self.buffer_size:
            self._clear_old()

    def compute_advantages(self, last_value=0):
        advantages = np.zeros(self.size, dtype=np.float32)
        last_adv = 0

        for t in reversed(range(self.size)):
            if t  == self.size -1:
                next_val = last_value
            else:
                next_val = self.values[t+1]

            delta = self.rewards[t] + self.gamma * next_val * (1 - self.dones[t]) - self.values[t]
            advantages[t] = delta + self.gamma * self.lambda_ * (1 - self.dones[t]) * last_adv
            last_adv = advantages[t]

        self.advantages[:self.size] = advantages

    def _clear_old(self):
        self.ptr = 0
        self.size = 0

    def sample(self, batch_size):
        if self.size == 0:
            return None
        
        batch_idxs = np.random.choice(self.size, batch_size, replace=self.size<batch_size)
        batch = dict(
            camera_obs=self.camera_obs[batch_idxs],
            vector_obs=self.vector_obs[batch_idxs],
            actions=self.actions[batch_idxs],
            rewards=self.rewards[batch_idxs],
            next_camera_obs=self.next_camera_obs[batch_idxs],
            next_vector_obs=self.next_vector_obs[batch_idxs],
            dones=self.dones[batch_idxs],
            values=self.values[batch_idxs],
            # advantages=self.advantages[batch_idxs]
        )

        return batch

    def __len__(self):
        return self.size
    

