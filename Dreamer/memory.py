import numpy as np
import torch
import pickle

class ExperienceReplay():
    def __init__(self, size, symbolic_env, observation_size, vector_obs, action_size, bit_depth, device):
        self.device = device
        self.symbolic_env = symbolic_env
        self.size = size
        self.camera_observation = np.empty((size, *observation_size), dtype=np.float32)
        self.vector_obs = np.empty((size, vector_obs), dtype=np.float32)
        self.action = np.empty((size, action_size), dtype=np.float32)
        self.reward = np.empty((size,), dtype=np.float32)
        self.nonterminals = np.empty((size,1), dtype=np.float32)
        self.idx = 0
        self.full = False
        self.steps, self.episodes = 0, 0
        self.bit_depth = bit_depth

    def add(self, observation, vector_obs, action, reward, done):
        
        self.camera_observation[self.idx] = observation
        self.vector_obs[self.idx] = vector_obs

        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.nonterminals[self.idx] = not done
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)

    def _sample_idx(self, L):
        valid_idx= False
        while not valid_idx:
            idx = np.random.randint(0, self.size if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.size
            valid_idx = not self.idx in idxs[1:]
        return idxs
    
    def preprocess_observation(self, observation, bit_depth):
        observation.div_(2**(8 - bit_depth)).floor_().div_(2**bit_depth).sub_(0.5) # Quantise and scale
        observation.add_(torch.rand_like(observation).div_(2**bit_depth)) # Dequantise with noise

    def postprocess_observation(self, observation, bit_depth):
        return np.clip(np.floor((observation + 0.5) * 2**bit_depth) * 2**(8 - bit_depth), 0, 255).astype(np.uint8) # Dequantise and un-normalise
    
    def _retrieve_batch(self, idxs, n, L):
        vec_idxs = idxs.transpose().reshape(-1)
        camera_obs = torch.tensor(self.camera_observation[vec_idxs], dtype=torch.float32)
        vector_obs = torch.tensor(self.vector_obs[vec_idxs], dtype=torch.float32)
        if not self.symbolic_env:
            self.preprocess_observation(camera_obs, self.bit_depth)
        return camera_obs.reshape(n, L, *camera_obs.shape[1:]), vector_obs.reshape(n,L,*vector_obs.shape[1:]), self.action[vec_idxs].reshape(n, L, -1), self.reward[vec_idxs].reshape(n, L), self.nonterminals[vec_idxs].reshape(n, L)
    
    def sample(self, n, L):
        batch = self._retrieve_batch(np.array([self._sample_idx(L) for _ in range(n)]), n, L)
        return [item.clone().to(self.device) if isinstance(item, torch.Tensor) else torch.Tensor(item).to(self.device) for item in batch]