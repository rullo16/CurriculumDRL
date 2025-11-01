import numpy as np


def _norm_reward(r,d,mean, std):
    r = r.copy()
    idx = (d == 0.0)
    r[idx] = (r[idx] - mean) / (std + 1e-8)
    return r

class RunningStat:
    def __init__(self, shape, eps=1e-4):
        self._mean = np.zeros(shape, dtype=np.float32)
        self._var = np.ones(shape, dtype=np.float32)
        self._count = eps

    def update(self, x):
        """
        Accepts either a single sample with shape == self._mean.shape
        or a batch with shape (B, *self._mean.shape).
        Works for scalars, vectors, images – no flattening needed.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return

        target_shape = self._mean.shape
        if x.shape == target_shape:                  # 1 sample, add batch-dim
            x = x.reshape((1, *target_shape))
        elif x.shape[1:] != target_shape:            # shape mismatch → loud fail
            raise ValueError(
                f"RunningStat update shape mismatch: got {x.shape}, "
                f"expected (*, {target_shape})."
            )

        # ---- standard incremental update -----------------------------------
        batch_mean  = x.mean(axis=0)
        batch_var   = x.var(axis=0)
        batch_count = x.shape[0]

        delta       = batch_mean - self._mean
        tot_count   = self._count + batch_count

        self._mean += delta * batch_count / tot_count
        m_a         = self._var * self._count
        m_b         = batch_var * batch_count
        m2          = m_a + m_b + (delta ** 2) * self._count * batch_count / tot_count
        self._var   = m2 / tot_count
        self._count = tot_count
    
    @property
    def mean(self): return self._mean
    @property
    def std(self): return np.sqrt(self._var + 1e-8)

class SAC_ExperienceBuffer:
    """
    Buffer for storing and sampling experience transitions.
    Supports calculating discounted returns and advantages.
    """
    def __init__(self, camera_obs_dim, vector_obs_dim, action_dim, params):
        self.buffer_size = params.get('buffer_size', 1_000_000)
        self.gamma = params.get('gamma', 0.99)
        self.lambda_ = params.get('lambda_', 0.95)
        self.camera_stat = RunningStat(camera_obs_dim)
        self.vector_stat = RunningStat(vector_obs_dim)
        self.reward_stat = RunningStat((1,))

        self.camera_obs = np.zeros((self.buffer_size, *camera_obs_dim), dtype=np.uint8)
        self.next_camera_obs = np.zeros((self.buffer_size, *camera_obs_dim), dtype=np.uint8)
        self.vector_obs = np.zeros((self.buffer_size, *vector_obs_dim), dtype=np.float32)
        self.next_vector_obs = np.zeros((self.buffer_size, *vector_obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, action_dim[0]), dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.priorities = np.zeros(self.buffer_size, dtype=np.float32)  # Add priorities

        self.ptr, self.size = 0, 0

    def store(self, camera_obs, vector_obs, action, reward, next_camera_obs, next_vector_obs, done, priority=1.0):
        idx = self.ptr % self.buffer_size
        self.camera_obs[idx] = camera_obs
        self.vector_obs[idx] = vector_obs
        self.actions[idx] = action
        self.dones[idx] = done

        
        self.reward_stat.update(reward)
        self.rewards[idx] = reward
        
        self.next_camera_obs[idx] = next_camera_obs
        self.next_vector_obs[idx] = next_vector_obs

        self.priorities[idx] = priority  # Store priority

        self.camera_stat.update(camera_obs)
        self.vector_stat.update(vector_obs)

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def store_joint(self, camera_obs, vector_obs, action, reward, next_camera_obs, next_vector_obs, done, priority=1.0, num_agents=4):
        
        for i in range(camera_obs.shape[0]):
            self.store(camera_obs[i], vector_obs[i], action[i], reward[i], next_camera_obs[i], next_vector_obs[i], done[i], priority)

    def sample(self, batch_size, alpha=0.6, beta=0.4, n_step=1):
        if self.size == 0:
            return None

        # Compute probabilities based on priorities
        scaled_priorities = self.priorities[:self.size] ** alpha
        prob_sum = scaled_priorities.sum()
        if prob_sum == 0 or np.isnan(scaled_priorities).any():
            probs = np.full(self.size, 1.0 / self.size, dtype=np.float32)
        else:
            probs = scaled_priorities / prob_sum

        batch_idxs = np.random.choice(self.size, batch_size, replace=self.size < batch_size, p=probs)
        k = np.arange(n_step, dtype=np.int32)
        idxs_k = np.minimum(batch_idxs[:, None] + k, self.size - 1)

        r_k = self.rewards[idxs_k]
        d_k = self.dones[idxs_k]

        discounts = (self.gamma ** k).astype(np.float32)
        mask = np.ones_like(d_k, dtype=np.float32)
        for i in range(1, n_step):
            mask[:, i] = mask[:,i-1]*(1.0-d_k[:, i-1])
        nstep_returns = (r_k * discounts * mask).sum(axis=1, keepdims=True)

        idx_n = np.minimum(batch_idxs + n_step, self.size - 1)
        
        weights = (self.size * probs[batch_idxs])**(-beta)
        weights = (weights / weights.max()).astype(np.float32)[:, None]

        norm_cam = lambda x: (x - self.camera_stat.mean) / self.camera_stat.std
        norm_vec = lambda x: (x - self.vector_stat.mean) / self.vector_stat.std

        return dict(
            camera_obs = norm_cam(self.camera_obs[batch_idxs]),
            vector_obs = norm_vec(self.vector_obs[batch_idxs]),
            actions = self.actions[batch_idxs],
            rewards = _norm_reward(self.rewards[batch_idxs,None], self.dones[batch_idxs,None], self.reward_stat.mean, self.reward_stat.std),
            rewards = self.rewards[batch_idxs, None],
            next_camera_obs = norm_cam(self.next_camera_obs[batch_idxs]),
            next_vector_obs = norm_vec(self.next_vector_obs[batch_idxs]),
            dones = self.dones[batch_idxs, None],
            indices = batch_idxs,
            weights = weights,
            # nstep_returns = _norm_reward(nstep_returns, d_k[:,0:1], self.reward_stat.mean, self.reward_stat.std),
            nstep_returns = nstep_returns,
            nstep_next_idxs = idx_n
        )
    
    def sample_joint(self, batch_size, alpha=0.6, beta=0.4, n_step=1, num_agents=4):
        
        assert self.size % num_agents == 0, "Buffer size must be divisible by number of agents."
        E = self.size // num_agents

        env_priorities = self.priorities[:E * num_agents].reshape(E, num_agents).max(axis=1)
        probs = env_priorities ** alpha
        probs_sum = probs.sum()
        if probs_sum == 0 or np.isnan(probs).any():
            probs = np.ones_like(env_priorities) / len(env_priorities)
        else:
            probs /= probs.sum()

        env_idxs = np.random.choice(E, batch_size, replace=E < batch_size, p=probs)

        idx_matrix = env_idxs[:, None] * num_agents + np.arange(num_agents)
        idx_flat = idx_matrix.reshape(-1)
        
        weights = (E * probs[env_idxs]) ** (-beta)
        weights /= weights.max()
        weights = np.repeat(weights, num_agents).astype(np.float32).reshape(batch_size, num_agents, 1)

        k = np.arange(n_step, dtype=np.int32)
        idxs_k = np.minimum(idx_flat[:, None] + k, self.size - 1)

        # norm_r = lambda x: (x - self.reward_stat.mean) / (self.reward_stat.std + 1e-8)
        # r_k = norm_r(self.rewards[idxs_k])
        # d_k = self.dones[idxs_k]

        # disconts = (self.gamma ** k).astype(np.float32)
        # mask = np.cumprod(1.0-d_k, axis=1, dtype=np.float32)
        # mask[:, 0] = 1.0  # Ensure the first step is always included in the mask
        # first_step = (d_k[:,0:1] == 1.0).astype(np.float32)
        # nstep_ret = first_step + r_k[:, 0:1] + (1-first_step) * (r_k * disconts * mask).sum(axis=1, keepdims=True)
        r_k = self.rewards[idxs_k]
        d_k = self.dones[idxs_k]

        disconts = (self.gamma ** k).astype(np.float32)

        mask = np.cumprod(1.0 - d_k, axis=1, dtype=np.float32)
        mask[:, 0] = 1.0  # Ensure the first step is always included in the mask

        n_step_ret = (r_k * disconts * mask).sum(axis=1, keepdims=True)

        idx_n_flat = np.minimum(idx_flat + n_step, self.size - 1)

        def norm_cam(x):
            return (x - self.camera_stat.mean) / self.camera_stat.std
        def norm_vec(x):
            return (x - self.vector_stat.mean) / self.vector_stat.std
        
        batch = dict(
            camera_obs = norm_cam(self.camera_obs[idx_flat]).reshape(batch_size, num_agents, *self.camera_obs.shape[1:]),
            vector_obs = norm_vec(self.vector_obs[idx_flat]).reshape(batch_size, num_agents, -1),
            actions = self.actions[idx_flat].reshape(batch_size, num_agents, -1),
            # rewards = _norm_reward(self.rewards[idx_flat], self.dones[idx_flat], self.reward_stat.mean,self.reward_stat.std).reshape(batch_size, num_agents, 1),
            rewards = self.rewards[idx_flat].reshape(batch_size, num_agents, 1),
            next_camera_obs = norm_cam(self.next_camera_obs[idx_flat]).reshape(batch_size, num_agents, *self.camera_obs.shape[1:]),
            next_vector_obs = norm_vec(self.next_vector_obs[idx_flat]).reshape(batch_size, num_agents, -1),
            dones = self.dones[idx_flat].reshape(batch_size, num_agents, 1),
            indices = idx_flat.reshape(batch_size, num_agents),
            weights = weights.astype(np.float32),  # Ensure weights is 2D
            # nstep_returns = _norm_reward(n_step_ret, d_k[:,:1], self.reward_stat.mean, self.reward_stat.std).reshape(batch_size, num_agents, 1),
            nstep_returns = n_step_ret.reshape(batch_size, num_agents, 1),
            nstep_next_idxs = idx_n_flat.reshape(batch_size, num_agents),
        )

        return batch
    
    def update_priorities(self, indices, priorities):
        assert indices.shape[0] == priorities.shape[0], "Indices and priorities must have the same length."
        self.priorities[indices] = priorities.astype(np.float32)

    def __len__(self):
        return self.size

    def save(self, path):
        np.savez_compressed(path,
                            camera_obs=self.camera_obs,
                            vector_obs=self.vector_obs,
                            actions=self.actions,
                            rewards=self.rewards,
                            next_camera_obs=self.next_camera_obs,
                            next_vector_obs=self.next_vector_obs,
                            dones=self.dones,
                            priorities=self.priorities)

    def load(self, path):
        data = np.load(path)
        
        for k in ('camera_obs', 'vector_obs', 'actions', 'rewards',
                  'next_camera_obs', 'next_vector_obs', 'dones', 'priorities'):
            setattr(self, k, data[k])

        self.size = len(self.rewards)
        self.ptr = self.size % self.buffer_size

        self.camera_stat.update(self.camera_obs)
        self.vector_stat.update(self.vector_obs)
        self.reward_stat.update(self.rewards)


    

