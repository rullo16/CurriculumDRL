import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class SAC_ExperienceBuffer:
    """
    Buffer for storing and sampling experience transitions.
    Supports calculating discounted returns and advantages.
    """
    def __init__(self, camera_obs_dim, vector_obs_dim, action_dim, params, device='cpu'):
        self.device = device  # ensure device is stored
        self.camera_obs_dims = camera_obs_dim
        self.vector_obs_dims = vector_obs_dim
        self.action_dim = action_dim
        self.params = params  # Save hyperparameters for later use
        self.max_steps = params.max_steps
        self.batch_size = params.batch_size
        self.mini_batch_size = params.mini_batch_size

        # Pre-allocate memory on the chosen device
        self.camera_obs = torch.zeros(self.max_steps + 1, *self.camera_obs_dims, device=self.device)
        self.vector_obs = torch.zeros(self.max_steps + 1, *self.vector_obs_dims, device=self.device)
        self.actions = torch.zeros(self.max_steps + 1, *self.action_dim, device=self.device)
        # For SAC, store the log-probability of the chosen action.
        self.action_log_probs = torch.zeros(self.max_steps + 1, device=self.device)
        self.rewards = torch.zeros(self.max_steps + 1, device=self.device)
        self.done_flags = torch.zeros(self.max_steps + 1, device=self.device)
        self.value_estimates = torch.zeros(self.max_steps+1, device=self.device)

        self.step = 0
        self.full = False  # Indicates if the buffer has been filled at least once
        self.returns = None
        self.advantages = None

    def add(self, camera_obs, vector_obs, action, value, action_log_prob, reward, done):
        """
        Add a new transition to the buffer.
        """
        self.camera_obs[self.step].copy_(torch.as_tensor(camera_obs, dtype=torch.float32, device=self.device))
        self.vector_obs[self.step].copy_(torch.as_tensor(vector_obs, dtype=torch.float32, device=self.device))
        self.actions[self.step].copy_(torch.as_tensor(action, dtype=torch.float32, device=self.device))
        self.action_log_probs[self.step] = torch.as_tensor(action_log_prob, dtype=torch.float32, device=self.device)
        self.rewards[self.step] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.done_flags[self.step] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.value_estimates[self.step] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.step = (self.step + 1) % self.max_steps

    def add_final_state(self, camera_obs, vector_obs,values):
        """
        Add the final state to the buffer.
        """
        self.camera_obs[-1].copy_(torch.as_tensor(camera_obs, dtype=torch.float32, device=self.device))
        self.vector_obs[-1].copy_(torch.as_tensor(vector_obs, dtype=torch.float32, device=self.device))
        self.value_estimates[-1] = torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def clear(self):
        """Reset the buffer state."""
        self.step = 0
        self.full = False

    def __len__(self):
        """Return the current number of stored transitions."""
        return self.max_steps if self.full else self.step

    def sample(self):
        """
        Yields mini-batches of transitions using BatchSampler.
        Ensures the entire buffer is iterated over.
        """
        indices_generator = BatchSampler(
            SubsetRandomSampler(range(self.batch_size)), 
            self.mini_batch_size,
            drop_last=True
        )
        for batch_indices in indices_generator:
            yield (
                self.camera_obs[:-1].reshape(-1, *self.camera_obs_dims)[batch_indices],
                self.vector_obs[:-1].reshape(-1, *self.vector_obs_dims)[batch_indices],
                self.actions[batch_indices],
                self.action_log_probs[batch_indices],
                self.value_estimates[:-1].reshape(-1)[batch_indices],
                self.advantages[batch_indices],
                self.returns[batch_indices],
                self.rewards.reshape(-1)[batch_indices],
                self.done_flags[batch_indices]
            )

    def sample_batch(self):
        """
        Returns a single random mini-batch of transitions.
        Useful for off-policy agent updates.
        """
        current_size = len(self)
        indices = torch.randint(0, current_size, (self.mini_batch_size,))
        return (
            self.camera_obs[indices],
            self.vector_obs[indices],
            self.actions[indices],
            self.action_log_probs[indices],
            self.value_estimates[indices],
            self.advantages[indices],
            self.rewards[indices],
            self.done_flags[indices]
        )

    def compute_advantages_and_returns(self, normalize_advantages=True):
        # Pre-allocate tensors for advantages and returns.
        self.advantages = torch.zeros(self.max_steps+1, device=self.device)
        self.returns = torch.zeros(self.max_steps+1, device=self.device)

        # We'll compute the advantages in reverse.
        # It is assumed that the discount and GAE lambda factors are stored in self.params.
        for t in reversed(range(self.max_steps - 1)):
            # Use the final state value as next value for the last transition.
            next_value = self.value_estimates[-1] if t == self.max_steps - 1 else self.value_estimates[t + 1]
            # If an episode ended at time t, mask the future contributions.
            mask = 1.0 - self.done_flags[t]
            # Temporal-difference error.
            delta = self.rewards[t] + self.params.gamma * next_value * mask - self.value_estimates[t]
            # Recursive computation of advantage.
            next_advantage = 0.0 if t == self.max_steps - 1 else self.advantages[t + 1]
            self.advantages[t] = delta + self.params.gamma * self.params.lambda_factor * mask * next_advantage

        # The return at each time step is the sum of the estimated value and computed advantage.
        self.returns = self.advantages + self.value_estimates[:self.max_steps+1]

        # Normalize advantages if requested to improve training stability.
        if normalize_advantages:
            adv_mean = self.advantages.mean()
            adv_std = self.advantages.std(unbiased=False) + 1e-8
            self.advantages = (self.advantages - adv_mean) / adv_std