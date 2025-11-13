import numpy as np


class RolloutBuffer:

    """
    On-policy buffer for storing trajectories and computing advantages
    Unlike SAC's replay buffer which is off-policy, this buffer:
    - Onlu stores recent experience
    - Computes advantages using Generalized Advantage Estimation (GAE)
    - Gets cleared after each update

    I am using GAE because reduces variance in advantage estimates, balances bias-variance tradeoff with lambda parameter and makes lerning more stable.
    """

    def __init__(self, num_steps, num_agents, obs_shape, action_dim, gamma=0.99, gae_lambda=0.95):
        """
        num_steps: Number of steps to collect before updating (e.g., 2048)
        num_agents: Number of agents in the environment (4 drones)
        obs_shape: Shape of encoded observations (e.g., (320,) for 256 vision + 64 vector)
        action_dim: Dimension of action space (6 for your drones)
        gamma: Discount factor for future rewards
        gae_lambda: GAE lambda parameter (0.95 = good balance)
        """
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        #Storage arrays
        self.observations = np.zeros((num_steps, num_agents, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.values = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_agents), dtype=np.float32)

        #Compute after collection
        self.advantages = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.returns = np.zeros((num_steps, num_agents), dtype=np.float32)

        self.ptr = 0

    def store(self,obs, action, reward, done, value, log_prob):
        """
        Store one transition for all agents
        obs: (num_agents, obs_dim) - encoded observations
        action: (num_agents, action_dim) - actions taken
        reward: (num_agents,) - rewards received
        done: (num_agents,) - done flags
        value: (num_agents,) - value estimates from critic
        log_prob: (num_agents,) - log probability of actions
        """
        if self.ptr >= self.num_steps:
            raise ValueError(f"Buffer Full! Called store() {self.ptr+1} times but buffer size is {self.num_steps}")
        
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob

        self.ptr += 1

    def compute_returns_and_advantages(self, last_values):
        """
        Compute advatnages using GAE
        Core of PPO, GAE combines short-term accuracy (low bias) using actual rewards and long-term stability (low vairance) using value estimates

        A_t = δ_t + γλδ_{t+1} + (γλ)²δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)

        last_values: (num_agents,) - value estimates for the final state
        """
        if self.ptr != self.num_steps:
            raise ValueError(f"Buffer not full! Only {self.ptr}/{self.num_steps} transitions")
        
        #Bootstrap, append final value estimate "What would we get if we continued from last state"
        values_with_bootstrap = np.vstack([self.values, last_values[None, :]])

        #Compute TD errors (δ_t)
        #δ_t = r_t + γV(s_{t+1}) - V(s_t)
        #one-step TD error
        advantages = np.zeros_like(self.rewards)
        last_gae = 0

        #Work backwards (t= T-1, T-2, ..., 0)
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps-1:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = last_values

            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t+1]

            #TD Error: δ_t = r_t + γV(s_{t+1})(1-done) - V(s_t)
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]

            #GAE: A_t = δ_t + γλA_{t+1}(1-done)
            #Recursively builds the advantage estimate
            last_gae = delta + self.gamma*self.gae_lambda*next_non_terminal*last_gae
            advantages[t] = last_gae

        #Returns are just advantages + values
        #R_t = A_t + V(s_t)
        #What the value function should predict
        returns = advantages + self.values

        self.advantages = advantages
        self.returns = returns

    def get(self):
        """
        Get all stored data and reset buffer

        returns dict with all stored arrays
        """
        if self.ptr != self.num_steps:
            raise ValueError(f"Buffer not full!")
        
        #Normalize advantages (mean=0, std=1)
        #Makes learning more stable by keeping gradient magnitudes reasonable
        advantages_flat = self.advantages.reshape(-1)
        adv_mean = advantages_flat.mean()
        adv_std = advantages_flat.std()
        self.advantages = np.clip(self.advantages, -10, 10)

        self.ptr = 0

        return {
            'observations': self.observations.copy(),
            'actions': self.actions.copy(),
            'returns': self.returns.copy(),
            'advantages': self.advantages.copy(),
            'log_probs': self.log_probs.copy(),
            'values': self.values.copy() 
        }
