from HER import HindsightExperienceReplay
from Network import PPONet
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PPOAgent:
    def __init__(self, obs_space, action_space, params):
        self.params = params
        self.net = PPONet(obs_space, action_space).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=params["learning_rate"])
        self.trajectories = HindsightExperienceReplay(obs_space, action_space, params["buffer_size"], params["batch_size"], self.goal_sampling, self.params)
        self.min_batch = params["batch_size"]
        self.train_batch = params["train_batch"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def goal_sampling(self, goals):
        noise = torch.randn_like(goals) * 0.1
        return goals + noise

    def select_action(self,obs):
        state = obs[0]
        inv = obs[1]
        if not isinstance(state, torch.Tensor) and not isinstance(inv, torch.Tensor):
            state = np.array(state)
            inv = np.array(inv)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            inv = torch.tensor(inv, dtype=torch.float32).unsqueeze(0).to(self.device)
        policy, value = self.net((state,inv))
        value = value.squeeze(-1)
        action_probs = torch.nn.functional.log_softmax(policy, dim=1)
        action_probs = torch.clip(action_probs, -1)
        action_probs = torch.distributions.Categorical(logits=(action_probs))
        action = action_probs.sample()
        return action.detach().cpu().item(), action_probs.log_prob(action).cpu().detach().numpy(), value.cpu().detach().numpy()
    
    def train_action(self, obs):
        state = obs[0]
        inv = obs[1]
        if not isinstance(state, torch.Tensor) and not isinstance(inv, torch.Tensor):
            state = np.array(state)
            inv = np.array(inv)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            inv = torch.tensor(inv, dtype=torch.float32).unsqueeze(0).to(self.device)
        policy, value = self.net((state,inv))
        value = value.squeeze(-1)
        action_probs = torch.nn.functional.log_softmax(policy, dim=1)
        action_probs = torch.clip(action_probs, -1)
        action_probs = torch.distributions.Categorical(logits=(action_probs))
        action = action_probs.sample()
        return action.cpu().numpy(), action_probs, value  
    
    def improv_lr(self, optimizer, init_lr, timesteps, num_timesteps):
        lr = init_lr * (1 - timesteps / num_timesteps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer
    
    def train(self, steps):
        batch = self.params["n_steps"] // self.min_batch
        if batch < self.min_batch:
            batch = self.min_batch

        gradient_acc_steps = batch / self.min_batch
        gradient_steps = 1

        self.net.train()
        for _ in range(self.params["n_steps"]):
            samples = self.trajectories.sample()
            for sample in samples:
                obs_batch, inv_batch, action_batch, reward_batch, advantage_batch, value_batch,log_prob_batch, goal_obs, goal_inv = sample

                obs_batch = obs_batch.to(device)
                inv_batch = inv_batch.to(device)
                action_batch = action_batch.to(device)
                reward_batch = reward_batch.to(device)
                advantage_batch = advantage_batch.to(device)
                old_log_prob_batch = log_prob_batch.to(device)
                old_value_batch = value_batch.to(device)
                goal_obs = goal_obs.to(device)
                goal_inv = goal_inv.to(device)

                _, action_dist, value_batch = self.train_action((obs_batch, inv_batch))
                
                action_log_probs = action_dist.log_prob(action_batch)
                ratio = torch.exp(action_log_probs - old_log_prob_batch)
                unclipped_adv = ratio * advantage_batch
                clipped_adv = torch.clamp(ratio, 1-self.params["clip"], 1+self.params["clip"]) * advantage_batch
                action_loss = -torch.min(unclipped_adv, clipped_adv).mean()

                clipped_values = old_value_batch + (value_batch-old_value_batch).clamp(-self.params["clip"], self.params["clip"])
                unclipped_values = (value_batch-reward_batch).pow(2)
                clipped_values = (clipped_values-reward_batch).pow(2)
                value_loss = 0.5 * torch.max(unclipped_values, clipped_values).mean()

                entropy_beta = max(0.01, 0.01 * (1 - steps / self.params["n_steps"]))
                entropy_loss = action_dist.entropy().mean()
                loss = (action_loss+(0.5*value_loss)-(entropy_beta*entropy_loss))
                loss.backward()

                if gradient_steps % gradient_acc_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                gradient_steps += 1