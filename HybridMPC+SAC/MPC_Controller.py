import itertools
import torch

class MPCController:
    def __init__(self, sac_agent, horizon=5, candidates=256, top_k=24, iterations=5):
        self.sac_agent = sac_agent
        self.horizon = horizon
        self.candidates = candidates
        self.action_dim = sac_agent.action_dim
        self.top_k = top_k
        self.iterations = iterations
        self.device = sac_agent.device

    def plan(self, camera_obs, vector_obs, step):
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = torch.ones(self.horizon, self.action_dim, device=self.device)

        for _ in range(self.iterations):
            samples = torch.randn(self.candidates, self.horizon, self.action_dim, device=self.device) * std + mean
            samples = samples.clamp(-1, 1)

            values = []
            for i in range(self.candidates):
                value = 0.0
                for t in range(self.horizon):
                    a = samples[i,t].unsqueeze(0)
                    q1,q2 = self.sac_agent.get_critic(camera_obs, vector_obs, a, step)
                    values +=  torch.min(q1,q2).item()
                
                values.append(value)

            values = torch.tensor(values, device=self.device)
            elite_indices = torch.topk(values, self.top_k).indices
            elite_samples = samples[elite_indices]
            mean = elite_samples.mean(dim=0)
            std = elite_samples.std(dim=0)+1e-5


        return mean[0].detach().cpu().numpy()