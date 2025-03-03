from DistilledSACAgent import DistilledSACAgent

class MultiSAC(DistilledSACAgent):
    def __init__(self, camera_obs_dim, vector_obs_dim, action_dim, num_agents, params):
        super().__init__(camera_obs_dim, vector_obs_dim, action_dim, params)

        self.num_agents = num_agents
        self.agents = [DistilledSACAgent(camera_obs_dim, vector_obs_dim, action_dim, params) for _ in range(num_agents)]

    def get_action(self, camera_obs, vector_obs, train=False):
        actions = [agent.get_action(camera_obs[i], vector_obs[i], train) for i, agent in enumerate(self.agents)]
        return actions
    
    def update(self, trajectories):
        for agent in self.agents:
            agent.update(trajectories)

            