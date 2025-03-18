import mlagents
from mlagents_envs.environment import UnityEnvironment as UE
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv as UPZBE
from SAC_Distillation.DistilledSACAgent import DistilledSAC
from SAC_Distillation.Trajectories import SAC_ExperienceBuffer
from Hyperparameters import HYPERPARAMS as params
import numpy as np
import torch
import wandb



wandb.init(project="SAC_Distillation", entity="fede-")
wandb.config.update(params['sac_distilled'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.config.update({"device": device})


def relocate_agents(env):
    return list(env.agents)  # simplified

# New helper to extract observation data for an agent
def get_agent_obs(obs, agent):
    agent_data = obs[agent]
    return np.array(agent_data[1]), np.array(agent_data[2])



env = UE(file_name="DroneFlightv1", seed=1, side_channels=[], no_graphics_monitor=True, no_graphics=True)
env = UPZBE(env)
agents = relocate_agents(env)

Buffer = SAC_ExperienceBuffer(env.observation_space(agents[0])[1].shape, env.observation_space(agents[0])[2].shape,env.action_space(agents[0]).shape, params['sac_distilled'])
brain = DistilledSAC(env.observation_space(agents[0])[1].shape, env.observation_space(agents[0])[2].shape, env.action_space(agents[0]).shape,len(agents), params['sac_distilled'])

for s in range(1, params['sac_distilled'].seed_episodes):
    obs, done, t = env.reset(), [False for _ in env.agents], 0
    while not all(done) or t < params['sac_distilled'].n_steps_random_exploration:
        actions = {}
        log_probs = {}
        values = {}
        agents = relocate_agents(env)
        for agent in agents:
            actions[agent] = env.action_space(agent).sample()
            if agent not in obs.keys():
                continue
            obs1, obs2 = get_agent_obs(obs, agent)
            v1,v2 = brain.get_values(obs1,obs2, actions[agent],t)
            values[agent] = torch.min(v1,v2)


        next_obs, reward, done, _ = env.step(actions)

        for agent in agents:
            if agent not in next_obs.keys():
                continue
            next_obs1, next_obs2 = get_agent_obs(next_obs, agent)
            Buffer.store(obs1, obs2, actions[agent], reward[agent], next_obs1, next_obs2, done[agent])
        obs = next_obs
        done = [done[agent] for agent in agents if agent in done.keys()]
        t += 1
    print(f'Finished episode {s}')

# Buffer.compute_advantages()
print("Finished Rnd Exploration")
env.close()

brain.fine_tune_teacher(Buffer, epochs=2)
brain.train(Buffer,step = params['sac_distilled'].seed_episodes*params['sac_distilled'].n_steps_random_exploration)
Buffer._clear_old()
torch.save(brain.net.state_dict(), "SavedModels/SAC_distilled_checkpoint.pth")

env = UE(file_name="DroneFlightv1", seed=1, side_channels=[], no_graphics_monitor=True, no_graphics=True)
env = UPZBE(env)
agents = relocate_agents(env)

steps = 0
while steps < params['sac_distilled'].max_steps:
    obs, done, t = env.reset(), [False for _ in env.agents], 0
    episode_reward = 0
    while not all(done) or t < params['sac_distilled'].n_steps:
        actions = {}
        log_probs = {}
        values = {}
        agents = relocate_agents(env)
        for agent in agents:
            actions[agent] = env.action_space(agent).sample()
            if agent not in obs.keys():
                continue
            obs1, obs2 = get_agent_obs(obs, agent)
            v1,v2 = brain.get_values(obs1,obs2, actions[agent], steps+t)
            values[agent] = torch.min(v1,v2)


        next_obs, reward, done, _ = env.step(actions)

        for agent in agents:
            if agent not in next_obs.keys():
                continue
            next_obs1, next_obs2 = get_agent_obs(next_obs, agent)
            Buffer.store(obs1, obs2, actions[agent], reward[agent], next_obs1, next_obs2, done[agent])
        obs = next_obs
        done = [done[agent] for agent in agents if agent in done.keys()]
        tot_reward = [reward[agent] for agent in agents if agent in reward.keys()]
        t += 1
        
    obs_keys = list(obs.keys())
    mean_reward = np.mean(tot_reward)
    steps += t
    

    # SAC optimization step
    brain.train(Buffer, steps)

    brain.actor_optimizer = brain.adjust_lr(brain.actor_optimizer, params['sac_distilled'].actor_lr, steps, params['sac_distilled'].n_steps)
    brain.critic_optim = brain.adjust_lr(brain.critic_optim, params['sac_distilled'].critic_lr, steps, params['sac_distilled'].n_steps)
    brain.distill_optimizer = brain.adjust_lr(brain.distill_optimizer, params['sac_distilled'].distill_lr, steps, params['sac_distilled'].n_steps)
    wandb.log({"Mean Reward": mean_reward, "Steps": steps})
env.close()
torch.save(brain.net.state_dict(), "SavedModels/SAC_distilled_checkpoint.pth")

# Ensure the model is in evaluation mode
brain.net.eval()

# Create dummy input matching the expected input format of the model
dummy_input_1 = torch.randn(1, *env.observation_space(agents[0])[1].shape).to(device)
dummy_input_2 = torch.randn(1, *env.observation_space(agents[0])[2].shape).to(device)

# Export the model to ONNX format
torch.onnx.export(
    brain.net,
    (dummy_input_1, dummy_input_2),
    "SavedModels/SAC_distilled.onnx",
    export_params=True,
    opset_version=10,
    do_constant_folding=True,
    input_names=["observation1", "observation2"],
    output_names=["action"],
)
print("Model exported to ONNX format successfully.")

# Dispose of the dummy input tensors
del dummy_input_1
del dummy_input_2
torch.cuda.empty_cache()