#%%
import os
import json
import torch
import gymnasium as gym
from dcrl_env_harl_partialobs import DCRL

# Import HAPPO from here: /lustre/guillant/HARL/harl/algorithms/actors/happo.py
# But I am working on /lustre/guillant/dc-rl

import sys
sys.path.append('/lustre/guillant/HARL/harl')

import harl
from algorithms.actors.happo import HAPPO


#%%
# Checkpoint and config path:
checkpoint_path = os.path.join('/lustre/guillant/HARL/results/dcrl', 'CA/happo/ls_dc_bat/seed-00001-2024-05-28-23-23-34')
config_path = os.path.join(checkpoint_path, 'config.json')

# Read config_path
# Read the config file
with open(config_path, 'r') as f:
    config = json.load(f)

env_config = config['env_args']
# Create the dcrl environment
env = DCRL(env_config)

# Obtain from env_config how many agents is active: ''agents': ['agent_ls', 'agent_dc', 'agent_bat']'
# Obtain the number of active agents
num_agents = len(env_config['agents'])
agents = env_config['agents']
actors = {}
for agent_id, agent in enumerate(env_config['agents']):
    checkpoint = torch.load(checkpoint_path + "/models/actor_agent" + str(agent_id) + ".pt")
    
    # load_state_dict from checkpoint
    model_args = config['algo_args']['model']
    algo_args = config['algo_args']['algo']
    agent = HAPPO({**algo_args["model"], **algo_args["algo"]},
                  self.envs.observation_space[agent_id],
                  self.envs.action_space[agent_id],
                  device=self.device,
                  )
    actors[agent].load_state_dict(checkpoint['model_state_dict'])
    actors[agent].eval()

#%%
# Reset the environment
obs = env.reset()
done = False
total_reward = 0

while not done:
    # Get the actions for each actor
    actions = {}
    for agent in agents:
        actor = actors[agent]
        action = actor(torch.tensor(obs[agent]).float().unsqueeze(0)).detach().numpy()
        actions[env_config['agents'][agent_id]] = action
    
    obs[agent_id], reward, done, _ = env.step(action[0])
    action = model.predict(obs)

    # Take a step in the environment
    obs, reward, done, _ = env.step(action)

    # Accumulate the reward
    total_reward += reward

# Print the total reward
print("Total reward:", total_reward)

# Path to the trained model
model_path = '/path/to/trained/model.h5'

# Evaluate the model
evaluate_model(model_path)