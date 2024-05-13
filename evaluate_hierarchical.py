#%%
import sys
from tqdm import tqdm

import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from heirarchical_env import HeirarchicalDCRL, HeirarchicalDCRLWithHysterisis, HeirarchicalDCRLWithHysterisisMultistep, DEFAULT_CONFIG
from hierarchical_workload_optimizer import WorkloadOptimizer

#%%
trainer = Algorithm.from_checkpoint('./results/MultiStep/PPO_HeirarchicalDCRLWithHysterisisMultistep_6fc05_00000_0_2024-05-13_15-35-34/checkpoint_000450')
#%%
env = HeirarchicalDCRLWithHysterisis(DEFAULT_CONFIG)
greedy_optimizer = WorkloadOptimizer(env.datacenters.keys())
#%%
def compare_transfer_actions(actions1, actions2):
    """Compare transfer actions for equality on specific keys."""
    # Check if both actions have the same set of transfers
    if set(actions1.keys()) != set(actions2.keys()):
        return False

    # Iterate through each transfer action and compare
    for key in actions1:
        action1 = actions1[key]
        action2 = actions2[key]

        # Check the specific keys within each transfer action
        if (action1['receiver'] != action2['receiver'] or
            action1['sender'] != action2['sender'] or
            not np.array_equal(action1['workload_to_move'], action2['workload_to_move'])):
            return False

    return True

max_iterations = 4*24*30
results_all = []

# Initialize lists to store the 'current_workload' metric
workload_DC1 = [[], [], [], []]
workload_DC2 = [[], [], [], []]
workload_DC3 = [[], [], [], []]

# 3 Different agents (RL, One-step Greedy, Multi-step Greedy, Do nothing)

for i in [0, 1, 2, 3]:
    done = False
    obs, _ = env.reset(seed=123)    

    actions_list = []
    rewards_list = []
    total_reward = 0    
    
    with tqdm(total=max_iterations, ncols=150) as pbar:
        while not done:
            if i == 0:
                actions = trainer.compute_single_action(obs)
            elif i == 1:
                # One-step greedy
                ci = [obs[dc][-1] for dc in env.datacenters]
                actions = {'receiver': np.argmin(ci), 'sender': np.argmax(ci), 'workload_to_move': np.array([1.])}
                actions = {'transfer_1': actions}
            elif i == 2:
                # Multi-step greedy
                # sort the ci index with repect of their values
                ci = [obs[dc][-1] for dc in env.datacenters]
                sorted_ci = np.argsort(ci)
                # First create the 'transfer_1' action with the transfer from the datacenter with the highest ci to the lowest ci
                # Then, create the 'transfer_2' action with the transfer from the datacenter with the second highest ci to the second lowest ci
                actions = {}
                for j in range(len(sorted_ci)-1):
                    actions[f'transfer_{j+1}'] = {'receiver': np.argmin(ci), 'sender': sorted_ci[-(j+1)], 'workload_to_move': np.array([1.])}
                    
                 # Check if multi-step greedy actions are different from trainer actions
                trainer_action = trainer.compute_single_action(obs)
                # trainer_action['transfer_1']['workload_to_move'] = 0.23
                # Compare actions element by element
                # if not compare_transfer_actions(actions, trainer_action):
                #     print("WARNING: Multi-step greedy actions differ from trainer actions.")
                #     print("Trainer actions: ", trainer_action)
                #     print("Multi-step greedy actions: ", actions)
            else:
                # Do nothing
                actions = {'sender': 0, 'receiver': 0, 'workload_to_move': np.array([0.0])}
                actions = {'transfer_1': actions}

            
            obs, reward, terminated, done, info = env.step(actions)

            # Obtain the 'current_workload' metric for each datacenter using the low_level_infos -> agent_ls -> ls_original_workload
            workload_DC1[i].append(env.low_level_infos['DC1']['agent_ls']['ls_original_workload'])  
            workload_DC2[i].append(env.low_level_infos['DC2']['agent_ls']['ls_original_workload'])
            workload_DC3[i].append(env.low_level_infos['DC3']['agent_ls']['ls_original_workload'])
        
            total_reward += reward
    
            actions_list.append(actions['transfer_1'])
            rewards_list.append(reward)
            
            pbar.update(1)

    results_all.append((actions_list, rewards_list))
    print(f'Not computed workload: {env.not_computed_workload:.2f}')
    # pbar.close()

    print(total_reward)
#%%
import matplotlib.pyplot as plt
# Plot the 'current_workload' metric
controllers = ['RL', 'One-step Greedy', 'Multi-step Greedy', 'Do nothing']
for i in range(4):
    plt.figure(figsize=(10, 6))
    plt.plot(workload_DC1[i][:4*24*7], label='DC1')
    plt.plot(workload_DC2[i][:4*24*7], label='DC2')
    plt.plot(workload_DC3[i][:4*24*7], label='DC3')
    plt.title(f'Current Workload for {controllers[i]} Controller')
    plt.xlabel('Time Step')
    plt.ylabel('Current Workload')
    plt.legend()
    plt.grid('on', linestyle='--', alpha=0.5)
    plt.show()
#%%
# Print the average workload computed by each controller from the workload_DCx list
for i in range(4):
    print(f'Average workload for {controllers[i]}: {np.mean(workload_DC1[i]):.2f}, {np.mean(workload_DC2[i]):.2f}, {np.mean(workload_DC3[i]):.2f}, Total: {np.sum(workload_DC1[i]) + np.sum(workload_DC2[i]) + np.sum(workload_DC3[i]):.2f}')
# %%
