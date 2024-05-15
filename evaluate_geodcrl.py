import sys
from tqdm import tqdm
import glob
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from geo_dcrl import HARL_HierarchicalDCRL, DEFAULT_CONFIG
from hierarchical_workload_optimizer import WorkloadOptimizer

FOLDER = 'results/simulexchange/PPO_HARL_HierarchicalDCRL_a503b_00000_0_2024-05-13_23-08-01/'
CHECKPOINT_PATH = sorted(glob.glob(FOLDER + 'checkpoint_*'))[-1]
trainer = Algorithm.from_checkpoint(CHECKPOINT_PATH)

env = HARL_HierarchicalDCRL(DEFAULT_CONFIG)
greedy_optimizer = WorkloadOptimizer(env.datacenters.keys())

max_iterations = 4*24*30
# results_all = []

# Initialize lists to store the 'current_workload' metric
workload_DC1 = [[], [], [], [], []]
workload_DC2 = [[], [], [], [], []]
workload_DC3 = [[], [], [], [], []]

# 3 Different agents (RL, One-step Greedy, Multi-step Greedy, Do nothing)

for idx, control_case in enumerate(["RL", "1_step_greedy", "multistep_greedy_agp", "multistep_greedy_vineet","do_nothing"]):
    done = False
    obs, _ = env.reset(seed=123)    

    # actions_list = []
    # rewards_list = []
    total_reward = 0    
    
    with tqdm(total=max_iterations, ncols=150) as pbar:
        while not done:
            if control_case == "RL":
                # RL
                actions = trainer.compute_single_action(obs)
            elif control_case == "1_step_greedy":
                # One-step greedy
                ci = [obs[dc][-1] for dc in env.datacenters]              
                actions = np.zeros(env.action_space.shape)
                sender_idx, receiver_idx = np.argmax(ci), np.argmin(ci)
                if sender_idx < receiver_idx:
                   val = 1.0
                   i,k = sender_idx, receiver_idx
                else:
                    val = -1.0
                    i,k = receiver_idx, sender_idx
                j = k-i-1
                offset = int(env.num_datacenters*i - i*(i+1)/2)
                actions[offset+j] = val
                
            elif control_case == "multistep_greedy_agp":

                # Multi-step greedy; sort the ci index with repect of their values
                ci = [obs[dc][-1] for dc in env.datacenters]
                sorted_ci = np.argsort(ci)
                # First create the 'transfer_1' action with the transfer from the datacenter with the highest ci to the lowest ci
                # Then, create the 'transfer_2' action with the transfer from the datacenter with the second highest ci to the second lowest ci
                actions = {}
                for j in range(len(sorted_ci)-1):
                    actions[f'transfer_{j+1}'] = {'receiver': np.argmin(ci), 'sender': sorted_ci[-(j+1)], 'workload_to_move': np.array([1.])}
                
                temp_actions = np.zeros(env.action_space.shape)
                for key, val in actions.items():
                    i, k = val['sender'], val['receiver']
                    if i < k:
                        val = 1.0
                    else:
                        i,k = k,i
                        val = -1.0
                    j = k-i-1
                    offset = int(env.num_datacenters*i - i*(i+1)/2)
                    temp_actions[offset+j] = val
                actions = temp_actions
                

            elif control_case == "multistep_greedy_vineet":
                actions = np.zeros(env.action_space.shape)
                _, transfer_matrix = greedy_optimizer.compute_adjusted_workload(obs)
                for send_key,val_dict in transfer_matrix.items():
                    for receive_key,val in val_dict.items():
                        if val!=0:
                            i,k = int(send_key[-1])-1, int(receive_key[-1])-1
                            if i > k:  # the ordering is not right; reverse it and als; also assuming val!= also weeds out i = k case
                                i,k = k,i
                                multiplier = -1
                            j = k-i-1
                            offset = int(env.num_datacenters*i - i*(i+1)/2)
                            actions[offset+j] = val*multiplier
            else:
                # Do nothing
                actions = np.zeros(env.action_space.shape)

            
            obs, reward, terminated, done, info = env.step(actions)

            # Obtain the 'current_workload' metric for each datacenter using the low_level_infos -> agent_ls -> ls_original_workload
            workload_DC1[idx].append(env.low_level_infos['DC1']['agent_ls']['ls_original_workload'])  
            workload_DC2[idx].append(env.low_level_infos['DC2']['agent_ls']['ls_original_workload'])
            workload_DC3[idx].append(env.low_level_infos['DC3']['agent_ls']['ls_original_workload'])
        
            total_reward += reward
    
            # actions_list.append(actions['transfer_1'])
            # rewards_list.append(reward)
            
            pbar.update(1)

    # results_all.append((actions_list, rewards_list))
    print(f'{control_case} : Not computed workload: {env.not_computed_workload:.2f}')
    # pbar.close()

    print(f'{control_case} total reward:  {total_reward}')
    
import matplotlib.pyplot as plt
# Plot the 'current_workload' metric
controllers = ["RL", "1_step_greedy", "multistep_greedy_agp", "multistep_greedy_vineet","do_nothing"]
for i,ctrl in enumerate(controllers):
    plt.figure(figsize=(20, 6))
    plt.plot(workload_DC1[i][:4*24*7], label='DC1')
    plt.plot(workload_DC2[i][:4*24*7], label='DC2')
    plt.plot(workload_DC3[i][:4*24*7], label='DC3')
    plt.title(f'Current Workload for \n {ctrl} Controller')
    plt.xlabel('Time Step')
    plt.ylabel('Current Workload')
    plt.legend()
    plt.grid('on', linestyle='--', alpha=0.5)
    plt.show()

# Print the average workload computed by each controller from the workload_DCx list
for i,ctrl in enumerate(controllers):
    print(f'Average workload for {ctrl}: {np.mean(workload_DC1[i]):.2f}, {np.mean(workload_DC2[i]):.2f},\
        {np.mean(workload_DC3[i]):.2f}, Total: {np.sum(workload_DC1[i]) + np.sum(workload_DC2[i])+ np.sum(workload_DC3[i]):.2f}')