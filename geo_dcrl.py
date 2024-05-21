import numpy as np
from tqdm import tqdm
from gymnasium.spaces import Box

from heirarchical_env import HeirarchicalDCRL, DEFAULT_CONFIG

class HierarchicalDCRLCombinatorial(HeirarchicalDCRL):
    
    def __init__(self, config): 

        super().__init__(config)

        N = len(self.datacenters)
        
        # Define action space:
        # We consider every combination of N datacenters. The sign determintes the direction 
        # of transfer 
        num_actions = N * (N - 1) // 2
        self.action_space = Box(-1, 1, shape=([num_actions]))
    
    def action_wrapper(self, actions):
        # actions is an array of floats in the range (-1.0, 1.0) of "ordered data centers" in the format
        # [DC1<->DC2, DC1<->DC3, ..., DC1<->DCN... DC2<->DC3...DC2<->DCN....DC(N-1)<->DCN] where N = self.num_datacenters
        # have to return as a list of same length as actions with each element being a dictionary of sender, receiver and workload_to_move
        N = len(self.datacenters)

        new_actions = {}
        cntr = 1
        for i in range(N-1):
            offset = int(N*i - i*(i+1)/2)
            for j in range(N-(i+1)):
                # depending on the sign choose the sender and receiver
                # if positive, i is sender and j+i+1 is receiver
                # if negative, j+i+1 is sender and i is receiver
                if actions[offset+j] >= 0:
                    new_actions[f'transfer_{cntr}'] = \
                        {'sender': i, 'receiver': j+i+1, \
                            'workload_to_move': np.array([actions[offset+j]])}
                else:
                    new_actions[f'transfer_{cntr}'] = \
                        {'sender': j+i+1, 'receiver': i, \
                            'workload_to_move': np.array([-actions[offset+j]])}
                cntr += 1    
            return new_actions
        
    def step(self, actions):
        actions = self.action_wrapper(actions)
        
        return super().step(actions)
        
def main():
    """Main function."""
    
    env = HierarchicalDCRLCombinatorial(DEFAULT_CONFIG)
    # env = HARL_HierarchicalDCRL_v2(DEFAULT_CONFIG)
    done = False
    obs, _ = env.reset(seed=0)
    total_reward = 0
    
    max_iterations = 4*24*30
    with tqdm(total=max_iterations) as pbar:
        while not done:
    
            # Random actions
            actions = env.action_space.sample()

            obs, reward, _, truncated, _ = env.step(actions)
            done = truncated
            total_reward += reward

            # Update the progress bar
            pbar.update(1)

    # After simulation, calculate average metrics for each environment
    average_metrics = {
        env_id: {metric: sum(values) / len(values) for metric, values in env_metrics.items()}
        for env_id, env_metrics in env.metrics.items()
    }

    # Print average metrics for each environment
    for env_id, env_metrics in average_metrics.items():
        print(f"Average Metrics for {env.datacenters[env_id].location}:")
        for metric, value in env_metrics.items():
            print(f"\t{metric}: {value:,.2f}")
        print()  # Blank line for readability

    # Sum metrics across datacenters
    print("Summed metrics across all DC:")
    total_metrics = {}
    for metric in env_metrics:
        total_metrics[metric] = 0.0
        for env_id in average_metrics:
            total_metrics[metric] += average_metrics[env_id][metric]

        print(f'\t{metric}: {total_metrics[metric]:,.2f}')

    print("Total reward = ", total_reward)


if __name__ == "__main__":
    main()
