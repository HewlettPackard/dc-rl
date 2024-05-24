import numpy as np
from tqdm import tqdm
from gymnasium.spaces import Box
from utils.helper_methods import non_linear_combine, RunningStats
from heirarchical_env import HeirarchicalDCRL, DEFAULT_CONFIG

class HierarchicalDCRLCombinatorial(HeirarchicalDCRL):
    
    def __init__(self, config): 

        super().__init__(config)

        N = len(self.datacenters)
        self.num_datacenters = N
        
        # Define action space:
        # We consider every combination of N datacenters. The sign determintes the direction 
        # of transfer 
        num_actions = N * (N - 1) // 2
        self.action_space = Box(-1, 1, shape=([num_actions]))
        
        self.stats1 = RunningStats()
        self.stats2 = RunningStats()
        
        self.cfp_reward =  0.0  # includes both dcrl reward and hysterisis reward
        self.workload_violation_rwd = 0.0 # excess workload is penalized
        self.combined_reward = 0.0  # cfp_reward + workload_violation_rwd
        self.cost_of_moving_mw = 0.0
        self.action_choice = 0.0
        self.overassigned_wkld_penalty = config["overassigned_wkld_penalty"]
        self.hysterisis_penalty = config["hysterisis_penalty"]
    
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
        self.action_choice = actions
        actions = self.action_wrapper(actions)
        
        return super().step(actions)
    
    def calc_reward(self):
        self.cfp_reward =  super().calc_reward()  # includes both dcrl reward and hysterisis reward
        self.workload_violation_rwd = -1.0*self.overassigned_wkld_penalty*sum([i[-1] for i in self.overassigned_workload])  # excess workload is penalized
        self.combined_reward = non_linear_combine(self.cfp_reward, self.workload_violation_rwd, self.stats1, self.stats2)  # cfp_reward + workload_violation_rwd  # 
        return self.combined_reward

    # def get_dc_variables(self, dc_id: str) -> np.ndarray:
    #     dc = self.datacenters[dc_id]

    #     # TODO: check if the variables are normalized with the same values or with min_max values
    #     obs = np.array([
    #         dc.datacenter_capacity_mw,
    #         dc.workload_m.get_current_workload(),
    #         dc.weather_m.get_current_weather(),
    #         self.low_level_infos[dc_id]['agent_dc'].get('dc_total_power_kW', 0)/1000.0,
    #         dc.ci_m.get_current_ci(),
    #     ])
    #     # obs = {
    #     #     'dc_capacity': dc.datacenter_capacity_mw,
    #     #     'curr_workload': dc.workload_m.get_current_workload(),
    #     #     'weather': dc.weather_m.get_current_weather(),
    #     #     'total_power_kw': self.low_level_infos[dc_id]['agent_dc'].get('dc_total_power_kW', 0),
    #     #     'ci': dc.ci_m.get_current_ci(),
    #     # }

    #     return obs
    
    def set_hysterisis(self, mwh_to_move: float, sender: str, receiver: str):
        PENALTY = self.hysterisis_penalty
        
        self.cost_of_moving_mw = mwh_to_move * PENALTY

        self.datacenters[sender].dc_env.set_workload_hysterisis(self.cost_of_moving_mw)
        self.datacenters[receiver].dc_env.set_workload_hysterisis(self.cost_of_moving_mw)
            
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
