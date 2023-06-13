import ray
from ray.rllib.algorithms.a2c import A2CConfig

from utils.rllib_callbacks import CustomCallbacks
from train import train
from dcrl_eplus_env import DCRLeplus
from dcrl_env import DCRL

# Data collection config
TIMESTEP_PER_HOUR = 4
COLLECTED_DAYS = 1
NUM_AGENTS = 3
NUM_WORKERS = 24

CONFIG = (
        A2CConfig()
        .environment(
            env=DCRL,
            env_config={
                # Agents active
                'agents': ['agent_ls', 'agent_dc', 'agent_bat'],

                # Datafiles
                'location': 'ny',
                'cintensity_file': 'NYIS_NG_&_avgCI.csv',
                'weather_file': 'USA_NY_New.York-Kennedy.epw',

                # Battery capacity
                'max_bat_cap_Mw': 2,
                
                'reward_method': 'default_dc_reward'
            }
        )
        .framework("torch")
        .rollouts(num_rollout_workers=NUM_WORKERS)
        .training(
            gamma=0.99, 
            lr=1e-6,
            use_gae=True, 
            train_batch_size=24 * TIMESTEP_PER_HOUR * COLLECTED_DAYS * NUM_WORKERS * NUM_AGENTS,
            model={'fcnet_hiddens': [128, 64, 16], 'fcnet_activation': 'relu'}, 
        )
        .callbacks(CustomCallbacks)
        .resources(num_cpus_per_worker=1, num_gpus=0)
    )

NAME = "test"
RESULTS_DIR = './results'

if __name__ == '__main__':

    ray.init(ignore_reinit_error=True)
    # ray.init(local_mode=True, ignore_reinit_error=True)

    train(
        algorithm="A2C",
        config=CONFIG,
        results_dir=RESULTS_DIR,
        name=NAME,
    )