import gym
import ray
from ray import air, tune
from ray.rllib.policy.policy import PolicySpec

from maddpg import MADDPGConfigStable, MADDPGStable
from utils.rllib_callbacks import CustomCallbacks
from train import train
from dcrl_eplus_env import DCRLeplus
from dcrl_env import DCRL

CONFIG = (
        MADDPGConfigStable()
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

                # MADDPG returns logits instead of discrete actions
                "actions_are_logits": True,
            }
        )
        .framework("tf")
        .rollouts(num_rollout_workers=24)
        .training(
            gamma=0.99, 
            lr=1e-7,
            model={'fcnet_hiddens':[128, 64, 16], 'fcnet_activation': 'relu'},
            use_local_critic=False,
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
        algorithm=MADDPGStable,
        config=CONFIG,
        results_dir=RESULTS_DIR,
        name=NAME,
    )
