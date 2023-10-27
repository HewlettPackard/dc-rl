from typing import Union
import random

import ray
from ray import air, tune
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.tune.schedulers import PopulationBasedTraining

from create_trainable import create_wrapped_trainable

from deepmerge import always_merger
import pprint
import csv
from datetime import datetime

def train(
    algorithm: Union[str, Algorithm], 
    config: AlgorithmConfig,
    results_dir: str = './results',
    name: str = 'latest_experiment',
    overrides: dict = {}
    ):
    """
    Starts the training process for a given algorithm.

    Args:
        algorithm (rllib.algorithm): RL algorithm to use for training.
        config (algorithm_config): Algorithm training configuration.
        results_dir (string): Directory to save the results
        overrides (dict): Extra configuration

    """

    
    hyperparam_mutations = {
                    "actor_lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6],
                    "critic_lr":[1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6],
                    "train_batch_size":[16*12, 32*12, 64*12, 128*12, 256*12],
                    "gamma": [0.9999, 0.9995, 0.999, 0.995, 0.99, 0.95, 0.9, 0.8, 0.5, 0.3, 0.1],
                    "model": {
                            "attention_num_transformer_units": tune.choice([1, 2, 4, 8, 16]),
                            "attention_dim": tune.choice([32, 64, 128, 256]),
                            "attention_num_heads": tune.choice([1, 2, 4, 8, 16]),
                            "attention_head_dim": tune.choice([16, 32, 64, 128]),
                            "attention_memory_inference": tune.choice([25, 50, 100, 200]),
                            "attention_memory_training": tune.choice([25, 50, 100, 200]),
                            "attention_position_wise_mlp_dim": tune.choice([16, 32, 64, 128]),
                            "attention_init_gru_gate_bias": tune.choice([1.0, 2.0, 3.0]),
                            "attention_use_n_prev_actions": tune.choice([0, 1, 2, 3]),
                            "attention_use_n_prev_rewards": tune.choice([0, 1, 2, 3]),
                            },
                    }

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=420,
        resample_probability=0.25,
        burn_in_period=300,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations=hyperparam_mutations,
    )
    
    # Create a dummy environment to get obs. and action space
    dummy_env = config.env(config.env_config)
    ls_env, dc_env, bat_env = dummy_env.ls_env, dummy_env.dc_env, dummy_env.bat_env 

    config = config.multi_agent(
                policies={
                    "agent_ls": PolicySpec(
                        None,
                        ls_env.observation_space,
                        ls_env.action_space,
                        config={"agent_id" : 0},
                    ),
                    "agent_dc": PolicySpec(
                        None,
                        dc_env.observation_space,
                        dc_env.action_space,
                        config={"agent_id" : 1},
                    ),
                    "agent_bat": PolicySpec(
                        None,
                        bat_env.observation_space,
                        bat_env.action_space,
                        config={"agent_id" : 2},
                    ),
                },
                policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            )

    # Only include agents as specified in env_config['agents']
    for agent in list(config.policies.keys()):
        if agent not in config.env_config['agents']:
            config.policies.pop(agent)

    # Reassign agent ids
    for i, policy in enumerate(config.policies.values()):
        policy.config['agent_id'] = i
    
    config = config.to_dict()
    config.update(overrides)
    
    # config = always_merger.merge(config, param_space)

    results = tune.Tuner(
                create_wrapped_trainable(algorithm),
                tune_config=tune.TuneConfig(
                    metric="episode_reward_mean",
                    mode="max",
                    scheduler=pbt,
                    num_samples=7,
                ),
                param_space=config,
                run_config=air.RunConfig(stop={"training_iteration": 300_000},
                    verbose=0,
                    local_dir=results_dir,
                    name=name,
                    checkpoint_config=ray.air.CheckpointConfig(
                        checkpoint_frequency=5,
                        num_to_keep=5,
                        checkpoint_score_attribute="episode_reward_mean",
                        checkpoint_score_order="max"
                    ),
                )
            ).fit()

    best_result = results.get_best_result()

    print("Best performing trial's final set of hyperparameters:\n")
    hyperparams = {k: v for k, v in best_result.config.items() if k in hyperparam_mutations}
    pprint.pprint(hyperparams)

    print("\nBest performing trial's final reported metrics:\n")

    metrics_to_print = [
        "episode_reward_mean",
        "episode_reward_max",
        "episode_reward_min",
        "episode_len_mean",
    ]
    pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})
    
    # Save hyperparameters to a CSV file
    current_date = datetime.now().strftime('%Y-%m-%d')

    filename = f'best_hyperparameters_ai04_{current_date}.csv'

    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Hyperparameter', 'Value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for k, v in hyperparams.items():
            writer.writerow({'Hyperparameter': k, 'Value': v})