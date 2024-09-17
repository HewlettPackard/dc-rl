#%%
import os
import json
import yaml
import subprocess
import random
from shutil import copyfile
from datetime import datetime

# Path to the train_sustaindc.py script
TRAINING_SCRIPT = "train_sustaindc.py"

# Custom JSON-like YAML dump
def custom_yaml_dump(data, file_path):
    """Custom dump function to save the YAML file with JSON-like format."""
    yaml_content = yaml.dump(data, default_flow_style=False, sort_keys=False)
    # Ensure that null values are written as 'null' instead of '~'
    yaml_content = yaml_content.replace('~', 'null')
    with open(file_path, 'w') as f:
        f.write(yaml_content)
        
# Function to generate random values for specific parameters
def generate_random_config(base_config):
    # Modify specific parameters in base_config with random values
    base_config['train']['num_env_steps'] = 10000000
    # Randomize rollout threads and episode length
    base_config['train']['n_rollout_threads'] = random.choice([1, 2, 4])
    base_config['train']['episode_length'] = random.choice([128, 256, 512, 1024, 2048])
    
    # Randomize network sizes
    # base_config['model']['hidden_size_policy'] = [random.choice([4, 8, 16]), random.choice([2, 4, 8, 16])]
    # base_config['model']['hidden_size_value'] = [random.choice([4, 8, 16]), random.choice([2, 4, 8, 16])]
    # num_policy_layers = random.choice([1, 2])
    # if num_policy_layers == 1:
    #     base_config['model']['hidden_size_policy'] = [random.choice([8, 16, 32, 64])]
    # else:
    #     base_config['model']['hidden_size_policy'] = [random.choice([8, 16, 32, 64]), random.choice([8, 16, 32, 64])]

    # # Choose between 1 or 2 layers for the value network
    # num_value_layers = random.choice([1, 2])
    # if num_value_layers == 1:
    #     base_config['model']['hidden_size_value'] = [random.choice([8, 16, 32, 64, 128])]
    # else:
    #     base_config['model']['hidden_size_value'] = [random.choice([8, 16, 32, 64, 128]), random.choice([8, 16, 32, 64])]
    
    
    
    # Randomize learning rates
    base_config['model']['lr'] = round(random.uniform(0.00005, 0.001), 6)
    base_config['model']['critic_lr'] = round(random.uniform(0.00005, 0.001), 6)
    base_config['model']['opti_eps'] = round(random.uniform(0.00000001, 0.001), 9)
    
    # Randomize the last layer activation
    base_config['model']['action_squash_method'] = 'none' #random.choice(['none', 'tanh'])
    
    # Randomize use_valuenorm: True or False
    base_config['train']['use_valuenorm'] = random.choice([True, False])
    
    # Randomize use_feature_normalization: True
    base_config['model']['use_feature_normalization'] = random.choice([True, False])
    
    # Randomize discount factor (gamma)
    base_config['algo']['gamma'] = round(random.uniform(0.90, 0.999), 3)
    
    # Randomize GAE lambda
    base_config['algo']['gae_lambda'] = round(random.uniform(0.8, 0.99), 2)
    
    # Randomize PPO clip parameter
    base_config['algo']['clip_param'] = round(random.uniform(0.1, 0.4), 2)
    
    # Randomize entropy coefficient
    base_config['algo']['entropy_coef'] = round(random.uniform(0.01, 0.1), 4)
    
    # Randomize mini-batches
    base_config['algo']['actor_num_mini_batch'] = random.choice([2, 4, 8])
    base_config['algo']['critic_num_mini_batch'] = random.choice([2, 4, 8])
    
    # Randomize PPO epochs
    base_config['algo']['ppo_epoch'] = random.choice([3, 5, 10, 15, 20])
    base_config['algo']['critic_epoch'] = random.choice([3, 5, 10, 15, 20])
    
    # Randomize gradient clipping
    base_config['algo']['max_grad_norm'] = round(random.uniform(1.0, 5.0), 2)
    
    # Randomize value loss coefficient
    base_config['algo']['value_loss_coef'] = round(random.uniform(0.3, 0.7), 2)
    return base_config


# Function to run the training script with a specific configuration
def run_training(config_path, exp_name):
    # Create the command to run the training script
    command = [
        "python", TRAINING_SCRIPT,
        "--load_config", config_path,
        "--exp_name", exp_name
    ]
    
    # Run the command
    subprocess.run(command)
#%%
# Load the base configuration (you can load from an existing config or create one from scratch)
with open('harl/configs/algos_cfgs/happo.yaml', 'r') as f:
    base_config = yaml.safe_load(f)

# Number of training runs
num_runs = 25

# Directory to save configurations
config_dir = "generated_configs"
os.makedirs(config_dir, exist_ok=True)
#%%
# Loop through and create multiple training runs
for i in range(num_runs):
    # Generate a new random configuration
    random_config = generate_random_config(base_config.copy())
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a filename for the config file
    config_filename = f"config_run_{i}_{current_time}.yaml"
    config_filepath = os.path.join(config_dir, config_filename)
    
    # Save the generated configuration
    custom_yaml_dump(random_config, config_filepath)
    
    # Add current datetime to experiment name (down to seconds)
    exp_name = f"random_run_{i}_{current_time}_discrete_nopenalty_includingci"
    
    # Run the training script with the generated config
    run_training(config_filepath, exp_name) 

# %%
