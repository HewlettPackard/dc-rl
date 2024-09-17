"""
Train the environment using the selected algorithm.

The original code is from https://github.com/PKU-MARL/HARL
Several modifications are made to adapt to the SustainDC environment.
"""

import os
import sys
import warnings
import argparse
import json
import yaml

warnings.filterwarnings('ignore')
# sys.path.insert(0, os.getcwd())

from harl.utils.configs_tools import get_defaults_yaml_args, update_args

def main():
    """Main function to train the environment using the selected algorithm."""
    # Create an argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add command-line arguments
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="sustaindc",
        choices=["sustaindc"],
        help="Environment name. Choose from: sustaindc."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="DEBUG_RUN",
        help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file."
    )
    
    # Parse known arguments and process unknown arguments
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        """Evaluate the argument if possible, otherwise return the argument as is."""
        try:
            return eval(arg)
        except:
            return arg

    # Process unparsed arguments to a dictionary
    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}

    # Convert args to dictionary
    args = vars(args)

    # Load configuration
    if args["load_config"] != "":
        # Load config from existing config file
        # if the file of load_config starts with "generated_configs/", then load using yaml
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
        
        if args["load_config"].startswith("generated_configs/"):
            with open(args["load_config"], "r") as file:
                all_config = yaml.safe_load(file)
        else:
            with open(args["load_config"], encoding="utf-8") as file:
                all_config = json.load(file)
        algo_args = all_config
        # env_args = all_config["env_args"]
    else:
        # Load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])

    # Update args from command line
    update_args(unparsed_dict, algo_args, env_args)

    # Start training
    from harl.runners import RUNNER_REGISTRY

    # Initialize and run the selected algorithm
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
