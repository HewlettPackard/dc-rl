"""
Train the environment using the selected algorithm.

The original code is from https://github.com/PKU-MARL/HARL
Several modifications are made to adapt to the SustainDC environment.

Hyperparameters: harl/configs/algos_cfgs/<algo>.yaml and harl/configs/envs_cfgs/sustaindc.yaml, then
    --config my.yaml          sections env: (env config) and seed:/device:/train:/eval:/model:/algo:/... (algo config)
    --<leaf_key> <value>      HARL's original overrides by leaf key name, e.g. --lr 1e-4 --location ny
    --set train.lr=1e-4       explicit dotted overrides (repeatable), highest precedence
Training is logged and evaluated with the common suite (sdc_eval/harl.py): rollout/, train/, health/, perf/, eval/
in <run>/metrics.csv and <run>/logs (TensorBoard); models/ = best fixed-seed work-adjusted CO2, models_final/ = last.
Runs go to <out_root>/sustaindc/<location>/<algo>/<exp_name>/seed-XXXXX-<run_id>, out_root = --out_root, else
$SDC_OUT_ROOT, else logger.log_dir from the yaml (./results).
"""

import os
import sys
import warnings
import argparse
import json

warnings.filterwarnings('ignore')
# sys.path.insert(0, os.getcwd())

from harl.utils.configs_tools import get_defaults_yaml_args, update_args
from sdc_eval.config import add_config_args, load_config, merge, parse_overrides
from sdc_eval.harl import add_eval_defaults
from sdc_eval.paths import ENV_VAR, resolve_root

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
        default="installtest",
        help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file."
    )
    
    # Parse known arguments and process unknown arguments
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Name the run directory seed-<seed>-<run_id> instead of using a timestamp. Running again with the same "
             "run_id resumes from the run's latest checkpoint (e.g. pass the Slurm job id, which a job requeued after "
             "preemption keeps)."
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=500_000,
        help="Save a resumable checkpoint (networks, optimizers, RNG states, replay buffer) every N env steps; 0 = off."
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=None,
        help=f"Output root (default: ${ENV_VAR} if set, else logger.log_dir from the algo yaml)."
    )
    add_config_args(parser)
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
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:
        # Load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])

    # Hyperparameters: yaml < --config < --<leaf_key> flags < --set (sdc_eval/config.py)
    add_eval_defaults(algo_args)
    cfg = merge({**algo_args, "env": env_args}, load_config(args["config"]))
    env_args = cfg.pop("env")
    algo_args = cfg
    update_args(unparsed_dict, algo_args, env_args)
    cfg = merge({**algo_args, "env": env_args}, parse_overrides(args["overrides"]))
    env_args = cfg.pop("env")
    algo_args = cfg
    if args["out_root"] or ("log_dir" not in unparsed_dict and os.environ.get(ENV_VAR)):
        algo_args["logger"]["log_dir"] = resolve_root(args["out_root"])

    # Start training
    from harl.runners import RUNNER_REGISTRY

    # Initialize and run the selected algorithm
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
