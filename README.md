# DCRL-Green

This repository contains the datasets and code for the paper DCRL_Green: Sustainable Data Center Environment and Benchmark for Multi-Agent Reinforcement Learning.

<p align="center">
  <img src="https://github.com/HewlettPackard/dc-rl/blob/main/sphinx/images/DCRL-sim1.png" align="centre" width="500" />
</p>

## Introduction
DCRL-green is a framework for testing multi-agent Reinforcement Learning (MARL) algorithm that optimizes data centers for multiple objectives of carbon footprint reduction, energy consumption, and energy cost. It uses OpenAI Gym standard and supports modeling and control of three different types of problems: Carbon aware flexible load shifting, Data center HVAC cooling energy optimization and carbon aware battery auxiliary supply.

Currently, we provide two versions for the data center dynamics. 

`DCRL (dcrl_env.py)`: This default version is implemented in Python and can be used with the prerequisites listed below. 

`DCRLeplus (dcrl_eplus_env.py)`: This uses the [EnergyPlus](https://energyplus.net/) model of a data center from the [Sinergym](https://github.com/ugr-sail/sinergym) repository. We also provide a seperate docker that incorporates this environment.


## Documentation and Installation
Refer to the [docs](https://hewlettpackard.github.io/dc-rl/) for documentation of the DCRL-Green.

# Quick Start Guide

## Prerequisites
- Linux OS (Ubuntu 20.04)
- Conda
- [Optional] Docker (see [docs](https://docs.docker.com/get-docker/)) (Only required for the Sinergym-EnergyPlus Environment)


## Installation
First, download the repository. If using HTML, execute:
```bash
$ git clone https://github.com/HewlettPackard/dc-rl.git
```
If using SSH, execute:
```bash
$ git clone git@github.com:HewlettPackard/dc-rl.git
```

Make sure you have conda installed. For more instructions on installing conda please check the [documentation](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html#install-linux-silent).

Change the current working directory to the dc-rl folder:

```bash
$ cd dc-rl
```

Create a conda environment and install dependencies:
```bash
$ conda create -n dcrl python=3.10
$ conda activate dcrl
$ pip install -r requirements.txt
```

## Usage
### Running the DCRL environment
To run an episode of the environment with a random agent:
```bash
$ python dcrl_env.py
```

### Training an RL agent on the DCRL environment
To start training, run the following command:

(Note: The `episode_reward_mean` will be `nan` for the first few iterations until 1 episode is completed)

For PPO:
```bash
$ python train_ppo.py
```

For MADDPG:
```bash
$ python train_maddpg.py
```

For A2C:
```bash
$ python train_a2c.py
```

### Training on the DCRLeplus environment

```bash
$ docker pull agnprz/carbon_sustain:v3
$ docker run -t -i -v $PWD:/sinergym/dc-rl --shm-size=10.24gb agnprz/carbon_sustain:v3
$ cd dc-rl
$ EPLUS=1 python train_ppo.py
```

### Monitoring Training
Monitor the training using TensorBoard. By default, the location of the training data is at ```./results```. To visualize, run:

```bash
$ tensorboard --logdir ./results
```

### Running in Background Mode
If you want to run the DCRL-Green framework in background mode use the following command:

```bash
$ hohup python PYTHON_SCRIPT > OUTPUT_FILE  &
```
where ```PYTHON_SCRIPT``` is the script you want to run (e.g., ```train_ppo.py```) and ```OUTPUT_FILE``` is the name of the file that will contain the output (e.g. ```latest_experiment_output.txt```).

## Contributing
Contributions are welcome. For major changes, please open an issue first to discuss what you would like to change. Please ensure to update tests as appropriate.

## Contact
For any questions or concerns, please open an issue in this repository or contact [add email].
