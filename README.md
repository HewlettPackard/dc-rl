# DCRL-Green

This repository contains the datasets and code for the paper DCRL_Green: Sustainable Data Center Environment and Benchmark for Multi-Agent Reinforcement Learning.

<p align="center">
  <img src="https://github.com/HewlettPackard/dc-rl/blob/main/sphinx/images/DCRL-sim1.png" align="centre" width="500" />
</p>

## Introduction
DCRL-green is a framework for testing multi-agent Reinforcement Learning (MARL) algorithm that optimizes data centers for multiple objectives of carbon footprint reduction, energy consumption, and energy cost. It uses OpenAI Gym standard and supports modeling and control of three different types of problems: Carbon aware flexible load shifting, Data center HVAC cooling energy optimization and carbon aware battery auxiliary supply.

Currently, we provide two version for the data center dynamics. The default version is implemented in Python and can be used with the prerequisites listed below. We also provide a seperate docker that incorporates the Energy plus model of a data center from the [Sinergym](https://github.com/ugr-sail/sinergym) repository.

## Documentation and Installation
Refer to the [docs](https://hewlettpackard.github.io/dc-rl/) for documentation of the DCRL-Green.

# Quick Start Guide

## Prerequisites
- Linux OS
- Python 3.9
- PyTorch
- Gymnasium
- RLLib
- Pandas
- [Optional] TensorBoard (for training monitoring). One way to do this is with pip ([link](https://pypi.org/project/tensorboard/))
- [Optional] Docker (see [docs](https://docs.docker.com/get-docker/)) (If user plans to try Sinergym-Energy Plus Environment)


## Installation
First, download the repository. If using HTML, execute:
```bash
git clone https://github.com/HewlettPackard/dc-rl.git
```
If using SSH, execute
```bash
git clone git@github.com:ugr-sail/sinergym.git
```


Change the current working directory to the DC-CFR_Benchmark folder:

```bash
cd DC-CFR_Benchmark
```

## Usage
### Running scripts
To start the optimization script, run the following command:

For PPO
```bash
python train_ppo.py
```

For A2C
```bash
python train_a2c.py
```

For MADDPG
```bash
python train_maddpg.py
```

### Monitoring Training
Monitor the training loop using TensorBoard. By default, the location of the training data is at ```/results/```. To visualize the training data, run:

```bash
tensorboard --logdir results/latest_experiment
```

## Contributing
Contributions are welcome. For major changes, please open an issue first to discuss what you would like to change. Please ensure to update tests as appropriate.

## Contact
For any questions or concerns, please open an issue in this repository or contact [add email].
