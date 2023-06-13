# DCRL-Green

This repository contains the datasets and code for the paper DCRL_Green: Sustainable Data Center Environment and Benchmark for Multi-Agent Reinforcement Learning.

## Introduction
DCRL-green is a framework for testing multi-agent Reinforcement Learning (MARL) algorithm that optimizes data centers for multiple objectives of carbon footprint reduction, energy consumption, and energy cost. It uses OpenAI Gym standard and supports modeling and control of three different types of problems: Carbon aware flexible load shifting, Data center HVAC cooling energy optimization and carbon aware battery auxiliary supply.

<p align="center">
  <img src="https://github.com/HewlettPackard/dc-rl/blob/main/sphinx/images/DCRL-sim1.png" align="centre" width="500" />
</p>

## Documentation and Installation
Refer to the [docs](https://hewlettpackard.github.io/dc-rl/) for documentation of the DCRL-Green.

# Quick Start Guide

## Prerequisites
- Docker (see [docs](https://docs.docker.com/get-docker/))
- Python 3.x (see [instructions](https://python.land/installing-python))
- TensorBoard (for training monitoring). One way to do this is with pip ([link](https://pypi.org/project/tensorboard/))
- Pandas
- RLLib
- Gymnasium

## Installation
First, download the repository. If using HTML, execute:
```bash
git clone https://github.com/HewlettPackard/dc-rl.git
```

Change the current working directory to the DC-CFR_Benchmark folder:

```bash
cd DC-CFR_Benchmark
```

## Usage
### Running scripts
To start the optimization script, run the following command:

```bash
python train_ppo.py
```

### Monitoring Training
Monitor the training loop using TensorBoard. By default, the location of the training data is at ```/results/```. To visualize the training data, run:

```bash
tensorboard --logdir results/latest_experiment
```

### Checkpoints
If you wish to save checkpoints during training, edit the variable ```checkpoints = False``` to ```True``` in the ```sustainability_env.py``` file.

## Contributing
Contributions are welcome. For major changes, please open an issue first to discuss what you would like to change. Please ensure to update tests as appropriate.

## Contact
For any questions or concerns, please open an issue in this repository or contact [add email].

## License
[The license]