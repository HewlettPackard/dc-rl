# SustainDC (DCRL-Green) - Benchmarking for Sustainable Data Center Control

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start Guide](#quick-start-guide)
5. [Environment Details](#environment-details)
6. [Customization](#customization)
7. [Benchmarking Algorithms](#benchmarking-algorithms)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Dashboard](#dashboard)
10. [Contributing](#contributing)
11. [Contact](#contact)
12. [License](#license)


## Introduction
SustainDC is a set of Python environments for benchmarking multi-agent reinforcement learning (MARL) algorithms in data centers (DC). It focuses on sustainable DC operations, including workload scheduling, cooling optimization, and auxiliary battery management. This repository contains the code and datasets for the paper SustainDC - Benchmarking for Sustainable Data Center Control.

<p align="center">
  <img src="media/SustainDC.png" alt="SustainDC" width="1000">
</p>

Demo of SustainDC
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XF92aR6nVYxENrviHeFyuRu0exKBb-nh?usp=sharing)

TODO: This demo should be updated

### Documentation and Installation
Refer to the [docs](https://hewlettpackard.github.io/dc-rl/) for broader documentation of SustainDC.

## Features
- **Highly Customizable Environments:** Allows users to define and modify various aspects of DC operations, including server configurations, cooling systems, and workload traces.
- **Multi-Agent Support:** Enables the testing of MARL controllers with both homogeneous and heterogeneous agents, facilitating the study of collaborative and competitive strategies in DC management.
- **Gymnasium Integration:** Environments are wrapped in the Gymnasium `Env` class, making it easy to benchmark different control strategies using standard reinforcement learning libraries.
- **Realistic External Variables:** Incorporates real-world data such as weather conditions, carbon intensity, and workload traces to simulate the dynamic and complex nature of DC operations.
- **Collaborative Reward Mechanisms:** Supports the design of custom reward structures to promote collaborative optimization across different DC components.
- **Benchmarking Suite:** Includes scripts and tools for evaluating the performance of various MARL algorithms, providing insights into their effectiveness in reducing energy consumption and carbon emissions.

<p align="center">
  <img src="media/RLLoopv3.png" alt="RL Loop" width="500">
</p>


## Installation
### Prerequisites
- Python 3.7+
- Dependencies listed in `requirements.txt`

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/HewlettPackard/dc-rl.git
    ```
2. Navigate to the repository directory:
    ```bash
    cd dc-rl
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```


## Quick Start Guide
1. **Setup Configuration:**
   Customize the `dc_config.json` file to specify your DC environment settings.

2. **Environment Configuration:**
   The main environment for wrapping the environments is `dclr_env_harl_partialobs.py`, which reads configurations from `env_config` and manages the external variables using managers for weather, carbon intensity, and workload. 

4. **Run Example:**
   Execute a simple example to get started:
   ```bash
   python examples/run_random.py
   ```

5. **Run Example:**
   Execute a simple example to get started:
   ```bash
   python examples/evaluate.py
   ```


## Environment Details

### Workload Environment
The **Workload Environment** in SustainDC manages the execution and scheduling of delayable workloads within the DC. It includes open-source workload traces from [Alibaba](https://github.com/alibaba/clusterdata) and [Google](https://github.com/google/cluster-data) DC, which represent the computational demand placed on the DC. Users can customize this environment by adding new workload traces or modifying the existing ones. The workload traces are used to simulate the tasks that the DC needs to process, providing a realistic and dynamic workload for benchmarking purposes.

#### Observation Space
- Time of Day and Year: The sine and cosine representation of the hour of the day and the day of the year, providing a periodic understanding of time.
- Grid Carbon Intensity (CI): Current and forecasted carbon intensity values, allowing the agent to optimize workload scheduling based on carbon emissions.
- Rescheduled Workload Left: The amount of workload that has been rescheduled but not yet executed.
  
#### Action Space
- Store Delayable Tasks: The agent can choose to store delayable tasks for future execution.
- Compute All Immediate Tasks: The agent can decide to process all current tasks immediately.
- Maximize Throughput: The agent can maximize the throughput by balancing immediate and delayed tasks based on the current carbon intensity.

<p align="center">
  <img src="media/agent_ls_explanation.png" alt="LS Agent" width="200">
</p>


### Data Center Environment
The **Data Center Environment** models the IT and HVAC systems of a DC, focusing on optimizing energy consumption and cooling. This environment simulates the electrical and thermal behavior of the DC components, including servers, cooling systems, and other infrastructure. Users can customize various parameters such as the number of servers, cooling setpoints, and the configuration of the HVAC system. This environment helps evaluate the performance of different control strategies aimed at reducing energy consumption and improving the overall efficiency of the DC.
A representation of the DC modelled can be seen in the following figure:
<p align="center">
  <img src="media/Data_center_modelled.png" alt="Data Center Modelled" width="1000">
</p>

#### Observation Space
- Time of Day and Year: The sine and cosine representation of the hour of the day and the day of the year, providing a periodic understanding of time.
- Ambient Weather (Dry Bulb Temperature): Current outside temperature affecting the cooling load.
- IT Room Temperature: Current temperature inside the data center, crucial for maintaining optimal server performance.
- Previous Step Energy Consumption: Historical data on cooling and IT energy consumption for trend analysis.
- Grid Carbon Intensity (CI): Forecasted carbon intensity values to optimize cooling strategies.

#### Action Space
- Decrease Setpoint: Lower the cooling setpoint to increase cooling, consuming more energy on the cooling but less on the IT.
- Maintain Setpoint: Keep the current cooling setpoint constant.
- Increase Setpoint: Raise the cooling setpoint to reduce cooling, using less energy on the cooling but more on the IT.

<p align="center">
  <img src="media/agent_dc_explanation.png" alt="DC Agent" width="200">
</p>


### Battery Environment
The **Battery** Environment simulates the charging and discharging cycles of batteries used in the DC. It models how batteries can be charged from the grid during periods of low carbon intensity and provide auxiliary energy during periods of high carbon intensity. This environment helps in assessing the effectiveness of battery management strategies in reducing the carbon footprint and optimizing energy usage in the DC.

#### Observation Space
- Time of Day and Year: The sine and cosine representation of the hour of the day and the day of the year, providing a periodic understanding of time.
- State of Charge (SoC): Current energy level of the battery.
- Grid Energy Consumption: Combined energy consumption of IT and cooling systems.
- Grid Carbon Intensity (CI): Current and forecasted carbon intensity values to determine optimal charging and discharging times.

#### Action Space
- Charge Battery: Store energy in the battery during periods of low carbon intensity.
- Hold Energy: Maintain the current state of charge.
- Discharge Battery: Provide auxiliary energy to the DC during periods of high carbon intensity.

<p align="center">
  <img src="media/agent_bat_explanation.png" alt="BAT Agent" width="250">
</p>


### Connections Between Environments
The three environments in SustainDC are interconnected to provide a comprehensive simulation of DC operations:

- The **Workload Environment** generates the computational demand that the **Data Center Environment** must process. This includes managing the scheduling of delayable tasks to optimize energy consumption and reduce the carbon footprint.

- The **Data Center Environment** handles the cooling and IT operations required to process the workloads. It is directly influenced by the workload generated, as higher computational demand results in increased heat generation, necessitating more cooling and energy consumption.

- The **Battery Environment** supports the DC by providing auxiliary energy during periods of high carbon intensity, helping to reduce the overall carbon footprint of the DC's operations. It is affected by both the **Workload Environment** and the **Data Center Environment**. The workload affects heat generation, which in turn impacts the cooling requirements and energy consumption of the DC, thereby influencing the battery's charging and discharging cycles.


Together, these interconnected environments provide a realistic and dynamic platform for benchmarking multi-agent reinforcement learning algorithms aimed at enhancing the sustainability and efficiency of DC operations.


### External Variables
SustainDC uses several external variables to provide a realistic simulation environment:

#### Workload
The Workload external variable in SustainDC represents the computational demand placed on the DC. By default, SustainDC includes a collection of open-source workload traces from Alibaba and Google DCs. Users can customize this component by adding new workload traces to the `data/Workload` folder or specifying a path to existing traces in the `dcrl_env_harl_partialobs.py` file under the `workload_file` configuration.

![Comparison between two workload traces of Alibaba trace (2017) and Google (2011).](media/workload_comparison.png)

#### Weather
The Weather external variable in SustainDC captures the ambient environmental conditions impacting the DC's cooling requirements. By default, SustainDC includes weather data files in the .epw format from various locations where DCs are commonly situated. These locations include Arizona, California, Georgia, Illinois, New York, Texas, Virginia, and Washington. Users can customize this component by adding new weather files to the `data/Weather` folder or specifying a path to existing weather files in the `dcrl_env_harl_partialobs.py` file under the `weather_file` configuration.

Each .epw file contains hourly data for various weather parameters, but for our purposes, we focus on the ambient temperature.

![Comparison between external temperature of the different selected locations.](media/weather_all_locations.png)

#### Carbon Intensity
The Carbon Intensity (CI) external variable in SustainDC represents the carbon emissions associated with electricity consumption. By default, SustainDC includes CI data files for various locations: Arizona, California, Georgia, Illinois, New York, Texas, Virginia, and Washington. These files are located in the `data/CarbonIntensity` folder and are extracted from [https://api.eia.gov/bulk/EBA.zip](https://api.eia.gov/bulk/EBA.zip). Users can customize this component by adding new CI files to the `data/CarbonIntensity` folder or specifying a path to existing files in the `dcrl_env_harl_partialobs.py` file under the `cintensity_file` configuration.

![Comparison of carbon intensity across the different selected locations.](media/ci_all_locations.png)

Furthermore, in the figure below, we show the average daily carbon intensity against the average daily coefficient of variation (CV) for various locations. This figure highlights an important perspective on the variability and magnitude of carbon intensity values across different regions. Locations with a high CV indicate greater fluctuation in carbon intensity, offering more "room to play" for DRL agents to effectively reduce carbon emissions through dynamic actions. Additionally, locations with a high average carbon intensity value present greater opportunities for achieving significant carbon emission reductions. The selected locations are highlighted, while other U.S. locations are also plotted for comparison. Regions with both high CV and high average carbon intensity are identified as prime targets for DRL agents to maximize their impact on reducing carbon emissions.

![Average daily carbon intensity versus average daily coefficient of variation (CV) for the grid energy provided from US. Selected locations are remarked. High CV indicates more fluctuation, providing more opportunities for DRL agents to reduce carbon emissions. High average carbon intensity values offer greater potential gains for DRL agents.](media/average_CI_vs_avgerage_CV.png)

Below is a summary of the selected locations, typical weather values, and carbon emissions characteristics:

<div align="center">

| Location   | Typical Weather                      | Carbon Emissions                 |
|------------|--------------------------------------|----------------------------------|
| Arizona    | Hot, dry summers; mild winters       | High avg CI, High variation      |
| California | Mild, Mediterranean climate          | Medium avg CI, Medium variation  |
| Georgia    | Hot, humid summers; mild winters     | High avg CI, Medium variation    |
| Illinois   | Cold winters; hot, humid summers     | High avg CI, Medium variation    |
| New York   | Cold winters; hot, humid summers     | Medium avg CI, Medium variation  |
| Texas      | Hot summers; mild winters            | Medium avg CI, High variation    |
| Virginia   | Mild climate, seasonal variations    | Medium avg CI, Medium variation  |
| Washington | Mild, temperate climate; wet winters | Low avg CI, Low variation        |

</div>


## Customization
SustainDC offers extensive customization options to tailor the environments to specific needs and configurations. Users can modify various parameters and components across the **Workload**, **Data Center**, and **Battery** environments, as well as external variables like weather carbon intensity data, and workload trace.

<p align="center">
  <img src="media/schematic.png" alt="Schematic" width="1000">
</p>


### Main Configuration File
The main environment for wrapping the environments is `dclr_env_harl_partialobs.py`, which reads configurations from `dc_config.json` and manages the external variables using managers for weather, carbon intensity, and workload.

#### Example Configuration
```bash
env_config = {
    # Agents active
    'agents': ['agent_ls', 'agent_dc', 'agent_bat'],

    # Datafiles
    'location': 'ny',
    'cintensity_file': 'NYIS_NG_&_avgCI.csv',
    'weather_file': 'USA_NY_New.York-Kennedy.epw',
    'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',

    # Data Center maximum capacity
    'datacenter_capacity_mw': 1,
    
    # Battery capacity
    'max_bat_cap_Mw': 2,
    
    # Collaborative weight in the reward
    'individual_reward_weight': 0.8,
    
    # Flexible load ratio
    'flexible_load': 0.4,
    
    # Specify reward methods
    'ls_reward': 'default_ls_reward',
    'dc_reward': 'default_dc_reward',
    'bat_reward': 'default_bat_reward'
}
```


### Data Center Configuration File
The customization of the DC is done through the `dc_config.json` file located in the `utils` folder. This file allows users to specify every aspect of the DC environment design.

#### Example Configuration File Structure
```json
{
    "data_center_configuration": {
        "NUM_ROWS": 4,
        "NUM_RACKS_PER_ROW": 5,
    },
    "hvac_configuration": {
        "C_AIR": 1006,
        "RHO_AIR": 1.225,
        "CRAC_SUPPLY_AIR_FLOW_RATE_pu": 0.00005663,
        "CRAC_REFRENCE_AIR_FLOW_RATE_pu": 0.00009438,
        "CRAC_FAN_REF_P": 150,
        "CHILLER_COP_BASE": 5.0,
        "CHILLER_COP_K": 0.1,
        "CHILLER_COP_T_NOMINAL": 25.0,
        "CT_FAN_REF_P": 1000,
        "CT_REFRENCE_AIR_FLOW_RATE": 2.8315,
        "CW_PRESSURE_DROP": 300000,
        "CW_WATER_FLOW_RATE": 0.0011,
        "CW_PUMP_EFFICIENCY": 0.87,
        "CT_PRESSURE_DROP": 300000,
        "CT_WATER_FLOW_RATE": 0.0011,
        "CT_PUMP_EFFICIENCY": 0.87
    },
    "server_characteristics": {
        "CPU_POWER_RATIO_LB": [0.01, 1.00],
        "CPU_POWER_RATIO_UB": [0.03, 1.02],
        "IT_FAN_AIRFLOW_RATIO_LB": [0.01, 0.225],
        "IT_FAN_AIRFLOW_RATIO_UB": [0.225, 1.0],
    }
}
```

### Adding New Workload Data

#### Overview
By default, SustainDC includes workload traces from [Alibaba](https://github.com/alibaba/clusterdata) and [Google](https://github.com/google/cluster-data) DC. These traces are used to simulate the tasks that the DC needs to process, providing a realistic and dynamic workload for benchmarking purposes.

#### Data Source
The default workload traces are extracted from:
- Alibaba 2017 CPU Data ([https://github.com/alibaba/clusterdata](https://github.com/alibaba/clusterdata))
- Google 2011 CPU Data ([https://github.com/google/cluster-data](https://github.com/google/cluster-data))

#### Expected File Format
Workload trace files should be in CSV format, with two columns: a timestamp or index (must be unnamed), and the corresponding DC Utilization (`cpu_load`). The CPU load must be expressed as a fraction of the DC utilization (between 0 and 1). The workload file must contain one year of data with an hourly periodicity (365*24=8760 rows). 

#### Example Workload Trace File
```csv
,index,cpu_load
1,0.380
2,0.434
3,0.402
4,0.485
...
```

#### Integration Steps
1. Place the new workload trace file in the `data/Workload` folder.
2. Update the workload_file entry in env_config with the path to the new workload trace file.


### Adding New Carbon Intensity Data

#### Overview
Carbon Intensity (CI) data represents the carbon emissions associated with electricity consumption. SustainDC includes CI data files for various locations to simulate the carbon footprint of the DC's energy usage.

#### Data Source
The default carbon intensity data files are extracted from:
- U.S. Energy Information Administration (EIA) API: [https://api.eia.gov/bulk/EBA.zip](https://api.eia.gov/bulk/EBA.zip)

#### Expected File Format
Carbon intensity files should be in CSV format, with columns representing different energy sources and the average carbon intensity. The columns typically include:
- `timestamp`: The time of the data entry.
- `WND`: Wind energy.
- `SUN`: Solar energy.
- `WAT`: Water (hydropower) energy.
- `OIL`: Oil energy.
- `NG`: Natural gas energy.
- `COL`: Coal energy.
- `NUC`: Nuclear energy.
- `OTH`: Other energy sources.
- `avg_CI`: The average carbon intensity.

#### Example Carbon Intensity File
```csv
timestamp,WND,SUN,WAT,OIL,NG,COL,NUC,OTH,avg_CI
2022-01-01 00:00:00+00:00,1251,0,3209,0,15117,2365,4992,337,367.450
2022-01-01 01:00:00+00:00,1270,0,3022,0,15035,2013,4993,311,363.434
2022-01-01 02:00:00+00:00,1315,0,2636,0,14304,2129,4990,312,367.225
2022-01-01 03:00:00+00:00,1349,0,2325,0,13840,2334,4986,320,373.228
...
```

#### Integration Steps
1. Add the new carbon intensity data files to the `data/CarbonIntensity` folder.
2. Update the `cintensity_file` entry in `env_config` with the path to the new carbon intensity file.


### Adding New Weather Data

#### Overview
Weather data captures the ambient environmental conditions that impact the DC's cooling requirements. SustainDC includes weather data files in the .epw format from various locations where DCs are commonly situated.

#### Data Source
The default weather data files are extracted from:
- EnergyPlus Weather Data: [https://energyplus.net/weather](https://energyplus.net/weather)

#### Expected File Format
Weather data files should be in the .epw (EnergyPlus Weather) format, which includes hourly data for various weather parameters such as temperature, humidity, wind speed, etc.

#### Example Weather Data File
Below is a partial example of an .epw file:
```epw
LOCATION,New York JFK Int'l Ap NY USA,USA,NY,716070,40.63,-73.78,-5.0,3.4
DESIGN CONDITIONS,1,New York JFK Int'l Ap Ann Clg .4% Condns DB=>MWB,25.0
...
DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31
...
2022,1,1,1,60.1,45.1,1004.1,0,0,4.1,80.1,30.0
2022,1,1,2,59.8,44.9,1004.0,0,0,4.1,80.0,29.8
...
```

#### Integration Steps
1. Add the new weather data files in the .epw format to the `data/Weather` folder.
2. Update the `weather_file` entry in `env_config` with the path to the new weather file.


### Custom Reward Structures
SustainDC allows users to define custom reward structures to promote collaborative optimization across different DC components. Users can modify the reward functions in the `utils/reward_creator.py` file to suit their specific optimization goals.

By leveraging these customization options, users can create highly specific and optimized simulations that reflect the unique requirements and challenges of their DC operations.


## Benchmarking Algorithms

SustainDC supports a variety of reinforcement learning algorithms for benchmarking. This section provides an overview of the supported algorithms and highlights their differences.

### Supported Algorithms

#### PPO (Proximal Policy Optimization)
PPO is a popular reinforcement learning algorithm that strikes a balance between simplicity and performance. It uses a clipped objective function to ensure that policy updates are not too drastic, which helps stabilize training. PPO is known for its robustness and is widely used in various applications.

#### IPPO (Independent Proximal Policy Optimization)
IPPO is a variant of PPO designed for multi-agent systems where each agent operates independently. Each agent has its own policy and value function, and they do not share information directly. This approach allows each agent to learn its own strategy independently of the others, which can be beneficial in environments where agents have distinct roles.

#### MAPPO (Multi-Agent Proximal Policy Optimization)
MAPPO extends PPO to multi-agent settings by using a centralized value function that takes into account the states and actions of all agents. This centralized approach allows for better coordination among agents, as the value function can provide a more comprehensive evaluation of the joint actions. MAPPO is particularly useful in cooperative tasks where agents need to work together closely.

#### HAPPO (Heterogeneous Agent Proximal Policy Optimization)
HAPPO is a variant of MAPPO designed for environments with heterogeneous agents, where different agents may have different observation spaces, action spaces, and reward structures. HAPPO allows each agent to have its own policy and value function, but it also incorporates mechanisms for coordination among the heterogeneous agents. This approach is useful in complex environments where agents have diverse roles and capabilities.

#### HAA2C (Heterogeneous Agent Advantage Actor-Critic)
HAA2C is a multi-agent extension of the Advantage Actor-Critic (A2C) algorithm for heterogeneous agents. Each agent has its own actor and critic networks, but they share a common environment. The algorithm uses the advantage function to reduce variance in policy updates, making training more stable. HAA2C is effective in scenarios where agents have different types of observations and actions.

#### HAD3QN (Heterogeneous Agent Dueling Double Deep Q-Network)
HAD3QN is a variant of the Dueling Double Deep Q-Network (D3QN) algorithm tailored for heterogeneous multi-agent environments. It combines the benefits of dueling network architectures and double Q-learning to improve learning stability and performance. Each agent has its own dueling network, which separately estimates the state value and the advantages of each action. This approach helps in environments where agents need to make fine-grained distinctions between actions.

#### HASAC (Heterogeneous Agent Soft Actor-Critic)
HASAC is an extension of the Soft Actor-Critic (SAC) algorithm for heterogeneous multi-agent systems. SAC is an off-policy actor-critic algorithm that uses entropy regularization to encourage exploration. HASAC adapts this approach to multi-agent settings with heterogeneous agents, allowing each agent to have its own policy and value function while coordinating through shared entropy-based objectives. This algorithm is suitable for environments requiring continuous action spaces and high exploration.

### Differences and Use Cases
- **PPO vs. IPPO:** PPO is designed for single-agent environments, whereas IPPO is adapted for multi-agent environments where each agent learns independently.
- **IPPO vs. MAPPO:** While IPPO treats agents independently, MAPPO uses a centralized value function to coordinate agents, making it better for cooperative tasks.
- **MAPPO vs. HAPPO:** Both use centralized value functions, but HAPPO is tailored for environments with heterogeneous agents with different capabilities and roles.
- **HAPPO vs. HAA2C:** HAPPO is a PPO-based algorithm, whereas HAA2C extends A2C to multi-agent settings, offering different stability and performance trade-offs.
- **HAA2C vs. HAD3QN:** HAA2C is an actor-critic method, while HAD3QN is a value-based method with dueling and double Q-learning enhancements.

By supporting a diverse set of algorithms, SustainDC allows researchers to benchmark and compare the performance of various reinforcement learning approaches in the context of sustainable DC control.

  
### Running Benchmarks
To configure the used algorithm, TBC..........


## Evaluation Metrics
- Carbon Footprint (CFP): Cumulative carbon emissions over the evaluation period.
- HVAC Energy: Energy consumed by the DC cooling system.
- IT Energy: Energy consumed by the DC servers.
- Water Usage: Efficient utilization of water for cooling.
- Task Dropped: Number of dropped tasks due to workload scheduling.


## Dashboard

To get an in-depth look at the SustainDC dashboard and see real-time metrics, watch the video demonstration. The video showcases the dynamic plotting of variables from the agents, environments, and metrics, providing a comprehensive view of the DC operations.

Click on the screenshot below to watch the video (right-click and select "Open link in new tab" to view in a new tab):

[![Dashboard, click it to visualize it](media/DCRL_screenshot2.png)](https://www.dropbox.com/scl/fi/85gumlvjgbbk5kwjhee3i/Data-Center-Green-Dashboard-ver2.mp4?rlkey=w3mu21qqdk9asi826cjyyutzl&dl=0)

In the video, you will see:
- **Real-time plotting of agent variables:** Watch how the agents' actions and states are visualized dynamically.
- **Environment metrics:** Observe the DC's performance metrics, including energy consumption, cooling requirements, and workload distribution.
- **Interactive dashboard features:** Learn about the various interactive elements of the dashboard that allow for detailed analysis and monitoring.

If you wish to download the video directly, [click here](https://www.dropbox.com/scl/fi/85gumlvjgbbk5kwjhee3i/Data-Center-Green-Dashboard-ver2.mp4?rlkey=w3mu21qqdk9asi826cjyyutzl&dl=1).


## Contributing
We welcome contributions from the community! Whether it's bug fixes, new features, or improvements to the documentation, your help is appreciated. Please follow the guidelines below to contribute to SustainDC.

### How to Contribute

1. **Fork the Repository:**
   Fork the SustainDC repository to your own GitHub account.

2. **Clone the Repository:**
   Clone the forked repository to your local machine:
   ```bash
   git clone https://github.com/your-username/dc-rl.git
   ```
   
3. **Create a Branch:**
   Create a new branch for your feature or bug fix:
   ```bash
    git checkout -b feature-or-bugfix-name
   ```
   
4. **Make Changes:**
   Make your changes to the codebase.
   
5. **Commit Changes:**
   Commit your changes with a clear and descriptive commit message:
   ```bash
   git commit -m "Description of your changes"
   ```

6. **Push Changes:**
   Push your changes to your forked repository:
   ```bash
   git push origin feature-or-bugfix-name
   ```

7. **Create a Pull Request:**
  Open a pull request on the original SustainDC repository. Provide a clear description of what your changes do and any relevant information for the review process.

### Code of Conduct
Please note that we have a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

### Reporting Issues
If you find a bug or have a feature request, please create an issue on the [GitHub Issues](https://github.com/HewlettPackard/dc-rl/issues) page. Provide as much detail as possible to help us understand and address the issue.

### Guidelines
- Ensure your code follows the project's coding standards and conventions.
- Write clear, concise commit messages.
- Update documentation if necessary.
- Add tests to cover your changes if applicable.

Thank you for contributing to SustainDC!

## Contact
If you have any questions, suggestions, or issues, please feel free to contact us. We are here to help and support you in using SustainDC.

### Contact Information
- **Email:** [soumyendu.sarkar@hpe.com](mailto:soumyendu.sarkar@hpe.com)
- **GitHub Issues:** [GitHub Issues Page](https://github.com/HewlettPackard/dc-rl/issues)

## License

### MIT License

The majority of this project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

### Creative Commons Attribution-NonCommercial 4.0 International License

Some parts of this project are derived from code originally published under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License. The original authors retain the copyright to these portions, and they are used here under the terms of the CC BY-NC 4.0 license.

#### Attribution for CC BY-NC 4.0 Licensed Material
- [Original Repository](https://github.com/facebookresearch/CarbonExplorer)

### Combined Work
This combined work is available under both the MIT License for the new contributions and the CC BY-NC 4.0 License for the portions derived from the original work.


