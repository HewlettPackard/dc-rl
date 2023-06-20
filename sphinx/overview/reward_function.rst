===============
Reward function
===============

DCRL-Green provides an interface where the end-user can choose to train three agents independently of each other's reward feedback, or consider a collaborative reward approach. 

Independent rewards
-------------------

.. csv-table::
   :file: ../tables/agent_rewards.csv
   :header-rows: 1

Total EC refers o the total building energy consumption (HVAC+IT), CI is the carbon intensity in the power grid indicating the inverse of the availability of green energy, and UL refers to the amount of unassigned flexible workload.
A penalty is attributed to the load shifting agent if it fails to schedule all the required load within the time horizon N.

Collaborative rewards
---------------------

The reward-sharing mechanism allows the agents to estimate the feedback from their actions in other environments.

.. csv-table::
   :file: ../tables/agent_colab_rewards.csv
   :header-rows: 1

Users have an option to choose between the two reward mechanisms. This can be done by specifying the :math:`\eta` value in the script.

+----------------+---------------------------------------+
| Reward Scheme  |   Implementation                      | 
+================+=======================================+
| Collaborative  | .. code-block:: bash                  |
|                |                                       |   
|                |    'individual_reward_weight': 0.33   |
+----------------+---------------------------------------+
| Independent    | .. code-block:: bash                  |
|                |                                       |   
|                |    'individual_reward_weight': 1.0    |
+----------------+---------------------------------------+
| Default        | .. code-block:: bash                  |
| (weighted)     |                                       |   
|                |    'individual_reward_weight': 0.8    |
+----------------+---------------------------------------+

Custom rewards
--------------

DCRL-Green inludes an option to include custom rewards to match user's the desired optimization objective. 

**Step 1:** 
users can define these reward functions in :code:`utils/reward_creator.py`. The function can follow the schema:

.. code-block:: bash

    def custom_agent_reward(params: dict) -> float:
        #read reward input parameters from dict object
        #custom reward calculations 
        custom_reward = 0.0 #update with custom reward shaping 
        return custom_reward

**Step 2:**
Add the new custom reward function to the :code:`REWARD_METHOD_MAP` dictionary.

.. code-block:: bash

    REWARD_METHOD_MAP = {
    'default_dc_reward' : default_dc_reward,
    'default_bat_reward': default_bat_reward,
    'default_ls_reward' : default_ls_reward,
    # Additional rewards methods
    'tou_reward' : tou_reward,
    'renewable_energy_reward' : renewable_energy_reward,
    'energy_efficiency_reward' : energy_efficiency_reward,
    'energy_PUE_reward' : energy_PUE_reward,
    'temperature_efficiency_reward' : temperature_efficiency_reward,
    # Add custom reward methods here
    'custom_agent_reward' : custom_agent_reward,
    'how_you_want_to_call_it' : how_is_the_function_called,
    }


A dictionary of environment parameters is made available to the users. Based on the custom objective and requirements, a combination of these parameters can be utilised in defining the rewards.

.. csv-table::
   :file: ../tables/reward_params.csv
   :header-rows: 1

Some examples of custom rewards are listed below:

*Example 1*

.. code-block:: bash

    def energy_PUE_reward(params: dict) -> float:
        """
        Calculates a reward value based on Power Usage Effectiveness (PUE).

        Args:
            params (dict): Dictionary containing parameters:
                total_energy_consumption (float): Total energy consumption of the data center.
                it_equipment_energy (float): Energy consumed by the IT equipment.

        Returns:
            float: Reward value.
        """
        total_energy_consumption = params['total_energy_consumption']  
        it_equipment_energy = params['it_equipment_energy']  
        
        # Calculate PUE
        pue = total_energy_consumption / it_equipment_energy if it_equipment_energy != 0 else float('inf')
        
        # We aim to get PUE as close to 1 as possible, hence we take the absolute difference between PUE and 1
        # We use a negative sign since RL seeks to maximize reward, but we want to minimize PUE
        reward = -abs(pue - 1)
        
        return reward

*How to use this custom reward function*

To use this custom reward function, the reward function must be declared in the :code:`REWARD_METHOD_MAP`.
In this case, by default, the reward function is already declared as :code:`'energy_PUE_reward' : energy_PUE_reward,`.
Therefore, to use the reward function as a reward function of an agent (i.e., agent dc, the agent that modifies the HVAC setpoint), in the algorithm definition (i.e., :code:`train_ppo.py`),
the reward definition of the dc agent must be declared as :code:`dc_reward:energy_PUE_reward` within the env_config dictionary.

The following piece of code show how to declare the :code:`'energy_PUE_reward'` as reward function for the dc agent:

.. code-block:: bash

    CONFIG = (
        PPOConfig()
        .environment(
            env=DCRL if not os.getenv('EPLUS') else DCRLeplus,
            env_config={
                # Agents active
                'agents': ['agent_ls', 'agent_dc', 'agent_bat'],
                
                # Other configurations ....
                
                # Specify reward methods
                'ls_reward': 'default_ls_reward',
                **'dc_reward': 'energy_PUE_reward',**
                'bat_reward': 'default_bat_reward'
            }

Other reward function definitions can be found here:
*Example 2*

.. code-block:: bash

    def tou_reward(params: dict) -> float:
        """
        Calculates a reward value based on the Time of Use (ToU) of energy.

        Args:
            params (dict): Dictionary containing parameters:
                energy_usage (float): The energy usage of the agent.
                hour (int): The current hour of the day (24-hour format).

        Returns:
            float: Reward value.
        """
        
        # ToU dict: {Hour, price}
        tou = {0: 0.25,
            1: 0.25,
            2: 0.25,
            3: 0.25,
            4: 0.25,
            5: 0.25,
            6: 0.41,
            7: 0.41,
            8: 0.41,
            9: 0.41,
            10: 0.41,
            11: 0.30,
            12: 0.30,
            13: 0.30,
            14: 0.30,
            15: 0.30,
            16: 0.27,
            17: 0.27,
            18: 0.27,
            19: 0.27,
            20: 0.27,
            21: 0.27,
            22: 0.25,
            23: 0.25,
            }
        
        # Obtain the price of electricity at the current hour
        current_price = tou[params['hour']]
        # Obtain the energy usage
        energy_usage = params['bat_total_energy_with_battery_KWh']
        
        # The reward is negative as the agent's objective is to minimize energy cost
        tou_reward = -1.0 * energy_usage * current_price

        return tou_reward

*Example 3*

.. code-block:: bash

    def renewable_energy_reward(params: dict) -> float:
        """
        Calculates a reward value based on the usage of renewable energy sources.

        Args:
            params (dict): Dictionary containing parameters:
                renewable_energy_ratio (float): Ratio of energy coming from renewable sources.
                total_energy_consumption (float): Total energy consumption of the data center.

        Returns:
            float: Reward value.
        """
        assert params.get('renewable_energy_ratio') is not None, 'renewable_energy_ratio is not defined. This parameter should be included using some external dataset and added to the reward_info dictionary'
        renewable_energy_ratio = params['renewable_energy_ratio'] # This parameter should be included using some external dataset
        total_energy_consumption = params['bat_total_energy_with_battery_KWh']
        factor = 1.0 # factor to scale the weight of the renewable energy usage

        # Reward = maximize renewable energy usage - minimize total energy consumption
        reward = factor * renewable_energy_ratio  -1.0 * total_energy_consumption
        return reward

*Example 4*

.. code-block:: bash

    def energy_efficiency_reward(params: dict) -> float:
        """
        Calculates a reward value based on energy efficiency.

        Args:
            params (dict): Dictionary containing parameters:
                ITE_load (float): The amount of energy spent on computation (useful work).
                total_energy_consumption (float): Total energy consumption of the data center.

        Returns:
            float: Reward value.
        """
        it_equipment_power = params['dc_ITE_total_power_kW']  
        total_power_consumption = params['dc_total_power_kW']  
        
        reward = it_equipment_power / total_power_consumption
        return reward

*Example 5*

.. code-block:: bash

    def temperature_efficiency_reward(params: dict) -> float:
        """
        Calculates a reward value based on the efficiency of cooling in the data center.

        Args:
            params (dict): Dictionary containing parameters:
                current_temperature (float): Current temperature in the data center.
                optimal_temperature_range (tuple): Tuple containing the minimum and maximum optimal temperatures for the data center.

        Returns:
            float: Reward value.
        """
        assert params.get('optimal_temperature_range') is not None, 'optimal_temperature_range is not defined. This parameter should be added to the reward_info dictionary'
        current_temperature = params['dc_int_temperature'] 
        optimal_temperature_range = params['optimal_temperature_range']
        min_temp, max_temp = optimal_temperature_range
        
        if min_temp <= current_temperature <= max_temp:
            reward = 1.0
        else:
            if current_temperature < min_temp:
                reward = -abs(current_temperature - min_temp)
            else:
                reward = -abs(current_temperature - max_temp)
        return reward