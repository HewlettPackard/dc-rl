=================================
Custom Reward Functions 
=================================

|F| provides an interface where the end-user can choose to train agents independently of each other's reward feedback, or consider a collaborative reward approach. 

Independent reward functions
----------------------------------

.. csv-table::
   :file: ../tables/agent_rewards.csv
   :header-rows: 1

Total EC refers to the total building energy consumption (HVAC+IT), CI is the carbon intensity in the power grid indicating the inverse of the availability of green energy, and UL refers to the amount of unassigned flexible computational workload.
A penalty is attributed to the load shifting agent if it fails to schedule all the required load within the time horizon N.

Collaborative reward functions
----------------------------------

The reward-sharing mechanism allows the agents to estimate the feedback from their actions in other environments. Users have an option to choose the level of colaboration between the agents. This can be done by specifying the :math:`\eta` value in the script.

.. csv-table::
   :file: ../tables/agent_colab_rewards.csv
   :header-rows: 1

Example :math:`\eta` values to set up a collaborative, independent and custom weighted environment are given in the table below. The higher the value of :math:`\eta`, the less collaboration between the agents in the environment.   

+----------------+-------------------------------------------+
| Reward Scheme  |   Implementation                          | 
+================+===========================================+
| Collaborative  | :code:`individual_reward_weight': 0.33`   |
+----------------+-------------------------------------------+
| Independent    | :code:`individual_reward_weight': 1.0`    |
+----------------+-------------------------------------------+
| Default        | :code:`individual_reward_weight': 0.8`    |
| (weighted)     |                                           |   
+----------------+-------------------------------------------+

Custom reward functions
--------------------------------

|F| allows users to define custom reward structures to promote collaborative optimization across different DC components. Users can modify the reward functions in the :code:`utils/reward_creator.py` file to suit their specific optimization goals. Those function should follow the schema:

.. code-block:: python

    def custom_agent_reward(params: dict) -> float:
        #read reward input parameters from dict object
        #custom reward calculations 
        custom_reward = 0.0 #update with custom reward shaping 
        return custom_reward

Next, users need to add the new custom reward function(s) to the :code:`REWARD_METHOD_MAP` dictionary:

.. code-block:: python

    REWARD_METHOD_MAP = {
    'default_dc_reward' : default_dc_reward,
    'default_bat_reward': default_bat_reward,
    'default_ls_reward' : default_ls_reward,
    # Add custom reward methods here
    'custom_agent_reward' : custom_agent_reward,
    'tou_reward' : tou_reward,
    'renewable_energy_reward' : renewable_energy_reward,
    'energy_efficiency_reward' : energy_efficiency_reward,
    'energy_PUE_reward' : energy_PUE_reward,
    'temperature_efficiency_reward' : temperature_efficiency_reward,
    'water_usage_efficiency_reward' : water_usage_efficiency_reward,
    }


A dictionary of the environment parameters (:code:`reward_params`) is available to users in :code:`sustaindc_env.py`.
This object consists of the information dictionary of each environment, and some other global variables such as time, day, carbon intensity, outside temperature, etc.
If a user wants to add additional custom parameters, they must be added in the dictionary :code:`reward_params` so that those variables are visible in the reward function.
Within the dictionary, the following environment parameters are available to users:

.. csv-table::
   :file: ../tables/reward_params.csv
   :header-rows: 1

Depending on the objective and requirements, users can utilize a combination of these parameters to define their customized reward functions, or use one of already provided reward functions.

Some examples of custom rewards are listed below:

*Example 1: Reward function based on power usage effectivness (PUE)*

.. code-block:: python

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

*Example 2: Reward function based on time of use (ToU) of energy*

.. code-block:: python

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

*Example 3: Reward function based on the usage of renewable energy sources*

.. code-block:: python

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

*Example 4: Reward function based on energy efficiency*

.. code-block:: python

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

*Example 5: Reward function based on the efficiency of cooling in the data center*

.. code-block:: python

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

*Example 6: Reward function based on the efficiency of water usage in the data center*

.. code-block:: python

    def water_usage_efficiency_reward(params: dict) -> float:
    """
    Calculates a reward value based on the efficiency of water usage in the data center.
    
    A lower value of water usage results in a higher reward, promoting sustainability
    and efficiency in water consumption.

    Args:
        params (dict): Dictionary containing parameters:
            dc_water_usage (float): The amount of water used by the data center in a given period.

    Returns:
        float: Reward value. The reward is higher for lower values of water usage, 
        promoting reduced water consumption.
    """
    dc_water_usage = params['dc_water_usage']
    
    # Calculate the reward. This is a simple inverse relationship; many other functions could be applied.
    # Adjust the scalar as needed to fit the scale of your rewards or to emphasize the importance of water savings.
    reward = -0.01 * dc_water_usage
    
    return reward

By leveraging these customization options, users can create highly specific and optimized simulations that reflect the unique requirements and challenges of their DC operations.