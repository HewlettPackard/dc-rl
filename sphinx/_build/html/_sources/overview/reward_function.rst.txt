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
        # Add custom reward methods here
        'custom_agent_reward' : custom_agent_reward,
        
    }

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

    REWARD_METHOD_MAP = {
        'default_dc_reward' : default_dc_reward,
        'default_bat_reward': default_bat_reward,
        'default_ls_reward' : default_ls_reward,
        # Add custom reward methods here
        'custom_agent_reward' : custom_agent_reward,
        'energy_PUE_reward': energy_PUE_reward,
    }

