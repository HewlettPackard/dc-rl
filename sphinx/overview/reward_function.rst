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
|                |    'individual_reward_weight': 0.0    |
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
        renewable_energy_ratio = params['renewable_energy_ratio']
        total_energy_consumption = params['total_energy_consumption']

        reward = -1.0 * renewable_energy_ratio * total_energy_consumption
        return reward

    REWARD_METHOD_MAP = {
        'default_dc_reward' : default_dc_reward,
        'default_bat_reward': default_bat_reward,
        'default_ls_reward' : default_ls_reward,
        # Add custom reward methods here
        'custom_agent_reward' : custom_agent_reward,
        'renewable_energy_reward': renewable_energy_reward,
    }

