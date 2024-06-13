==================
Reward Functions
==================

|F| provides an interface where the end-user can choose to train agents independently of each other's reward feedback, or consider a collaborative reward approach. 

Independent rewards
-------------------

.. csv-table::
   :file: ../tables/agent_rewards.csv
   :header-rows: 1

Total EC refers to the total building energy consumption (HVAC+IT), CI is the carbon intensity in the power grid indicating the inverse of the availability of green energy, and UL refers to the amount of unassigned flexible computational workload.
A penalty is attributed to the load shifting agent if it fails to schedule all the required load within the time horizon N.

Collaborative rewards
---------------------

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