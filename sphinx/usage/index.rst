=====
Usage
=====

Reinforcement learning algorithms
---------------------------------

DCRL-Green supports the two multi-agent training approaches: independent learning approach such as IPPO :cite:p:`dewitt2020independent`, IA2C that decompose the problem into single-agent tasks for learning
and centralised learning approach such as MADDPG :cite:p:`lowe2020multiagent` that uses a centralised critic with decentralised execution.

The following command can be used to run the MARL algorithm of choice:

+--------+---------------------------+------------------------------------+
| Name   |   DCRL Implementation     | DCRLeplus Implementation           |
+========+===========================+====================================+
| IPPO   | .. code-block:: bash      | .. code-block:: bash               |
|        |                           |                                    |
|        |    python train_PPO.py    |   EPLUS=1 python train_ppo.py      |
+--------+---------------------------+------------------------------------+
| IA2C   | .. code-block:: bash      | .. code-block:: bash               |
|        |                           |                                    |
|        |    python train_a2c.py    |    EPLUS=1 python train_a2c.py     |
+--------+---------------------------+------------------------------------+
| MADDPG | .. code-block:: bash      | .. code-block:: bash               |
|        |                           |                                    |
|        |    python train_MADDPG.py |    EPLUS=1  python train_MADDPG.py |
+--------+---------------------------+------------------------------------+

.. note::
   User configurations described in the following sections can be modified by directly editing the above python scripts.

Carbon Intensity data
---------------------

This dataset includes hourly average carbon intensity (CI) information for a given location. DCRL-Green includes CI data for 3 locations (NY, AZ, WA) collected by 
`US Energy Information Administration <eia>`_.
Users can upload CI dataset of choice in the repository, under :code:`dc-rl/data/CarbonIntensity` and use the variable :code:`'cintensity_file'` to specify the filename.

.. _eia: https://www.eia.gov/environment/emissions/state/

.. note::
   CI dataset should be saved in .csv format and should contain CI data for one year (24*365)

Example:

.. code-block:: bash

   'cintensity_file': 'NYIS_NG_&_avgCI.csv'

The above default dataset consists CI data for a period of one year starting from 1/1/2022 to 12/31/2022.

Weather data
------------

Description of weather dataset. What info does it give? The default weather data in DCRL-Green was obtained from the EnergyPlus :cite:p:`crawley2000energy`
weather files collection. Users can upload weather dataset of choice in the repository, under :code:`dc-rl/data/Weather` and use the variable :code:`'weather_file'` to specify the filename.

.. note::
   Weather dataset should be saved in .epw format

Example:

.. code-block:: bash
   
   'weather_file': 'USA_NY_New.York-Kennedy.epw'

Workload data
-------------

This dataset provides hourly IT workload information. The default weather data in DCRL-Green was obtained from the Alibaba open source database :cite:p:`alibaba2018`. Users can upload IT workload dataset of choice in the repository, under :code:`dc-rl/data/Workload` and use the variable :code:`"workload_file"` to specify the filename.

.. note::
   Workload dataset should be saved in .csv format and should contain data for one year (24*365)
   
Agent configuration
-------------------

DCRL-Green supports three MARL agents to optimize energy usage and reduce carbon footprint of data centers. More details of the agents' operations are discussed in :ref:`Agents` section. Based on the requirement, users can include agents of choice in the training script. The agents that are not involved in training will select the :code:`Idle` action by default and will not contribute to the optimization process. The variable :code:`"agents"` can be used to specify the required agents.

.. note::
   Agent names must be provided as a list of strings, where :code:`"agent_ls"`, :code:`"agent_dc"`, :code:`"agent_bat"` represents load shifting agent, DC cooling agent, battery control agent respectively

Example:

.. code-block:: bash
   
   'agents': ['agent_ls','agent_dc', 'agent_bat']

Hyperparameter configuration
----------------------------

The hyperparameters are specific to the MARL algorithms discussed above. The following table represent the default values used and method to modify the hyperparameters. 

.. csv-table::
   :file: ../tables/hperparameters_table.csv
   :header-rows: 1