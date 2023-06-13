=====
Usage
=====

Reinforcement learning algorithms
---------------------------------

DCRL-Green supports the two multi-agent training approaches: independent learning approach such as IPPO :cite:p:`dewitt2020independent`, IA2C that decompose the problem into single-agent tasks for learning
and centralised learning approach such as MADDPG :cite:p:`lowe2020multiagent` that uses a centralised critic with decentralised execution.

:code:`--algorithm` argument can be used to provide the MARL algorithm of choice.

+--------+---------------------------+
| Name   |   Implementation          | 
+========+===========================+
| IPPO   | :code:`--algorithm IPPO`  |
+--------+---------------------------+
| IA2C   | :code:`--algorithm IA2C`  |
+--------+---------------------------+
| MADDPG | :code:`--algorithm MADDPG`|
+--------+---------------------------+

DC simulation
-------------

The data center simulation framework can be constructed using the following two options:
EnergyPlus enabled execution , Custom built execution

:code:`--EPmodel` argument can be used to provide the DC simulation of choice.

+------------+---------------------------+
| Name       |   Implementation          | 
+============+===========================+
| EnergyPlus | :code:`--EPmodel option`  |
+------------+---------------------------+
| EnergyPlus | :code:`--EPmodel option`  |
+------------+---------------------------+

Carbon Intensity data
---------------------

This dataset includes hourly average carbon intensity (CI) information for a given location. DCRL-Green includes CI data for 3 locations (NY, AZ, WA) collected by 
`US Energy Information Administration <eia>`_.
Users can upload CI dataset of choice in the repository, under :code:`dc-rl/data/CarbonIntensity` and use the argument :code:`--cintensity` to specify the filename.

.. _eia: https://www.eia.gov/environment/emissions/state/

.. note::
   CI dataset should be saved in .csv format

Example:

.. code-block:: bash

   python filename.py --cintensity NYIS.csv

Weather data
------------

Description of weather dataset. What info does it give? The default weather data in DCRL-Green was obtained from the EnergyPlus :cite:p:`crawley2000energy`
weather files collection. Users can upload weather dataset of choice in the repository, under :code:`dc-rl/data/Weather` and use the argument :code:`--weather` to specify the filename.

.. note::
   Weather dataset should be saved in .epw format

Example:

.. code-block:: bash
   
   python filename.py --weather NYIS.epw

Workload data
-------------

This dataset provides hourly IT workload information. The default weather data in DCRL-Green was obtained from the Alibaba open source database :cite:p:`alibaba2018`. Users can upload IT workload dataset of choice in the repository, under :code:`dc-rl/data/Workload` and use the argument :code:`--workload` to specify the filename.

.. note::
   Workload dataset should be saved in .csv format

Example:

.. code-block:: bash
   
   python filename.py --workload Alibaba_CPU_Data_Hourly.csv

Miscellaneous Inputs
--------------------

