===========
Environment
===========

DCRL-Green provides a simulation framework for Data Centers and evaluates multi objective, collaborative MARL agents in this environment. 
Along with the flexibility to adjust other external settings, such as the workload profile, the external weather pattern, and the grid carbon intensity (CI) as discussed in :ref:`Usage`, it provides customizable JSON object files for three main sections:

1. DC Geometry (rows, racks, workload, CRAC characteristics)
2. Server Characteristics (power consumption)
3. HVAC Characteristics (cooling mechanisms)

The model calculates Total IT power, CRAC Fan power, and CRAC Evaporator power, enabling real-time calculations for energy optimization.

Configure the DC Environment:
-----------------------------

Users can configure the simulation of the DC environment to suit their requirements. It may be configured by modifying the :code:`utils/dc_config.json` file in the Github_ repository.

.. _Github: https://github.com/HewlettPackard/dc-rl/blob/main/utils/dc_config.json

1. Configure DC Geometry
~~~~~~~~~~~~~~~~~~~~~~~~

The following DC specifications could be modified under the :code:`data_center_configuration` subsection as follows:

.. csv-table::
   :file: ../tables/DC_geometry.csv
   :header-rows: 1

2. Configure Server Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following DC specifications could be modified under the :code:`server_characteristics` subsection as follows:

.. csv-table::
   :file: ../tables/DC_server_chars.csv
   :header-rows: 1

3. Configure HVAC Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following DC specifications could be modified under the :code:`hvac_configuration` subsection as follows:

.. csv-table::
   :file: ../tables/DC_hvac_chars.csv
   :header-rows: 1