===========
Environment
===========

DCRL-Green consists of multiple sub-environments that together form a comprehensive simulation framework for data centers. Each sub-environment focuses on a specific aspect of data center operations and includes adjustable parameters that allow researchers to customize the simulation according to their specific needs and research goals.
The adjustable parameters vary depending on the sub-environment and enable researchers to investigate and optimize different components of data center sustainability. 

Here are the key sub-environments in DCRL-Green along with their adjustable parameters:

**Data Center Environment:**
This sub-environment simulates the overall operation of the data center.
Along with the flexibility to adjust other external settings such as the workload profile, the external weather pattern, and the grid carbon intensity (CI) as discussed in the :ref:`Usage` section, it provides customizable a JSON object file for three main sections:

1. DC geometry (rows, racks, workload, CRAC characteristics)
2. Server characteristics (power consumption)
3. HVAC characteristics (cooling mechanisms)

The model calculates Total IT power, CRAC fan power, and CRAC evaporator power, enabling real-time calculations for energy optimization.

**Load Shifting Environment:**
The load shifting sub-environment focuses on the scheduling and allocation of flexible workloads within the data center. 
Adjustable parameters in this sub-environment include:

1. Flexible workload ratio
2. Time window for rescheduling

**Battery Environment:**
This sub-environment models the battery energy storage system in the data center. Adjustable parameters in the battery environment include:

1. Maximum battery capacity
2. Charging and discharging rates
3. Time window for scheduling

The adjustable parameters in each sub-environment allow researchers to explore different scenarios, evaluate the impact of various factors, and optimize data center sustainability. By adjusting these parameters, users can conduct comprehensive simulations, investigate optimization strategies, and customize the simulation environment to their specific research needs.

It is worth mentioning that the adjustable parameters outlined above represent a subset of the available options in each sub-environment. DCRL-Green offers extensive flexibility and customization capabilities, providing researchers with a powerful platform to study and improve data center sustainability.

Feel free to modify and enhance the content based on your specific sub-environments and adjustable parameters in DCRL-Green.

Configure the DC Environment:
-----------------------------

Users can configure the simulation of the DC environment to suit their requirements. It may be configured by modifying the :code:`utils/dc_config.json` file in the GitHub_ repository.

.. _GitHub: https://github.com/HewlettPackard/dc-rl/blob/main/utils/dc_config.json

.. figure:: ../images/Labeled_dc_view_2.png
   :scale: 20 %
   :alt:  Data center view in 6Sigma DCX from a given configuration file
   :align: center

   Data center view in 6Sigma DCX from a given configuration file

1. Configure DC geometry
~~~~~~~~~~~~~~~~~~~~~~~~

The following DC specifications could be modified under the :code:`data_center_configuration` subsection:

.. csv-table::
   :file: ../tables/DC_geometry.csv
   :header-rows: 1

2. Configure server characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following DC specifications could be modified under the :code:`server_characteristics` subsection:

.. csv-table::
   :file: ../tables/DC_server_chars.csv
   :header-rows: 1

3. Configure HVAC Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following DC specifications could be modified under the :code:`hvac_configuration` subsection:

.. csv-table::
   :file: ../tables/DC_hvac_chars.csv
   :header-rows: 1

Configure the Load Shifting Environment:
----------------------------------------

It may be configured by modifying the source code as shown below:

.. csv-table::
   :file: ../tables/ls_config_table.csv
   :header-rows: 1

Configure the Battery Environment:
----------------------------------

It may be configured by modifying the source code as shown below:

.. csv-table::
   :file: ../tables/bat_config_table.csv
   :header-rows: 1
