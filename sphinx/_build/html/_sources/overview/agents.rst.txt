======
Agents
======

DCRL-Green includes three different types of agents for energy and carbon footprint optimization as shown in the figure below. The three types of agents are: 

- Flexible Load Shifting agent: to schedule data center workloads 
- HVAC Cooling agent: to model the data center operations under the given workload, grid and weather conditions, and to make decisions on the temperature setpoint of the cooling system 
- Energy Storage agent: usually a typical data center Uninterrupted Power System (UPS) with a scheduler that decides when to charge or discharge the battery based on data center power demand and grid carbon intensity

.. figure:: ../images/dependency.png
   :scale: 30 %
   :alt: Internal and external system dependencies
   :align: center

   Internal and external system dependencies

