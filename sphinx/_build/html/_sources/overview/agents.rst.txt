======
Agents
======

DCRL-Green includes three different agents for energy and carbon footprint optimization as shown in the figure. It includes the Flexible Load
Shifting agent to schedule data center workloads, HVAC Cooling agent to model the data center operations under the given workload, grid 
and weather conditions, to makes decisions on the temperature setpoint of the cooling system and the Energy Storage agent is a typical data center Uninterrupted Power System (UPS) 
with a scheduler that decides when to charge or discharge the battery based on data center power demand and grid Carbon intensity.

.. figure:: ../images/dependency.png
   :scale: 30 %
   :alt: Internal and External System Dependencies
   :align: center

   Internal and External System Dependencies

