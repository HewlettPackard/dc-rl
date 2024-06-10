========
Overview
========

In this section we will discuss each of the components of our framework in more details. |F| consist of three main environments:

* Workload Envronment - model and control the execution and scheduling of delay-tolerant workloads within the DC 
* Data Center Environment - model and manage the servers in the IT room cabinets that process workloads and the HVAC system and components 
* Battery Environment - simulates the DC battery charging behavior during off-peak hours and provides auxiliary energy to the DC during peak grid carbon intensity periods

.. image:: images/schematic.png
   :scale: 60 %
   :alt: Overview of the SustainDC main environments
   :align: center


.. update this 
.. toctree::
   :maxdepth: 1

   workload
   datacenter
   battery

   .. model
   environment
   agents
   observations
   actions
   reward_function

   