========
Overview
========

Data Center Model 
-----------------------
High-level overview of the operational model of a |F| data center is given in the figure below.

.. _sustaindc_model:

.. figure:: ../images/SustainDC.png
   :scale: 40 %
   :alt: Overview of the SustainDC main environments
   :align: center


Workloads are uploaded to the DC from a proxy client. A fraction of these jobs can be flexible or delayed to different time periods. The servers that process these jobs generate heat that needs to be removed from the DC. This is done by a complex HVAC system that ensures optimal temperature in the DC. As shown in the figure below, the warm air leaves the servers and is moved to the Computer Room Air Handler (CRAH) by the forced draft of the HVAC fan. Next, the hot air is cooled down to optimal setpoint using a chilled water loop and then send back to the IT room. Parallely, a second water loop transfers the removed heat to a cooling tower, where it is rejected to the outside environment. 

.. _sustaindc_hvac:

.. image:: ../images/Data_center_modelled.png
   :scale: 60 %
   :alt: Overview of the SustainDC HVAC system
   :align: center

Big data centers also incorporate battery banks. Batteries can be charged from the grid during low Carbon Intensity (CI) periods. During higher CI periods, they provide auxiliary energy to the DC.  

Core Environments 
-----------------------

|F| consist of three main environments reflecting different aspects of a data center that can be optimized to reduce carbon footprint or other sustainability metrics:

* Workload Envronment - model and control the execution and scheduling of delay-tolerant workloads within the DC 
* Data Center Environment - model and manage the servers in the IT room cabinets that process workloads and the HVAC system and components 
* Battery Environment - simulates the DC battery charging behavior during off-peak hours and provides auxiliary energy to the DC during peak grid carbon intensity periods

|F| enables a comprehensive set of customizations for each of the three environmentsdeveloped in Python. A high-level overview that highlights their individual components, customization capabilities, and associated control problems is given in th figure below.

.. _sustaindc_envs:

.. image:: ../images/schematic.png
   :scale: 60 %
   :alt: Overview of the SustainDC components and parameters
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

   