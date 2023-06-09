.. dc-rl documentation master file, created by
   sphinx-quickstart on Thu Jun  1 22:41:38 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DCRL-Green
==========

DCRL-green is a framework for testing Multi-Agent Reinforcement Learning (MARL) algorithms that optimizes data centers for multiple objectives of carbon footprint reduction, energy consumption, and energy
cost.

It uses OpenAI Gym standard and supports modeling and control of three different types of problems:

- Carbon-aware flexible load shifting
- Data center HVAC cooling energy optimization
- Carbon-aware battery auxiliary supply

.. image:: images/DCRL-sim1.png
   :scale: 30 %
   :alt: Overview of the physical and digital systems
   :align: center



Main contributions of DCRL-Green:

- The first OpenAI framework, to our knowledge, focused on carbon footprint reduction for data centers
- Modular design, i.e users can utilize pre-defined modules for load shifting, energy and battery or build their own
- Scalable architecture that allows multiple different types of modules and connections between them
- Robust data center model that provides in-depth customization to fit users' needs
- Provides pre-defined reward functions as well as interface to create custom reward functions
- Built-in mechanisms for reward shaping focused on degree of cooperation between the agents and level of prioritization of carbon footprint reduction versus other metrics such as energy cost
- Reward shaping through custom reward functions
- Built-in MARL algorithms, with ability to incorporate user-specified custom agents

.. toctree::
   :hidden:
   
   installation/index
   gettingstarted
   usage/index
   overview/index
   code/index
   train_evaluate/index
   contribution_guidelines
   references