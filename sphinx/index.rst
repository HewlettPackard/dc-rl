.. dc-rl documentation master file, created by
   sphinx-quickstart on Thu Jun  1 22:41:38 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|F|
==========

|F| is a set of Python environments for benchmarking multi-agent reinforcement learning (MARL) algorithms in data centers (DC). It focuses on sustainable DC operations, including workload scheduling, cooling optimization, and auxiliary battery management. This page contains the documentation for the GitHub `repository <https://github.com/HewlettPackard/dc-rl>`_ for the paper `"SustainDC: Benchmarking for Sustainable Data Center Control" <https://openreview.net/forum?id=UYgE9IfQIV>`

>Avisek Naug, Antonio Guillen, Ricardo Luna Gutierrez, Vineet Gundecha, Desik Rengarajan, Sahand Ghorbanpour, Sajad Mousavi, Ashwin Ramesh Babu, Dejan Markovikj, Lekhapriya Dheeraj Kashyap, Soumyendu Sarkar, "SustainDC: Benchmarking for Sustainable Data Center Control" in _Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track_[Online]. Available: [pdf](https://openreview.net/attachment?id=UYgE9IfQIV&name=pdf).

<details markdown="block">
<summary>BibTeX</summary>

```tex
@inproceedings{naug2024sustaindc,
    title = {{SustainDC}: Benchmarking for Sustainable Data Center Control},
    author = {Naug, Avisek and Guillen, Antonio and Luna Gutierrez, Ricardo and Gundecha, Vineet and Rengarajan, Desik and Ghorbanpour, Sahand and Mousavi, Sajad and Babu, Ashwin Ramesh and Markovikj, Dejan and Dheeraj Kashyap, Lekhapriya and Sarkar, Soumyendu},
    year = 2024,
    month = 12,
    booktitle = {Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    address = {Vancouver, Canada},
    url = {https://openreview.net/forum?id=UYgE9IfQIV}
}
```
</details>

It uses OpenAI Gym standard and supports modeling and control of three different types of problems:

- Carbon-aware flexible load shifting
- Data center HVAC cooling energy optimization
- Carbon-aware battery auxiliary supply

.. image:: images/DCRL-sim1.png
   :scale: 30 %
   :alt: Overview of the physical and digital systems
   :align: center



Main contributions of |F|:

- the first OpenAI framework, to the best of our knowledge, focused on carbon footprint reduction for data center clusters
- support for hierarchical data center clusters modeling where users have flexibility in specifying the cluster architecture
- Modular design, i.e users can utilize pre-defined modules for load shifting, energy and battery or build their own
- Scalable architecture that allows multiple different types of modules and connections between them
- Robust data center model that provides in-depth customization to fit users' needs
- Provides pre-defined reward functions as well as interface to create custom reward functions
- Built-in mechanisms for reward shaping focused on degree of cooperation between the agents and level of prioritization of carbon footprint reduction versus other metrics such as energy cost
- Reward shaping through custom reward functions
- Built-in MARL algorithms, with ability to incorporate user-specified custom agents

.. toctree::
   :hidden:
   
   installation
   gettingstarted
   overview/index
   usage/index
   code/index
   train_evaluate/index
   contribution_guidelines
   references