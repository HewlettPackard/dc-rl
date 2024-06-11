===============================
Benchmarking Algorithms
===============================


|F| supports a variety of reinforcement learning algorithms for benchmarking. This section provides an overview of the supported algorithms and highlights their differences.

Supported Algorithms
--------------------------

.. csv-table::
   :file: ../tables/supported_algorithms.csv
   :header-rows: 1


Differences and Use Cases
--------------------------

  - **PPO vs. IPPO:** PPO is for single-agent setups, while IPPO suits multi-agent environments with independent learning.
  - **IPPO vs. MAPPO:** IPPO treats agents independently; MAPPO coordinates agents with a centralized value function, ideal for cooperative tasks.
  - **MAPPO vs. HAPPO:** Both use centralized value functions, but HAPPO is for heterogeneous agents with different capabilities.
  - **HAPPO vs. HATRPO:** HAPPO uses PPO-based updates; HATRPO adapts TRPO for more stable and robust policy updates in heterogeneous settings.
  - **HAPPO vs. HAA2C:** HAPPO is PPO-based; HAA2C extends A2C to multi-agent systems, offering stability and performance trade-offs.
  - **HAA2C vs. HAD3QN:** HAA2C is an actor-critic method; HAD3QN uses value-based learning with dueling and double Q-learning
  - **HAD3QN vs. HASAC:** HAD3QN is value-based; HASAC uses entropy regularization for environments with continuous action spaces.

.. csv-table::
   .. :file: ../tables/algorithms_usecases.csv
   .. :header-rows: 1

By supporting a diverse set of algorithms, |F| allows researchers to benchmark and compare the performance of various reinforcement learning approaches in the context of sustainable DC control.