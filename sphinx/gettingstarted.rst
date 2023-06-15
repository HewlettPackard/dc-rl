===============
Getting Started
===============

1. Example
----------

This is the simplest example to simulate one episode and view the roll-out information.

.. code-block:: python

    python dcrl_env.py 

2. Train the agents
-------------------

Run the following command to simulate the DC enivronment and optimize the agents with default configurations:

2.1 Train with DCRL Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    python train_ppo.py 

2.2 Train with DCRLeplus Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For execution using EnergyPlus model, launch the :code:`Sinergym` environment before running the script (details can be found in :ref:`EnergyPlus enabled DC simulation (DCRLeplus)`)

.. code-block:: python

    EPLUS=1 python train_ppo.py

.. note::
   The :code:`episode_reward_mean` will display nan values for the first few iterations until 1 episode is completed

3. Monitor the results
----------------------

The training logs and the results of each trial are stored in the specified local directory, under a sub-folder called :code:`/results/` and can be visualized with TensorBoard by specifying the output directory of the results.

Example:

.. code-block:: python

    tensorboard --logdir ./results

In this example, :code:`test` is the default name of the experiment.


A detailed description of the configurations are provided in the Usage section. The Default configurations in DCRL-green are as follows:

.. csv-table::
   :file: tables/default_args.csv
   :header-rows: 1