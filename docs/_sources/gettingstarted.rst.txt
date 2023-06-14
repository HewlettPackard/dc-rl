===============
Getting Started
===============

1. Example
----------

This is the simplest example to simulate one episode and view the roll-out information.

TODO

2. Train the agents
-------------------

Run the following command to simulate the DC enivronment and optimize the agents with default configurations:

.. note::
   For execution using EnergyPlus model, launch the Sinergym environment before running the script (details can be found in :ref:`EnergyPlus enabled DC simulation`)

.. code-block:: python

    python train_ppo.py 

2.1 Monitor the results
~~~~~~~~~~~~~~~~~~~~~~~

The training logs and the results of each trial are stored in the specified local directory, under a sub-folder called :code:`/results/` and can be visualized with TensorBoard by specifying the output directory of the results.

Example:

.. code-block:: python

    tensorboard --logdir results/latest_experiment

In this example, :code:`latest_experiment` is the default name of the experiment.


A detailed description of the configurations are provided in the Usage section. The Default configurations in DCRL-green are as follows:

.. csv-table::
   :file: tables/default_args.csv
   :header-rows: 1