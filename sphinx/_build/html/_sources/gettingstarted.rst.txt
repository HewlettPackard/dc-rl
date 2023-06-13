===============
Getting Started
===============

Within the docker container, run the following command to simulate the DC enivronment and optimize the agents with default configurations:

.. code-block:: python

    python filename.py 

The training logs and the results of each trial are stored in the specified local directory, under a sub-folder called :code:`/results/` and can be visualized with TensorBoard by specifying the output directory of the results.

Example:

.. code-block:: python

    tensorboard --logdir results/latest_experiment

In this example, :code:`latest_experiment` is the default name of the experiment.


A detailed description of the configurations are provided in the Usage section. The Default configurations in DCRL-green are as follows:

.. csv-table::
   :file: tables/default_args.csv
   :header-rows: 1