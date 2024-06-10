===============
Getting Started
===============

1. Basic example
----------------

Run the |F| environment to simulate one episode and view the roll-out information using the following command:

.. code-block:: python

    python sustaindc_env.py 

2. Train the agents
-------------------

Run the following commands to simulate the DC enivronment and optimize the agents with default configurations:

.. code-block:: python

    python train_ppo.py 

3. Running in background mode
-------------------

If you want to run the |F| framework in background mode use the following command:

.. code-block:: python

    nohup python PYTHON_SCRIPT > OUTPUT_FILE.txt  &

where :code:`PYTHON_SCRIPT` is the script you want to run (e.g., :code:`train_ppo.py`) and :code:`OUTPUT_FILE` is the name of the file that will contain the output (e.g. :code:`latest_experiment_output`).

4. Monitor the results
----------------------

The training logs and the results of each trial are stored in the specified local directory, under a sub-folder called :code:`results` and can be visualized with TensorBoard by specifying the output directory of the results.

Example:

.. code-block:: python

    tensorboard --logdir ./results

In this example, :code:`test` is the default name of the experiment.


A detailed description of the configurations are provided in the :doc:`Usage <usage>` section. 