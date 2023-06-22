===============
Getting Started
===============

1. Basic example
----------------

Run the DCRL-Green environment to simulate one episode and view the roll-out information using the following command:

.. code-block:: python

    python dcrl_env.py 

2. Train the agents
-------------------

Run the following commands to simulate the DC enivronment and optimize the agents with default configurations:

2.1 Train using the DCRL environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    python train_ppo.py 

2.2 Train using the DCRLeplus environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If not done already, launch the docker container by executing the command:

.. code-block:: console

   docker run -t -i -v $PWD:/sinergym/dc-rl --shm-size=10.24gb agnprz/carbon_sustain:v3

If the docker container is launched successfully, the isolated :code:`sinergym` environment is enabled. Navigate to the to the source directory :code:`dc-rl` to be able to execute DCRL-Green:

.. code-block:: console

   cd dc-rl

.. note::
   Some useful Docker commands could be found here_
   
.. _here: https://docs.docker.com/engine/reference/commandline/cli/

Next, run the script:

.. code-block:: python

    EPLUS=1 python train_ppo.py

.. note::
   The :code:`episode_reward_mean` will display nan values for the first few iterations until 1 episode is completed

2.3 Running in background mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to run the DCRL-Green framework in background mode use the following command:

.. code-block:: python

    nohup python PYTHON_SCRIPT > OUTPUT_FILE.txt  &

where :code:`PYTHON_SCRIPT` is the script you want to run (e.g., :code:`train_ppo.py`) and :code:`OUTPUT_FILE` is the name of the file that will contain the output (e.g. :code:`latest_experiment_output`).

3. Monitor the results
----------------------

The training logs and the results of each trial are stored in the specified local directory, under a sub-folder called :code:`results` and can be visualized with TensorBoard by specifying the output directory of the results.

Example:

.. code-block:: python

    tensorboard --logdir ./results

In this example, :code:`test` is the default name of the experiment.


A detailed description of the configurations are provided in the Usage section. The default configurations in DCRL-Green are:

.. csv-table::
   :file: tables/default_args.csv
   :header-rows: 1