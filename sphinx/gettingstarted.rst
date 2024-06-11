===============
Getting Started
===============

1. **Setup Configuration**

   Customize the :code:`dc_config.json` file to specify your DC environment settings. To learn more about the DC parameters you can customize, check **#TODO: ADD LINK TO DC PARAMS**.

2. **Environment Configuration**

   The main environment for wrapping the environments is :code:`dclr_env_harl_partialobs.py`, which reads configurations from the :code:`EnvConfig` class and manages the external variables using managers for weather, carbon intensity, and workload. For instructions how to customize the enviroment configuration, check **#TODO: ADD LINK TO ENV PARAMS**.

3. **Run Example:**

   Execute a simple example to get started: **#TODO: UPDATE THIS**


.. code-block:: python

   python examples/run_random.py


4. **Run Example:**

   Execute a simple example to get started:

.. code-block:: python
   
   python examples/evaluate.py


5. **Running in background mode**

If you want to run the |F| framework in background mode use:

.. code-block:: python

    nohup python PYTHON_SCRIPT > OUTPUT_FILE.txt  &

where :code:`PYTHON_SCRIPT` is the script you want to run (e.g., :code:`run_random.py`) and :code:`OUTPUT_FILE` is the name of the file that will contain the output (e.g. :code:`latest_experiment_output`).

6. **Monitor the results**

The training logs and the results of each trial are stored in the specified local directory, under a sub-folder called :code:`results` and can be visualized with TensorBoard by specifying the output directory of the results.

Example:

.. code-block:: python

    tensorboard --logdir ./results

In this example, :code:`test` is the default name of the experiment.


A detailed description of the configurations are provided in the :doc:`Usage <usage>` section. **#TODO:UPDATE THIS**