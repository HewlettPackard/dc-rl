===============
Getting Started
===============

1. **Setup Configuration**

   Customize the :code:`dc_config.json` file to specify your DC environment settings. To learn more about the DC parameters you can customize, check :ref:`dcconf_ref`

2. **Environment Configuration**

   The main environment for wrapping the environments is :code:`sustaindc_env.py`, which reads configurations from the :code:`EnvConfig` class and manages the external data sources using managers for weather, carbon intensity, and workload. For instructions how to customize the enviroment configuration, check :ref:`mainconf_ref`

3. **Train Example:**

   Specify :code:`location` inside :code:`harl.configs.envs_cfgs.sustaindc.yaml`. Specify other algorithm hyperparameteres in :code:`harl.configs.algos_cfgs.happo.yaml`. User can also specify the choice of reinforcement learning vs baseline agents in the :code:`happo.yaml`
   
.. code-block:: bash
      
      python train_sustaindc.py --algo happo --exp_name happo


4. **Evaluation Example:**

   To evaluate the trained model run:

.. code-block:: bash
   
   python eval_sustaindc.py

The results are stored in the :code:`SAVE_EVAL` folder. This can be modified inside :code:`eval_sustaindc.py` with other experiment identifiers such as :code:`checkpoint`, :code:`location` and :code:`run`

5. **Running in background mode**

If you want to run the |F| framework in background mode use:

.. code-block:: bash

    nohup python PYTHON_SCRIPT > OUTPUT_FILE.txt  &

where :code:`PYTHON_SCRIPT` is the script you want to run (e.g., :code:`train_sustaindc.py`) and :code:`OUTPUT_FILE` is the name of the file that will contain the output (e.g. :code:`latest_experiment_output`)

6. **Monitor the results**

The training logs and the results of each trial are stored in the specified local directory, under a sub-folder called :code:`results` and can be visualized with TensorBoard by specifying the output directory of the results

Example:

.. code-block:: bash

    tensorboard --logdir ./results/dcrl/<location>/happo


A detailed description of the configurations are provided in the :ref:`usage_ref` section.