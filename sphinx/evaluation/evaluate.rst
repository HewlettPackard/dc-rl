How to evaluate |F| model
================================


In addition to monitoring the training process, it is essential to evaluate the performance of the |F| model after training. 

To evaluate your model:

  - Modify the constants in :code:`eval_harl.py` module to point to the training data file (by default it points to :code:`./trained_models/dcrl/az`)
  - There are multiple trained models grouped by location in the :code:`./trained_models/dcrl/` directory
  - Run :code:`python eval_harl.py` for evaluation. The model is evaluated on the specifics, such as the RL algorithm, DC location, and the Environment type used during training.

The default constant values in :code:`eval_harl.py` are given below: 

.. code-block:: python

  MODEL_PATH = 'trained_models'
  SAVE_EVAL = "results"
  ENV = 'dcrl'
  LOCATION = "az"
  AGENT_TYPE = "haa2c"
  RUN = "seed-00001-2024-06-04-20-41-56"
  ACTIVE_AGENTS = ['agent_ls', 'agent_dc', 'agent_bat']
  NUM_EVAL_EPISODES = 1