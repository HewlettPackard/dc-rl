How to evaluate |F| model
================================

**#TODO: COMPLETE THIS!**

In addition to monitoring the training process, it is essential to evaluate the performance of the |F| model after training. 

To evaluate your model:

  - Modify the :code:`CHECKPOINT` variable in :code:`evaluate.py` module to point to the training data file (i.e. :code:`CHECKPOINT = './results/test/PPO_DCRL_c2f2a_00000_0_2024-06-16_16-51-50/checkpoint_001215/'`)
  - Run :code:`python eval_harl.py` for evaluation. The model is evaluated on the specifics, such as the RL algorithm, the Environment type used during training.
