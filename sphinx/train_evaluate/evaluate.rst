How to evaluate DCRL-Green model
================================

In addition to monitoring the training process using TensorBoard, it is essential to evaluate the performance of the DCRL-Green model after training. The following steps outline the evaluation procedure:

**Step 1:** 

Modify the :code:`CHECKPOINT` variable in :code:`evaluate.py` module to point to the training data file.

Example:

.. code-block:: python

    #path to checkpoint
    CHECKPOINT = './results/test/PPO_DCRL_c2f2a_00000_0_2023-06-16_16-51-50/checkpoint_001215/'

**Step 2:** 

Run the following command for evaluation. The model is evaluated on the specifics, such as the RL algorithm, the Environment type used during training.

.. code-block:: python

    python evaluate.py

The evaluation process prints a table with the Carbon Footprint and Total Energy consumption values for a period of one year.
Using the steps outlined above, you can effectively evaluate the DCRL-Green model and make necessary adjustments to improve its performance.