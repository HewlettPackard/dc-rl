How to evaluate DCRL-Green model
================================

In addition to monitoring the training process using TensorBoard, it is essential to evaluate the performance of the DCRL-Green model after training. The following steps outline the evaluation procedure:

1. Modifiy the :code:`CHECKPOINT` variable to point to the training data file

2. Load the trained RL agent, the environment, and the evaluation data using the :code:`run` function provided in the :code:`evaluate_model.py` module.

3. Run the evaluation using the :code:`evaluate_model.py` module.

4. Analyze and visualize the evaluation results to determine the model's performance using the csv file created.

Using the steps outlined above, you can effectively evaluate the DCRL-Green model and make necessary adjustments to improve its performance.