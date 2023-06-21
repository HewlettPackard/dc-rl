How to Monitor Training Results Using TensorBoard
=================================================

TensorBoard is a powerful tool for monitoring and visualizing the training process of reinforcement learning algorithms. DCRL-Green provides a custom callback implementation called :code:`CustomCallbacks` found in :code:`utils/rllib_callbacks.py` which can be used to track the performance of the model during training with TensorBoard.

After starting the training process, you can view the results using TensorBoard by running the following command in your terminal: :code:`tensorboard --logdir=/results/` and then navigating to the localhost URL that is displayed.

Once TensorBoard is running, you can view various metrics such as the average total energy with battery (:code:`total_energy_with_battery`), the average CO2 footprint (:code:`CO2_footprint_mean`), and the total load left (:code:`load_left_mean`). You can also monitor the model's progress by viewing graphs of the various metrics during training (such as :code:`episode_reward_mean`).

- The :code:`total_energy_with_battery` metric represents the average total energy consumed by all the agents in the environment, including energy stored in batteries. This metric is a useful indicator of the overall energy efficiency of the system.

- The :code:`CO2_footprint_mean` metric represents the average amount of CO2 emissions produced by all the agents in the environment. This metric is an important environmental metric that can be used to evaluate the carbon footprint of the data center.

- The :code:`load_left_mean` metric represents the total amount of workload left unassigned by the load shifting agent in the environment. This metric is a useful indicator of the efficiency of the load shifting module and can help to identify if the agents are not computing all of the planned workload.

How to add Custom Metrics
=========================

To add new custom metrics to track during training with TensorBoard, you can modify the :code:`CustomCallbacks` class as follows:

**Step 1:** 

Create a new key in the :code:`episode.user_data` dictionary in the :code:`on_episode_start` method.

Example to track the average battery SoC along one episode:

.. code-block:: python

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
      episode.user_data["total_battery_SoC"] = 0

In this example, :code:`total_battery_SoC` is the new key that we initiate.

**Step 2:** 

Store or collect the value of the desired metric in the :code:`on_episode_step` method.

Example to track the average battery SoC along one episode:

.. code-block:: python

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs) -> None:
      episode.user_data["total_battery_SoC"] += total_battery_SoC

**Step 3:** 

Continue to store or modify the value of the desired metric in the :code:`on_episode_end` method, a function that is called at the end of each episode in the training process and store the final metric value using the :code:`episode.custom_metrics` dictionary.

Example to track the average battery SoC along one episode:

.. code-block:: python

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
      episode.custom_metrics["average_battery_SoC"] = episode.user_data["total_battery_SoC"] / episode.user_data["step_count"]

Once you have added the custom metric to the :code:`CustomCallbacks` class, you can view it in TensorBoard by selecting the appropriate metric (i.e. :code:`average_battery_SoC` in the previous example) from the dropdown list of metrics in the TensorBoard dashboard.

Overall, adding custom metrics in this way gives you greater flexibility and control over the training process, allowing you to track specific metrics that are relevant to your use case and goals.

.. figure:: ../images/tensorboard.png
   :scale: 40 %
   :alt: TensorBoard dashboard
   :align: center

   An example of the tensorboard dashboard