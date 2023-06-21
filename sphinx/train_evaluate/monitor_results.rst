How to Monitor Training Results Using TensorBoard
=================================================

TensorBoard is a powerful tool for monitoring and visualizing the training process of reinforcement learning algorithms. DCRL-Green provides a custom callback implementation called :code:`CustomCallbacks` which can be used to track the performance of the model during training with TensorBoard.

After starting the training process, you can view the results using TensorBoard by running the following command in your terminal: :code:`tensorboard --logdir=/results/` and then navigating to the localhost URL that is displayed.

Once TensorBoard is running, you can view various metrics such as the average total energy with battery (:code:`total_energy_with_battery`), the average CO2 footprint (:code:`CO2_footprint_mean`), and the total load left (:code:`load_left_mean`). You can also monitor the model's progress by viewing graphs of the various metrics during training (such as :code:`episode_reward_mean`).

- The :code:`total_energy_with_battery` metric represents the average total energy consumed by all the agents in the environment, including energy stored in batteries. This metric is a useful indicator of the overall energy efficiency of the system.

- The :code:`CO2_footprint_mean` metric represents the average amount of CO2 emissions produced by all the agents in the environment. This metric is an important environmental metric that can be used to evaluate the carbon footprint of the data center.

- The :code:`load_left_mean` metric represents the total amount of workload left unassigned by the load shifting agent in the environment. This metric is a useful indicator of the efficiency of the load shifting module and can help to identify if the agents are not computing all of the planned workload.

Adding Custom Metrics
=====================

To add new custom metrics to track during training with TensorBoard, you can modify the :code:`CustomCallbacks` class by adding new attributes to the :code:`episode.custom_metrics` dictionary in the :code:`on_episode_end` method.
Previously, the user should create a new key in the :code:`episode.user_data` dictionary in the :code:`on_episode_start` method.

Then, the user should store the value of the desired metric in the :code:`on_episode_step` method.

:code:`episode.user_data["new_metric_sum"] += new_metric`

Finally, the user should store the value of the desired metric :code:`episode.custom_metrics` using the desired key to be used in the tensorboard in the :code:`on_episode_end` method.

For example, if you wanted to track the average battery SoC along one episode, you could add the following line to the :code:`on_episode_end` method:

:code:`episode.custom_metrics["average_battery_SoC"] = episode.user_data["total_battery_SoC"] / episode.user_data["step_count"]`

Once you have added the custom metric to the :code:`CustomCallbacks` class, you can view it in TensorBoard by selecting the appropriate metric (i.e. :code:`average_battery_SoC` in the previous example) from the dropdown list of metrics in the TensorBoard dashboard.

Overall, adding custom metrics in this way gives you greater flexibility and control over the training process, allowing you to track specific metrics that are relevant to your use case and goals.
