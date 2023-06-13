import logging
from typing import List, Optional, Type

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn.dqn import DQN
from ray.rllib.algorithms.maddpg.maddpg_tf_policy import MADDPGTFPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.maddpg import MADDPGConfig, MADDPG


class MADDPGConfigStable(MADDPGConfig):
    """A light wrapper over the original MADDPG config which fixes the stability issue"""

    def __init__(self, algo_class=None):
        """Initializes the MADDPG config."""
        super().__init__(algo_class=algo_class or MADDPG)

    @override(AlgorithmConfig)
    def validate(self) -> None:
        """
        Adds the `before_learn_on_batch` hook to the config.

        This hook is called explicitly prior to `train_one_step()` in the
        `training_step()` methods of DQN and APEX.
        """
        # Call super's validation method.
        super().validate()

        def f(batch, workers, config):

            # This could potentially return policies in a scrambled
            # order, so make a temp copy.
            _policies = dict(
                        workers.local_worker().foreach_policy_to_train(lambda p, i: (i, p))
                    )
            
            # Use the original config to iterate over the policies. This makes sure
            # the policies are in the right order
            policies = {}
            for p in config.multiagent['policies'].keys():
                policies[p] = _policies[p]

            return before_learn_on_batch(batch, policies, config["train_batch_size"])

        self.before_learn_on_batch = f


def before_learn_on_batch(multi_agent_batch, policies, train_batch_size):
    samples = {}

    # Modify keys.
    for pid, p in policies.items():
        i = p.config["agent_id"]
        keys = multi_agent_batch.policy_batches[pid].keys()
        keys = ["_".join([k, str(i)]) for k in keys]
        samples.update(dict(zip(keys, multi_agent_batch.policy_batches[pid].values())))

    # Make ops and feed_dict to get "new_obs" from target action sampler.
    new_obs_ph_n = [p.new_obs_ph for p in policies.values()]
    new_obs_n = list()
    for k, v in samples.items():
        if "new_obs" in k:
            new_obs_n.append(v)

    for i, p in enumerate(policies.values()):
        feed_dict = {new_obs_ph_n[i]: new_obs_n[i]}
        new_act = p.get_session().run(p.target_act_sampler, feed_dict)
        samples.update({"new_actions_%d" % i: new_act})

    # Share samples among agents.
    policy_batches = {pid: SampleBatch(samples) for pid in policies.keys()}
    return MultiAgentBatch(policy_batches, train_batch_size)


class MADDPGStable(DQN):
    @classmethod
    @override(DQN)
    def get_default_config(cls) -> AlgorithmConfig:
        return MADDPGConfigStable()

    @classmethod
    @override(DQN)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        return MADDPGTFPolicy
