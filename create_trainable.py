from typing import Union, Dict, List

from ray.tune.trainable import Trainable
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ResultDict
from ray.rllib.algorithms import Algorithm
from ray.tune.registry import get_trainable_cls

def create_wrapped_trainable(alg: Union[str, Algorithm]) -> Algorithm:
    
    if isinstance(alg, str):
        base_algorithm = get_trainable_cls(alg)
    elif issubclass(alg, Algorithm):
        base_algorithm = alg
    else:
        raise TypeError

    class WrappedTrainable(base_algorithm):
        """A wrapper around any trainable that prints out training progress in an orderly fashion"""

        KEYS_TO_PRINT = [
            'training_iteration', 
            'episodes_total',
            'num_env_steps',
            'episode_reward_mean',
            'policy_reward_mean',
            ]

        @staticmethod
        def flatten(d: Dict, res: List) -> List:
            for key, val in d.items():
                if isinstance(val, dict):
                    flatten(val, res)
                else:
                    if 'min' in key or 'max' in key:
                        continue
                    res.append(f'{key}: {round(val, 2)}')
            return res

        def display_results(self, results: ResultDict) -> None:
            display_cols = []
            for key in self.KEYS_TO_PRINT:
                try:
                    val = results[key]
                except:
                    continue
                
                if isinstance(val, dict):
                    display_cols = self.flatten(val, display_cols)
                else:
                    display_cols.append(f'{key}: {round(val, 2)}')
            
            print(', '.join(display_cols))

        @override(Trainable)
        def train(self) -> ResultDict:
            results = super(WrappedTrainable, self).train()
            self.display_results(results)

            return results

        def log_result(self, result: Dict):
            pass
    
    # Ensure that trainable name is same as the base algorithm
    WrappedTrainable.__name__ = base_algorithm.__name__

    return WrappedTrainable