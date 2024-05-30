from typing import Union, Dict, List

from ray.tune.trainable import Trainable
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ResultDict
from ray.rllib.algorithms import Algorithm
from ray.tune.registry import get_trainable_cls

def create_wrapped_trainable(alg: Union[str, Algorithm]) -> Algorithm:
    """
    Create a wrapped trainable algorithm that prints out training progress in an orderly fashion.

    Args:
        alg (Union[str, Algorithm]): The trainable instance or string id of a registered algorithm.

    Returns:
        Algorithm: An instance of the wrapped trainable algorithm.

    Raises:
        TypeError: If `alg` is not an instance of `str` or `Algorithm`.

    """

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
            'episode_reward_mean',
            'policy_reward_mean'
            ]

        def flatten(self, d: Dict, res: Dict = {}, flattened_key: str = '') -> Dict:
            """
            Flatten a nested dictionary into a dictionary with a single level of keys.

            Args:
                d (Dict): The nested dictionary to flatten.
                res (Dict): A dictionary to contain the flattened dictionary.
                flattened_key (str): A prefix string to be added to all resulting flattened keys.

            Returns:
                Dict: A dictionary with flattened keys.

            """
            for key, val in d.items():
                if isinstance(val, dict):
                    self.flatten(val, res, flattened_key + key + '/' )
                else:
                    # We are only interested in the mean
                    if 'min' in key or 'max' in key:
                        continue
                    res[flattened_key + key] = val
            return res

        def display_results(self, results: ResultDict) -> None:
            """
            Display the specified results in a human-readable format.

            Args:
                self (WrappedTrainable): The current instance of the wrapped trainable algorithm.
                results (ResultDict): The results to display.

            Returns:
                None.

            """
            display = []
            results = self.flatten(results)
            
            for key_to_print in self.KEYS_TO_PRINT:
                matching_keys = [key for key in results.keys() if key.startswith(key_to_print)]

                for key in matching_keys:
                    val = results[key]
                    
                    k = key.split('/')[-1]
                    display.append(f'{k}: {round(val, 2)}')
            
            print(', '.join(display))

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