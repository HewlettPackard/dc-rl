"""Implementation of custom Gym environments."""

import random
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np

from custom_sinergym.utils.common import is_wrapped
from custom_sinergym.utils.logger import CSVLogger


class NormalizeObservation(gym.ObservationWrapper):

    def __init__(self,
                 env: Any,
                 ranges: Dict[str, Sequence[Any]],
                 add_sincos: bool = False):
        """Observations normalized to range [0, 1].

        Args:
            env (Any): Original Sinergym environment.
            ranges (Dict[str, Sequence[Any]]): Observation variables ranges to apply normalization (rely on environment).
        """
        super(NormalizeObservation, self).__init__(env)
        self.unwrapped_observation = None
        self.ranges = ranges
        self.add_sincos = add_sincos

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Applies normalization to observation.

        Args:
            obs (np.ndarray): Original observation.

        Returns:
            np.ndarray: Normalized observation.
        """
        # Save original obs in class attribute
        self.unwrapped_observation = obs.copy()

        # NOTE: If you want to record day, month and hour, you should add that
        # variables as keys
        for i, variable in enumerate(self.env.variables['observation']):
            # normalization (handle DivisionbyZero Error)
            if (self.ranges[variable][1] -
                    self.ranges[variable][0] == 0):
                obs[i] = max(
                    self.ranges[variable][0], min(
                        obs[i], self.ranges[variable][1]))
            else:
                if self.add_sincos:
                    if variable in 'coshour':
                        norm_hour = obs[i] / 23
                        obs[i] = np.cos(norm_hour * np.pi * 2)
                    elif variable in 'sinhour':
                        norm_hour = obs[i] / 23
                        obs[i] = np.sin(norm_hour * np.pi * 2)
                    elif variable in 'sindayOTY':
                        norm_dayOTY = (obs[i]-1) / 365
                        obs[i] = np.sin(norm_dayOTY * np.pi * 2)
                    elif variable in 'cosdayOTY':
                        norm_dayOTY = (obs[i]-1) / 365
                        obs[i] = np.cos(norm_dayOTY * np.pi * 2)
                    else:
                        obs[i] = ((obs[i] - self.ranges[variable][0]) / \
                                (self.ranges[variable][1] - self.ranges[variable][0]) - 0.5 ) * 2
                else:
                    obs[i] = (obs[i] - self.ranges[variable][0]) / \
                            (self.ranges[variable][1] - self.ranges[variable][0])

            # If value is out
            if np.isnan(obs[i]):
                obs[i] = 0
            # elif obs[i] > 1:
            #     obs[i] = 1
            # elif obs[i] < 0:
            #     obs[i] = 0
        # Return obs values in the SAME ORDER than obs argument.
        return np.array(obs)

    def get_unwrapped_obs(self) -> Optional[np.ndarray]:
        """Get last environment observation without normalization.

        Returns:
            Optional[np.ndarray]: Last original observation. If it is the first observation, this value is None.
        """
        return self.unwrapped_observation


class MultiObsWrapperwOnlyTemp(gym.Wrapper):
    def __init__(self, env: Any, n: int=5, flatten: bool=True,
                 add_sincos: bool=False, weather_variables: int=3) -> None:
        """Stack of observations.

        Args:
            env (Any): Original Gym environment.
            n (int, optional): Number of observations to be stacked. Defaults to 5.
            flatten (bool, optional): Whether or not flat the observation vector. Defaults to True.
        """
        super(MultiObsWrapperwOnlyTemp, self).__init__(env)
        self.n = n
        self.ind_flat = flatten
        self.weather_variables = weather_variables
        self.history = []
        self.add_sincos = add_sincos
        shape = env.observation_space.shape
        # The new shape is shape[0] + (shape[0]-3)*(n-1) because I remove the 3 observations of the date/time of the past observation.
        # and -3 again because I remove the observations related with forecast weather data.
        if self.add_sincos:
            date_vars = 4
        else:
            date_vars = 2
        self.date_vars = date_vars
        # Observation: {date/time)}:date_vars, {ext_temp, setpoint, int_temp, HVACe, Totale, ITEe}:6, {weather_forecast_timesteps}:3
        
        # len_obs is only the data that is refered to the datacenter observation
        self.len_obs = shape[0] - date_vars - self.weather_variables # -3 because I remove the 3 observations of the weather forecast data.
        past_obse = 2 # External relative temperature + Internal temperature
        self.past_obse = past_obse
        new_shape = (shape[0] + past_obse * (n-1),) # 2 because I only store the past observation that are refered to the ext_temp and int_temp.
        
        # new_shape = (shape[0] + self.len_obs * (n-1),)

        # new_shape = (shape[0] + 2 * (n-1),)
        # new_shape = (shape[0] * n,) if flatten else ((n,) + shape)
        self.observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=new_shape, dtype=np.float32)

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str,
                                     Any]] = None) -> Tuple[np.ndarray,
                                                            Dict[str,
                                                                 Any]]:
        """Resets the environment.

        Returns:
            np.ndarray: Stacked previous observations.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        # The first observation is appended completely to the stack.
        self.history = []
        self.history.extend(obs)
        
        # The remaining observations are appended to the stack removing the date/time part of the observation.
        # 3 because the first 3 items are month, day, hour.
        
        for _ in range(self.n - 1):
            self.history.append(obs[self.date_vars + 6]) # External relative temperature
            self.history.append(obs[self.date_vars + 2]) # Internal temperature
        return self._get_obs(), info

    def step(self, action: Union[int, np.ndarray]
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Performs the action in the new environment.

        Args:
            action (Union[int, np.ndarray]): Action to be executed in environment.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Tuple with next observation, reward, bool for terminated episode and dict with extra information.
        """

        observation, reward, terminated, truncated, info = self.env.step(
            action)
        # Check this
        past_obs = self.history[-(self.n-1)*self.past_obse:-self.past_obse]
        last_obs = self.history[self.date_vars:self.date_vars+self.len_obs]
        self.history = list(observation) + [last_obs[0], last_obs[2]] + past_obs
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Get observation history.

        Returns:
            np.array: Array of previous observations.
        """
        if self.ind_flat:
            return np.array(self.history).reshape(-1,)
        else:
            return np.array(self.history)

class tinyMultiObsWrapperwWeather(gym.Wrapper):
    def __init__(self, env: Any, n: int = 5, flatten: bool = True, add_sincos: bool = False) -> None:
        """Stack of observations.

        Args:
            env (Any): Original Gym environment.
            n (int, optional): Number of observations to be stacked. Defaults to 5.
            flatten (bool, optional): Whether or not flat the observation vector. Defaults to True.
        """
        super(tinyMultiObsWrapperwWeather, self).__init__(env)
        self.n = n
        self.ind_flat = flatten
        self.history = []
        shape = env.observation_space.shape
        # The new shape is shape[0] + (shape[0]-3)*(n-1) because I remove the 3 observations of the date/time of the past observation.
        # and -3 again because I remove the observations related with forecast weather data.
        new_shape = (shape[0] + (shape[0] - 3 - 3) * (n-1),)
        self.len_obs = shape[0] - 3 - 3
        # new_shape = (shape[0] * n,) if flatten else ((n,) + shape)
        self.observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=new_shape, dtype=np.float32)

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str,
                                     Any]] = None) -> Tuple[np.ndarray,
                                                            Dict[str,
                                                                 Any]]:
        """Resets the environment.

        Returns:
            np.ndarray: Stacked previous observations.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        # The first observation is appended completely to the stack.
        self.history = []
        self.history.extend(obs)
        
        # The remaining observations are appended to the stack removing the date/time part of the observation.
        # 3 because the first 3 items are month, day, hour.
        for _ in range(self.n - 1):
            self.history.extend(obs[3:-3])
        return self._get_obs(), info

    def step(self, action: Union[int, np.ndarray]
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Performs the action in the new environment.

        Args:
            action (Union[int, np.ndarray]): Action to be executed in environment.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Tuple with next observation, reward, bool for terminated episode and dict with extra information.
        """

        observation, reward, terminated, truncated, info = self.env.step(
            action)
        past_obs = self.history[-(self.n-1)*self.len_obs:-self.len_obs]
        last_obs = self.history[3:self.len_obs+3]
        self.history = list(observation) + last_obs + past_obs
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Get observation history.

        Returns:
            np.array: Array of previous observations.
        """
        if self.ind_flat:
            return np.array(self.history).reshape(-1,)
        else:
            return np.array(self.history)

class tinyMultiObsWrapper(gym.Wrapper):
    def __init__(self, env: Any, n: int = 5, flatten: bool = True, add_sincos: bool = False) -> None:
        """Stack of observations.

        Args:
            env (Any): Original Gym environment.
            n (int, optional): Number of observations to be stacked. Defaults to 5.
            flatten (bool, optional): Whether or not flat the observation vector. Defaults to True.
        """
        super(tinyMultiObsWrapper, self).__init__(env)
        self.n = n
        self.ind_flat = flatten
        self.history = []
        shape = env.observation_space.shape
        self.add_sincos = add_sincos
        if add_sincos:
            self.date_vars = 4
        else:
            self.date_vars = 2
        # The new shape is shape[0] + (shape[0]-3)*(n-1) because I remove the 3 observations of the date/time of the past observation.
        new_shape = (shape[0] + (shape[0] - self.date_vars) * (n-1),)
        # new_shape = (shape[0] + (3) * (n-1),) # 3 is the observation vector that refers to the external/setpoint/internal temperatures.
        self.len_obs = shape[0] - self.date_vars # DC Observations
        # new_shape = (shape[0] * n,) if flatten else ((n,) + shape)
        self.observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=new_shape, dtype=np.float32)

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str,
                                     Any]] = None) -> Tuple[np.ndarray,
                                                            Dict[str,
                                                                 Any]]:
        """Resets the environment.

        Returns:
            np.ndarray: Stacked previous observations.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        # The first observation is appended completely to the stack.
        self.history = []
        self.history.extend(obs)
        
        # The remaining observations are appended to the stack removing the date/time part of the observation.
        # 3 because the first 3 items are month, day, hour.
        for _ in range(self.n - 1):
            # self.history.extend(obs[3:])
            self.history.extend(obs[self.date_vars:self.date_vars+self.len_obs]) # 3 because the first 3 items are day_of_the_year, sinhour, coshour.
        return self._get_obs(), info

    def step(self, action: Union[int, np.ndarray]
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Performs the action in the new environment.

        Args:
            action (Union[int, np.ndarray]): Action to be executed in environment.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Tuple with next observation, reward, bool for terminated episode and dict with extra information.
        """

        observation, reward, terminated, truncated, info = self.env.step(action)
        
        past_obs = self.history[-(self.n-1)*self.len_obs:-self.len_obs]
        last_obs = self.history[self.date_vars:self.date_vars+self.len_obs]
        self.history = list(observation) + last_obs + past_obs
        
        #! self.history = list(observation) + self.history[3:]
        #! self.history = self.history[:-(len(observation)-3)]
        # self.history.append(observation)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Get observation history.

        Returns:
            np.array: Array of previous observations.
        """
        if self.ind_flat:
            return np.array(self.history).reshape(-1,)
        else:
            return np.array(self.history)
        
        
class MultiObsWrapper(gym.Wrapper):
    def __init__(self, env: Any, n: int = 5, flatten: bool = True) -> None:
        """Stack of observations.

        Args:
            env (Any): Original Gym environment.
            n (int, optional): Number of observations to be stacked. Defaults to 5.
            flatten (bool, optional): Whether or not flat the observation vector. Defaults to True.
        """
        super(MultiObsWrapper, self).__init__(env)
        self.n = n
        self.ind_flat = flatten
        self.history = deque([], maxlen=n)
        shape = env.observation_space.shape
        new_shape = (shape[0] * n,) if flatten else ((n,) + shape)
        self.observation_space = gym.spaces.Box(
            low=-5e6, high=5e6, shape=new_shape, dtype=np.float32)

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str,
                                     Any]] = None) -> Tuple[np.ndarray,
                                                            Dict[str,
                                                                 Any]]:
        """Resets the environment.

        Returns:
            np.ndarray: Stacked previous observations.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.n):
            self.history.append(obs)
        return self._get_obs(), info

    def step(self, action: Union[int, np.ndarray]
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Performs the action in the new environment.

        Args:
            action (Union[int, np.ndarray]): Action to be executed in environment.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Tuple with next observation, reward, bool for terminated episode and dict with extra information.
        """

        observation, reward, terminated, truncated, info = self.env.step(
            action)
        self.history.append(observation)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Get observation history.

        Returns:
            np.array: Array of previous observations.
        """
        if self.ind_flat:
            return np.array(self.history).reshape(-1,)
        else:
            return np.array(self.history)


class LoggerWrapper(gym.Wrapper):

    def __init__(
        self,
        env: Any,
        logger_class: Callable = CSVLogger,
        monitor_header: Optional[List[str]] = None,
        progress_header: Optional[List[str]] = None,
        flag: bool = True,
    ):
        """CSVLogger to log interactions with environment.

        Args:
            env (Any): Original Gym environment.
            logger_class (CSVLogger): CSV Logger class to use to log all information.
            monitor_header: Header for monitor.csv in each episode. Default is None (default format).
            progress_header: Header for progress.csv in whole simulation. Default is None (default format).
            flag (bool, optional): State of logger (activate or deactivate). Defaults to True.
        """
        gym.Wrapper.__init__(self, env)
        # Headers for csv logger
        monitor_header_list = monitor_header if monitor_header is not None else ['timestep'] + env.variables['observation'] + env.variables['action'] + [
            'time (seconds)', 'reward', 'power_penalty', 'comfort_penalty', 'abs_comfort', 'terminated']
        self.monitor_header = ''
        for element_header in monitor_header_list:
            self.monitor_header += element_header + ','
        self.monitor_header = self.monitor_header[:-1]
        progress_header_list = progress_header if progress_header is not None else [
            'episode_num',
            'cumulative_reward',
            'mean_reward',
            'cumulative_power_consumption',
            'mean_power_consumption',
            'cumulative_comfort_penalty',
            'mean_comfort_penalty',
            'cumulative_power_penalty',
            'mean_power_penalty',
            'comfort_violation (%)',
            'mean_comfort_violation',
            'std_comfort_violation',
            'cumulative_comfort_violation',
            'length(timesteps)',
            'time_elapsed(seconds)']
        self.progress_header = ''
        for element_header in progress_header_list:
            self.progress_header += element_header + ','
        self.progress_header = self.progress_header[:-1]

        # Create simulation logger, by default is active (flag=True)
        self.logger = logger_class(
            monitor_header=self.monitor_header,
            progress_header=self.progress_header,
            log_progress_file=env.simulator._env_working_dir_parent +
            '/progress.csv',
            flag=flag)

    def step(self, action: Union[int, np.ndarray]
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Sends action to the environment. Logging new information in monitor.csv.

        Args:
            action (Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]): Action selected by the agent.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Observation for next timestep, reward obtained, Whether the episode has ended or not, Wheather episode has been truncated or not, and a dictionary with extra information
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        # We added some extra values (month,day,hour) manually in env, so we
        # need to delete them.
        if is_wrapped(self, NormalizeObservation):
            # Record action and new observation in simulator's csv
            self.logger.log_step_normalize(
                obs=obs,
                action=info['action'],
                reward=reward,
                terminated=terminated,
                info=info)
            # Record original observation too
            self.logger.log_step(
                obs=self.env.get_unwrapped_obs(),
                action=info['action'],
                reward=reward,
                terminated=terminated,
                info=info)
        else:
            # Only record observation without normalization
            self.logger.log_step(
                obs=obs,
                action=info['action'],
                reward=reward,
                terminated=terminated,
                info=info)

        return obs, reward, terminated, truncated, info

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str,
                                     Any]] = None) -> Tuple[np.ndarray,
                                                            Dict[str,
                                                                 Any]]:
        """Reset the environment. Recording episode summary in logger

        Args:
            seed (Optional[int]): The seed that is used to initialize the environment's episode (np_random). if value is None, a seed will be chosen from some source of entropy. Defaults to None.
            options (Optional[Dict[str, Any]]):Additional information to specify how the environment is reset. Defaults to None.

        Returns:
            Tuple[np.ndarray,Dict[str,Any]]: Current observation and info context with additional information.
        """
        # It isn't first episode simulation, so we can logger last episode
        if self.env.simulator._episode_existed:
            self.env.simulator.logger_main.debug(
                'End of episode, recording summary (progress.csv) if logger is active')
            self.logger.log_episode(episode=self.env.simulator._epi_num)

        # Then, reset environment
        obs, info = self.env.reset(seed=seed, options=options)

        # Create monitor.csv for information of this episode
        self.env.simulator.logger_main.debug(
            'Creating monitor.csv for current episode (episode ' + str(
                self.env.simulator._epi_num) + ') if logger is active')
        self.logger.set_log_file(
            self.env.simulator._eplus_working_dir + '/monitor.csv')

        if is_wrapped(self, NormalizeObservation):
            # Store initial state of simulation (normalized)
            self.logger.log_step_normalize(obs=obs,
                                           action=[None for _ in range(
                                               len(self.env.variables['action']))],
                                           reward=None,
                                           terminated=False,
                                           info=None)
            # And store original obs
            self.logger.log_step(obs=self.env.get_unwrapped_obs(),
                                 action=[None for _ in range(
                                     len(self.env.variables['action']))],
                                 reward=None,
                                 terminated=False,
                                 info=None)
        else:
            # Only store original step
            self.logger.log_step(obs=obs,
                                 action=[None for _ in range(
                                     len(self.env.variables['action']))],
                                 reward=None,
                                 terminated=False,
                                 info=None)

        return obs, info

    def close(self) -> None:
        """Recording last episode summary and close env.
        """
        # Record last episode summary before end simulation
        self.env.simulator.logger_main.debug(
            'End of episode, recording summary (progress.csv) if logger is active')
        self.logger.log_episode(episode=self.env.simulator._epi_num)

        # Then, close env
        self.env.close()

    def activate_logger(self) -> None:
        """Activate logger if its flag False.
        """
        self.logger.activate_flag()

    def deactivate_logger(self) -> None:
        """Deactivate logger if its flag True.
        """
        self.logger.deactivate_flag()

# ---------------------- Specific environment wrappers ---------------------#


class OfficeGridStorageSmoothingActionConstraintsWrapper(
        gym.ActionWrapper):  # pragma: no cover
    def __init__(self, env):
        assert env.idf_path.split(
            '/')[-1] == 'OfficeGridStorageSmoothing.idf', 'OfficeGridStorageSmoothingActionConstraintsWrapper: This wrapper is not valid for this environment.'
        super().__init__(env)

    def action(self, act: np.ndarray) -> np.ndarray:
        """Due to Charge rate and Discharge rate can't be more than 0.0 simultaneously (in OfficeGridStorageSmoothing.idf),
           this wrapper clips one of the to 0.0 when both have a value upper than 0.0 (randomly).

        Args:
            act (np.ndarray): Action to be clipped

        Returns:
            np.ndarray: Action Clipped
        """
        if self.flag_discrete:
            null_value = 0.0
        else:
            # -1.0 is 0.0 when action space transformation to simulator action space.
            null_value = -1.0
        if act[2] > null_value and act[3] > null_value:
            random_rate_index = random.randint(2, 3)
            act[random_rate_index] = null_value
        return act
