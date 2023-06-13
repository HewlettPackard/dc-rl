"""
Gym environment for simulation with EnergyPlus.
"""

import os
import glob
import datetime
from sqlite3 import DatabaseError
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd

from custom_sinergym.simulators import EnergyPlus
from custom_sinergym.utils.common import export_actuators_to_excel
from custom_sinergym.utils.constants import PKG_DATA_PATH
from custom_sinergym.utils.rewards import ExpReward, LinearReward


class EplusEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    # ---------------------------------------------------------------------------- #
    #                            ENVIRONMENT CONSTRUCTOR                           #
    # ---------------------------------------------------------------------------- #
    def __init__(
        self,
        idf_file: str,
        weather_file: str,
        observation_space: gym.spaces.Box = gym.spaces.Box(
            low=-5e6, high=5e6, shape=(4,), dtype=np.float32),
        observation_variables: List[str] = [],
        action_space: Union[gym.spaces.Box, gym.spaces.Discrete] = gym.spaces.Box(
            low=0, high=0, shape=(0,), dtype=np.float32),
        action_variables: List[str] = [],
        action_mapping: Dict[int, Tuple[float, ...]] = {},
        weather_variability: Optional[Tuple[float]] = None,
        reward: Any = LinearReward,
        reward_kwargs: Optional[Dict[str, Any]] = {},
        act_repeat: int = 1,
        max_ep_data_store_num: int = 10,
        action_definition: Optional[Dict[str, Any]] = None,
        env_name: str = 'eplus-env-v1',
        config_params: Optional[Dict[str, Any]] = None,
        delta_actions: bool = False,
        temp_range: List[float] = [10, 30],
        add_forecast_weather: bool = False,
        seed: Optional[str] = "",
        add_sincos: bool = False,
        relative_obs: bool = True,
        external_cpu_scheme: str = '',
        add_cpu_usage: bool = False
    ):
        """Environment with EnergyPlus simulator.

        Args:
            idf_file (str): Name of the IDF file with the building definition.
            weather_file (str): Name of the EPW file for weather conditions.
            observation_space (gym.spaces.Box, optional): Gym Observation Space definition. Defaults to an empty observation_space (no control).
            observation_variables (List[str], optional): List with variables names in IDF. Defaults to an empty observation variables (no control).
            action_space (Union[gym.spaces.Box, gym.spaces.Discrete], optional): Gym Action Space definition. Defaults to an empty action_space (no control).
            action_variables (List[str],optional): Action variables to be controlled in IDF, if that actions names have not been configured manually in IDF, you should configure or use extra_config. Default to empty List.
            action_mapping (Dict[int, Tuple[float, ...]], optional): Action mapping list for discrete actions spaces only. Defaults to empty list.
            weather_variability (Optional[Tuple[float]], optional): Tuple with sigma, mu and tao of the Ornstein-Uhlenbeck process to be applied to weather data. Defaults to None.
            reward (Any, optional): Reward function instance used for agent feedback. Defaults to LinearReward.
            reward_kwargs (Optional[Dict[str, Any]], optional): Parameters to be passed to the reward function. Defaults to empty dict.
            act_repeat (int, optional): Number of timesteps that an action is repeated in the simulator, regardless of the actions it receives during that repetition interval.
            max_ep_data_store_num (int, optional): Number of last sub-folders (one for each episode) generated during execution on the simulation.
            action_definition (Optional[Dict[str, Any]): Dict with building components to being controlled by Sinergym automatically if it is supported. Default value to None.
            env_name (str, optional): Env name used for working directory generation. Defaults to eplus-env-v1.
            config_params (Optional[Dict[str, Any]], optional): Dictionary with all extra configuration for simulator. Defaults to None.
        """

        # ---------------------------------------------------------------------------- #
        #                          Energyplus, BCVTB and paths                         #
        # ---------------------------------------------------------------------------- #
        eplus_path = os.environ['EPLUS_PATH']
        bcvtb_path = os.environ['BCVTB_PATH']
        self.pkg_data_path = PKG_DATA_PATH

        self.idf_path = os.path.join(self.pkg_data_path, 'buildings', idf_file)
        self.weather_path = os.path.join(
            self.pkg_data_path, 'weather', weather_file)

        # ---------------------------------------------------------------------------- #
        #                             Variables definition                             #
        # ---------------------------------------------------------------------------- #
        self.variables = {}
        self.variables['observation'] = observation_variables
        self.variables['action'] = action_variables

        # ---------------------------------------------------------------------------- #
        #                                   Simulator                                  #
        # ---------------------------------------------------------------------------- #
        self.seed = seed
        self.relative_obs = relative_obs
        self.env_name = env_name
        self.external_cpu_scheme = external_cpu_scheme
        # self.external_cpu_scheme = pd.read_csv(external_cpu_scheme, usecols=['cpu_load', 'day', 'hour', 'minute'])
        # self.external_cpu_scheme = self.external_cpu_scheme.pivot_table(values='cpu_load', index=['day', 'hour', 'minute']).to_dict()['cpu_load']
        
        self.simulator = EnergyPlus(
            env_name=env_name,
            eplus_path=eplus_path,
            bcvtb_path=bcvtb_path,
            idf_path=self.idf_path,
            weather_path=self.weather_path,
            variables=self.variables,
            act_repeat=act_repeat,
            max_ep_data_store_num=max_ep_data_store_num,
            action_definition=action_definition,
            config_params=config_params,
            seed=seed,
            add_forecast_weather=add_forecast_weather,
            add_sincos=add_sincos,
            relative_obs=relative_obs,
            external_cpu_scheme=external_cpu_scheme,
            add_cpu_usage=add_cpu_usage
        )
        
        self.timesteps_per_hour = self.simulator._eplus_run_stepsize / 60
        # ---------------------------------------------------------------------------- #
        #                       Detection of controllable planners                     #
        # ---------------------------------------------------------------------------- #
        self.schedulers = self.get_schedulers()

        # ---------------------------------------------------------------------------- #
        #        Adding simulation date to observation (not needed in simulator)       #
        # ---------------------------------------------------------------------------- #
        self.add_forecast_weather = add_forecast_weather
        self.add_sincos = add_sincos
        self.add_cpu_usage = add_cpu_usage
        
        if self.add_forecast_weather:
            # self.variables['observation'] = ['month', 'day', 'hour'] + self.variables['observation'] + ['extTemp1', 'extTemp2', 'extTemp3']
            if self.add_sincos:
                # self.variables['observation'] = ['day_of_the_month', 'sinhour', 'coshour'] + self.variables['observation'] + ['extTemp1', 'extTemp2', 'extTemp3']
                # self.variables['observation'] = ['sindayOTY', 'cosdayOTY', 'sinhour', 'coshour'] + self.variables['observation'] + ['extTemp1', 'extTemp2', 'extTemp3']
                self.variables['observation'] = ['sindayOTY', 'cosdayOTY', 'sinhour', 'coshour'] + self.variables['observation'] + \
                    ['extTemp1', 'extTemp2', 'extTemp3'] + ['extDP1', 'extDP2', 'extDP3'] + ['extHum1', 'extHum2', 'extHum3']  + ['extSolar1', 'extSolar2', 'extSolar3']

            else:
                self.variables['observation'] = ['day_of_the_month', 'hour'] + self.variables['observation'] + ['extTemp1', 'extTemp2', 'extTemp3']
            
            if 'NY' in self.weather_path:
                self.myweather_path = '/sinergym/projectDC/USA_NY_New.York-J.F.Kennedy.Intl.AP_v2.csv'
            elif 'AZ' in self.weather_path:
                self.myweather_path = '/sinergym/projectDC/USA_AZ_Davis-Monthan.AFB.csv'
            elif 'WA' in self.weather_path:
                self.myweather_path = '/sinergym/projectDC/USA_WA_Port.Angeles-William.csv'

            self.weather_i = 0
            # # self.weather_df = pd.read_csv(self.weather_path, skiprows=8, header=None, delimiter=',', names=['year', 'month', 'day', 'hour', 'minute', 'data_source_and_uncertainty_flags', 'dry_bulb_temperature', 'dew_point_temperature', 'relative_humidity', 'atmospheric_station_pressure', 'extraterrestrial_horizontal_radiation', 'extraterrestrial_direct_normal_radiation', 'horizontal_infrared_radiation_intensity', 'global_horizontal_radiation', 'direct_normal_radiation', 'diffuse_horizontal_radiation', 'global_horizontal_illuminance', 'direct_normal_illuminance', 'diffuse_horizontal_illuminance', 'zenith_luminance', 'wind_direction', 'wind_speed', 'total_sky_cover', 'opaque_sky_cover', 'visibility', 'ceiling_height', 'present_weather_observation', 'present_weather_codes', 'precipitable_water', 'aerosol_optical_depth', 'snow_depth', 'days_since_last_snowfall', 'albedo', 'liquid_precipitation_depth', 'liquid_precipitation_quantity'])
            # # ToDo: Obtain the correct path to the weather file
            # # If seed = '': read all files that Starts with Eplus-env-{env_name}- and obtain the last element with the highest resXXX number
            # # Else: read the file located in the path specified by seed like: Eplus-env-{env_name}/{seed}/-res1/Eplus-env-sub_runXXX, being XXX the highest runXXX number
            # myweather_path = path = r'/sinergym/Eplus-env-datacenter-mixed-continuous-v1-res119/Eplus-env-sub_run1/USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3_Random_4.0_1.0_0.1.epw'
            # # define the column names
            # columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Data Source and Uncertainty Flags',
            #         'Dry Bulb Temperature (°C)', 'Dew Point Temperature (°C)', 'Relative Humidity (%)',
            #         'Atmospheric Station Pressure (Pa)', 'Extraterrestrial Horizontal Radiation (Wh/m²)',
            #         'Extraterrestrial Direct Normal Radiation (Wh/m²)', 'Horizontal Infrared Radiation Intensity (Wh/m²)',
            #         'Global Horizontal Radiation (Wh/m²)', 'Direct Normal Radiation (Wh/m²)',
            #         'Diffuse Horizontal Radiation (Wh/m²)', 'Global Horizontal Illuminance (lux)',
            #         'Direct Normal Illuminance (lux)', 'Diffuse Horizontal Illuminance (lux)', 'Zenith Luminance (cd/m²)',
            #         'Wind Direction (°)', 'Wind Speed (m/s)', 'Total Sky Cover (tenths)', 'Opaque Sky Cover (tenths)',
            #         'Visibility (km)', 'Ceiling Height (m)', 'Present Weather Observation', 'Present Weather Codes',
            #         'Precipitable Water (mm)', 'Aerosol Optical Depth', 'Snow Depth (cm)', 'Days Since Last Snowfall',
            #         'Albedo', 'Liquid Precipitation Depth (mm)', 'Liquid Precipitation Quantity (hr)']
            # columns_to_select = ['Month', 'Day', 'Hour', 'Dry Bulb Temperature (°C)']
            # read the weather file into a pandas DataFrame with the defined column names
            # self.myweather_data = pd.read_csv(myweather_path, delimiter=',', skiprows=8, names=columns, usecols=columns_to_select)
            # self.weather_dict = self.myweather_data.pivot_table(values='Dry Bulb Temperature (°C)', index=['Month', 'Day', 'Hour']).to_dict()['Dry Bulb Temperature (°C)']

            # self.myweather_data = pd.read_csv(self.myweather_path)
            # self.weather_dict = self.myweather_data.pivot_table(values='Site Outdoor Air Drybulb Temperature(Environment)', index=['month', 'day', 'hour', 'minute']).to_dict()['Site Outdoor Air Drybulb Temperature(Environment)']
            # self.weather_dict = {}
            # for month, days in [(6, 31), (7, 32), (8, 32)]:
            #     for day in range(1, days):
            #         for hour in range(24):
            #             for minute in range(0, 60, 15):
            #                 weather_data = self.myweather_data[(self.myweather_data['month'] == month) & (self.myweather_data['day'] == day) & (self.myweather_data['hour'] == hour) & (self.myweather_data['minute'] == minute)]
            #                 self.weather_dict[(month, day, hour, minute)] = weather_data['Site Outdoor Air Drybulb Temperature(Environment)'].values[0]
            # myweather_path = self.get_weather_file_path(env_name=env_name, seed=self.seed)
            # self.weather_dict = self.read_weather_file(myweather_path)
        else:
            # self.variables['observation'] = ['month', 'day', 'hour'] + self.variables['observation']
            if self.add_sincos:
                self.variables['observation'] = ['sindayOTY', 'cosdayOTY', 'sinhour', 'coshour'] + self.variables['observation']
            else:
                self.variables['observation'] = ['day_of_the_year', 'hour'] + self.variables['observation']
            
            if self.add_cpu_usage:
                self.variables['observation'] = self.variables['observation'] + ['cpuUsage', 'carbonIntensity', 'batterySoC']

        # ---------------------------------------------------------------------------- #
        #                          reset default options                               #
        # ---------------------------------------------------------------------------- #
        self.default_options = {}
        # Weather Variability
        if weather_variability:
            self.default_options['weather_variability'] = weather_variability
        # ... more reset option implementations here

        # ---------------------------------------------------------------------------- #
        #                               Observation Space                              #
        # ---------------------------------------------------------------------------- #
        self._observation_space = observation_space

        # ---------------------------------------------------------------------------- #
        #                                 Action Space                                 #
        # ---------------------------------------------------------------------------- #
        # Action space type
        self.flag_discrete = (
            isinstance(
                action_space,
                gym.spaces.Discrete))

        # Discrete
        if self.flag_discrete:
            self.action_mapping = action_mapping
            self._action_space = action_space
            self.temp_range = temp_range

        # Continuous
        else:
            # Defining action values setpoints (one per value)
            self.setpoints_space = action_space
            self.temp_range = [self.setpoints_space.low, self.setpoints_space.high]
            self._action_space = gym.spaces.Box(
                # continuous_action_def[2] --> shape
                low=np.array(
                    np.repeat(-1, action_space.shape[0]), dtype=np.float32),
                high=np.array(
                    np.repeat(1, action_space.shape[0]), dtype=np.float32),
                dtype=action_space.dtype
            )

        self.delta_actions = delta_actions
        self.last_setpoint = 0
        # ---------------------------------------------------------------------------- #
        #                                    Reward                                    #
        # ---------------------------------------------------------------------------- #
        self.reward_fn = reward(self, **reward_kwargs)
        self.obs_dict = None

        # ---------------------------------------------------------------------------- #
        #                        Environment definition checker                        #
        # ---------------------------------------------------------------------------- #
        self.temp_state = None
        self._check_eplus_env()

    def read_weather_file(self, weather_file_path):
        print('DEBUG: Reading weather file: {}'.format(weather_file_path))
        # define the column names
        columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Data Source and Uncertainty Flags',
                'Dry Bulb Temperature (°C)', 'Dew Point Temperature (°C)', 'Relative Humidity (%)',
                'Atmospheric Station Pressure (Pa)', 'Extraterrestrial Horizontal Radiation (Wh/m²)',
                'Extraterrestrial Direct Normal Radiation (Wh/m²)', 'Horizontal Infrared Radiation Intensity (Wh/m²)',
                'Global Horizontal Radiation (Wh/m²)', 'Direct Normal Radiation (Wh/m²)',
                'Diffuse Horizontal Radiation (Wh/m²)', 'Global Horizontal Illuminance (lux)',
                'Direct Normal Illuminance (lux)', 'Diffuse Horizontal Illuminance (lux)', 'Zenith Luminance (cd/m²)',
                'Wind Direction (°)', 'Wind Speed (m/s)', 'Total Sky Cover (tenths)', 'Opaque Sky Cover (tenths)',
                'Visibility (km)', 'Ceiling Height (m)', 'Present Weather Observation', 'Present Weather Codes',
                'Precipitable Water (mm)', 'Aerosol Optical Depth', 'Snow Depth (cm)', 'Days Since Last Snowfall',
                'Albedo', 'Liquid Precipitation Depth (mm)', 'Liquid Precipitation Quantity (hr)']
        columns_to_select = ['Month', 'Day', 'Hour', 'Dry Bulb Temperature (°C)', 'Dew Point Temperature (°C)', 'Relative Humidity (%)', 'Global Horizontal Radiation (Wh/m²)']
        # read the weather file into a pandas DataFrame with the defined column names
        self.myweather_data = pd.read_csv(weather_file_path, delimiter=',', skiprows=8, names=columns, usecols=columns_to_select)
        self.weather_dict_DBT = self.myweather_data.pivot_table(values='Dry Bulb Temperature (°C)', index=['Month', 'Day', 'Hour']).to_dict()['Dry Bulb Temperature (°C)']
        self.weather_dict_DPT = self.myweather_data.pivot_table(values='Dew Point Temperature (°C)', index=['Month', 'Day', 'Hour']).to_dict()['Dew Point Temperature (°C)']
        self.weather_dict_RH = self.myweather_data.pivot_table(values='Relative Humidity (%)', index=['Month', 'Day', 'Hour']).to_dict()['Relative Humidity (%)']
        self.weather_dict_GHR = self.myweather_data.pivot_table(values='Global Horizontal Radiation (Wh/m²)', index=['Month', 'Day', 'Hour']).to_dict()['Global Horizontal Radiation (Wh/m²)']
        
        # return self.weather_dict_DBT, self.weather_dict_DPT, self.weather_dict_RH, self.weather_dict_GHR

    '''
    'Site Outdoor Air Drybulb Temperature(Environment)',
    'Zone Thermostat Cooling Setpoint Temperature(West Zone)',
    'Zone Air Temperature(West Zone)',
    '''
    def forecast_weather(self, timesteps, obs):        
        # month, day, hour = obs[:3]
        # day_of_the_month, hour = obs[:2]
        
        # if day_of_the_month <= 30:
        #     month = 6
        #     day = day_of_the_month
        # elif day_of_the_month <= 61:
        #     month = 7
        #     day = day_of_the_month - 30
        # else:
        #     month = 8
        #     day = day_of_the_month - 61
        # create datetime object for January 1 of the same year
        dt = datetime.datetime(2020, 1, 1)
        day_of_year, hour = obs[0], obs[2]
        # add the number of days in the day_of_year value minus 1
        dt += datetime.timedelta(days=day_of_year-1)
        month = dt.month
        day = dt.day
        # Considering the current date and time, I need to obtain the weather data for the next timesteps defined in timestep variable.
        # For example, if the current date is 2020-01-01 00:00 and self.weather_i is 0, the next timesteps will be 2020-01-01 00:15, 2020-01-01 00:30 and 2020-01-01 00:45, if timesteps is [1,2,3]
        # If timestep is [1,2,4,8], I will add 15 minutes to the current date and time multiplied by the value of each timestep located in the timestep variable.
        # For example, if the current date is 2020-01-01 00:30, and timestep is [1,2,4,8], the next timesteps will be 2020-01-01 00:45, 2020-01-01 01:00, 2020-01-01 01:30 and 2020-01-01 02:30
        current_time = datetime.datetime(year=1989, month=month, day=day, hour=hour, minute=int(self.weather_i*self.timesteps_per_hour))
        forecast_temp_aux = []
        forecast_DP_aux = []
        forecast_RH_aux = []
        forecast_GHR_aux = []
        for timestep in timesteps:
            delta = datetime.timedelta(hours=timestep)
            future_time = current_time + delta
            month, day, hour = future_time.month, future_time.day, future_time.hour
            # forecast_data = self.weather_dict[(month, day, hour, minute)]
            # forecast_data = self.weather_dict[(month, day, hour+1)] # +1 because the hour is 1-indexed in the weather_file.epw
            forecast_temp = self.weather_dict_DBT[(month, day, hour+1)]
            forecast_DP_aux.append(self.weather_dict_DPT[(month, day, hour+1)])
            forecast_RH_aux.append(self.weather_dict_RH[(month, day, hour+1)])
            forecast_GHR_aux.append(self.weather_dict_GHR[(month, day, hour+1)])
            # forecast_weather.append(forecast_data['Site Outdoor Air Drybulb Temperature(Environment)'].values[0])
            # If relative future temperature: Out relative temperature = Dblist[idx_external] - Dblist[idx_internal]
            if self.relative_obs:
                curr_outdoor_temp = obs[4+0] + obs[4+2]
                forecast_temp_aux.append(forecast_temp - curr_outdoor_temp)
            else:
                forecast_temp_aux.append(forecast_temp)
            # delta = datetime.timedelta(minutes=self.timesteps_per_hour * timestep)
            # future_time = current_time + delta
            # month, day, hour, minute = future_time.month, future_time.day, future_time.hour, future_time.minute
            # forecast_data = self.myweather_data[(self.myweather_data['month'] == month) & (self.myweather_data['day'] == day) & (self.myweather_data['hour'] == hour) & (self.myweather_data['minute'] == minute)]

            # curr_outdoor_temp = obs[2] + obs[4]
            # forecast_weather.append(forecast_data['Site Outdoor Air Drybulb Temperature(Environment)'].values[0] - curr_outdoor_temp)
        
        self.weather_i += 1
        self.weather_i %= 4
        
        return forecast_temp_aux + forecast_DP_aux + forecast_RH_aux + forecast_GHR_aux
    
    
    def get_weather_file_path(self, env_name, seed=''):
        root_dir = os.getcwd()
        if seed == '':
            # Read all files that start with 'Eplus-env-{env_name}-'
            matches = [os.path.join(root_dir, dirname)
                        for dirname in os.listdir(root_dir)
                        if os.path.isdir(os.path.join(root_dir, dirname)) and dirname.startswith(f'Eplus-env-{env_name}-')]

            # Obtain the last folder with the highest 'resXXX' number
            max_res_num = -1
            res_folder = None
            for match in matches:
                res_num = int(match.split('-')[-1].replace('res', ''))
                if res_num > max_res_num:
                    max_res_num = res_num
                    res_folder = match

            # Obtain the file with extension '.epw' inside the res_folder
            for root, _, filenames in os.walk(res_folder+'/'):
                for filename in filenames:
                    if filename.endswith('.epw'):
                        return os.path.join(root, filename)
        else:
            # ToDo: Now I need to test if this else if correct, the previous if works
            # Read the file located in the path specified by 'seed'
            seed_dir = os.path.join(root_dir, f'Eplus-env-{env_name}', seed[1:-1], '-res1')
            if not os.path.isdir(seed_dir):
                raise ValueError(f"Seed directory '{seed_dir}' doesn't exist")

            # Read the file located in the path specified by 'seed'
            run_folders = [os.path.join(seed_dir, dirname)
                            for dirname in os.listdir(seed_dir)
                            if 'run' in dirname]
        
            # Obtain the file with extension '.epw' inside the run_folder with the highest run number
            max_run_num = -1
            run_folder = None
            for folder in run_folders:
                run_num = int(folder.split('_')[-1].replace('run', ''))
                if run_num > max_run_num:
                    max_run_num = run_num
                    run_folder = folder
            
            for root, _, filenames in os.walk(run_folder):
                for filename in filenames:
                    if filename.endswith('.epw'):
                        return os.path.join(root, filename)
        # If no file is found, return None
        return None
        
    # ---------------------------------------------------------------------------- #
    #                                     RESET                                    #
    # ---------------------------------------------------------------------------- #
    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str,
                                     Any]] = None) -> Tuple[np.ndarray,
                                                            Dict[str,
                                                                 Any]]:
        """Reset the environment.

        Args:
            seed (Optional[int]): The seed that is used to initialize the environment's episode (np_random). if value is None, a seed will be chosen from some source of entropy. Defaults to None.
            options (Optional[Dict[str, Any]]):Additional information to specify how the environment is reset. Defaults to None.

        Returns:
            Tuple[np.ndarray,Dict[str,Any]]: Current observation and info context with additional information.
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Change to next episode
        # if no options specified and environment has default reset options
        if not options and len(self.default_options) > 0:
            obs, info = self.simulator.reset(
                self.default_options)
        else:
            obs, info = self.simulator.reset(
                options)
                
        if self.add_forecast_weather:
            myweather_path = self.get_weather_file_path(env_name=self.env_name, seed=self.seed)
            self.read_weather_file(myweather_path)
            self.weather_i = 0
            obs.extend(self.forecast_weather([1,2,4], obs))
        
        self.CI = 0.69
        self.SoC = 0.69
        
        obs = np.asarray(np.hstack((obs, self.CI, self.SoC)))
        obs = np.array(obs, dtype=np.float32)
        
        self.obs_dict = dict(zip(self.variables['observation'], obs))
        if self.relative_obs:
            self.last_setpoint = self.obs_dict['Zone Thermostat Cooling Setpoint Temperature(West Zone)'] + self.obs_dict['Zone Air Temperature(West Zone)']
        else:
            self.last_setpoint = self.obs_dict['Zone Thermostat Cooling Setpoint Temperature(West Zone)']
        
        self.load_i = 0
        self.shifted_wklds = 0
        # myweather_path = r'/sinergym/Eplus-env-datacenter-mixed-continuous-v1-res119/Eplus-env-sub_run1/USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3_Random_4.0_1.0_0.1.epw'

        info['DC_load'] = self.obs_dict['Facility Total Electricity Demand Rate(Whole Building)']/1e3
        info['action'] = -1
        self.temp_state = obs

        return obs, info

    def obtain_cpu_load_deprecated(self):
        # Considering the day and the hour/minute, I select the cpu utilization
        # The file is self.external_cpu_scheme
        day_of_year, hour = self.obs_dict['sindayOTY'], self.obs_dict['sinhour']
        day_of_year %= 7
        minute = int(self.load_i*self.timesteps_per_hour)
        
        load = self.external_cpu_scheme[(day_of_year, hour, minute)]
        
        # Sanity clip
        load = np.clip(load, 0, 1)
        
        self.load_i += 1
        self.load_i %= 4
        
        return load
    
    def set_CI_SoC(self, CI_SoC):
        # print(f'Updating shifted_wklds to {shifted_wklds}')
        CI, SoC = CI_SoC
        self.CI = CI
        self.SoC = SoC

    # set_CI_SoC
    def set_shifted_wklds(self, shifted_wklds):
        # print(f'Updating shifted_wklds to {shifted_wklds}')
        self.shifted_wklds = shifted_wklds
    # ---------------------------------------------------------------------------- #
    #                                     STEP                                     #
    # ---------------------------------------------------------------------------- #
    def step(self,
             action: Union[int,
                           float,
                           np.integer,
                           np.ndarray,
                           List[Any],
                           Tuple[Any]]
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Sends action to the environment

        Args:
            action (Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]): Action selected by the agent.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Observation for next timestep, reward obtained, Whether the episode has ended or not, Wheather episode has been truncated or not, and a dictionary with extra information
        """

        # Get action
        # print('Action: ', action)
        action_ = self._get_action(action)
        raw_action = action
        if not self.external_cpu_scheme == '':
            # cpu_load = self.obtain_cpu_load()
            # # Concat action_ with cpu_load
            # action = [action, cpu_load*100]
            action = [action, self.shifted_wklds*100]
            action_.append(self.shifted_wklds)

        # Send action to the simulator
        self.simulator.logger_main.debug(action_)
        # Execute action in simulation
        obs, terminated, truncated, info = self.simulator.step(action_)
        
        # ToDo: change terminated to one week
        
        # self.load_shifting_model(action_load)
        
        if self.add_forecast_weather:
            obs.extend(self.forecast_weather([1,2,4], obs))
        
        obs.extend([self.CI, self.SoC])
        obs = np.array(obs, dtype=np.float32)
        
        # Create dictionary with observation
        self.obs_dict = dict(zip(self.variables['observation'], obs))

        # Calculate reward
        reward, terms = self.reward_fn()

        # info update with reward information
        info.update({
            'total_power': terms.get('total_energy'),
            'total_power_no_units': terms.get('reward_energy'),
            'comfort_penalty': terms.get('reward_comfort'),
            'abs_comfort': terms.get('abs_comfort'),
            'temperatures': terms.get('temperatures'),
            'action_': action_,
            'action': action,
            'raw_action': raw_action})
        
        info['DC_load'] = self.obs_dict['Facility Total Electricity Demand Rate(Whole Building)']/1e3

        self.temp_state = obs
        
        return obs, reward, terminated, truncated, info

    def update_state(self):
        self.temp_state[-3] = self.shifted_wklds
        self.temp_state[-2] = self.CI
        self.temp_state[-1] = self.SoC
        return self.temp_state
    # ---------------------------------------------------------------------------- #
    #                                RENDER (empty)                                #
    # ---------------------------------------------------------------------------- #
    def render(self, mode: str = 'human') -> None:
        """Environment rendering.

        Args:
            mode (str, optional): Mode for rendering. Defaults to 'human'.
        """
        pass

    # ---------------------------------------------------------------------------- #
    #                                     CLOSE                                    #
    # ---------------------------------------------------------------------------- #
    def close(self) -> None:
        """End simulation."""

        self.simulator.end_env()

    # ---------------------------------------------------------------------------- #
    #                           Environment functionality                          #
    # ---------------------------------------------------------------------------- #
    def get_schedulers(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Extract all schedulers available in the building model to be controlled.

        Args:
            path (str, optional): If path is specified, then this method export a xlsx file version in addition to return the dictionary.

        Returns:
            Dict[str, Any]: Python Dictionary: For each scheduler found, it shows type value and where this scheduler is present (Object name, Object field and Object type).
        """
        schedulers = self.simulator._config.schedulers
        if path is not None:
            export_actuators_to_excel(actuators=schedulers, path=path)
        return schedulers

    def get_zones(self) -> List[str]:
        """Get the zone names available in the building model of that environment.

        Returns:
            List[str]: List of the zone names.
        """
        return self.simulator._config.idf_zone_names

    def _get_action(self, action: Any) -> Union[int,
                                                float,
                                                np.integer,
                                                np.ndarray,
                                                List[Any],
                                                Tuple[Any]]:
        """Transform the action for sending it to the simulator."""
        # print(f"Action inside _get_action: {action}")
        # Get action depending on flag_discrete
        if self.flag_discrete:
            # Index for action_mapping
            if np.issubdtype(type(action), np.integer):
                if isinstance(action, int): # or isinstance(action, np.integer):
                    setpoints = self.action_mapping[action]
                else:
                    setpoints = self.action_mapping[action.item()]
                    if self.delta_actions:
                        setpoints = self._setpoints_transform_delta_discrete(setpoints)
            # Manual action
            elif isinstance(action, tuple) or isinstance(action, list):
                # stable-baselines DQN bug prevention
                if len(action) == 1:
                    setpoints = self.action_mapping[action[0]]
                else:
                    setpoints = action
            elif isinstance(action, np.ndarray):
                setpoints = self.action_mapping[action.item()]
            else:
                raise RuntimeError(
                    'action type not supported by Sinergym environment')
            action_ = list(setpoints)
        else:
            # transform action to setpoints simulation
            action_ = self._setpoints_transform(action)

        return action_
    
    def _setpoints_transform_delta_discrete(self,
                                    action: Union[int,
                                                float,
                                                np.integer,
                                                np.ndarray,
                                                List[Any],
                                                Tuple[Any]]) -> Union[int,
                                                                        float,
                                                                        np.integer,
                                                                        np.ndarray,
                                                                        List[Any],
                                                                        Tuple[Any]]:
        """ This method transforms an action defined in gym (-1,1 in all continuous environment) action space to simulation real action space.
        Considering that the action is a delta action, consider the last setpoint to obtain the next setpoint.
        The delta action is limited to increase the setpoint -0.5 and 0.5 C.

        Args:
            action (Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]): Action received in environment

        Returns:
            Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]: Action transformed in simulator action space.
        """
                
        # Transform the last setpoint to the next setpoint considering the delta action
        action_ = []
        
        value = self.last_setpoint + np.array([action])

        # Check if action is out of range
        if value <= self.temp_range[0]: #self.action_space.low[0]:
            value = np.array([self.temp_range[0]]) #self.action_space.low[0]
        
        if value >= self.temp_range[1]: #self.action_space.high[0]:
            value = np.array([self.temp_range[1]]) #self.action_space.high[0]
        
        self.last_setpoint = value
        action_.append(value[0])
        return action_
    
    def _setpoints_transform(self,
                             action: Union[int,
                                           float,
                                           np.integer,
                                           np.ndarray,
                                           List[Any],
                                           Tuple[Any]]) -> Union[int,
                                                                 float,
                                                                 np.integer,
                                                                 np.ndarray,
                                                                 List[Any],
                                                                 Tuple[Any]]:
        """ This method transforms an action defined in gym (-1,1 in all continuous environment) action space to simulation real action space.

        Args:
            action (Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]): Action received in environment

        Returns:
            Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]: Action transformed in simulator action space.
        """
        action_ = []

        for i, value in enumerate(action):
            if self._action_space.low[i] <= value <= self._action_space.high[i]:
                a_max_min = self._action_space.high[i] - \
                    self._action_space.low[i]
                sp_max_min = self.setpoints_space.high[i] - \
                    self.setpoints_space.low[i]

                action_.append(
                    self.setpoints_space.low[i] +
                    (
                        value -
                        self._action_space.low[i]) *
                    sp_max_min /
                    a_max_min)
            else:
                # If action is outer action_space already, it don't need
                # transformation
                action_.append(value)

        return action_

    def _check_eplus_env(self) -> None:
        """This method checks that environment definition is correct and it has not inconsistencies.
        """
        # OBSERVATION
        assert len(self.variables['observation']) == self._observation_space.shape[
            0], 'Observation space has not the same length than variable names specified.'

        # ACTION
        if self.flag_discrete:
            assert hasattr(
                self, 'action_mapping'), 'Discrete environment: action mapping should have been defined.'
            assert not hasattr(
                self, 'setpoints_space'), 'Discrete environment: setpoints space should not have been defined.'
            assert self._action_space.n == len(
                self.action_mapping), 'Discrete environment: The length of the action_mapping must match the dimension of the discrete action space.'
            # for values in self.action_mapping.values():
            #     assert len(values) == len(
            #         self.variables['action']), 'Discrete environment: Action mapping tuples values must have the same length than action variables specified.'
        else:
            assert len(self.variables['action']) == self._action_space.shape[
                0], 'Action space shape must match with number of action variables specified.'
            assert hasattr(
                self, 'setpoints_space'), 'Continuous environment: setpoints_space attribute should have been defined.'
            assert not hasattr(
                self, 'action_mapping'), 'Continuous environment: action mapping should not have been defined.'
            assert len(self._action_space.low) == len(self.variables['action']) and len(self._action_space.high) == len(
                self.variables['action']), 'Continuous environment: low and high values action space definition should have the same number of values than action variables.'

    # ---------------------------------------------------------------------------- #
    #                                  Properties                                  #
    # ---------------------------------------------------------------------------- #
    @property
    def action_space(
        self,
    ) -> gym.spaces.Space[Any] | gym.spaces.Space[Any]:
        return self._action_space

    @action_space.setter
    def action_space(self, space: gym.spaces.Space[Any]):
        self._action_space = space

    @property
    def observation_space(
        self,
    ) -> gym.spaces.Space[Any] | gym.spaces.Space[Any]:
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: gym.spaces.Space[Any]):
        self._observation_space = space
