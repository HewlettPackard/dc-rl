import os
from typing import Optional, Tuple
import numpy as np
import pandas as pd

import pyfmi
import gymnasium as gym
from gymnasium import spaces

import envs.datacenter as DataCenter
from utils import reward_creator

class dc_gymenv(gym.Env):
    
    def __init__(self, observation_variables : list,
                       observation_space : spaces.Box,
                       action_variables: list,
                       action_space : spaces.Discrete,
                       action_mapping: dict,
                       ranges : dict,  # this data frame should be time indexed for the code to work
                       add_cpu_usage : bool,
                       min_pump_speed : float,
                       max_pump_speed : float,
                       min_supply_temp : float,
                       max_supply_temp : float,
                       action_definition : dict,
                       DC_Config : dict,
                       seed : int = 123,
                       episode_length_in_time : pd.Timedelta = None,  # can be 1 week in minutes eg pd.Timedelta('7days')
                       ):
        """Creates the data center environment

        Args:
            observation_variables (list[str]): The partial list of variables that will be evaluated inside this evironment.The actual
                                                gym space may include other variables like sine cosine of hours, day of year, cpu usage,
                                                carbon intensity and battery state of charge.
            observation_space (spaces.Box): The gym observations space following gymnasium standard
            action_variables (list[str]): The list of action variables for the environment. It is used to create the info dict returned by
                                        the environment
            action_space (spaces.Discrete): The gym action space following gymnasium standard
            action_mapping (dict): A mapping from agent discrete action choice to actual delta change in setpoint. The mapping is defined in
                                    utils.make_pyeplus_env.py
            ranges (dict[str,list]): The upper and lower bounds on the observation_variables
            max_pump_speed (float): The maximum temperature allowed for the CRAC setpoint
            min_pump_speed (float): The minimum temperature allowed for the CRAC setpoint
            action_definition (dict): A mapping of the action name to the default or initialized value. Specified in utils.make_pyeplus_env.py
            episode_length_in_time (pd.Timedelta, optional): The maximum length after which the done flag should be True. Defaults to None. 
                                                            Setting none causes done to be True after data set is exausted.
        """
        
        self.observation_variables = observation_variables
        self.observation_space = observation_space
        self.action_variables = action_variables
        self.action_space = action_space
        self.action_mapping = action_mapping
        self.ranges = ranges
        self.seed = seed
        self.add_cpu_usage = add_cpu_usage
        self.ambient_temp = 20
        self.scale_obs = False
        self.obs_max = []
        self.obs_min = []
        self.DC_Config = DC_Config
        
        path2fmu = "SLC_MIMO.fmu"

        fmu_path = os.path.join(os.getcwd(), 'envs', path2fmu)
        # Check if the FMU file exists
        try:
            # Load the FMU from the path + current working directory
            fmu = pyfmi.load_fmu(fmu_path, kind='CS')
            # Continue with the rest of the code
            self.fmu = fmu
            print(f"FMU file loaded correctly: {path2fmu}")
        except Exception as e:
            print(f"Error loading FMU file: {e}")

        self.step_size = 15*60 # 15 Minutes
        self.liquid_guideline_temp = 32  # ASHRAE W32 Guidelines
        
        # similar to reset
        self.dc = DataCenter.DataCenter_ITModel(num_racks=self.DC_Config.NUM_RACKS,
                                                rack_supply_approach_temp_list=self.DC_Config.RACK_SUPPLY_APPROACH_TEMP_LIST,
                                                rack_CPU_config=self.DC_Config.RACK_CPU_CONFIG,
                                                max_W_per_rack=self.DC_Config.MAX_W_PER_RACK,
                                                DC_ITModel_config=self.DC_Config)
        
        
        self.CRAC_Fan_load, self.CRAC_cooling_load, self.Compressor_load, self.CW_pump_load, self.CT_pump_load = None, None, None, None, None
        self.HVAC_load = self.ranges['Facility Total HVAC Electricity Demand Rate(Whole Building)'][0]
        self.rackwise_cpu_pwr, self.rackwise_itfan_pwr, self.rackwise_outlet_temp = [], [], []
        self.cpu_load_frac = 0.5
        self.bat_SoC = 300*1e3  # all units are SI
        
        self.raw_curr_state = None
        self.raw_next_state = None
        self.raw_curr_stpt = action_definition['cooling setpoints']['initial_value']
        self.max_pump_speed = max_pump_speed
        self.min_pump_speed = min_pump_speed
        
        self.max_supply_temp = max_supply_temp
        self.min_supply_temp = min_supply_temp
        
        self.consecutive_actions = 0
        self.last_action = None
        self.action_scaling_factor = 1  # Starts with a scale factor of 1
        
        # IT + HVAC
        self.power_lb_kW = (self.ranges['Facility Total Building Electricity Demand Rate(Whole Building)'][0] + self.ranges['Facility Total HVAC Electricity Demand Rate(Whole Building)'][0]) / 1e3
        self.power_ub_kW = (self.ranges['Facility Total Building Electricity Demand Rate(Whole Building)'][1] + self.ranges['Facility Total HVAC Electricity Demand Rate(Whole Building)'][1]) / 1e3
        
        
        # Initialize running min/max for each observation variable
        self.min_values = np.array([np.inf] * len(self.observation_variables))
        self.max_values = np.array([-np.inf] * len(self.observation_variables))
        super().__init__()
    
    
    def reset(self, *, seed=None, options=None):

        """
        Reset `dc_gymenv` to initial state.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Environment options.

        Returns:
            raw_curr_state (List[float]): Current state of the environmment
            {} (dict): A dictionary that containing additional information about the environment state
        """

        super().reset(seed=self.seed)

        # Reset the FMU to the initial state
        self.fmu.reset()
        
        # Initialize the model
        self.fmu.setup_experiment(start_time=0, stop_time=60*60*30)
        self.fmu.initialize()
        self.current_time = 0
        self.pump_speed = (np.random.random()*(self.max_pump_speed-self.min_pump_speed)) + self.min_pump_speed
        self.temp_at_mixer = None
        self.coo_mov_flow_actual = 0.5
        self.coo_m_flow_nominal = 0.5
        self.prev_server_temps = 32  # ASHRAE W32 Guidelines
        
        self.CRAC_Fan_load, self.CRAC_cooling_load, self.Compressor_load, self.CW_pump_load, self.CT_pump_load = None, None, None, None, None
        self.HVAC_load = self.ranges['Facility Total HVAC Electricity Demand Rate(Whole Building)'][0]
        self.rackwise_cpu_pwr, self.rackwise_itfan_pwr, self.rackwise_outlet_temp = [], [], []
        self.water_usage = None
        
        self.raw_curr_state = self.get_obs()
                
        if self.scale_obs:
            return self.normalize(self.raw_curr_state), {}  
    
    
    def step(self, action):
        """
        Makes an environment step in`dc_gymenv.

        Args:
            action_id (int): Action to take.

        Returns:
            observations (List[float]): Current state of the environmment
            reward (float): reward value.
            done (bool): A boolean value signaling the if the episode has ended.
            info (dict): A dictionary that containing additional information about the environment state
        """
        
        # pump_speed_delta = self.action_mapping[action]
        
        # self.pump_speed += pump_speed_delta
        # self.pump_speed = max(min(self.pump_speed, self.max_pump_speed), self.min_pump_speed)
        
        # action = max(min(action, self.max_pump_speed), self.min_pump_speed)
        # alpha = 0.75
        # if self.pump_speed is None:
        #     self.pump_speed = action  # Initialize on first use with the current action
        # else:
        #     self.pump_speed = alpha * action + (1 - alpha) * self.pump_speed
            
        # Now the action is continuous
        # Apply a sigmoid function to the action to scale it between 0 and 1
        # action = 1 / (1 + np.exp(-np.array(action)))
        
        if len(action) == 1:
            # Only pump speed
            action[0] = action[0] * (self.max_pump_speed - self.min_pump_speed) + self.min_pump_speed
            self.pump_speed = max(min(action[0], self.max_pump_speed), self.min_pump_speed)
            supply_liquid_temp = self.liquid_guideline_temp + 273.15 #  Kelvins. Following the ASHRAE Guidelines
            
            # Only Supply temp
            # action[0] = action[0] * (self.max_supply_temp - self.min_supply_temp) + self.min_supply_temp
            # self.pump_speed = 0.25
            # supply_liquid_temp = max(min(action[0], self.max_supply_temp), self.min_supply_temp) + 273.15 #  Kelvins.
        else:
            # Transform the action to the [0,1] range using the self.action_space
            action = (action - self.action_space.low)/(self.action_space.high - self.action_space.low)
            
            # Scale the action to the correct range 
            action[0] = action[0] * (self.max_pump_speed - self.min_pump_speed) + self.min_pump_speed
            action[1] = action[1] * (self.max_supply_temp - self.min_supply_temp) + self.min_supply_temp
            
            
            self.pump_speed = max(min(action[0], self.max_pump_speed), self.min_pump_speed)
            supply_liquid_temp = max(min(action[1], self.max_supply_temp), self.min_supply_temp) + 273.15 #  Kelvins.
        pump_speed_delta = self.pump_speed # max(min(action, self.max_pump_speed), self.min_pump_speed)
        ITE_load_pct_list = [self.cpu_load_frac*100 for _ in range(self.DC_Config.NUM_RACKS)] 
        
    
        # Util, Setpoint, Average return temperature, Average CRAC Ret Temp, DC ITE power, CT power, Chiller power
        # 0, 15, 26.26, 23.16, 656710
        # 0, 21, 31.97, 28.87, 871450
        # 100, 15, 35.45, 32.44, 1248170
        # 100, 21, 36.88, 26.78, 1462910

        input_var_names = ['processer_utilization',  #  Percentaje of utilization of the servers
                           'stPT',                   #  Supply temperature of the liquid to the chiller. Kelvins
                           'm_flow_in',              #  Prescribed mass flow rate [kg/s] or [l/s]
                           ]
        
        cpu_util = self.cpu_load_frac*100 #  % Percentage
        input_ts_list = [cpu_util, supply_liquid_temp, self.pump_speed]
        
        self.fmu.set(input_var_names, input_ts_list)
        self.fmu.do_step(current_t=self.current_time, step_size=self.step_size)
        self.current_time += self.step_size # 15 Minutes
        
        # # Let's obtain the limits of the variables under cpu_util = 0 and cpu_util = 100, and self.pump_speed = 0.05 and self.pump_speed = 0.5
        # for temp_cpu_util in [0, 100]:
        #     for temp_pump_speed in [0.05, 0.5]:
        #         input_ts_list = [temp_cpu_util, supply_liquid_temp, temp_pump_speed]
        #         self.fmu.set(input_var_names, input_ts_list)
        #         self.fmu.do_step(current_t=self.current_time, step_size=self.step_size*10) # To simulate steady state
        #         self.current_time += self.step_size*10

        #         modelica_pump_power = self.fmu.get('mov.P')[0] * 15000 * 20 # W. To Simulate 20 Pumps
        #         inlet_temp_liq = self.fmu.get('tempout.T')[0]   # Kelvins
        #         return_temp_liq = self.fmu.get('tempin.T')[0]   # Kelvins
        #         coo_m_flow_nominal = np.abs(self.fmu.get('coo.m_flow_nominal'))[0]  #Kg/s
        #         self.coo_mov_flow_actual = np.abs(self.fmu.get('mov.m_flow_actual'))[0]  #Kg/s
        #         coo_Q_flow = np.abs(self.fmu.get('coo.Q_flow'))[0] * 100 #  To simulate a datacenter with 300 cabinets
        #         self.temp_at_mixer = self.fmu.get('tempatmixer.T')[0]  # Kelvins
                
        #         pipe1_temp = self.fmu.get('pipe1.sta_b.T')[0]  # Kelvins
        #         pipe2_temp = self.fmu.get('pipe2.sta_b.T')[0]  # Kelvins
        #         pipe3_temp = self.fmu.get('pipe3.sta_b.T')[0]  # Kelvins
                
        #         server1_temp = self.fmu.get('serverblock1.heatCapacitor.T')[0]  # Kelvins
        #         server2_temp = self.fmu.get('serverblock2.heatCapacitor.T')[0]  # Kelvins
        #         server3_temp = self.fmu.get('serverblock3.heatCapacitor.T')[0]  # Kelvins
                
        #         server_temps = [server1_temp, server2_temp, server3_temp]

        #         ITE_load_pct_list = [temp_cpu_util for _ in range(self.DC_Config.NUM_RACKS)] 

        #         self.rackwise_cpu_pwr, _, _ = \
        #             self.dc.compute_datacenter_IT_load_outlet_temp(ITE_load_pct_list=ITE_load_pct_list, CRAC_setpoint=self.liquid_guideline_temp, server_temps=server_temps)
        #         # Now print the values
        #         print(f"CPU Utilization: {temp_cpu_util}%, Pump Speed: {temp_pump_speed}")
        #         print(f"Modelica Pump Power: {modelica_pump_power} W")
        #         print(f"Inlet Temp Liquid: {inlet_temp_liq-273.15} C")
        #         print(f"Return Temp Liquid: {return_temp_liq-273.15} C")
        #         print(f"Cooling Mass Flow Nominal: {coo_m_flow_nominal} Kg/s")
        #         print(f"Cooling Mass Flow Actual: {self.coo_mov_flow_actual} Kg/s")
        #         print(f"Cooling Q Flow: {coo_Q_flow} W")
        #         print(f"Temp at Mixer: {self.temp_at_mixer-273.15} C")
        #         print(f"Pipe1 Temp: {pipe1_temp-273.15} C")
        #         print(f"Pipe2 Temp: {pipe2_temp-273.15} C")
        #         print(f"Pipe3 Temp: {pipe3_temp-273.15} C")
        #         print(f"Server1 Temp: {server1_temp-273.15} C")
        #         print(f"Server2 Temp: {server2_temp-273.15} C")
        #         print(f"Server3 Temp: {server3_temp-273.15} C")
        #         print(f"ITE Load: {sum(self.rackwise_cpu_pwr)} W")
        #         print("\n")
                
                
        
        # We can extract the pump power consumption from 'mov.P', althought this is a normalized value, so, the value should be multiplied by 15 to obtain the power consumption in KW
        # More info:
        '''
        For a liquid-cooled data center with a 1MW compute capacity, a typical pump in the primary loop might consume approximately 13 to 15 kW. 
        This estimate assumes a nominal head of around 20 meters and a flow rate sufficient to maintain a 30°C temperature difference across the
        cooling system, with typical pump efficiencies.
        '''
        
        modelica_pump_power = self.fmu.get('mov.P')[0] * 15000 * 3 # W. To Simulate 3 Pumps
        inlet_temp_liq = self.fmu.get('tempout.T')[0]   # Kelvins
        return_temp_liq = self.fmu.get('tempin.T')[0]   # Kelvins
        self.coo_m_flow_nominal = np.abs(self.fmu.get('coo.m_flow_nominal'))[0]  #Kg/s
        self.coo_mov_flow_actual = np.abs(self.fmu.get('mov.m_flow_actual'))[0]  #Kg/s
        self.coo_Q_flow = np.abs(self.fmu.get('coo.Q_flow'))[0] * 350 #  To simulate a datacenter with 350 cabinets and a total power of 1MW
        self.temp_at_mixer = self.fmu.get('tempatmixer.T')[0]  # Kelvins
        
        pipe1_temp = self.fmu.get('pipe1.sta_b.T')[0]  # Kelvins
        pipe2_temp = self.fmu.get('pipe2.sta_b.T')[0]  # Kelvins
        pipe3_temp = self.fmu.get('pipe3.sta_b.T')[0]  # Kelvins
        
        server1_temp = self.fmu.get('serverblock1.heatCapacitor.T')[0]  # Kelvins
        server2_temp = self.fmu.get('serverblock2.heatCapacitor.T')[0]  # Kelvins
        server3_temp = self.fmu.get('serverblock3.heatCapacitor.T')[0]  # Kelvins
        
        server_temps = [server1_temp, server2_temp, server3_temp]
        self.rackwise_cpu_pwr, _, _ = \
            self.dc.compute_datacenter_IT_load_outlet_temp(ITE_load_pct_list=ITE_load_pct_list, CRAC_setpoint=inlet_temp_liq-273.15, server_temps=server_temps)

        data_center_total_ITE_Load = sum(self.rackwise_cpu_pwr)
        data_center_total_ITE_Load = (self.coo_Q_flow + data_center_total_ITE_Load)/2

        # print(f'Q_flow: {coo_Q_flow} W, ITE Load: {data_center_total_ITE_Load} W, average of those metrics: {(coo_Q_flow + data_center_total_ITE_Load)/2} W')
        
        self.CT_Fan_pwr, self.CRAC_cooling_load, self.chiller_power, self.power_water_pump_CT  = DataCenter.calculate_HVAC_power(CRAC_setpoint=inlet_temp_liq-273.15,
                                                                                                                                 chiller_heat_removed=data_center_total_ITE_Load*0.2,
                                                                                                                                 ambient_temp=self.ambient_temp,
                                                                                                                                 DC_Config=self.DC_Config)
        self.HVAC_load = self.CT_Fan_pwr + self.CRAC_cooling_load + self.power_water_pump_CT + modelica_pump_power# + self.chiller_power

        
        # Set the additional attributes for the cooling tower water usage calculation
        self.dc.hot_water_temp = return_temp_liq - 273.15 # °C
        self.dc.cold_water_temp = inlet_temp_liq - 273.15  # °C
        self.dc.wet_bulb_temp = self.wet_bulb  # °C from weather data

        # Calculate the cooling tower water usage
        self.water_usage = self.dc.calculate_cooling_tower_water_usage()  # liters per 15 minutes
        # self.water_usage = self.dc.calculate_cooling_tower_water_usage_v2(Q_IT=data_center_total_ITE_Load*0.2, T_hot=self.dc.hot_water_temp, T_cold=self.dc.cold_water_temp, T_wb=self.dc.wet_bulb_temp)
        # water_usage_meth2 = DataCenter.calculate_water_consumption_15min(self.CRAC_Cooling_load,  self.dc.hot_water_temp, self.dc.cold_water_temp)
        # print(f"Estimated cooling tower water usage method1 (liters per 15 min): {water_usage}")
        # print(f"Estimated cooling tower water usage method2 (liters per 15 min): {water_usage_meth2}")

        # calculate reward
        self.reward = 0
                
        # calculate self.raw_next_state
        self.raw_next_state = self.get_obs()
        
        # Update the last action
        self.last_action = pump_speed_delta
        
        # add info dictionary 
        self.info = {
            'dc_ITE_total_power_kW': data_center_total_ITE_Load / 1e3,
            'dc_CT_total_power_kW': self.CT_Fan_pwr / 1e3,
            'dc_Compressor_total_power_kW': (modelica_pump_power + self.coo_Q_flow) / 1e3,
            'dc_HVAC_total_power_kW': (self.HVAC_load) / 1e3,
            'dc_total_power_kW': (data_center_total_ITE_Load + self.HVAC_load) / 1e3,
            'dc_crac_setpoint_delta': pump_speed_delta,
            'dc_crac_setpoint': self.pump_speed,
            'dc_cpu_workload_fraction': self.cpu_load_frac,
            'dc_int_temperature': return_temp_liq - 273.15,
            'dc_exterior_ambient_temp': self.ambient_temp,
            'dc_power_lb_kW': self.power_lb_kW,
            'dc_power_ub_kW': self.power_ub_kW,
            'dc_CW_pump_power_kW': self.CW_pump_load,
            'dc_CT_pump_power_kW': self.CT_pump_load,
            'dc_water_usage': self.water_usage,
            'dc_pump_power_kW': modelica_pump_power / 1e3,
            'dc_heat_removed': self.coo_Q_flow/1e3,
            'dc_coo_m_flow_nominal': self.coo_m_flow_nominal,
            'dc_coo_mov_flow_actual': self.coo_mov_flow_actual,
            'dc_supply_liquid_temp': inlet_temp_liq - 273.15,
            'dc_return_liquid_temp': return_temp_liq - 273.15,
            'dc_average_pipe_temp': (pipe1_temp + pipe2_temp + pipe3_temp) / 3 - 273.15,
            'dc_average_server_temp': (server1_temp + server2_temp + server3_temp) / 3 - 273.15,
            'dc_current_servers_temps': np.average(server_temps) - 273.15,
            'dc_previous_servers_temps': self.prev_server_temps,
        }
        self.prev_server_temps = np.average(server_temps) - 273.15

        #Done and truncated are managed by the main class, implement individual function if needed
        truncated = False
        done = False 
        # return processed/unprocessed state to agent
        if self.scale_obs:
            # # Update the min/max values based on the new observation
            # self.update_observation_ranges(self.raw_next_state)
            
            # # Normalize the current observation
            # normalized_observation = self.normalize_observations(self.raw_next_state)
            # return normalized_observation, self.reward, done, truncated, self.info

            return self.normalize(self.raw_next_state), self.reward, done, truncated, self.info


    def NormalizeObservation(self,):
        """
        Obtains the value for normalizing the observation.
        """
        self.scale_obs = True
        for obs_var in self.observation_variables:
            self.obs_min.append(self.ranges[obs_var][0])
            self.obs_max.append(self.ranges[obs_var][1])
        
        self.obs_min = np.array(self.obs_min)
        self.obs_max = np.array(self.obs_max)
        self.obs_delta = self.obs_max - self.obs_min

    def update_observation_ranges(self, observations):
        # Update the minimum and maximum values
        # self.min_values = np.minimum(self.min_values, observations)
        # self.max_values = np.maximum(self.max_values, observations)
        for i, value in enumerate(observations):
            if value < self.min_values[i]:
                print(f"New minimum for {self.observation_variables[i]}: {value} (previous min: {self.min_values[i]})")
                self.min_values[i] = value
            
            if value > self.max_values[i]:
                print(f"New maximum for {self.observation_variables[i]}: {value} (previous max: {self.max_values[i]})")
                self.max_values[i] = value


    
    def normalize_observations(self, observations):
        # Ensure no division by zero by adding a small epsilon to the denominator
        epsilon = 1e-8
        return (observations - self.min_values) / (self.max_values - self.min_values + epsilon)


    def normalize(self,obs):
        """
        Normalizes the observation.
        """
        return 2 * np.float32((obs-self.obs_min)/self.obs_delta) - 1


    def get_obs(self):
        """
        Returns the observation at the current time step.
        
        The observation is a list of the following variables:
            'Site Outdoor Air Drybulb Temperature(Environment)',
            'Actual Pump Speed',
            'Temp at Mixer', # Temperature at the mixer, previous to the pump of the liquid cooling system
            'Facility Total HVAC Electricity Demand Rate(Whole Building)',  # 'HVAC POWER'
            'Facility Total Building Electricity Demand Rate(Whole Building)'  #  'IT POWER'
    

        Returns:
            observation (List[float]): Current state of the environmment.
        """
        actual_pump_speed = self.coo_mov_flow_actual  # In l/s
        nominal_pump_speed = self.coo_m_flow_nominal  # In l/s
        
        temp_at_mixer = self.liquid_guideline_temp  # C. Following the ASHRAE Guidelines
        if self.temp_at_mixer:
            temp_at_mixer = self.temp_at_mixer - 273.15  # Convert to Celsius

        # 'Facility Total HVAC Electricity Demand Rate(Whole Building)'  ie 'HVAC POWER'
        hvac_power = self.HVAC_load #self.CT_Cooling_load + self.Compressor_load

        # 'Facility Total Building Electricity Demand Rate(Whole Building)' ie 'IT POWER'
        if self.rackwise_cpu_pwr:
            it_power = (self.coo_Q_flow + sum(self.rackwise_cpu_pwr))/2
        else:
            it_power = self.ranges['Facility Total Building Electricity Demand Rate(Whole Building)'][0]

        return [self.ambient_temp, actual_pump_speed, temp_at_mixer, hvac_power, it_power]
        # return [actual_pump_speed, temp_at_mixer, hvac_power+it_power]


    def set_shifted_wklds(self, cpu_load):
        """
        Updates the current CPU workload. fraction between 0.0 and 1.0
        """
        if 0.0 > cpu_load or cpu_load > 1.0:
            print('CPU load out of bounds')
        assert 0.0 <= cpu_load <= 1.0, 'CPU load out of bounds'
        self.cpu_load_frac = cpu_load


    def set_ambient_temp(self, ambient_temp, wet_bulb):
        """
        Updates the external temperature.
        """
        self.ambient_temp = ambient_temp
        self.wet_bulb = wet_bulb
    
    
    def set_bat_SoC(self, bat_SoC):
        """
        Updates the battery state of charge.
        """
        self.bat_SoC = bat_SoC


    # def reset_fmu(self):
    #     """
    #     Reset the FMU.
    #     """
    #     self.fmu.reset()