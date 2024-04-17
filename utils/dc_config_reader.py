"""
This file is used to read the data center configuration from  user inputs provided inside dc_config.json. It also performs some auxiliary steps to calculate the server power specifications based on the given parameters.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

class DC_Config:
    def __init__(self, dc_config_file='dc_config.json', datacenter_capacity_mw=1):
        """
        Initializes a new instance of the DC_Config class, loading configuration
        data from the specified JSON configuration file.

        Args:
            dc_config_file (str): The path to the data center configuration JSON file.
        """
        # Determine the full path to the configuration file
        self.config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), dc_config_file)
        
        # Define the maximum compute power capacity of the datacenter on MW
        self.datacenter_capacity_mw = datacenter_capacity_mw
        
        # Load the JSON data from the configuration file
        self.config_data = self._load_config()

        # Set up configuration parameters
        self._setup_config()
        
    def _load_config(self):
        """
        Loads the data center configuration from the specified JSON file.

        Returns:
            dict: A dictionary containing the loaded configuration data.
        """
        with open(self.config_path, 'r') as file:
            return json.load(file)
    
    def _setup_config(self):
        """
        Sets up various configuration parameters based on the loaded JSON data.
        """
        ##################################################################
        #################### GEOMETRY DEPENDENT PARAMETERS ###############
        ##################################################################
        json_obj = self.config_data
        # Data Center Geometric configuration
        self.NUM_ROWS = json_obj['data_center_configuration']['NUM_ROWS']  # number of rows in which data centers are arranged
        self.NUM_RACKS_PER_ROW = json_obj['data_center_configuration']['NUM_RACKS_PER_ROW']  # number of racks/ITcabinets in each row
        self.NUM_RACKS = self.NUM_ROWS * self.NUM_RACKS_PER_ROW  # calculate total number of racks/ITcabinets in the data center model

        self.TOTAM_MAX_PWR = self.datacenter_capacity_mw * 1e6  # specify maximum allowed power consumption (W) for the entire data center
        self.MAX_W_PER_RACK = int(self.TOTAM_MAX_PWR/self.NUM_RACKS)  # calculate maximum allowable power consumption for each rack/ITcabinet


        # CFD may be used to precompute the "supply/return approach temperature" for each rack under given
        # geometry, containment, CRAC Air flow rate, Load
        
        # Supply approach temperature: It is the delta T i.e. the temperature difference between 
        # CRAC_setpoint and the actual inlet temperature to the rack .Its value depends on the geometry
        # of the data center rack arrangements and can be pre-computed from CFD analysis. The length of
        # the list should be the same as NUM_RACKS; Default values are populated from paper [3] assuming:
        # Scenario # 19 from Table 5
        self.RACK_SUPPLY_APPROACH_TEMP_LIST = json_obj['data_center_configuration']['RACK_SUPPLY_APPROACH_TEMP_LIST']

        # Return approach temperature: It is the delta T i.e. the temperature difference between 
        # CRAC return temperature and the rack Outlet temperature .Its value also depends on the geometry
        # of the data center rack arrangements and can be pre-computed from CFD analysis. The length of
        # the list should be the same as NUM_RACKS; Default values are populated from paper [3] assuming:
        # Scenario # 19 from Table 5
        # we add some variation to the default values to highlight change in geometry
        self.RACK_RETURN_APPROACH_TEMP_LIST = json_obj['data_center_configuration']['RACK_RETURN_APPROACH_TEMP_LIST']

        # how many servers are assigned in each rack. The actual number of servers per rack may be limited while
        self.CPUS_PER_RACK = json_obj['data_center_configuration']['CPUS_PER_RACK']  

        ##################################################################
        #################### SERVER CONFIGURATION ########################
        ##################################################################

        # Specify the CPU Config for each cpu/server in each rack 
        # The full load power and the idle power may be populated using spec sheets from common servers in use
        # This value may be ignored internally if total rack load exceeds MAX_W_PER_RACK


        # CPU Power Parameters
        self.DEFAULT_SERVER_POWER_CHARACTERISTICS = json_obj['server_characteristics']['DEFAULT_SERVER_POWER_CHARACTERISTICS']

        # This list should be of length NUM_RACKS; Here DEFAULT_SERVER_POWER_CHARACTERISTICS is of same length as NUM_RACKS
        assert len(self.DEFAULT_SERVER_POWER_CHARACTERISTICS) == self.NUM_RACKS, "DEFAULT_SERVER_POWER_CHARACTERISTICS should be of length as NUM_RACKS"
        # self.RACK_CPU_CONFIG = [[{'full_load_pwr' : j[0],
                            # 'idle_pwr': j[-1]} for _ in range(int(self.CPUS_PER_RACK))] for j in self.DEFAULT_SERVER_POWER_CHARACTERISTICS]

        # Parallelize the construction of RACK_CPU_CONFIG
        def construct_cpu_config(server_power_characteristics):
            """Function to construct CPU configuration for a single server."""
            return [{'full_load_pwr': j[0], 'idle_pwr': j[-1]} for _ in range(int(self.CPUS_PER_RACK)) for j in server_power_characteristics]
        
        # Use ThreadPoolExecutor to parallelize the operation
        with ThreadPoolExecutor() as executor:
            # Submit tasks to the executor
            futures = [executor.submit(construct_cpu_config, [j]) for j in self.DEFAULT_SERVER_POWER_CHARACTERISTICS]
            
            # Wait for the futures to complete and collect the results
            self.RACK_CPU_CONFIG = [future.result() for future in as_completed(futures)]

        # A default value of HP_PROLIANT server for standalone testing
        self.HP_PROLIANT = json_obj["server_characteristics"]['HP_PROLIANT']

        # Serve/cpu parameters; Obtained from [3]
        self.CPU_POWER_RATIO_LB = json_obj['server_characteristics']['CPU_POWER_RATIO_LB']
        self.CPU_POWER_RATIO_UB = json_obj['server_characteristics']['CPU_POWER_RATIO_UB']
        self.IT_FAN_AIRFLOW_RATIO_LB = json_obj['server_characteristics']['IT_FAN_AIRFLOW_RATIO_LB']
        self.IT_FAN_AIRFLOW_RATIO_UB = json_obj['server_characteristics']['IT_FAN_AIRFLOW_RATIO_UB']
        self.IT_FAN_FULL_LOAD_V = json_obj['server_characteristics']['IT_FAN_FULL_LOAD_V']
        self.ITFAN_REF_V_RATIO, self.ITFAN_REF_P = json_obj['server_characteristics']['ITFAN_REF_V_RATIO'], json_obj['server_characteristics']['ITFAN_REF_P']
        self.INLET_TEMP_RANGE = json_obj['server_characteristics']['INLET_TEMP_RANGE']

        ##################################################################
        #################### HVAC CONFIGURATION ##########################
        ##################################################################

        # Air parameters
        self.C_AIR = json_obj['hvac_configuration']['C_AIR']  # J/kg.K
        self.RHO_AIR = json_obj['hvac_configuration']['RHO_AIR']  # kg/m3

        # CRAC Unit paramters
        self.CRAC_SUPPLY_AIR_FLOW_RATE_pu = json_obj['hvac_configuration']['CRAC_SUPPLY_AIR_FLOW_RATE_pu']
        self.CRAC_REFRENCE_AIR_FLOW_RATE_pu = json_obj['hvac_configuration']['CRAC_REFRENCE_AIR_FLOW_RATE_pu']
        self.CRAC_FAN_REF_P = json_obj['hvac_configuration']['CRAC_FAN_REF_P']

        # Chiller Stats
        self.CHILLER_COP = json_obj['hvac_configuration']['CHILLER_COP']
        self.CW_PRESSURE_DROP = json_obj['hvac_configuration']['CW_PRESSURE_DROP'] #Pa 
        self.CW_WATER_FLOW_RATE = json_obj['hvac_configuration']['CW_WATER_FLOW_RATE'] #m3/s
        self.CW_PUMP_EFFICIENCY = json_obj['hvac_configuration']['CW_PUMP_EFFICIENCY'] #%

        # Cooling Tower parameters
        self.CT_FAN_REF_P = json_obj['hvac_configuration']['CT_FAN_REF_P']
        self.CT_REFRENCE_AIR_FLOW_RATE = json_obj['hvac_configuration']['CT_REFRENCE_AIR_FLOW_RATE']
        self.CT_PRESSURE_DROP = json_obj['hvac_configuration']['CT_PRESSURE_DROP'] #Pa 
        self.CT_WATER_FLOW_RATE = json_obj['hvac_configuration']['CT_WATER_FLOW_RATE']#m3/s
        self.CT_PUMP_EFFICIENCY = json_obj['hvac_configuration']['CT_PUMP_EFFICIENCY'] #%


#References:
#[1]: Postema, Björn Frits. "Energy-efficient data centres: model-based analysis of power-performance trade-offs." (2018).
#[2]: Raghunathan, S., & Vk, M. (2014). Power management using dynamic power state transitions and dynamic voltage frequency scaling controls in virtualized server clusters. Turkish Journal of Electrical Engineering and Computer Sciences, 24(4). doi: 10.3906/elk-1403-264
#[3]: Sun, Kaiyu, et al. "Prototype energy models for data centers." Energy and Buildings 231 (2021): 110603.
#[4]: Breen, Thomas J., et al. "From chip to cooling tower data center modeling: Part I influence of server inlet temperature and temperature rise across cabinet." 2010 12th IEEE Intersociety Conference on Thermal and Thermomechanical Phenomena in Electronic Systems. IEEE, 2010.
#[5]: https://h2ocooling.com/blog/look-cooling-tower-fan-efficiences/#:~:text=The%20tower%20has%20been%20designed,of%200.42%20inches%20of%20water.
