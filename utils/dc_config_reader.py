"""
This file is used to read the data center configuration from  user inputs provided inside dc_config.json. It also performs some auxiliary steps to calculate the server power specifications based on the given parameters.
"""

import json

import os

file_path = os.path.abspath(__file__)
PATH = os.path.split(os.path.dirname(file_path))[0]

use_config = "dc_config.json"  # "dc_config_per_cpu.json" use this for custom cpu config

##################################################################
#################### READ DATA CENTER CONFIGURATIONS #############
##################################################################
with open(PATH + '/utils/' + use_config, 'r') as myfile:
    data=myfile.read()
json_obj = json.loads(data)

##################################################################
#################### GEOMETRY DEPENDENT PARAMETERS ###############
##################################################################

# Data Center Geometric configuration
NUM_ROWS = json_obj['data_center_configuration']['NUM_ROWS']  # number of rows in which data centers are arranged
NUM_RACKS_PER_ROW = json_obj['data_center_configuration']['NUM_RACKS_PER_ROW']  # number of racks/ITcabinets in each row
NUM_RACKS = NUM_ROWS * NUM_RACKS_PER_ROW  # calculate total number of racks/ITcabinets in the data center model

TOTAM_MAX_PWR = 2400 * 1e3  # specify maximum allowed power consumption for the data center
MAX_W_PER_RACK = int(TOTAM_MAX_PWR/NUM_RACKS)  # calculate maximum allowable power consumption for each rack/ITcabinet


# CFD may be used to precompute the "supply/return approach temperature" for each rack under given
# geometry, containment, CRAC Air flow rate, Load
 
# Supply approach temperature: It is the delta T i.e. the temperature difference between 
# CRAC_setpoint and the actual inlet temperature to the rack .Its value depends on the geometry
# of the data center rack arrangements and can be pre-computed from CFD analysis. The length of
# the list should be the same as NUM_RACKS; Default values are populated from paper [3] assuming:
# Scenario # 19 from Table 5
RACK_SUPPLY_APPROACH_TEMP_LIST = json_obj['data_center_configuration']['RACK_SUPPLY_APPROACH_TEMP_LIST']

# Return approach temperature: It is the delta T i.e. the temperature difference between 
# CRAC return temperature and the rack Outlet temperature .Its value also depends on the geometry
# of the data center rack arrangements and can be pre-computed from CFD analysis. The length of
# the list should be the same as NUM_RACKS; Default values are populated from paper [3] assuming:
# Scenario # 19 from Table 5
# we add some variation to the default values to highlight change in geometry
RACK_RETURN_APPROACH_TEMP_LIST = json_obj['data_center_configuration']['RACK_RETURN_APPROACH_TEMP_LIST']

# how many servers are assigned in each rack. The actual number of servers per rack may be limited while
if use_config == "dc_config.json":
    CPUS_PER_RACK = json_obj['data_center_configuration']['CPUS_PER_RACK']  

##################################################################
#################### SERVER CONFIGURATION ########################
##################################################################

# Specify the CPU Config for each cpu/server in each rack 
# The full load power and the idle power may be populated using spec sheets from common servers in use
# This value may be ignored internally if total rack load exceeds MAX_W_PER_RACK


# CPU Power Parameters
DEFAULT_SERVER_POWER_CHARACTERISTICS = json_obj['server_characteristics']['DEFAULT_SERVER_POWER_CHARACTERISTICS']

# This list should be of length NUM_RACKS; Here DEFAULT_SERVER_POWER_CHARACTERISTICS is of same length as NUM_RACKS
assert len(DEFAULT_SERVER_POWER_CHARACTERISTICS) == NUM_RACKS, "DEFAULT_SERVER_POWER_CHARACTERISTICS should be of length as NUM_RACKS"
if use_config == "dc_config_per_cpu.json":
    RACK_CPU_CONFIG = DEFAULT_SERVER_POWER_CHARACTERISTICS
else:
    RACK_CPU_CONFIG = [[{'full_load_pwr' : j[0],
                     'idle_pwr': j[-1]} for _ in range(CPUS_PER_RACK)] for j in DEFAULT_SERVER_POWER_CHARACTERISTICS]

# A default value of HP_PROLIANT server for standalone testing
HP_PROLIANT = json_obj["server_characteristics"]['HP_PROLIANT']

# Serve/cpu parameters; Obtained from [3]
CPU_POWER_RATIO_LB = json_obj['server_characteristics']['CPU_POWER_RATIO_LB']
CPU_POWER_RATIO_UB = json_obj['server_characteristics']['CPU_POWER_RATIO_UB']
IT_FAN_AIRFLOW_RATIO_LB = json_obj['server_characteristics']['IT_FAN_AIRFLOW_RATIO_LB']
IT_FAN_AIRFLOW_RATIO_UB = json_obj['server_characteristics']['IT_FAN_AIRFLOW_RATIO_UB']
IT_FAN_FULL_LOAD_V = json_obj['server_characteristics']['IT_FAN_FULL_LOAD_V']
ITFAN_REF_V_RATIO, ITFAN_REF_P = json_obj['server_characteristics']['ITFAN_REF_V_RATIO'], json_obj['server_characteristics']['ITFAN_REF_P']
INLET_TEMP_RANGE = json_obj['server_characteristics']['INLET_TEMP_RANGE']

##################################################################
#################### HVAC CONFIGURATION ##########################
##################################################################

# Air parameters
C_AIR = json_obj['hvac_configuration']['C_AIR']  # J/kg.K
RHO_AIR = json_obj['hvac_configuration']['RHO_AIR']  # kg/m3

# CRAC Unit paramters
CRAC_SUPPLY_AIR_FLOW_RATE_pu = json_obj['hvac_configuration']['CRAC_SUPPLY_AIR_FLOW_RATE_pu']
CRAC_REFRENCE_AIR_FLOW_RATE_pu = json_obj['hvac_configuration']['CRAC_REFRENCE_AIR_FLOW_RATE_pu']
CRAC_FAN_REF_P = json_obj['hvac_configuration']['CRAC_FAN_REF_P']

# Chiller Stats
CHILLER_COP = json_obj['hvac_configuration']['CHILLER_COP']
CW_PRESSURE_DROP = json_obj['hvac_configuration']['CW_PRESSURE_DROP'] #Pa 
CW_WATER_FLOW_RATE = json_obj['hvac_configuration']['CW_WATER_FLOW_RATE'] #m3/s
CW_PUMP_EFFICIENCY = json_obj['hvac_configuration']['CW_PUMP_EFFICIENCY'] #%

# Cooling Tower parameters
CT_FAN_REF_P = json_obj['hvac_configuration']['CT_FAN_REF_P']
CT_REFRENCE_AIR_FLOW_RATE = json_obj['hvac_configuration']['CT_REFRENCE_AIR_FLOW_RATE']
CT_PRESSURE_DROP = json_obj['hvac_configuration']['CT_PRESSURE_DROP'] #Pa 
CT_WATER_FLOW_RATE = json_obj['hvac_configuration']['CT_WATER_FLOW_RATE']#m3/s
CT_PUMP_EFFICIENCY = json_obj['hvac_configuration']['CT_PUMP_EFFICIENCY'] #%


#References:
#[1]: Postema, Björn Frits. "Energy-efficient data centres: model-based analysis of power-performance trade-offs." (2018).
#[2]: Raghunathan, S., & Vk, M. (2014). Power management using dynamic power state transitions and dynamic voltage frequency scaling controls in virtualized server clusters. Turkish Journal of Electrical Engineering and Computer Sciences, 24(4). doi: 10.3906/elk-1403-264
#[3]: Sun, Kaiyu, et al. "Prototype energy models for data centers." Energy and Buildings 231 (2021): 110603.
#[4]: Breen, Thomas J., et al. "From chip to cooling tower data center modeling: Part I influence of server inlet temperature and temperature rise across cabinet." 2010 12th IEEE Intersociety Conference on Thermal and Thermomechanical Phenomena in Electronic Systems. IEEE, 2010.
#[5]: https://h2ocooling.com/blog/look-cooling-tower-fan-efficiences/#:~:text=The%20tower%20has%20been%20designed,of%200.42%20inches%20of%20water.
