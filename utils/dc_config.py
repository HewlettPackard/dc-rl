"""
This file contains the data center configurations that may be customized by the user for designing the data center. The use of this file has been deprecated. Any changes to this file will not be reflected in the actual data center design. Instead, modify `utils/dc_config.json` to design the data center.
"""
##################################################################
#################### GEOMETRY DEPENDENT PARAMETERS ###############
##################################################################

# Data Center Geometric configuration
NUM_ROWS = 4
NUM_RACKS_PER_ROW = 5
NUM_RACKS = NUM_ROWS * NUM_RACKS_PER_ROW

# CFD may be used to precompute the supply approach temperature for each rack under given
# geometry, containment, CRAC Air flow rate, LOad rate
# 
# Supply approach temperature: It is the delta T i.e. the temperature difference between 
# CRAC_setpoint and the actual inlet temperature to the rack .Its value depends on the geometry
# of the data center rack arrangements and can be pre-computed from CFD analysis. The length of
# the list should be the same as NUM_RACKS; Default values are populated from paper [3] assuming:
# Scenario # 19 from Table 5
TOTAM_MAX_PWR = 2400 * 1e3
MAX_W_PER_RACK = int(TOTAM_MAX_PWR/NUM_RACKS)
# we add some variation to the default values to highlight change in geometry
RACK_SUPPLY_APPROACH_TEMP_LIST = [
    5.3, 5.3, 5.3, 5.3,  # row 1
    5.0, 5.0, 5.0, 5.0,  # row 2
    5.0, 5.0, 5.0, 5.0,  # row 3
    5.3, 5.3, 5.3, 5.3,  # row 4
]
# Return approach temperature: It is the delta T i.e. the temperature difference between 
# CRAC return temperature and the rack Outlet temperature .Its value also depends on the geometry
# of the data center rack arrangements and can be pre-computed from CFD analysis. The length of
# the list should be the same as NUM_RACKS; Default values are populated from paper [3] assuming:
# Scenario # 19 from Table 5
# we add some variation to the default values to highlight change in geometry
RACK_RETURN_APPROACH_TEMP_LIST = [
    -3.7, -3.7, -3.7, -3.7,  # row 1
    -2.5, -2.5, -2.5, -2.5,  # row 2
    -2.5, -2.5, -2.5, -2.5,  # row 3
    -3.7, -3.7, -3.7, -3.7,  # row 4
]

##################################################################
#################### SERVER CONFIGURATION ########################
##################################################################

# Specify the CPU Config for each cpu/server in each rack 
# The full load power and the idle power may be populated using spec sheets from common servers in use
CPUS_PER_RACK = 300  # This value may be overridden internally if total rack load exceeds MAX_W_PER_RACK

# This list should be of length NUM_RACKS
RACK_CPU_CONFIG : list[list[dict]] = [
    [{'full_load_pwr' :170, 'idle_pwr':110} for _ in range(CPUS_PER_RACK)],
    [{'full_load_pwr' :120, 'idle_pwr':60} for _ in range(CPUS_PER_RACK)],
    [{'full_load_pwr' :870, 'idle_pwr':110} for _ in range(CPUS_PER_RACK)],
    [{'full_load_pwr' :650, 'idle_pwr':100} for _ in range(CPUS_PER_RACK)],
    [{'full_load_pwr' :420, 'idle_pwr':130} for _ in range(CPUS_PER_RACK)],
    [{'full_load_pwr' :160, 'idle_pwr':100} for _ in range(CPUS_PER_RACK)],
    [{'full_load_pwr' :300, 'idle_pwr':60} for _ in range(CPUS_PER_RACK)],
    [{'full_load_pwr' :530, 'idle_pwr':80} for _ in range(CPUS_PER_RACK)],
    [{'full_load_pwr' :670, 'idle_pwr':110} for _ in range(CPUS_PER_RACK)],
    [{'full_load_pwr' :660, 'idle_pwr':100} for _ in range(CPUS_PER_RACK)],
    [{'full_load_pwr' :670, 'idle_pwr':110} for _ in range(CPUS_PER_RACK)],
    [{'full_load_pwr' :190, 'idle_pwr':120} for _ in range(CPUS_PER_RACK)],
    [{'full_load_pwr' :170, 'idle_pwr':110} for _ in range(CPUS_PER_RACK)],
    [{'full_load_pwr' :170, 'idle_pwr':110} for _ in range(CPUS_PER_RACK)],
    [{'full_load_pwr' :220, 'idle_pwr':150} for _ in range(CPUS_PER_RACK)],
    [{'full_load_pwr' :280, 'idle_pwr':180} for _ in range(CPUS_PER_RACK)],
    [{'full_load_pwr' :110, 'idle_pwr':70} for _ in range(CPUS_PER_RACK)],
    [{'full_load_pwr' :170, 'idle_pwr':110} for _ in range(CPUS_PER_RACK)],
    [{'full_load_pwr' :170, 'idle_pwr':110} for _ in range(CPUS_PER_RACK)],
    [{'full_load_pwr' :170, 'idle_pwr':110} for _ in range(CPUS_PER_RACK)]
]


# Sample CPU power parameters
HP_PROLIANT = (110,170)  # HP Proliant Server Wattage at 0% and 100% utilization [2]

# Serve/cpu parameters
CPU_POWER_RATIO_LB = (0.22, 1.00)  # [3] taken from the figure
CPU_POWER_RATIO_UB = (0.24, 1.02)  # [3] taken from the figure
IT_FAN_AIRFLOW_RATIO_LB = (0.0,0.6)  # [3] taken from the figure
IT_FAN_AIRFLOW_RATIO_UB = (0.7,1.3)  # [3] taken from the figure
IT_FAN_FULL_LOAD_V = 0.05  # TODO: Have to check these values m3/s
ITFAN_REF_V_RATIO, ITFAN_REF_P = 1.0, 10
INLET_TEMP_RANGE = (18, 27)  # [3]

##################################################################
#################### HVAC CONFIGURATION ##########################
##################################################################

# Air parameters
C_AIR = 1006  # J/kg.K
RHO_AIR = 1.225  # kg/m3

# CRAC Unit paramters
CRAC_SUPPLY_AIR_FLOW_RATE_pu = 120/(2119 * 1e3) #m3/s.w [3] Table 5; fixing AHU supply air flow rate based on geometry; the usual values are in per Kw
CRAC_REFRENCE_AIR_FLOW_RATE_pu = 200/(2119 * 1e3) #m3/s.w [3] Table 5; fixing AHU supply air flow rate based on geometry TODO: Have to update this value
CRAC_FAN_REF_P = 150  # TODO: Have to check these values watt?

# Chiller Stats
CHILLER_COP = 6.0  # Have to check these values;usual values are in (2.5, 3.5) with as high as 7.5 for variable speed high efficiency chillers. Using 6.0 from [4]

# Cooling Tower parameters
CT_FAN_REF_P = 1000 # [5]
CT_REFRENCE_AIR_FLOW_RATE = 6000/2119


#References:
#[1]: Postema, Björn Frits. "Energy-efficient data centres: model-based analysis of power-performance trade-offs." (2018).
#[2]: Raghunathan, S., & Vk, M. (2014). Power management using dynamic power state transitions and dynamic voltage frequency scaling controls in virtualized server clusters. Turkish Journal of Electrical Engineering and Computer Sciences, 24(4). doi: 10.3906/elk-1403-264
#[3]: Sun, Kaiyu, et al. "Prototype energy models for data centers." Energy and Buildings 231 (2021): 110603.
#[4]: Breen, Thomas J., et al. "From chip to cooling tower data center modeling: Part I influence of server inlet temperature and temperature rise across cabinet." 2010 12th IEEE Intersociety Conference on Thermal and Thermomechanical Phenomena in Electronic Systems. IEEE, 2010.
#[5]: https://h2ocooling.com/blog/look-cooling-tower-fan-efficiences/#:~:text=The%20tower%20has%20been%20designed,of%200.42%20inches%20of%20water.
