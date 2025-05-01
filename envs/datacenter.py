import os
import numpy as np

class CPU():

    def __init__(self, full_load_pwr = None, idle_pwr = None, cpu_config = None): 
        """CPU class in charge of the energy consumption and termal calculations of the individuals CPUs
            in a Rack.

        Args:
            full_load_pwr (float, optional): Power at full capacity.
            idle_pwr (float, optional): Power while idle.
            cpu_config (config): Configuration for the DC.
        """
        self.cpu_config = cpu_config
        
        self.full_load_pwr = full_load_pwr  if not None else self.cpu_config.HP_PROLIANT[0]
        self.idle_pwr = idle_pwr if not None else self.cpu_config.HP_PROLIANT[1]

        self.m_cpu = None
        self.c_cpu = None
        self.m_itfan = None
        self.c_itfan = None
        self.cpu_curve1()
        self.itfan_curve2()
        
        self.v_fan = None  # needed later for calculating outlet temperature
        self.itfan_v_ratio_at_inlet_temp = None
        self.total_DC_full_load = None

    def cpu_curve1(self,):
        """
        initialize the  cpu power ratio curve at different IT workload ratios as a function of inlet temperatures [3]
        """
        # curve parameters at lowest ITE utilization 0%
        self.m_cpu  = (self.cpu_config.CPU_POWER_RATIO_UB[0]-self.cpu_config.CPU_POWER_RATIO_LB[0])/(self.cpu_config.INLET_TEMP_RANGE[1]-self.cpu_config.INLET_TEMP_RANGE[0])
        self.c_cpu  = self.cpu_config.CPU_POWER_RATIO_UB[0] - self.m_cpu*self.cpu_config.INLET_TEMP_RANGE[1]
        # max vertical shift in power ratio curve at a given point in inlet temperature for 100% change in ITE input load pct
        self.ratio_shift_max_cpu = self.cpu_config.CPU_POWER_RATIO_LB[1] - self.cpu_config.CPU_POWER_RATIO_LB[0]

    def itfan_curve2(self,):
        """
        initialize the itfan velocity ratio curve at different IT workload ratios as a function of inlet temperatures [3]
        """
        # curve parameters at ITE utilization 25%
        self.m_itfan = (self.cpu_config.IT_FAN_AIRFLOW_RATIO_UB[0]-self.cpu_config.IT_FAN_AIRFLOW_RATIO_LB[0])/(self.cpu_config.INLET_TEMP_RANGE[1]-self.cpu_config.INLET_TEMP_RANGE[0])
        self.c_itfan =  self.cpu_config.IT_FAN_AIRFLOW_RATIO_UB[0] - self.m_itfan*self.cpu_config.INLET_TEMP_RANGE[1]
        # max vertical shift in fan flow ratio curve at a given point for 75% change in ITE input load pct
        self.ratio_shift_max_itfan = self.cpu_config.IT_FAN_AIRFLOW_RATIO_LB[1] - self.cpu_config.IT_FAN_AIRFLOW_RATIO_LB[0]


class Rack():

    def __init__(self, CPU_config_list, max_W_per_rack = 10000, rack_config = None):  # [3] Table 2 Mid-tier data center
        """Defines the rack as a collection of CPUs

        Args:
            CPU_config_list (config): CPU configuration
            max_W_per_rack (int): Maximun power allowed for a whole rack. Defaults to 10000.
            rack_config (config): Rack configuration. Defaults to None.
        """
        
        self.rack_config = rack_config
        
        self.CPU_list = []
        self.current_rack_load = 0
        for CPU_config in CPU_config_list:
            self.CPU_list.append(CPU(full_load_pwr = CPU_config['full_load_pwr'], idle_pwr = CPU_config['idle_pwr'], cpu_config = self.rack_config))
            self.current_rack_load += self.CPU_list[-1].full_load_pwr
            if self.current_rack_load>= max_W_per_rack:
                self.CPU_list.pop()
                break
            
        self.num_CPUs = len(self.CPU_list)
        self.cpu_and_fan_init()
        self.v_fan_rack = None  # will be later pointing to a 1d np array whose shape is the number of cpus in the rack class
  
    def cpu_and_fan_init(self,):
        """
        Initialize the CPU and Fan parameters for the servers in each rack with the specified data center configurations
        """
        
        #common to both cpu and fan 
        inlet_temp_lb,inlet_temp_ub =[],[]
        
        # only for cpu
        m_cpu,c_cpu,ratio_shift_max_cpu,\
            idle_pwr, full_load_pwr = [],[],[],[],[]
            
        # only for it fan
        m_itfan,c_itfan,ratio_shift_max_itfan,\
        ITFAN_REF_P,ITFAN_REF_V_RATIO,IT_FAN_FULL_LOAD_V = \
            [],[],[],[],[],[]
        self.m_coefficient = 10 #1 -> 10 
        self.c_coefficient = 5 #1 -> 5
        self.it_slope = 20 #100 -> 20
            
        for CPU_item in self.CPU_list:
            
            #common to both cpu and fan
            inlet_temp_lb.append(CPU_item.cpu_config.INLET_TEMP_RANGE[0])
            inlet_temp_ub.append(CPU_item.cpu_config.INLET_TEMP_RANGE[1])
            
            # only for cpu
            m_cpu.append(CPU_item.m_cpu)
            c_cpu.append(CPU_item.c_cpu)
            ratio_shift_max_cpu.append(CPU_item.ratio_shift_max_cpu)
            idle_pwr.append(CPU_item.idle_pwr)
            full_load_pwr.append(CPU_item.full_load_pwr)
            
            # only for itfan
            m_itfan.append(CPU_item.m_itfan)
            c_itfan.append(CPU_item.c_itfan)
            ratio_shift_max_itfan.append(CPU_item.ratio_shift_max_itfan)
            ITFAN_REF_P.append(CPU_item.cpu_config.ITFAN_REF_P)
            ITFAN_REF_V_RATIO.append(CPU_item.cpu_config.ITFAN_REF_V_RATIO)
            IT_FAN_FULL_LOAD_V.append(CPU_item.cpu_config.IT_FAN_FULL_LOAD_V)
            
            
        # commont to both cpu and itfan
        self.inlet_temp_lb,self.inlet_temp_ub =np.array(inlet_temp_lb),\
                                    np.array(inlet_temp_ub)
        
        # only for cpu
        self.m_cpu,self.c_cpu,self.ratio_shift_max_cpu,\
            self.idle_pwr, self.full_load_pwr = \
                                    np.array(m_cpu),np.array(c_cpu),\
                                    np.array(ratio_shift_max_cpu),\
                                    np.array(idle_pwr), np.array(full_load_pwr)
        
        # only for itfan
        self.m_itfan,self.c_itfan,self.ratio_shift_max_itfan,\
        self.ITFAN_REF_P,self.ITFAN_REF_V_RATIO,self.IT_FAN_FULL_LOAD_V = \
            np.array(m_itfan),np.array(c_itfan),np.array(ratio_shift_max_itfan),\
            np.array(ITFAN_REF_P),np.array(ITFAN_REF_V_RATIO),np.array(IT_FAN_FULL_LOAD_V) 
            
    def compute_instantaneous_pwr(self,inlet_temp, ITE_load_pct):
        """Calculate the power consumption of the whole rack at the current step

        Args:
            inlet_temp (float): Room temperature
            ITE_load_pct (float): Current CPU usage

        Returns:
            cpu_power (float): Current CPU power usage
        """
        # Assuming that all CPUs have the same parameters
        cpu = self.CPU_list[0]
        tot_cpu_pwr = cpu.compute_instantaneous_cpu_pwr(inlet_temp, ITE_load_pct)*self.num_CPUs

        tot_itfan_pwr = []
        for CPU_item in self.CPU_list:
            tot_itfan_pwr.append(CPU_item.compute_instantaneous_fan_pwr(inlet_temp, ITE_load_pct))

        return tot_cpu_pwr, np.array(tot_itfan_pwr).sum()

    def compute_instantaneous_pwr_vecd(self, inlet_temp, ITE_load_pct):
        """Calculate the power consumption of the whole rack at the current step in a vectorized manner

        Args:
            inlet_temp (float): Room temperature
            ITE_load_pct (float): Current CPU usage
        Returns:
            cpu_power (float): Current CPU power usage
        """
        
        # CPU
        base_cpu_power_ratio = (self.m_cpu+0.05)*inlet_temp + self.c_cpu
        cpu_power_ratio_at_inlet_temp = base_cpu_power_ratio + self.ratio_shift_max_cpu*(ITE_load_pct/100)
        temp_arr = np.concatenate((self.idle_pwr.reshape(1,-1),
                                   (self.full_load_pwr*cpu_power_ratio_at_inlet_temp).reshape(1,-1)),
                                  axis=0)
        cpu_power = np.max(temp_arr,axis=0)
        
        # itfan
        base_itfan_v_ratio = self.m_itfan*self.m_coefficient*inlet_temp + self.c_itfan*self.c_coefficient
        itfan_v_ratio_at_inlet_temp = base_itfan_v_ratio + self.ratio_shift_max_itfan*(ITE_load_pct/self.it_slope)
        itfan_pwr = self.ITFAN_REF_P * (itfan_v_ratio_at_inlet_temp/self.ITFAN_REF_V_RATIO)  # [4] Eqn (3)
        self.v_fan_rack = self.IT_FAN_FULL_LOAD_V*itfan_v_ratio_at_inlet_temp
        
        return np.sum(cpu_power), np.sum(itfan_pwr)

    def get_average_rack_fan_v(self,):
        """Calculate the average fan velocity for each rack

        Returns:
            (float): Average fan flow rate for the rack
        """
            
        return self.v_fan_rack[0]
    
    def get_total_rack_fan_v(self,):
        """Calculate the total fan velocity for each rack

        Returns:
            (float): Total fan flow rate for the rack
        """
        return np.sum(self.v_fan_rack)
    
    
    def get_current_rack_load(self,):
        """Returns the total power consumption of the rack

        Returns:
            float: Total power consumption of the rack
        """
        return self.current_rack_load

    def clamp_supply_approach_temp(self,supply_approach_temperature):
        """Returns the clamped delta/ supply approach temperature between the range [3.8, 5.3]

        Returns:
            float: Supply approach temperature
        """
        return max(3.8, min(supply_approach_temperature, 5.3))
    
class DataCenter_ITModel():

    def __init__(self, num_racks, hru, rack_supply_approach_temp_list, rack_CPU_config, max_W_per_rack = 10000, DC_ITModel_config = None, chiller_sizing = False) -> None:
        """Creates the DC from a giving DC configuration

        Args:
            num_racks (int): Number of racks in the DC
            rack_supply_approach_temp_list (list[float]): models the supply approach temperature for each rack based on geometry and estimated from CFD
            rack_CPU_config (list[list[dict]]): A list of lists where each list is associated with a rack. 
            It is a list of dictionaries with their full load and idle load values in W
            chiller_sizing (bool): Whether to perform Chiller Power Sizing
        """
        self.DC_ITModel_config = DC_ITModel_config
        self.racks_list = []
        self.rack_supply_approach_temp_list = rack_supply_approach_temp_list
        self.rack_CPU_config = rack_CPU_config
        self.hru = hru
        
        self.rackwise_inlet_temp = []
        
        for _, CPU_config_list in zip(range(num_racks),self.rack_CPU_config):
            self.racks_list.append(Rack(CPU_config_list, max_W_per_rack = max_W_per_rack, rack_config=self.DC_ITModel_config))
        
        self.total_datacenter_full_load()
        
        self.tower_flow_rate = np.round(DC_ITModel_config.CT_WATER_FLOW_RATE * 3600, 4)  # m³/hr
        self.hot_water_temp = None  # °C
        self.reduced_hot_water_temp = None # °C
        self.cold_water_temp = None  # °C
        self.wet_bulb_temp = None  # °C
        self.cycles_of_concentration = 5
        self.drift_rate = 0.01
        
        
        
    def compute_datacenter_IT_load_outlet_temp(self,ITE_load_pct_list, CRAC_setpoint):
        
        """Calculate the average outlet temperature of all the racks

        Args:
            ITE_load_pct_list (List[float]): CPU load for each rack
            CRAC_setpoint (float): CRAC setpoint

        Returns:
            rackwise_cpu_pwr (List[float]): Rackwise CPU power usage
            rackwise_itfan_pwr (List[float]):  Rackwise IT fan power usage
            rackwise_outlet_temp (List[float]): Rackwise outlet temperature
        """

        rackwise_cpu_pwr = [] 
        rackwise_itfan_pwr = []
        rackwise_outlet_temp = []
        rackwise_inlet_temp = []
      
        c = 1.918
        d = 1.096
        e = 0.824
        f = 0.526
        g = -14.01
        
        for rack, rack_supply_approach_temp, ITE_load_pct \
                                in zip(self.racks_list, self.rack_supply_approach_temp_list, ITE_load_pct_list):
            #clamp supply approach temperatures
            rack_supply_approach_temp = rack.clamp_supply_approach_temp(rack_supply_approach_temp)
            rack_inlet_temp = rack_supply_approach_temp + CRAC_setpoint 
            rackwise_inlet_temp.append(rack_inlet_temp)
            rack_cpu_power, rack_itfan_power = rack.compute_instantaneous_pwr_vecd(rack_inlet_temp, ITE_load_pct)
            # Max IT Power : 83000
            # Min IT Power : 28000
            rackwise_cpu_pwr.append(rack_cpu_power)
            rackwise_itfan_pwr.append(rack_itfan_power)
            
            power_term = (rack_cpu_power + rack_itfan_power)**d
            
            airflow_term = self.DC_ITModel_config.C_AIR*self.DC_ITModel_config.RHO_AIR*rack.get_total_rack_fan_v()**e * f
            log_term = 0# h * np.log(max(power_term / airflow_term, 1))  # Log term, avoid log(0)
            # rackwise_outlet_temp.append((rack_inlet_temp + (rack_cpu_power+rack_itfan_power)/(self.DC_ITModel_config.C_AIR*self.DC_ITModel_config.RHO_AIR*rack.get_total_rack_fan_v()*a) + b * (rack_cpu_power+rack_itfan_power) - c))
            outlet_temp = rack_inlet_temp + c * power_term / airflow_term + g 
            if outlet_temp > 60:
                print(f'WARNING, the outlet temperature is higher than 60C: {outlet_temp:.3f}')
            
            if outlet_temp - rack_inlet_temp < 2:
                print(f'There is something wrong with the delta calculation because is too small: {outlet_temp - rack_inlet_temp:.3f}')
                print(f'Inlet Temp: {rack_inlet_temp:.3f}, Util: {ITE_load_pct}, ITE Power: {rack_cpu_power+rack_itfan_power:.3f}')
                print(f'Power term: {power_term:.3f}, Airflow term: {airflow_term:.3f}, Log term: {log_term:.3f}')
                print(f'Delta: {c * power_term / airflow_term + g + log_term:.3f}')
                raise Exception("Sorry, no numbers below 2")

            if ITE_load_pct > 95 and CRAC_setpoint < 16.5:
                if outlet_temp - rack_inlet_temp < 2:
                    print(f'There is something wrong with the delta calculation for MAX is too small: {outlet_temp - rack_inlet_temp:.3f}')
                    print(f'Inlet Temp: {rack_inlet_temp:.3f}, ITE Power: {rack_cpu_power+rack_itfan_power:.3f}')
                    print(f'Power term: {power_term:.3f}, Airflow term: {airflow_term:.3f}, Log term: {log_term:.3f}')
                    print(f'Delta: {c * power_term / airflow_term + g + log_term:.3f}')
                    raise Exception("Sorry, no numbers below 2")
            rackwise_outlet_temp.append(outlet_temp)

            # rackwise_outlet_temp.append(
            #                     rack_inlet_temp + \
            #                     (rack_cpu_power+rack_itfan_power)/(self.DC_ITModel_config.C_AIR*self.DC_ITModel_config.RHO_AIR*rack.get_total_rack_fan_v())
            #                     )
        self.rackwise_inlet_temp = rackwise_inlet_temp
            
        return rackwise_cpu_pwr, rackwise_itfan_pwr, rackwise_outlet_temp
    
    def total_datacenter_full_load(self,):
        """Calculate the total DC IT(IT CPU POWER + IT FAN POWER) power consumption
        """
        x = [rack_item.get_current_rack_load() for rack_item in self.racks_list]
        self.total_DC_full_load = sum(x)

    def calculate_cooling_tower_water_usage(self):
        """
        Calculate the estimated water usage of the cooling tower.

        This function uses the attributes set in the class to estimate the water usage based 
        [Sharma, R.K., Shah, A., Bash, C.E., Christian, T., & Patel, C.D. (2009). Water efficiency management in datacenters: Metrics and methodology. 2009 IEEE International Symposium on Sustainable Systems and Technology, 1-6.]
        [Mohammed Shublaq, Ahmad K. Sleiti., (2020).  Experimental analysis of water evaporation losses in cooling towers using filters]
        https://spxcooling.com/water-calculator/
        """
        # We're assuming m³/hr, which is standard
        # Calculations doubles as 'self' variables are passed

        # Calculate the range (difference between hot and cold water temperature)
        range_temp = self.hot_water_temp - self.cold_water_temp
        reduced_range_temp  = self.reduced_hot_water_temp - self.cold_water_temp
        
        y_intercept = 0.3528 * range_temp + 0.101
        reduced_y_intercept = 0.3528 * reduced_range_temp + 0.101
        
        # The water usage estimation formula would need to be derived from the graph you provided.
        norm_water_usage = 0.044 * self.wet_bulb_temp + y_intercept
        reduced_norm_water_usage = 0.044 * self.wet_bulb_temp + reduced_y_intercept
        
        water_usage = np.clip(norm_water_usage, 0, None)
        reduced_water_usage = np.clip(reduced_norm_water_usage, 0, None)
        
        water_usage += water_usage * self.drift_rate  # adjust for drift
        reduced_water_usage += reduced_water_usage * self.drift_rate  # adjust for drift


        # Convert m³/hr to the desired unit (e.g., liters per 15 minutes) if necessary
        # There are 1000 liters in a cubic meter. There are 4 15-minute intervals in an hour.
        water_usage_liters_per_15min = np.round((water_usage * 1000) / 4, 4)
        reduced_water_usage_liters_per_15min = np.round((reduced_water_usage * 1000) / 4, 4)
        water_savings_liters_per_15min = water_usage_liters_per_15min - reduced_water_usage_liters_per_15min

        return water_usage_liters_per_15min, reduced_water_usage_liters_per_15min, water_savings_liters_per_15min


def calculate_chiller_power(max_cooling_cap, load, ambient_temp):
    """
    Calculate the chiller power consumption based on load and operating conditions.
    
    Obtained from:
        1) https://github.com/NREL/EnergyPlus/blob/9bb39b77a871dee7543c892ae53b0812c4c17b0d/testfiles/AirCooledElectricChiller.idf
        2) https://github.com/NREL/EnergyPlus/issues/763
        3) https://dmey.github.io/EnergyPlusFortran-Reference/proc/calcelectricchillermodel.html
        4) https://github.com/NREL/EnergyPlus/blob/9bb39b77a871dee7543c892ae53b0812c4c17b0d/tst/EnergyPlus/unit/ChillerElectric.unit.cc#L95

    Args:
        max_cooling_cap (float): Maximum cooling capacity of the chiller (Watts).
        load (float): The heat load to be removed by the chiller (Watts).
        ambient_temp (float): Current ambient temperature (Celsius).
        
    Returns:
        float: Estimated power consumption of the chiller (Watts).
    """
    
    # Coefficients from https://github.com/NREL/EnergyPlus/blob/9bb39b77a871dee7543c892ae53b0812c4c17b0d/tst/EnergyPlus/unit/ChillerElectric.unit.cc#L95
    capacity_coefficients = [0.94483600, -0.05700880, 0.00185486]
    power_coefficients = [2.333, -1.975, 0.6121]
    full_load_factor = [0.03303, 0.6852, 0.2818]

    # Chiller design specifications
    min_plr = 0.05
    max_plr = 1.0
    design_cond_temp = 35.0
    design_evp_out_temp = 6.67
    chiller_nominal_cap = max_cooling_cap
    temp_rise_coef = 2.778
    rated_cop = 3.0
    
    # Calculate the delta temperature for capacity adjustment
    delta_temp = (ambient_temp - design_cond_temp)/temp_rise_coef - (design_evp_out_temp - design_cond_temp)
    
    # Calculate available nominal capacity ratio
    avail_nom_cap_rat = capacity_coefficients[0] + capacity_coefficients[1] * delta_temp + capacity_coefficients[2] * delta_temp**2
    
    # Calculate available chiller capacity
    available_capacity = chiller_nominal_cap * avail_nom_cap_rat if avail_nom_cap_rat != 0 else 0

    # Calculate power ratio
    full_power_ratio = power_coefficients[0] + power_coefficients[1] * avail_nom_cap_rat + power_coefficients[2] * avail_nom_cap_rat**2
    
    # Determine part load ratio (PLR)
    part_load_ratio = max(min_plr, min(load / available_capacity, max_plr)) if available_capacity > 0 else 0
    
    # Calculate fractional full load power
    frac_full_load_power = full_load_factor[0] + full_load_factor[1] * part_load_ratio + full_load_factor[2] * part_load_ratio**2
    
    # Determine operational part load ratio (OperPartLoadRat)
    # If the PLR is less than Min PLR calculate the actual PLR for calculations. The power will then adjust for the cycling.
    if available_capacity > 0:
        if load / available_capacity < min_plr:
            oper_part_load_rat = load / available_capacity
        else:
            oper_part_load_rat = part_load_ratio
    else:
        oper_part_load_rat = 0.0
    
    # Operational PLR for actual conditions
    if oper_part_load_rat < min_plr:
        frac = min(1.0, oper_part_load_rat / min_plr)
    else:
        frac = 1.0

    # Calculate the chiller compressor power
    power = frac_full_load_power * full_power_ratio * available_capacity / rated_cop * frac

    #  Total heat rejection is the sum of the cooling capacity and the power input
    total_heat_rejection = load + power

    return power if oper_part_load_rat > 0 else 0


def calculate_HVAC_power(CRAC_setpoint, hru, avg_CRAC_return_temp, ambient_temp, data_center_full_load, DC_Config, max_IT_load, ctafr=None):
    """Calculate the HVAV power attributes

        Args:
            CRAC_Setpoint (float): The control action
            avg_CRAC_return_temp (float): The average of the temperatures from all the Racks + their corresponding return approach temperature (Delta)
            ambient_temp (float): outside air temperature
            data_center_full_load (float): total data center capacity

        Returns:
            CRAC_Fan_load (float): CRAC fan power
            CT_Fan_pwr (float):  Cooling tower fan power
            CRAC_cooling_load (float): CRAC cooling load
            Compressor_load (float): Chiller compressor load - called chiller power
            CRAC_cooling_load_reduction (float): Reduction in cooling requirements due to heat recovery
        """
    # Air system calculations
    m_sys = DC_Config.RHO_AIR * DC_Config.CRAC_SUPPLY_AIR_FLOW_RATE_pu * data_center_full_load
    CRAC_cooling_load = m_sys * DC_Config.C_AIR * max(0.0, avg_CRAC_return_temp - CRAC_setpoint) # coo.Q_thistime
    CRAC_Fan_load = DC_Config.CRAC_FAN_REF_P * (DC_Config.CRAC_SUPPLY_AIR_FLOW_RATE_pu / DC_Config.CRAC_REFRENCE_AIR_FLOW_RATE_pu)**3

    #Call heat recovery code to determine how much CRAC_cooling_load changes due to heat transferred to DC office and nearby office building
    if hru:
        CRAC_cooling_load_reduction = min(heat_recovery(max_IT_load, ambient_temp, DC_Config), 0.25* data_center_full_load)
    else:
        CRAC_cooling_load_reduction = 0
    #print(heat_recovery(max_IT_load, ambient_temp, DC_Config), "What it want to recover")
    #print(CRAC_cooling_load_reduction, "WHAT IT IS ALLOWED TO RECOVER")
    #print(data_center_full_load, "ITE LOAD ATM")
    CRAC_cooling_required = CRAC_cooling_load - CRAC_cooling_load_reduction 

    #How was the cooling towers max capacity picked for this, why are the variables from the config files different to the calculate_chiller_power? 
    chiller_power = calculate_chiller_power(DC_Config.CT_FAN_REF_P, CRAC_cooling_required, ambient_temp)

    # Pumping Chiller power calculation
    power_consumed_CW = (DC_Config.CW_PRESSURE_DROP * DC_Config.CW_WATER_FLOW_RATE) / DC_Config.CW_PUMP_EFFICIENCY

    # Chilled water pump power calculation for Cooling Tower
    power_consumed_CT = (DC_Config.CT_PRESSURE_DROP*DC_Config.CT_WATER_FLOW_RATE)/DC_Config.CT_PUMP_EFFICIENCY

    if ambient_temp < 5:
        return CRAC_Fan_load, 0.0, CRAC_cooling_required, chiller_power, power_consumed_CW, power_consumed_CT

    # Cooling tower fan power calculations
    Cooling_tower_air_delta = max(50 - (ambient_temp - CRAC_setpoint), 1)
    m_air = CRAC_cooling_required / (DC_Config.C_AIR * Cooling_tower_air_delta)
    v_air = m_air / DC_Config.RHO_AIR
    
    # Reference cooling tower air flow rate
    if ctafr is None:
        ctafr = DC_Config.CT_REFRENCE_AIR_FLOW_RATE
    CT_Fan_pwr = DC_Config.CT_FAN_REF_P * (min(v_air / ctafr, 1))**3
    
    # ToDo: exploring the new chiller_power method
    return CRAC_Fan_load, CT_Fan_pwr, CRAC_cooling_required, chiller_power, power_consumed_CW, power_consumed_CT, CRAC_cooling_load_reduction

def chiller_sizing(DC_Config, min_CRAC_setpoint=16, max_CRAC_setpoint=22, max_ambient_temp=40.0):
    '''
    Calculates the chiller sizing for a data center based on the given configuration and parameters.
    
    Parameters:
        DC_Config (object): The data center configuration object.
        min_CRAC_setpoint (float): The minimum CRAC setpoint temperature in degrees Celsius. Default is 16.
        max_CRAC_setpoint (float): The maximum CRAC setpoint temperature in degrees Celsius. Default is 22.
        max_ambient_temp (float): The maximum ambient temperature in degrees Celsius. Default is 40.0.
    
    Returns:
        tuple: A tuple containing the cooling tower reference air flow rate (ctafr) and the rated load of the cooling tower (CT_rated_load).
    '''
    dc = DataCenter_ITModel(num_racks=DC_Config.NUM_RACKS,
                            hru=False,
                            rack_supply_approach_temp_list=DC_Config.RACK_SUPPLY_APPROACH_TEMP_LIST,
                            rack_CPU_config=DC_Config.RACK_CPU_CONFIG,
                            max_W_per_rack=DC_Config.MAX_W_PER_RACK,
                            DC_ITModel_config=DC_Config)
    
    cpu_load = 100.0
    ITE_load_pct_list = [cpu_load for i in range(DC_Config.NUM_RACKS)]
        
    rackwise_cpu_pwr, rackwise_itfan_pwr, rackwise_outlet_temp = \
                        dc.compute_datacenter_IT_load_outlet_temp(ITE_load_pct_list=ITE_load_pct_list, CRAC_setpoint=max_CRAC_setpoint)
                        
    avg_CRAC_return_temp = calculate_avg_CRAC_return_temp(rack_return_approach_temp_list=DC_Config.RACK_RETURN_APPROACH_TEMP_LIST,
                                                                         rackwise_outlet_temp=rackwise_outlet_temp)
    
    data_center_total_ITE_Load = sum(rackwise_cpu_pwr) + sum(rackwise_itfan_pwr)
    
    m_sys = DC_Config.RHO_AIR * DC_Config.CRAC_SUPPLY_AIR_FLOW_RATE_pu * data_center_total_ITE_Load
    
    CRAC_cooling_load = m_sys*DC_Config.C_AIR*max(0.0, avg_CRAC_return_temp-min_CRAC_setpoint) 
    #Do not understand how this delta is determine or why, seems to set the temperature delta at 26K, where does the 50 come from?)
    Cooling_tower_air_delta = max(50 - (max_ambient_temp-min_CRAC_setpoint), 1)  
    
    #Should the cooling tower load not also account for the added heat caused by the HVAC/Chiller?
    m_air = CRAC_cooling_load/(DC_Config.C_AIR*Cooling_tower_air_delta) 
    
    # Cooling Tower Reference Air FlowRate
    ctafr = m_air/DC_Config.RHO_AIR
    
    '''
    # Assuming that the HVAC consumes 43% of energy on a datacenter, and the rest is IT
    # We assume that on months where temperature is lower than 15C, the HVAC energy consumption is almost 0. (> 60% of time)
    # Also, the HVAC energy consumption follow the external temperature trend (almost a sin signal).
    # The average area for a sin period of a sin is 2*pi
    # The average power consumption of a day where the maximum temperature is raised and the lowest setpoint is used (MaxPower) is 2*pi*MaxPower [W]
    # The average energy for that day is (2*pi*MaxPower)[W]/24[hours] -> 0.26 * MAxPower [W/h]
    # So, to obtain an average HVAC energy consumption of 43%, we need to scale the total maximum energy consumption with a factor of 10
    # This value is obtained after a methodic literature search.
    '''
    #The cooling tower should also be accounting for the heat generated by the CRAC cooler which depends on COP
    CT_rated_load = CRAC_cooling_load #2 * data_center_total_ITE_Load * (43/(100-43))
    
    return ctafr, CT_rated_load, data_center_total_ITE_Load
    
def calculate_avg_CRAC_return_temp(rack_return_approach_temp_list,rackwise_outlet_temp):   
    """Calculate the CRAC return air temperature

        Args:
            rack_return_approach_temp_list (List[float]): The delta change in temperature from each rack to the CRAC unit
            rackwise_outlet_temp (float): The outlet temperature of each rack
        Returns:
            (float): CRAC return air temperature
        """
    n = len(rack_return_approach_temp_list)
    return sum([i + j for i,j in zip(rack_return_approach_temp_list,rackwise_outlet_temp)])/n  # CRAC return is averaged across racks

def heat_recovery(max_IT_load, ambient_temp, DC_Config):
    temperature_delta = max((DC_Config.OFFICE_GUIDE_TEMP - ambient_temp), 0) #don't allow negative numbers, heat can't be lost to outside if it is warmer than inside
    DC_office_heat = DC_Config.AVE_HLP * DC_Config.DC_AREA_PU * max_IT_load *  temperature_delta
    
    #Some heat can be sent to an office building immediatly next to the DC, this is not a District heating network model
    office_heat = DC_Config.AVE_HLP *DC_Config.OFFICE_BUILDING_AREA * temperature_delta
    #print("************************")
    #print(DC_office_heat, "DC HEAT SAVED")
    #print(office_heat, "OFFICE HEAT SAVED")
    #print("________________________")
    cooling_load_reduction = DC_office_heat + office_heat

    return cooling_load_reduction

"""
References:
[1]: Postema, Björn Frits. "Energy-efficient data centres: model-based analysis of power-performance trade-offs." (2018).
[2]: Raghunathan, S., & Vk, M. (2014). Power management using dynamic power state transitions and dynamic voltage frequency
     scaling controls in virtualized server clusters. Turkish Journal of Electrical Engineering and Computer Sciences, 24(4). doi: 10.3906/elk-1403-264
[3]: Sun, Kaiyu, et al. "Prototype energy models for data centers." Energy and Buildings 231 (2021): 110603.
[4]: Breen, Thomas J., et al. "From chip to cooling tower data center modeling: Part I influence of server inlet temperature and temperature 
     rise across cabinet." 2010 12th IEEE Intersociety Conference on Thermal and Thermomechanical Phenomena in Electronic Systems. IEEE, 2010.
[5]: https://h2ocooling.com/blog/look-cooling-tower-fan-efficiences/#:~:text=The%20tower%20has%20been%20designed,of%200.42%20inches%20of%20water.
"""