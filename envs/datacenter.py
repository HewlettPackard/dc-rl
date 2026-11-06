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
        # IT fan airflow follows IT_FAN_AIRFLOW_RATIO_LB/UB from the config. These multipliers had been raised to
        # 10 / 5 / 20 (from 1 / 1 / 100), which made server fans 20-70% of IT power and dominated the setpoint
        # trade-off; with the config values fans are ~2.5-12% of IT power.
        self.m_coefficient = 1
        self.c_coefficient = 1
        self.it_slope = 100
            
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
        # CPU power ratio vs inlet temperature follows CPU_POWER_RATIO_LB/UB from the config (+2% of full-load power
        # across INLET_TEMP_RANGE). A hard-coded "+0.05" per C used to be added to the slope here, which made a server
        # draw more than its rated full-load power even when idle and made raising the setpoint far too costly.
        base_cpu_power_ratio = self.m_cpu*inlet_temp + self.c_cpu
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
    
# CRAC setpoint (C) at which `datacenter_capacity_mw` is matched. IT power rises with inlet temperature, and 22 C is the
# highest setpoint evaluated when building the environment's observation ranges (the agent can use 15-21.6 C), so
# full-load IT power never exceeds the configured capacity.
FULL_LOAD_REFERENCE_CRAC_SETPOINT = 22.0


class DataCenter_ITModel():

    def __init__(self, num_racks, rack_supply_approach_temp_list, rack_CPU_config, max_W_per_rack = 10000, DC_ITModel_config = None, chiller_sizing = False) -> None:
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
        
        self.rackwise_inlet_temp = []
        
        for _, CPU_config_list in zip(range(num_racks),self.rack_CPU_config):
            self.racks_list.append(Rack(CPU_config_list, max_W_per_rack = max_W_per_rack, rack_config=self.DC_ITModel_config))
        
        self.total_datacenter_full_load()
        
        self.tower_flow_rate = np.round(DC_ITModel_config.CT_WATER_FLOW_RATE * 3600, 4)  # m³/hr
        self.hot_water_temp = None  # °C
        self.cold_water_temp = None  # °C
        self.wet_bulb_temp = None  # °C
        self.heat_rejected_W = 0.0  # set by the caller before calculate_cooling_tower_water_usage()
        self.cycles_of_concentration = 5
        self.drift_rate = 0.01

        # Scale extensive quantities (IT power, and through it cooling load, HVAC power and water use) so that the
        # data center's full-load IT power equals DC_Config.TOTAM_MAX_PWR (= datacenter_capacity_mw * 1e6 W).
        # Temperatures are computed from the unscaled reference racks, so they do not depend on the capacity.
        self.power_scale = 1.0
        capacity_W = getattr(DC_ITModel_config, 'TOTAM_MAX_PWR', None)
        if capacity_W:
            cpu_pwr, itfan_pwr, _ = self.compute_datacenter_IT_load_outlet_temp(
                ITE_load_pct_list=[100.0] * len(self.racks_list), CRAC_setpoint=FULL_LOAD_REFERENCE_CRAC_SETPOINT)
            self.power_scale = capacity_W / (sum(cpu_pwr) + sum(itfan_pwr))
        
        
        
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
                
        crac_air_delta_T = crac_air_temperature_rise(self.DC_ITModel_config)

        for rack, rack_supply_approach_temp, rack_return_approach_temp, ITE_load_pct \
                                in zip(self.racks_list, self.rack_supply_approach_temp_list,
                                       self.DC_ITModel_config.RACK_RETURN_APPROACH_TEMP_LIST, ITE_load_pct_list):
            #clamp supply approach temperatures
            rack_supply_approach_temp = rack.clamp_supply_approach_temp(rack_supply_approach_temp)
            rack_inlet_temp = rack_supply_approach_temp + CRAC_setpoint 
            rackwise_inlet_temp.append(rack_inlet_temp)
            rack_cpu_power, rack_itfan_power = rack.compute_instantaneous_pwr_vecd(rack_inlet_temp, ITE_load_pct)
            # Max IT Power : 83000
            # Min IT Power : 28000
            rackwise_cpu_pwr.append(rack_cpu_power)
            rackwise_itfan_pwr.append(rack_itfan_power)
            
            # Air-side energy balance. CRAC supply airflow is proportional to IT power (CRAC_SUPPLY_AIR_FLOW_RATE_pu,
            # m3/s per W), so the air carrying the IT heat back to the CRAC warms by a fixed amount:
            # return - setpoint = 1 / (rho * q_pu * c_p). The rack outlet sits above the CRAC return by the rack's
            # return approach. (The previous empirical outlet formula was fitted to the inflated CPU power above and
            # gives negative temperature rises with the configured server power.)
            rack_outlet_temp = CRAC_setpoint + crac_air_delta_T - rack_return_approach_temp
            rackwise_outlet_temp.append(rack_outlet_temp)

        self.rackwise_inlet_temp = rackwise_inlet_temp

        # Only the reported powers are scaled with datacenter_capacity_mw; temperatures do not depend on it.
        rackwise_cpu_pwr = [p * self.power_scale for p in rackwise_cpu_pwr]
        rackwise_itfan_pwr = [p * self.power_scale for p in rackwise_itfan_pwr]

        return rackwise_cpu_pwr, rackwise_itfan_pwr, rackwise_outlet_temp
    
    def total_datacenter_full_load(self,):
        """Calculate the total DC IT(IT CPU POWER + IT FAN POWER) power consumption
        """
        x = [rack_item.get_current_rack_load() for rack_item in self.racks_list]
        self.total_DC_full_load = sum(x)

    def calculate_cooling_tower_water_usage(self):
        """
        Estimate cooling tower water use (litres per 15-minute step) from the heat the tower rejects.

        Evaporation removes the heat (heat_rejected_W / latent heat of vaporization); blowdown keeps the dissolved
        solids at `cycles_of_concentration` (blowdown = evaporation / (cycles - 1)); drift adds `drift_rate` on top.
        `heat_rejected_W` (IT heat + chiller compressor power) must be set before calling.

        (The previous empirical formula took its only load dependence from the CRAC air temperature rise, which is
        constant once the cooling load obeys the energy balance; it also did not scale with heat rejected.)
        """
        latent_heat_J_per_kg = 2.45e6  # water at ~20-25 C
        evaporation_kg_s = max(self.heat_rejected_W, 0.0) / latent_heat_J_per_kg
        blowdown_kg_s = evaporation_kg_s / (self.cycles_of_concentration - 1)
        water_kg_s = (evaporation_kg_s + blowdown_kg_s) * (1 + self.drift_rate)
        return water_kg_s * 15 * 60  # 1 kg of water ~ 1 litre

def crac_air_temperature_rise(DC_Config):
    """CRAC return minus supply air temperature (K) implied by the air-side energy balance.

    CRAC airflow is CRAC_SUPPLY_AIR_FLOW_RATE_pu (m3/s per W of IT power), so removing the IT heat warms the air by
    1 / (RHO_AIR * CRAC_SUPPLY_AIR_FLOW_RATE_pu * C_AIR), independent of load (about 14.3 K with the default config).
    """
    return 1.0 / (DC_Config.RHO_AIR * DC_Config.CRAC_SUPPLY_AIR_FLOW_RATE_pu * DC_Config.C_AIR)


# Water-cooled reference chiller (EnergyPlus Chiller:Electric). All values from one consistent example:
# https://github.com/NREL/EnergyPlus/blob/9bb39b77a871dee7543c892ae53b0812c4c17b0d/tst/EnergyPlus/unit/ChillerElectric.unit.cc#L78
# Model equations: EnergyPlus PlantChillers.cc, ElectricChillerSpecs::calculate (CalcElectricChillerModel).
EPLUS_CHILLER = {
    'nominal_cop': 4.75,
    'min_plr': 0.15,
    'max_plr': 1.0,
    'design_cond_inlet_temp': 29.44,     # C
    'temp_rise_coef': 2.682759,
    'design_chw_outlet_temp': 6.667,     # C
    'chw_outlet_lower_limit': 5.0,       # C
    'capacity_coefs': (0.94483600, -0.05700880, -0.00185486),
    'power_coefs': (1.907846, -1.20498700, 0.26346230),
    'full_load_coefs': (0.03303, 0.6852, 0.2818),
}


def chiller_available_capacity_ratio(cond_inlet_temp, chw_outlet_temp):
    """EnergyPlus AvailNomCapRat: available / nominal capacity at the given condenser inlet and chilled water outlet temps."""
    ch = EPLUS_CHILLER
    delta_temp = (cond_inlet_temp - ch['design_cond_inlet_temp']) / ch['temp_rise_coef'] \
        - (chw_outlet_temp - ch['design_chw_outlet_temp'])
    a, b, c = ch['capacity_coefs']
    return a + b * delta_temp + c * delta_temp ** 2


def chiller_operating_temps(CRAC_setpoint, wet_bulb_temp, DC_Config):
    """Condenser inlet temperature (cooling-tower supply = wet bulb + tower approach) and chilled water outlet
    temperature (CRAC supply air setpoint - coil approach, not below the chiller's lower limit)."""
    cond_inlet_temp = wet_bulb_temp + DC_Config.CT_APPROACH_TEMP
    chw_outlet_temp = max(EPLUS_CHILLER['chw_outlet_lower_limit'], CRAC_setpoint - DC_Config.CHW_SUPPLY_APPROACH_TEMP)
    return cond_inlet_temp, chw_outlet_temp


def calculate_chiller_power(max_cooling_cap, load, cond_inlet_temp, chw_outlet_temp=EPLUS_CHILLER['design_chw_outlet_temp']):
    """Electric chiller power (W) using the EnergyPlus Chiller:Electric model (steady state).

    Args:
        max_cooling_cap (float): Nominal chiller capacity (W).
        load (float): Heat to remove (W).
        cond_inlet_temp (float): Condenser inlet (cooling-tower supply) water temperature (C).
        chw_outlet_temp (float): Chilled water outlet (supply) temperature (C).

    Returns:
        float: Chiller electric power (W). Efficiency falls as the condenser water gets warmer and rises with warmer
        chilled water, as in EnergyPlus. (The previous version subtracted (design chilled water - design condenser)
        instead of (chilled water - design chilled water), shifting the curve input by +28 K, and flipped the sign of
        the third capacity coefficient, which made the chiller more efficient in hot weather.)
    """
    if load <= 0:
        return 0.0
    ch = EPLUS_CHILLER
    avail_nom_cap_rat = chiller_available_capacity_ratio(cond_inlet_temp, chw_outlet_temp)
    available_capacity = max_cooling_cap * avail_nom_cap_rat
    if available_capacity <= 0:
        return 0.0
    p1, p2, p3 = ch['power_coefs']
    full_load_power_rat = p1 + p2 * avail_nom_cap_rat + p3 * avail_nom_cap_rat ** 2
    part_load_ratio = max(ch['min_plr'], min(load / available_capacity, ch['max_plr']))
    f1, f2, f3 = ch['full_load_coefs']
    frac_full_load_power = f1 + f2 * part_load_ratio + f3 * part_load_ratio ** 2
    oper_part_load_rat = load / available_capacity if load / available_capacity < ch['min_plr'] else part_load_ratio
    frac = min(1.0, oper_part_load_rat / ch['min_plr']) if oper_part_load_rat < ch['min_plr'] else 1.0
    return frac_full_load_power * full_load_power_rat * available_capacity / ch['nominal_cop'] * frac

def calculate_HVAC_power(CRAC_setpoint, avg_CRAC_return_temp, ambient_temp, data_center_full_load, DC_Config, ctafr=None,
                         wet_bulb_temp=None):
    """Calculate the HVAC power attributes

    Args:
        CRAC_setpoint (float): The control action (CRAC supply air temperature, C)
        avg_CRAC_return_temp (float): The average of the temperatures from all the Racks + their corresponding return approach temperature (Delta)
        ambient_temp (float): outside air temperature (dry bulb, C)
        data_center_full_load (float): total data center IT load in W
        DC_Config (config): configuration of the data center
        ctafr (float, optional): cooling tower reference air flow rate
        wet_bulb_temp (float, optional): outside wet bulb temperature (C); defaults to ambient_temp (conservative)

    Returns:
        CRAC_Fan_load (float): CRAC fan power
        CT_Fan_pwr (float):  Cooling tower fan power
        CRAC_cooling_load (float): CRAC cooling load (= IT heat)
        chiller_power (float): Chiller compressor load
        power_consumed_CW (float): Chilled water pump power
        power_consumed_CT (float): Cooling tower pump power
    """
    if wet_bulb_temp is None:
        wet_bulb_temp = ambient_temp

    # Steady-state energy balance: all IT electrical power ends up as heat that the CRAC must remove. (The previous
    # rho * q_pu * P_IT * c_p * (return - setpoint) only matched the IT heat near 70% load: it removed about half the
    # heat at idle and ~15% more than produced at full load, which made flattening the load look artificially cheap.)
    CRAC_cooling_load = data_center_full_load
    CRAC_Fan_load = DC_Config.CRAC_FAN_REF_P * (DC_Config.CRAC_SUPPLY_AIR_FLOW_RATE_pu / DC_Config.CRAC_REFRENCE_AIR_FLOW_RATE_pu)**3

    # Chiller: condenser water from the cooling tower, chilled water temperature follows the CRAC setpoint
    cond_inlet_temp, chw_outlet_temp = chiller_operating_temps(CRAC_setpoint, wet_bulb_temp, DC_Config)
    chiller_capacity = getattr(DC_Config, 'CHILLER_NOMINAL_CAP', None) or DC_Config.CT_FAN_REF_P
    chiller_power = calculate_chiller_power(chiller_capacity, CRAC_cooling_load, cond_inlet_temp, chw_outlet_temp)

    # Chilled water pump power calculation
    power_consumed_CW = (DC_Config.CW_PRESSURE_DROP * DC_Config.CW_WATER_FLOW_RATE) / DC_Config.CW_PUMP_EFFICIENCY
    # Cooling tower pump power calculation
    power_consumed_CT = (DC_Config.CT_PRESSURE_DROP*DC_Config.CT_WATER_FLOW_RATE)/DC_Config.CT_PUMP_EFFICIENCY

    if ambient_temp < 5:
        return CRAC_Fan_load, 0.0, CRAC_cooling_load, chiller_power, power_consumed_CW, power_consumed_CT

    # Cooling tower fan power calculations
    Cooling_tower_air_delta = max(50 - (ambient_temp - CRAC_setpoint), 1)
    m_air = CRAC_cooling_load / (DC_Config.C_AIR * Cooling_tower_air_delta)
    v_air = m_air / DC_Config.RHO_AIR

    # Reference cooling tower air flow rate
    if ctafr is None:
        ctafr = DC_Config.CT_REFRENCE_AIR_FLOW_RATE
    CT_Fan_pwr = DC_Config.CT_FAN_REF_P * (min(v_air / ctafr, 1))**3

    return CRAC_Fan_load, CT_Fan_pwr, CRAC_cooling_load, chiller_power, power_consumed_CW, power_consumed_CT

def chiller_sizing(DC_Config, min_CRAC_setpoint=16, max_CRAC_setpoint=22, max_ambient_temp=40.0, max_wet_bulb_temp=32.0):
    '''
    Size the cooling tower and chiller for a data center.

    Parameters:
        DC_Config (object): The data center configuration object.
        min_CRAC_setpoint (float): The minimum CRAC setpoint temperature in degrees Celsius. Default is 16.
        max_CRAC_setpoint (float): The maximum CRAC setpoint temperature in degrees Celsius. Default is 22.
        max_ambient_temp (float): The maximum ambient (dry bulb) temperature in degrees Celsius. Default is 40.0.
        max_wet_bulb_temp (float): The design maximum wet bulb temperature in degrees Celsius. Default is 32.0.

    Returns:
        tuple: cooling tower reference air flow rate (ctafr), rated cooling load (CT_rated_load, W) and chiller nominal
        capacity (W), sized so the chiller can still remove the peak heat at the hottest wet bulb and coldest
        chilled water temperature.
    '''
    dc = DataCenter_ITModel(num_racks=DC_Config.NUM_RACKS,
                            rack_supply_approach_temp_list=DC_Config.RACK_SUPPLY_APPROACH_TEMP_LIST,
                            rack_CPU_config=DC_Config.RACK_CPU_CONFIG,
                            max_W_per_rack=DC_Config.MAX_W_PER_RACK,
                            DC_ITModel_config=DC_Config)

    ITE_load_pct_list = [100.0 for i in range(DC_Config.NUM_RACKS)]
    rackwise_cpu_pwr, rackwise_itfan_pwr, _ = \
                        dc.compute_datacenter_IT_load_outlet_temp(ITE_load_pct_list=ITE_load_pct_list, CRAC_setpoint=max_CRAC_setpoint)

    # Peak heat to remove = peak IT power (energy balance)
    CRAC_cooling_load = sum(rackwise_cpu_pwr) + sum(rackwise_itfan_pwr)

    Cooling_tower_air_delta = max(50 - (max_ambient_temp-min_CRAC_setpoint), 1)
    m_air = CRAC_cooling_load/(DC_Config.C_AIR*Cooling_tower_air_delta)
    # Cooling Tower Reference Air FlowRate
    ctafr = m_air/DC_Config.RHO_AIR
    CT_rated_load = CRAC_cooling_load

    # Chiller nominal capacity: available capacity at the worst conditions must cover the peak heat
    cond_inlet_temp, chw_outlet_temp = chiller_operating_temps(min_CRAC_setpoint, max_wet_bulb_temp, DC_Config)
    worst_ratio = chiller_available_capacity_ratio(cond_inlet_temp, chw_outlet_temp)
    chiller_nominal_cap = CRAC_cooling_load / worst_ratio

    return ctafr, CT_rated_load, chiller_nominal_cap

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