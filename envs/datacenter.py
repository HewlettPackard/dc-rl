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
        self.total_DC_full_load = None

    def cpu_curve1(self,):
        """
        initialize the  cpu power ratio curve at different IT workload ratios as a function of inlet temperatures [3]
        """
        # curve parameters at lowest temperature
        self.m_cpu  = (self.cpu_config.CPU_POWER_RATIO_UB[0]-self.cpu_config.CPU_POWER_RATIO_LB[0])/(self.cpu_config.INLET_TEMP_RANGE[1]-self.cpu_config.INLET_TEMP_RANGE[0])
        self.c_cpu  = self.cpu_config.CPU_POWER_RATIO_UB[0] - self.m_cpu*self.cpu_config.INLET_TEMP_RANGE[1]
        # max vertical shift in power ratio curve at a given point for 100% change in ITE input load pct
        self.ratio_shift_max_cpu = self.cpu_config.CPU_POWER_RATIO_LB[1] - self.cpu_config.CPU_POWER_RATIO_LB[0]

    def itfan_curve2(self,):
        """
        initialize the itfan velocity ratio curve at different IT workload ratios as a function of inlet temperatures [3]
        """
        # curve parameters at lowest temperature
        self.m_itfan = (self.cpu_config.IT_FAN_AIRFLOW_RATIO_UB[0]-self.cpu_config.IT_FAN_AIRFLOW_RATIO_LB[0])/(self.cpu_config.INLET_TEMP_RANGE[1]-self.cpu_config.INLET_TEMP_RANGE[0])
        self.c_itfan =  self.cpu_config.IT_FAN_AIRFLOW_RATIO_UB[0] - self.m_itfan*self.cpu_config.INLET_TEMP_RANGE[1]
        # max vertical shift in power ratio curve at a given point for 100% change in ITE input load pct
        self.ratio_shift_max_itfan = self.cpu_config.IT_FAN_AIRFLOW_RATIO_LB[1] - self.cpu_config.IT_FAN_AIRFLOW_RATIO_LB[0]      

    def compute_instantaneous_cpu_pwr(self, inlet_temp, ITE_load_pct):
        """Calculate the power consumption of the CPUs at the current step

        Args:
            inlet_temp (float): Room temperature
            ITE_load_pct (float): Current CPU usage

        Returns:
            cpu_power (float): Current CPU power usage
        """
        assert ((inlet_temp>self.cpu_config.INLET_TEMP_RANGE[0]) & (inlet_temp<self.cpu_config.INLET_TEMP_RANGE[1])), f"Server Inlet Temp Outside 18C - 27C range. Current Val {inlet_temp}"
        base_cpu_power_ratio = (self.m_cpu+0.05)*inlet_temp + self.c_cpu
        cpu_power_ratio_at_inlet_temp = base_cpu_power_ratio + self.ratio_shift_max_cpu*(ITE_load_pct/100)
        cpu_power = max(self.idle_pwr, self.full_load_pwr*cpu_power_ratio_at_inlet_temp)

        return cpu_power
    
    def compute_instantaneous_fan_pwr(self, inlet_temp, ITE_load_pct):
        """Calculate the power consumption of the Fans of the IT equipment at the current step

        Args:
            inlet_temp (float): Room temperature
            ITE_load_pct (float): Current CPU usage

        Returns:
            cpu_power (float): Current Fans power usage
        """
        assert ((inlet_temp>self.cpu_config.INLET_TEMP_RANGE[0]) & (inlet_temp<self.cpu_config.INLET_TEMP_RANGE[1])), f"Server Inlet Temp Outside 18C - 27C range. Current Val {inlet_temp}"
        base_itfan_v_ratio = self.m_itfan*inlet_temp + self.c_itfan
        itfan_v_ratio_at_inlet_temp = base_itfan_v_ratio + self.ratio_shift_max_itfan*(ITE_load_pct/100)
        itfan_pwr = self.cpu_config.ITFAN_REF_P * (itfan_v_ratio_at_inlet_temp/self.cpu_config.ITFAN_REF_V_RATIO)**3  # [4] Eqn (3)
        
        self.v_fan = self.cpu_config.IT_FAN_FULL_LOAD_V*itfan_v_ratio_at_inlet_temp

        return itfan_pwr
    
class Rack():

    def __init__(self, CPU_config_list, max_W_per_rack = 10000,rack_config = None):  # [3] Table 2 Mid-tier data center
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
        
    def compute_instantaneous_pwr(self,inlet_temp, ITE_load_pct):
        """Calculate the power consumption of the whole rack at the current step

        Args:
            inlet_temp (float): Room temperature
            ITE_load_pct (float): Current CPU usage

        Returns:
            cpu_power (float): Current CPU power usage
        """
        tot_cpu_pwr = []
        tot_itfan_pwr = []
        for CPU_item in self.CPU_list:
            tot_cpu_pwr.append(CPU_item.compute_instantaneous_cpu_pwr(inlet_temp, ITE_load_pct))
            tot_itfan_pwr.append(CPU_item.compute_instantaneous_fan_pwr(inlet_temp, ITE_load_pct))

        return np.array(tot_cpu_pwr).sum(), np.array(tot_itfan_pwr).sum()
    
    def get_average_rack_fan_v(self,):
        fan_v = []
        for cpu_item in self.CPU_list:
            fan_v.append(cpu_item.v_fan)
            
        return np.array(fan_v).sum()
    
    def get_current_rack_load(self,):
        return self.current_rack_load
    
class DataCenter_ITModel():

    def __init__(self, num_racks, rack_supply_approach_temp_list, rack_CPU_config, max_W_per_rack = 10000, DC_ITModel_config = None) -> None:
        """
        Args:
            num_racks (int): _description_
            rack_supply_approach_temp_list (list[float]): models the supply approach temperature for each rack based on geometry and estimated from CFD
            rack_CPU_config (list[list[dict]]): A list of lists where each list is associated with a rack. 
            It is a list of dictionaries with their full load and idle load values in W
        """
        self.DC_ITModel_config = DC_ITModel_config
        self.racks_list = []
        self.rack_supply_approach_temp_list = rack_supply_approach_temp_list
        self.rack_CPU_config = rack_CPU_config
        
        for _, CPU_config_list in zip(range(num_racks),self.rack_CPU_config):
            self.racks_list.append(Rack(CPU_config_list, max_W_per_rack = max_W_per_rack, rack_config=self.DC_ITModel_config))
        
        self.total_datacenter_full_load()
        
        
        
            
    def compute_datacenter_IT_load_outlet_temp(self,ITE_load_pct_list, CRAC_setpoint):  # the outward facing method that will be called repeatedly
        
        rackwise_cpu_pwr = [] 
        rackwise_itfan_pwr = []
        rackwise_outlet_temp = []
        
        for rack, rack_supply_approach_temp, ITE_load_pct \
                                in zip(self.racks_list, self.rack_supply_approach_temp_list, ITE_load_pct_list):
            
            rack_inlet_temp = rack_supply_approach_temp + CRAC_setpoint
            rack_cpu_power, rack_itfan_power = rack.compute_instantaneous_pwr(rack_inlet_temp,ITE_load_pct)
            rackwise_cpu_pwr.append(rack_cpu_power)
            rackwise_itfan_pwr.append(rack_itfan_power)
            rackwise_outlet_temp.append((rack_inlet_temp + (rack_cpu_power+rack_itfan_power)/(self.DC_ITModel_config.C_AIR*self.DC_ITModel_config.RHO_AIR*rack.get_average_rack_fan_v()))*0.75)
            
        return rackwise_cpu_pwr, rackwise_itfan_pwr, rackwise_outlet_temp
    
    def total_datacenter_full_load(self,):
        x = [rack_item.get_current_rack_load() for rack_item in self.racks_list]
        self.total_DC_full_load = sum(x)


def calculate_HVAC_power(CRAC_setpoint, avg_CRAC_return_temp, ambient_temp,data_center_full_load, DC_Config):
    """
    CRAC_Setpoint : The control action
    avg_CRAC_return_temp : The average of the temperatures from all the Racks + their corresponding return approach temperature (Delta)
    ambient_temp : outside air temperature
    data_center_full_load : total data center capacity
    """
    
    m_sys = DC_Config.RHO_AIR * DC_Config.CRAC_SUPPLY_AIR_FLOW_RATE_pu * data_center_full_load
    CRAC_cooling_load = m_sys*DC_Config.C_AIR*(avg_CRAC_return_temp-CRAC_setpoint)  # [3] Eqn (5); unit is in watt ie J/s
    CRAC_Fan_load = DC_Config.CRAC_FAN_REF_P*(DC_Config.CRAC_SUPPLY_AIR_FLOW_RATE_pu/DC_Config.CRAC_REFRENCE_AIR_FLOW_RATE_pu)**3  # [4]
    
    Compressor_load = CRAC_cooling_load/DC_Config.CHILLER_COP  # [4]
    
    # air mass flow rate of cooling tower; we are assuming fixed delta T for air heating at the cooling tower
    if ambient_temp < 5:
        return CRAC_Fan_load, 0.0, CRAC_cooling_load, Compressor_load
    Cooling_tower_air_delta = max(50 - (ambient_temp-CRAC_setpoint), 5)  # TODO: See how we may modify this to make it a more correct function of ambient temperature
    m_air = CRAC_cooling_load/(DC_Config.C_AIR*Cooling_tower_air_delta)  # [4]
    v_air = m_air/DC_Config.RHO_AIR
    
    CT_Fan_pwr = DC_Config.CT_FAN_REF_P*(v_air/DC_Config.CT_REFRENCE_AIR_FLOW_RATE)**3  # [4]
    
    return CRAC_Fan_load, CT_Fan_pwr*1.05, CRAC_cooling_load, Compressor_load  # we are assuming this is a standalone data center with 5% overhead for other stuffs


def calculate_avg_CRAC_return_temp(rack_return_approach_temp_list,rackwise_outlet_temp):
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