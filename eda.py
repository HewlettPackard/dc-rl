#%%
from utils.dc_config_reader import DC_Config
import envs.datacenter as DataCenter
import numpy as np
import matplotlib.pyplot as plt

# Joint estimate of the range of values for delta change in temperature
dc_config = DC_Config()

rack_0 = dc_config.RACK_CPU_CONFIG[0]

done = False
iteration = 0
while not done:
    a = np.random.random()*0.5 + 0.1
    b = np.random.random()*0.05
    dc_config.IT_FAN_AIRFLOW_RATIO_LB = [0.1, a]
    dc_config.IT_FAN_AIRFLOW_RATIO_UB = [a, 0.8]
    dc_config.IT_FAN_FULL_LOAD_V = b

    DC_ITModel_config = dc_config

    rack_eg = DataCenter.Rack(rack_0, max_W_per_rack = 50000, rack_config = DC_ITModel_config)

    it_pct_curves = []

    temp_range = [17, 27]
    it_pct_range = [5, 100]

    for it_pct in it_pct_range:
        per_it_pct_curve = []
        for inlet_temp in temp_range:
            it_power, fan_power  = rack_eg.compute_instantaneous_pwr_vecd(inlet_temp=inlet_temp, ITE_load_pct=it_pct)

            v_fan = np.sum(rack_eg.v_fan_rack)

            per_it_pct_curve.append((it_power+fan_power)/(dc_config.C_AIR*dc_config.RHO_AIR*v_fan))  # DC_Config.IT_FAN_FULL_LOAD_V*0.01
        it_pct_curves.append(per_it_pct_curve)
    
    rang=it_pct_curves[-1][0] - it_pct_curves[0][0]
    inc_low = it_pct_curves[0][0]
    inc_high = it_pct_curves[-1][0]
    if inc_low < 5 and inc_high > 10:
        done = True
    iteration += 1
    
    if iteration > 1e6:
        print('Solution not found')
        done = True
#%%

for i,j in zip(it_pct_curves,it_pct_range):
    plt.plot(temp_range,i,marker='*',label=f'{j} percent')
plt.legend(bbox_to_anchor=(1.0,1.0))
# plt.show()
plt.title(f'a: {a:.3f}, b: {b:.3f}, range: {rang:3f}')
plt.tight_layout()
plt.savefig('test.png')