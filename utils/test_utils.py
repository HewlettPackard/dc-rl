#%%
from utils_cf import *
import matplotlib.pyplot as plt

#%%
init_day = get_init_day(6)
print(init_day)
#%%
wm = Weather_Manager(init_day=init_day, mu=0, theta=0, sigma=0, dt=0, weight=0)
temp = wm.reset()
temp = wm.get_total_weather()
plt.plot(temp[15000:15500])

wm = Weather_Manager(init_day=init_day)
temp = wm.reset()
temp = wm.get_total_weather()
plt.plot(temp[15000:15500])

temp = wm.reset()
temp = wm.get_total_weather()
plt.plot(temp[15000:15500])

temp = wm.reset()
temp = wm.get_total_weather()
plt.plot(temp[15000:15500])

temp = wm.reset()
temp = wm.get_total_weather()
plt.plot(temp[15000:15500])

temp = wm.reset()
temp = wm.get_total_weather()
plt.plot(temp[15000:15500])

#%%
wm = CI_Manager(mu=0, theta=0, sigma=0, dt=0, weight=0)
temp = wm.reset()
temp = wm.get_total_ci()
plt.plot(temp[15000:])

wm = CI_Manager(mu=0, theta=0, sigma=0, dt=0, weight=0.1)
temp = wm.reset()
temp = wm.get_total_ci()
plt.plot(temp[15000:])

temp = wm.reset()
temp = wm.get_total_ci()
plt.plot(temp[15000:])

temp = wm.reset()
temp = wm.get_total_ci()
plt.plot(temp[15000:])

temp = wm.reset()
temp = wm.get_total_ci()
plt.plot(temp[15000:])

temp = wm.reset()
temp = wm.get_total_ci()
plt.plot(temp[15000:])
# %%

wm = Workload_Manager(mu=0, theta=0, sigma=0, dt=0, weight=0)
temp = wm.reset()
temp = wm.get_total_wkl()
plt.plot(temp[15000:15200])

wm = Workload_Manager(mu=0, theta=0, sigma=0, dt=0, weight=0.001)
temp = wm.reset()
temp = wm.get_total_wkl()
plt.plot(temp[15000:15200])

temp = wm.reset()
temp = wm.get_total_wkl()
plt.plot(temp[15000:15200])

temp = wm.reset()
temp = wm.get_total_wkl()
plt.plot(temp[15000:15200])

temp = wm.reset()
temp = wm.get_total_wkl()
plt.plot(temp[15000:15200])
# %%

init_day = get_init_day(10)
t_m = Time_Manager(init_day=init_day)

workload_m = Workload_Manager(init_day=init_day, desired_std_dev=0.025, flexible_workload_ratio=0.1)
ci_m = CI_Manager(init_day=init_day)
weather_m = Weather_Manager(init_day=init_day) 
        
# %%
for i in range(2):
    workload = []
    workload.append(ci_m.reset()[0])
    for j in range(24*4*360):
        workload.append(ci_m.step()[0])
    plt.plot(workload)
# %%
