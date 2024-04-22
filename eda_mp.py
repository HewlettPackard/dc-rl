from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from utils.dc_config_reader import DC_Config
import envs.datacenter as DataCenter
from tqdm import tqdm

import math

def simulate(parameters):
    dc_config, rack_0, iteration_limit, show_progress = parameters
    
    if show_progress:
        pbar = tqdm(total=iteration_limit, desc="Processing", leave=False)


    iteration = 0
    done = False
    while not done and iteration < iteration_limit:
        a = np.random.uniform(0.434*0.5, 0.434*2)
        b = np.random.uniform(0.030*0.5, 0.030*2)
        c = np.random.uniform(1.978*0.5, 1.978*2)
        d = np.random.uniform(1.418*0.5, 1.418*2)
        e = np.random.uniform(1.299*0.5, 1.299*2)
        f = np.random.uniform(1.883*0.5, 1.883*2)
        g = np.random.uniform(8.185*0.5, 8.185*2)
        h = np.random.uniform(-14.247*0.5, -14.247*2)

        # a = np.random.uniform(0.2, 0.8)
        # b = np.random.uniform(0.02, 0.08)
        # c = np.random.uniform(0.5, 2)
        # d = np.random.uniform(0.5, 2)
        # e = np.random.uniform(0.5, 2)
        # f = np.random.uniform(0.5, 2)
        # g = np.random.uniform(-20, 20)  # Additive term
        # h = np.random.uniform(-20, 20)  # Coefficient for logarithmic term
        
        
        dc_config.IT_FAN_AIRFLOW_RATIO_LB = [0.01, a]
        dc_config.IT_FAN_AIRFLOW_RATIO_UB = [a, 1]
        dc_config.IT_FAN_FULL_LOAD_V = b
        
        rack_eg = DataCenter.Rack(rack_0, max_W_per_rack=50000, rack_config=dc_config)
        
        it_pct_curves = []
        # temp_range = [17, 20, 24, 27]
        # it_pct_range = [5, 10, 25, 50, 75, 100]
        
        temp_range = range(17,27,1)
        it_pct_range = [0,5,10,15,20,25,30,50,75,85,100]

        valid = True
        try:
            for it_pct in it_pct_range:
                per_it_pct_curve = []
                for inlet_temp in temp_range:
                    it_power, fan_power = rack_eg.compute_instantaneous_pwr_vecd(inlet_temp=inlet_temp, ITE_load_pct=it_pct)
                    if np.any(rack_eg.v_fan_rack < 0) or valid == False:
                        valid = False
                        break
                    v_fan = np.sum(rack_eg.v_fan_rack)
                    power_term = (it_power + fan_power)**d
                    airflow_term = (dc_config.C_AIR * dc_config.RHO_AIR * v_fan**e * f)
                    log_term = h * np.log(max(power_term / airflow_term, 1))  # Log term, avoid log(0)
                    per_it_pct_curve.append(c * power_term / airflow_term + g + log_term)
                it_pct_curves.append(per_it_pct_curve)

            rang = it_pct_curves[-1][0] - it_pct_curves[0][0]
            inc_low_low = it_pct_curves[0][0]
            inc_low_high = it_pct_curves[-1][0]
            
            inc_high_low = it_pct_curves[0][-1]
            inc_high_high = it_pct_curves[-1][-1]
            
            # 
            # if rang < 20 and inc_low_low > 0 and inc_low_low < 5 and valid:
            #     if inc_low_high > 10 and inc_low_high < 20:
            #         if inc_high_low < 20:
            #             if inc_high_high > 20 and inc_high_high < 40:
            #                 done = True
            
            if inc_low_low > 4 and inc_low_low < 6 and valid:
                if inc_high_low > 0 and inc_high_low < 5:
                    if inc_low_high > 10 and inc_low_high < 20:
                        if inc_high_high > 6 and inc_high_high < 12:
                            done = True

                            
        except:
            print('Exception trying that parameters')
        iteration += 1
        if show_progress:
            pbar.update(1)
        if done:
            if show_progress:
                pbar.close()
            return a, b, c, d, e, f, g, h, rang, it_pct_curves, temp_range, it_pct_range
    if show_progress:
        pbar.close()
    return None  # No valid solution found within the iteration limit

def main():
    dc_config = DC_Config()
    rack_0 = dc_config.RACK_CPU_CONFIG[0]
    
    num_workers = 120  # Adjust based on your machine's capabilities
    iteration_limit = 100000  # Limit for each worker to avoid infinite loops
    params = [(dc_config, rack_0, iteration_limit, idx == 0) for idx in range(num_workers)]
    
    with Pool(num_workers) as pool:
        results = [pool.apply_async(simulate, (param,)) for param in params]
        valid_results = [res.get() for res in results if res.get() is not None]

    if valid_results:
        # Assume taking the first result that meets the condition
        plt.figure(figsize=(12,4))
        a, b, c, d, e, f, g, h, rang, it_pct_curves, temp_range, it_pct_range = valid_results[0]
        for i, j in zip(it_pct_curves, it_pct_range):
            plt.plot(temp_range, i, marker='*', label=f'{j} percent')
        plt.legend(bbox_to_anchor=(1.0, 1.0))
        plt.title(f'a: {a:.3f}, b: {b:.3f}, c: {c:.3f}, d: {d:.3f}, e: {e:.3f}, f: {f:.3f}, g: {g:.3f}, h: {h:.3f}, range: {rang:.3f}')
        # plt.tight_layout()
        plt.savefig('test.png')
        print(f'a: {a:.3f}, b: {b:.3f}, c: {c:.3f}, d: {d:.3f}, e: {e:.3f}, f: {f:.3f}, g: {g:.3f}, h: {h:.3f}, range: {rang:.3f}')

    if not valid_results:
        print('No found any valid solution')

if __name__ == "__main__":
    main()
