#%%
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

#%%
temp_column = 6
original_temperature_data = pd.read_csv('/lustre/guillant/dc-rl/data/Weather/USA_NY_New.York-Kennedy.epw', skiprows=8, header=None).values[:,temp_column]
# original_temperature_data = pd.read_csv("/lustre/guillant/dc-rl/data/CarbonIntensity/AZPS_NG_&_avgCI.csv")['avg_CI'].values[:8760]

original_temperature_data = original_temperature_data.astype(float)

cut = 4000
win = 1000
temperature_data = original_temperature_data[cut:cut+win]
#%%
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# For simplicity, I'm using an arbitrary seasonal order. You'd adjust based on your data's seasonality.
seasonal_order = (1, 1, 1, 24)  # Assuming daily seasonality for hourly data

# Fit SARIMA model
model_sarima = SARIMAX(temperature_data, order=(2,1,4), seasonal_order=seasonal_order, trend="ct")
model_sarima_fit = model_sarima.fit(disp=0)
sarima_residuals = model_sarima_fit.resid

# Generating a simple exogenous variable for illustration
exog = np.random.rand(len(temperature_data))

# Fit SARIMAX model
model_sarimax = SARIMAX(temperature_data, exog=exog, order=(2,1,4), seasonal_order=seasonal_order, trend="ct")
model_sarimax_fit = model_sarimax.fit(disp=0)
sarimax_residuals = model_sarimax_fit.resid

# Best ARIMA Order: (2, 1, 4) with AIC: -5223.520015612823
# Best ARMA Order: (3, 0, 2) with AIC: -3389.44571240579
# Best AR Order: (4, 0, 0) with AIC: -1602.6987510207628
# Best MA Order: (0, 0, 4) with AIC: 22175.84142536604

# Fit ARIMA model
model_arima = ARIMA(temperature_data, order=(2,1,4))
model_arima_fit = model_arima.fit()
arima_residuals = model_arima_fit.resid

# Fit ARMA model
model_arma = ARIMA(temperature_data, order=(3, 0, 2), trend="ct")
model_arma_fit = model_arma.fit()
arma_residuals = model_arma_fit.resid

# Fit AR model
model_ar = ARIMA(temperature_data, order=(4,0,0), trend="ct")
model_ar_fit = model_ar.fit()
ar_residuals = model_ar_fit.resid

# Fit MA model
model_ma = ARIMA(temperature_data, order=(0,0,4), trend="ct")
model_ma_fit = model_ma.fit()
ma_residuals = model_ma_fit.resid

# Visualization
plt.figure(figsize=(15,8))

plt.subplot(2, 2, 1)
plt.plot(arima_residuals)
plt.title("ARIMA Residuals")

plt.subplot(2, 2, 2)
plt.plot(arma_residuals)
plt.title("ARMA Residuals")

plt.subplot(2, 2, 3)
plt.plot(ar_residuals)
plt.title("AR Residuals")

plt.subplot(2, 2, 4)
plt.plot(ma_residuals)
plt.title("MA Residuals")

plt.tight_layout()
plt.show()

# %%
# Adding the residuals as noise to the original data
noise_std = 1
temperature_with_sarima_noise = temperature_data + sarima_residuals# *np.random.normal(1, noise_std, size=len(arima_residuals))
temperature_with_sarimax_noise = temperature_data + sarimax_residuals# *np.random.normal(1, noise_std, size=len(arima_residuals))
temperature_with_arima_noise = temperature_data + arima_residuals# *np.random.normal(1, noise_std, size=len(arima_residuals))
temperature_with_arma_noise = temperature_data + arma_residuals# *np.random.normal(1, noise_std, size=len(arima_residuals))
temperature_with_ar_noise = temperature_data + ar_residuals# *np.random.normal(1, noise_std, size=len(arima_residuals))
temperature_with_ma_noise = temperature_data + ma_residuals# *np.random.normal(1, noise_std, size=len(arima_residuals))

# Visualization
plt.figure(figsize=(15,10))

plt.subplot(3, 2, 1)
plt.plot(temperature_data[10:], label="Original")
plt.plot(temperature_with_sarima_noise[10:], label="With SARIMA Noise", alpha=0.7)
plt.title("Temperature with SARIMA Noise")
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(temperature_data[10:], label="Original")
plt.plot(temperature_with_sarimax_noise[10:], label="With SARIMAX Noise", alpha=0.7)
plt.title("Temperature with SARIMAX Noise")
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(temperature_data[10:], label="Original")
plt.plot(temperature_with_arima_noise[10:], label="With ARIMA Noise", alpha=0.7)
plt.title("Temperature with ARIMA Noise")
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(temperature_data[10:], label="Original")
plt.plot(temperature_with_arma_noise[10:], label="With ARMA Noise", alpha=0.7)
plt.title("Temperature with ARMA Noise")
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(temperature_data[10:], label="Original")
plt.plot(temperature_with_ar_noise[10:], label="With AR Noise", alpha=0.7)
plt.title("Temperature with AR Noise")
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(temperature_data[10:], label="Original")
plt.plot(temperature_with_ma_noise[10:], label="With MA Noise", alpha=0.7)
plt.title("Temperature with MA Noise")
plt.legend()

plt.tight_layout()
plt.show()

#%%
# Visualization
plt.figure(figsize=(15,8))

# Original data
plt.plot(temperature_data, label="Original", linewidth=2)

# Temperature data with different noises
plt.plot(temperature_with_arima_noise, label="With ARIMA Noise", alpha=0.7)
plt.plot(temperature_with_arma_noise, label="With ARMA Noise", alpha=0.7)
plt.plot(temperature_with_ar_noise, label="With AR Noise", alpha=0.7)
plt.plot(temperature_with_ma_noise, label="With MA Noise", alpha=0.7)

plt.title("Temperature Data with Different Noises")
plt.legend()
plt.tight_layout()
plt.show()

# %% Adjust the values
'''
To select the optimal order for ARIMA, ARMA, AR, and MA models, we can use the AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion) values. The idea is to select the order that provides the lowest AIC or BIC values, as they strike a balance between the goodness of fit and model complexity.

Here's a step-by-step guide and the code:

Plot ACF and PACF:
By visualizing the Autocorrelation Function (ACF) and the Partial Autocorrelation Function (PACF) of the data, we can get a sense of possible lag orders.

Iteratively Fit Models and Check AIC & BIC:
This will involve a brute force method of trying different combinations of orders and selecting the one with the lowest AIC/BIC.

Let's start with the ACF and PACF plots:

'''
#%%
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set a consistent font for the plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

# Create the plots with aesthetic considerations
fig, ax = plt.subplots(2, 1, figsize=(3.5, 5))  # Adjust size to fit a single column

plot_acf(original_temperature_data, lags=30, ax=ax[0], color='dodgerblue')
ax[0].set_title('Autocorrelation Function (ACF)', fontsize=12)
ax[0].set_xlabel('Lags')
ax[0].set_ylabel('Correlation')
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].set_ylim([-1, 1.1])
ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)

plot_pacf(original_temperature_data, lags=30, ax=ax[1], color='dodgerblue')
ax[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12)
ax[1].set_xlabel('Lags')
ax[1].set_ylabel('Correlation')
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_ylim([-1, 1.1])
ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()

# Save the figure in high-resolution suitable for LaTeX
fig.savefig("ACF_PACF_single_column_CI.pdf", bbox_inches='tight', dpi=300)

#%%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF and PACF plots
plt.figure(figsize=(12,6))

plt.subplot(2, 1, 1)
plot_acf(original_temperature_data, lags=30, ax=plt.gca())
plt.title('ACF for Temperature Data')

plt.subplot(2, 1, 2)
plot_pacf(original_temperature_data, lags=30, ax=plt.gca())
plt.title('PACF for Temperature Data')

plt.tight_layout()
plt.show()

#%% Now, let's attempt to find the best orders for the models:

# Setting the max order for consideration
max_order = 5

# For ARIMA
best_aic = float('inf')
best_order = None
for p in tqdm(range(max_order), desc=" outer", position=0):
    for d in tqdm(range(max_order), desc=" inner loop", position=1, leave=False):
        for q in range(max_order):
            try:
                model = ARIMA(original_temperature_data, order=(p,d,q), trend="ct")
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p,d,q)
            except:
                continue
print(f'Best MA Order: {best_order} with AIC: {best_aic}')
# Best ARIMA Order: (2, 1, 4) with AIC: -5223.520015612823
# Best ARMA Order: (3, 0, 2) with AIC: -3389.44571240579
# Best AR Order: (4, 0, 0) with AIC: -1602.6987510207628
# Best MA Order: (0, 0, 4) with AIC: 22175.84142536604

# Similarly, you can find the best order for ARMA, AR, and MA using a similar approach.
#%%
'''
The above code will give you the best ARIMA order based on the AIC value. For AR, MA, and ARMA models, you can modify the above approach to try different combinations of p, d, and q.

This process can be computationally expensive, especially if the data series is long or the max_order is set too high.
So, you might want to set an appropriate max_order or consider using more advanced auto-selection algorithms that can be found in certain Python libraries.
'''
# %%
from tqdm import tqdm
np.random.seed(42)  # for reproducibility

# Fit ARIMA model
model_arima = ARIMA(temperature_data, order=(5,1,1))
model_arima_fit = model_arima.fit()

# Set noise standard deviation (you can adjust this value)
noise_std = 0.1


model = ARIMA(temperature_data, order=(4,1,2))
model_fit = model.fit()
forecasts = model_fit.forecast(steps=len(temperature_data))
    
# Add noise
noisy_forecasts = forecasts + np.random.normal(0, noise_std, size=len(temperature_data))


# Visualization
plt.figure(figsize=(15,6))

plt.plot(temperature_data, label="Original Data")
plt.plot(forecasts, label="Synthetic Data with Noise", linestyle="--")
plt.legend()
plt.title("Original vs. Synthetic Data with Noise")
plt.show()


# %%
# Generate a daily cyclical pattern for server load (higher during working hours, lower during off hours)
hourly_pattern = np.concatenate([np.linspace(0.6, 1, 8), np.linspace(1, 0.6, 16)])

# Replicate this pattern for the number of days in the temperature data
days = len(temperature_data) // 24
server_load_pattern = np.tile(hourly_pattern, days)

# Add a slight upward trend to indicate increasing demand on the data center over time
trend = np.linspace(1, 1.2, len(server_load_pattern))
server_load_pattern = server_load_pattern * trend

# Add some random noise to make it more realistic
server_load = server_load_pattern + 0.05 * np.random.randn(len(server_load_pattern))

# Visualization
plt.figure(figsize=(10,5))
plt.plot(server_load)
plt.title("Simulated Server Load")
plt.xlabel("Hours")
plt.ylabel("Server Load")
plt.tight_layout()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

# Values extracted from tables (using CO₂ in AZ as an example)
baseline = np.array([65.74, 52.16, 54.63, 52.10])
carbon_noise = np.array([64.42, 53.66, 54.50, 52.72])
weather_noise = np.array([63.60, 53.41, 54.92, 52.24])
both_noises = np.array([64.97, 53.93, 54.35, 53.19])

# Calculate relative performance degradation
relative_carbon_degradation = 100*(carbon_noise - baseline) / baseline
relative_weather_degradation = 100*(weather_noise - baseline) / baseline
relative_both_degradation = 100*(both_noises - baseline) / baseline

# Set aesthetic plot parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(6, 3.5))  # Adjust size as per your requirement

# Plotting
algorithms = ['A2C-FCN', 'A2C-GTrXL', 'MADDPG-FCN', 'MADDPG-GTrXL']
x = np.arange(len(algorithms))  # the label locations
width = 0.2  # the width of the bars

rects1 = ax.bar(x - width, relative_carbon_degradation, width, label='Carbon Noise', color='dodgerblue')
rects2 = ax.bar(x, relative_weather_degradation, width, label='Weather Noise', color='tomato')
rects3 = ax.bar(x + width, relative_both_degradation, width, label='Both Noises', color='mediumseagreen')

# Aesthetic configurations
ax.set_title('Relative performance degradation due to different noises (CO₂ in AZ)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(algorithms, rotation=0, ha="center", rotation_mode="anchor")
ax.set_ylabel('Relative Performance Degradation (%)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.axhline(0, color='black',linewidth=0.5)

plt.tight_layout()

# Save the figure in high-resolution suitable for LaTeX
fig.savefig("Relative_Performance_Degradation_AZ_CO2.pdf", bbox_inches='tight', dpi=300)

plt.show()

# %%


# Values extracted from tables (using CO₂ in AZ as an example)
baseline_energy = np.array([53.45, 42.83, 44.67, 42.73])
carbon_noise_energy = np.array([52.51, 44.10, 44.63, 43.32])
weather_noise_energy = np.array([51.76, 43.93, 44.90, 42.99])
both_noises_energy = np.array([52.77, 44.28, 44.43, 43.64])

# Calculate relative performance degradation
relative_carbon_degradation = 100*(carbon_noise_energy - baseline_energy) / baseline_energy
relative_weather_degradation = 100*(weather_noise_energy - baseline_energy) / baseline_energy
relative_both_degradation = 100*(both_noises_energy - baseline_energy) / baseline_energy

# Set aesthetic plot parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(6, 3.5))  # Adjust size as per your requirement

# Plotting
algorithms = ['A2C-FCN', 'A2C-GTrXL', 'MADDPG-FCN', 'MADDPG-GTrXL']
x = np.arange(len(algorithms))  # the label locations
width = 0.2  # the width of the bars

rects1 = ax.bar(x - width, relative_carbon_degradation, width, label='Carbon Noise', color='dodgerblue')
rects2 = ax.bar(x, relative_weather_degradation, width, label='Weather Noise', color='tomato')
rects3 = ax.bar(x + width, relative_both_degradation, width, label='Both Noises', color='mediumseagreen')

# Aesthetic configurations
ax.set_title('Relative performance degradation due to different noises (Energy in AZ)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(algorithms, rotation=0, ha="center", rotation_mode="anchor")
ax.set_ylabel('Relative Performance Degradation (%)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.axhline(0, color='black',linewidth=0.5)

plt.tight_layout()

# Save the figure in high-resolution suitable for LaTeX
fig.savefig("Relative_Performance_Degradation_AZ_Energy.pdf", bbox_inches='tight', dpi=300)

plt.show()

# %%

baseline = np.array([3.14, 2.99, 3.05, 2.91])
carbon_noise = np.array([3.11, 2.98, 3.06, 3.04])
weather_noise = np.array([3.12, 3.03, 3.09, 3.02])
both_noises = np.array([3.15, 3.04, 2.97, 3.03])

# Calculate relative performance degradation
relative_carbon_degradation = 100*(carbon_noise - baseline) / baseline
relative_weather_degradation = 100*(weather_noise - baseline) / baseline
relative_both_degradation = 100*(both_noises - baseline) / baseline

# Set aesthetic plot parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(6, 3.5))  # Adjust size as per your requirement

# Plotting
algorithms = ['A2C-FCN', 'A2C-GTrXL', 'MADDPG-FCN', 'MADDPG-GTrXL']
x = np.arange(len(algorithms))  # the label locations
width = 0.2  # the width of the bars

rects1 = ax.bar(x - width, relative_carbon_degradation, width, label='Carbon Noise', color='dodgerblue')
rects2 = ax.bar(x, relative_weather_degradation, width, label='Weather Noise', color='tomato')
rects3 = ax.bar(x + width, relative_both_degradation, width, label='Both Noises', color='mediumseagreen')

# Aesthetic configurations
ax.set_title('Relative performance degradation due to different noises (CO₂ in WA)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(algorithms, rotation=0, ha="center", rotation_mode="anchor")
ax.set_ylabel('Relative Performance Degradation (%)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.axhline(0, color='black',linewidth=0.5)

plt.tight_layout()

# Save the figure in high-resolution suitable for LaTeX
fig.savefig("Relative_Performance_Degradation_WA_CO2.pdf", bbox_inches='tight', dpi=300)

plt.show()
# %%
baseline_energy = np.array([29.66, 28.72, 29.14, 28.00])
carbon_noise_energy = np.array([29.66, 28.93, 29.17, 28.68])
weather_noise_energy = np.array([29.60, 29.05, 29.44, 28.98])
both_noises_energy = np.array([29.85, 29.09, 29.02, 28.97])

# Calculate relative performance degradation
relative_carbon_degradation = 100*(carbon_noise_energy - baseline_energy) / baseline_energy
relative_weather_degradation = 100*(weather_noise_energy - baseline_energy) / baseline_energy
relative_both_degradation = 100*(both_noises_energy - baseline_energy) / baseline_energy

# Set aesthetic plot parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(6, 3.5))  # Adjust size as per your requirement

# Plotting
algorithms = ['A2C-FCN', 'A2C-GTrXL', 'MADDPG-FCN', 'MADDPG-GTrXL']
x = np.arange(len(algorithms))  # the label locations
width = 0.2  # the width of the bars

rects1 = ax.bar(x - width, relative_carbon_degradation, width, label='Carbon Noise', color='dodgerblue')
rects2 = ax.bar(x, relative_weather_degradation, width, label='Weather Noise', color='tomato')
rects3 = ax.bar(x + width, relative_both_degradation, width, label='Both Noises', color='mediumseagreen')

# Aesthetic configurations
ax.set_title('Relative performance degradation due to different noises (Energy in WA)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(algorithms, rotation=0, ha="center", rotation_mode="anchor")
ax.set_ylabel('Relative Performance Degradation (%)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.axhline(0, color='black',linewidth=0.5)

plt.tight_layout()

# Save the figure in high-resolution suitable for LaTeX
fig.savefig("Relative_Performance_Degradation_WA_Energy.pdf", bbox_inches='tight', dpi=300)
# %%
import numpy as np
import matplotlib.pyplot as plt

# Extracted CO₂ values from the table
co2_ashrae = np.array([87.55, 23.99, 4.32])
co2_a2c_fcn = np.array([65.74, 18.55, 3.14])
co2_a2c_gtrxl = np.array([52.16, 17.86, 2.99])
co2_maddpg_fcn = np.array([54.63, 18.04, 3.05])
co2_maddpg_gtrxl = np.array([52.10, 17.58, 2.91])

locations = ["AZ", "NY", "WA"]
algorithms = ["ASHRAE", "A2C FCN", "A2C GTrXL", "MADDPG FCN", "MADDPG GTrXL"]

# Storing the CO₂ values for each location in a single matrix
data = np.array([co2_ashrae, co2_a2c_fcn, co2_a2c_gtrxl, co2_maddpg_fcn, co2_maddpg_gtrxl])

fig, axs = plt.subplots(1, 3, figsize=(10, 3))

# Define colors for better visualization
colors = ['gray', 'dodgerblue', 'royalblue', 'tomato', 'mediumseagreen']

# Plotting
for i, ax in enumerate(axs):
    ax.bar(algorithms, data[:, i], color=colors)
    ax.set_title(f'CO\N{SUBSCRIPT TWO} in {locations[i]}', fontsize=10)
    ax.set_ylabel('CO\N{SUBSCRIPT TWO} (Tonnes)', fontsize=8)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig("CO2_comparison.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.show()

#%% colors = ['gray', 'dodgerblue', 'royalblue', 'tomato', 'mediumseagreen']
import numpy as np
import matplotlib.pyplot as plt

# Extracted Energy values from the table
energy_ashrae = np.array([71.86, 45.12, 39.17])
energy_a2c_fcn = np.array([53.45, 34.61, 29.66])
energy_a2c_gtrxl = np.array([42.83, 33.15, 28.72])
energy_maddpg_fcn = np.array([44.67, 33.52, 29.14])
energy_maddpg_gtrxl = np.array([42.73, 32.87, 28.00])

locations = ["AZ", "NY", "WA"]
algorithms = ["ASHRAE", "A2C FCN", "A2C GTrXL", "MADDPG FCN", "MADDPG GTrXL"]

# Storing the Energy values for each location in a single matrix
data = np.array([energy_ashrae, energy_a2c_fcn, energy_a2c_gtrxl, energy_maddpg_fcn, energy_maddpg_gtrxl])

fig, axs = plt.subplots(1, 3, figsize=(10, 3))

# Define colors for better visualization
colors = ['gray', 'dodgerblue', 'royalblue', 'tomato', 'mediumseagreen']

# Plotting
for i, ax in enumerate(axs):
    ax.bar(algorithms, data[:, i], color=colors)
    ax.set_title(f'Energy in {locations[i]}', fontsize=10)
    ax.set_ylabel('Energy (Gwh)', fontsize=8)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()

# Save as a PDF
plt.savefig("Energy_comparison.pdf", format='pdf', bbox_inches='tight')
plt.show()

# %%
