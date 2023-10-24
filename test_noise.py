#%%
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

class ARNoise:
    def __init__(self, ar_params):
        self.ar_params = ar_params
        # For a pure AR model, there's no differencing and no MA component
        self.d = 0
        self.ma_params = []

    def generate(self, n_steps):
        ar_process = np.random.randn(n_steps)
        model = ARIMA(ar_process, order=(len(self.ar_params), self.d, len(self.ma_params)))
        model_fit = model.fit()
        residuals = model_fit.resid
        return residuals


class MANoise:
    def __init__(self, ma_params):
        self.ma_params = ma_params
        # For a pure MA model, there's no differencing and no AR component
        self.d = 0
        self.ar_params = []

    def generate(self, n_steps):
        ma_process = np.random.randn(n_steps)
        model = ARIMA(ma_process, order=(len(self.ar_params), self.d, len(self.ma_params)))
        model_fit = model.fit()
        residuals = model_fit.resid
        return residuals


class ARMANoise:
    def __init__(self, ar_params, ma_params):
        self.ar_params = ar_params
        self.ma_params = ma_params

    def generate(self, n_steps):
        arma_process = np.random.randn(n_steps)
        model = ARIMA(arma_process, order=(len(self.ar_params), 0, len(self.ma_params)))
        model_fit = model.fit()
        residuals = model_fit.resid
        return residuals
    

class ARIMANoise:
    def __init__(self, ar_params, d, ma_params):
        self.ar_params = ar_params
        self.d = d
        self.ma_params = ma_params

    def generate(self, n_steps):
        arima_process = np.random.randn(n_steps)
        model = ARIMA(arima_process, order=(len(self.ar_params), self.d, len(self.ma_params)))
        model_fit = model.fit()
        residuals = model_fit.resid
        return residuals

#%%
temp_column = 6
original_temperature_data = pd.read_csv('/lustre/guillant/dc-rl/data/Weather/USA_NY_New.York-Kennedy.epw', skiprows=8, header=None).values[:,temp_column]

original_temperature_data = original_temperature_data.astype(float)

cut = 4000
win = 500
temperature_data = original_temperature_data[cut:cut+win]
#%%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def add_noise_to_data(data, noise_model, *params):
    # Initialize the noise model
    noise_gen = noise_model(*params)

    # Generate the noise
    noise = noise_gen.generate(len(data))

    # Add the noise to the data
    noisy_data = data + noise

    return noisy_data

# Plotting function
def plot_noisy_data(original_data, noisy_data, title):
    plt.figure(figsize=(10, 6))
    plt.plot(original_data, label='Original Data')
    plt.plot(noisy_data, label='Noisy Data', alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.show()

# Fit ARIMA to original data
model = ARIMA(temperature_data, order=(2,2,2))
fit = model.fit()

# Extract the ARIMA order from fit.model_orders
ar_order = [fit.model_orders['ar']]  # Note the list brackets
d_order = 1 if fit.model_orders['reduced_ar'] or fit.model_orders['reduced_ma'] else 0
ma_order = [fit.model_orders['ma']]  # Note the list brackets

# Use these orders for generating noise
noisy_temp_ARIMA = add_noise_to_data(temperature_data, ARIMANoise, ar_order, d_order, ma_order)

# Plot the noisy data and original data
plot_noisy_data(temperature_data, noisy_temp_ARIMA, 'Temperature Data with ARIMA-based Noise')

# Generate and plot ARNoise
noisy_temp_AR = add_noise_to_data(temperature_data, ARNoise, ar_order)
plot_noisy_data(temperature_data, noisy_temp_AR, 'Temperature Data with AR Noise')

# Generate and plot MANoise
noisy_temp_MA = add_noise_to_data(temperature_data, MANoise, ma_order)
plot_noisy_data(temperature_data, noisy_temp_MA, 'Temperature Data with MA Noise')

# Generate and plot ARMANoise
noisy_temp_ARMA = add_noise_to_data(temperature_data, ARMANoise, ar_order, ma_order)
plot_noisy_data(temperature_data, noisy_temp_ARMA, 'Temperature Data with ARMA Noise')

# %%
# Modify the plotting function to accept multiple noisy data series
def plot_multiple_noisy_data(original_data, noisy_data_dict):
    plt.figure(figsize=(10, 6))
    plt.plot(original_data, label='Original Data', alpha=0.7)

    for title, data in noisy_data_dict.items():
        plt.plot(data, label=title, alpha=0.7)
    
    plt.legend()
    plt.show()

# Create a dictionary containing all the noisy data series
noisy_data_dict = {
    "ARIMA Noise": noisy_temp_ARIMA,
    "AR Noise": noisy_temp_AR,
    "MA Noise": noisy_temp_MA,
    "ARMA Noise": noisy_temp_ARMA
}

# Plot the original data along with each of the noisy datasets
plot_multiple_noisy_data(temperature_data, noisy_data_dict)
# %%

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

class ARIMANoise:
    def __init__(self, data, ar_order, d_order, ma_order):
        self.data = data
        self.ar_order = ar_order
        self.d_order = d_order
        self.ma_order = ma_order

    def generate(self):
        model = ARIMA(self.data, order=(self.ar_order, self.d_order, self.ma_order))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(self.data))
        residuals = self.data - forecast
        return residuals

# Load data
temp_column = 6
original_temperature_data = pd.read_csv('/lustre/guillant/dc-rl/data/Weather/USA_NY_New.York-Kennedy.epw', skiprows=8, header=None).values[:,temp_column]
original_temperature_data = original_temperature_data.astype(float)

cut = 4000
win = 500
temperature_data = original_temperature_data[cut:cut+win]

# Fit ARIMA to original data
model = ARIMA(temperature_data, order=(2,2,2))
fit = model.fit()

# Extract the ARIMA order from fit.model_orders
ar_order = [fit.model_orders['ar']]  # Note the list brackets
d_order = 1 if fit.model_orders['reduced_ar'] or fit.model_orders['reduced_ma'] else 0
ma_order = [fit.model_orders['ma']]  # Note the list brackets

# Use ARIMANoise to generate noise
noise_generator = ARIMANoise(temperature_data, ar_order, d_order, ma_order)
arima_noise = noise_generator.generate()

# Add ARIMA noise to temperature data
noisy_temp_ARIMA = temperature_data + arima_noise

# Plotting function
def plot_noisy_data(original_data, noisy_data, title):
    plt.figure(figsize=(10, 6))
    plt.plot(original_data, label='Original Data')
    plt.plot(noisy_data, label='Noisy Data', alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.show()

# Plot the noisy data and original data
plot_noisy_data(temperature_data, noisy_temp_ARIMA, 'Temperature Data with ARIMA-based Noise')

#%%

class RollingARIMANoise:
    def __init__(self, data, ar_order, d_order, ma_order):
        self.data = data
        self.ar_order = ar_order
        self.d_order = d_order
        self.ma_order = ma_order

    def generate(self):
        residuals = []

        # We start from the point after d_order + max(ar_order, ma_order) data points
        start_point = 1+2

        steps = 24
        # Iterating in steps of 24
        model = ARIMA(self.data, order=(self.ar_order, self.d_order, self.ma_order))
        model_fit = model.fit()
        for t in range(start_point, len(self.data) - steps, steps):
            forecast = model_fit.forecast(steps=steps)  # forecasting the next 24 hours
            residual = self.data[t:t+steps] - forecast
            residuals.extend(residual)
            
            # Fill the rest of the residuals with zeros
        remaining_points = len(self.data) - len(residuals)
        residuals.extend([0] * remaining_points)

        return np.array(residuals)

    
# Use RollingARIMANoise to generate noise
noise_generator = RollingARIMANoise(temperature_data, ar_order, d_order, ma_order)
rolling_arima_noise = noise_generator.generate()

# Add Rolling ARIMA noise to temperature data
noisy_temp_RollingARIMA = temperature_data + rolling_arima_noise

# Plot the noisy data and original data
plot_noisy_data(temperature_data, noisy_temp_RollingARIMA, 'Temperature Data with Rolling ARIMA-based Noise')
# %%
