#%%
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from tensorflow import config as tfconfig
import pandas as pd

temp_column = 6
temperature_data = pd.read_csv('/lustre/guillant/dc-rl/data/Weather/USA_NY_New.York-Kennedy.epw', skiprows=8, header=None).values[:,temp_column]

temperature_data = temperature_data.astype(float)

# Assuming hourly data, if not, adjust the freq parameter
start_date = '2020-01-01 00:00:00'  # Adjust the year as per your requirement
date_range = pd.date_range(start=start_date, periods=len(temperature_data), freq='H')
date_range = [str(d) for d in date_range]

# Create the DataFrame
df = pd.DataFrame({
    'Temperature': temperature_data
})

print(df)
#%%

# Define model parameters
gan_args = ModelParameters(batch_size=128,
                           lr=5e-4,
                           noise_dim=32,
                           layers_dim=128,
                           latent_dim=24,
                           gamma=1)

train_args = TrainParameters(epochs=10,
                             sequence_length=24,
                             number_sequences=1)

print(f'Number of GPUs : {tfconfig.list_physical_devices("GPU")}')
synth = TimeSeriesSynthesizer(modelname='timegan', model_parameters=gan_args)
synth.fit(df, train_args, num_cols=df.columns)
synth_data = synth.sample(len(temperature_data))
# %%
