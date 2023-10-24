#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA


#%%
dta = sm.datasets.sunspots.load_pandas().data
dta.index = pd.Index(sm.tsa.datetools.dates_from_range("1700", "2008"))
dta.index.freq = dta.index.inferred_freq
del dta["YEAR"]

#%%
arma_mod30 = ARIMA(dta, order=(7, 0, 0)).fit()

#%%
sunspot_activity = dta['SUNACTIVITY'].loc['1900-12-31':'1920-12-31'].values
predict_sunspots_nomod = arma_mod30.predict("1900", "1920", dynamic=True).values

#%%
print(arma_mod30.params)
print(arma_mod30.param_names)
print(arma_mod30._results.params)

#%%

# arma_mod30._results.params[0] = 35
#arma_mod30._results.params[1] = 1.35
arma_mod30._results.params[2] = -0.510
# arma_mod30._results.params[-2] = -8.58299449e-02

#%%
predict_sunspots_mod = arma_mod30.predict("1900", "1920", dynamic=True).values

plt.plot(sunspot_activity,'r*-',label='true')
plt.plot(predict_sunspots_nomod,'g*-',label='prediction(no weight change)')
plt.plot(predict_sunspots_mod,'b*-',label='prediction(weight change)')
plt.legend()
plt.show()

#%%