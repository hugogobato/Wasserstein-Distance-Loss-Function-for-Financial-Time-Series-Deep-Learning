import persim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from typing import Optional, Union, Tuple


import torch.nn.functional as F
from torch.distributions import Distribution
from torch.distributions import Bernoulli, Normal, StudentT, Poisson, NegativeBinomial

from torch.distributions import constraints


from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengers, AirPassengersPanel, AirPassengersStatic
from neuralforecast.losses.numpy import rmse, mape
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from WD_Loss_Function import *

df = pd.read_excel("SP100_vol.xlsx", index_col="Date",parse_dates=True)
columns=df.columns
df_static=pd.read_excel("Static_df.xlsx")
n_series=len(df.columns)

test_length=int(len(df['AMT'])*0.3)
df1=df.unstack().reset_index(name="Value").rename(columns={"level_0":"unique_id", "Date":"ds", "Value":"y"})

model = NBEATSx(h=10, input_size=84,
              loss=MSE_2DWD(),
              scaler_type='minmax',
              learning_rate=0.0005,
              stack_types=['identity','trend'],
              n_blocks=[3,3],
              mlp_units=[[512, 512], [512, 512]],
              windows_batch_size=32,
              num_lr_decays=5,
              val_check_steps=500,
              stat_exog_list=['AMT', 'COP', 'CVX', 'AXP', 'BAC', 'C', 'CB', 'CME', 'GS', 'JPM', 'MS',
     'PNC', 'USB', 'WFC', 'BA', 'CAT', 'GE', 'HON', 'LMT', 'MMM', 'UNP',
     'UPS', 'AAPL', 'ACN', 'ADBE', 'ADP', 'CMCSA', 'CRM', 'CSCO', 'DIS',
     'FIS', 'FISV', 'GOOG', 'IBM', 'INTC', 'INTU', 'MA', 'MSFT', 'NFLX',
     'NVDA', 'ORCL', 'QCOM', 'T', 'TXN', 'VZ', 'COST', 'KO', 'MO', 'PEP',
     'PG', 'WMT', 'D', 'DUK', 'SO', 'ABT', 'AMGN', 'BDX', 'BMY', 'BSX', 'CI',
     'CVS', 'DHR', 'GILD', 'ISRG', 'JNJ', 'LLY', 'MDT', 'MRK', 'PFE', 'SYK',
     'TMO', 'UNH', 'AMZN', 'HD', 'LOW', 'MCD', 'NKE', 'SBUX', 'TGT', 'TJX'],
              n_harmonics=0, n_polynomials=1,
              max_steps=350,
              random_seed=2251959)
fcst = NeuralForecast(models=[model],freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
forecasts = fcst.cross_validation(df=df1,val_size=10,static_df=df_static,n_windows=None, test_size=test_length-test_length%10,step_size=10)
forecasts = forecasts.dropna()
if "NBEATSx-median" not in list(forecasts.columns.values):
  Y_hat=forecasts["NBEATSx"].values
else:
  Y_hat=forecasts["NBEATSx-median"].values
Y_true=forecasts["y"].values
RMSE=np.sqrt(np.sum(((Y_true-Y_hat)**2))/len(Y_true))
QLIKE=(np.sum(Y_true/Y_hat-np.log(abs(Y_true)/abs(Y_hat))-1)/len(Y_true))
MAE=np.sum(abs(Y_true-Y_hat))/len(Y_true)
print(f"RMSE: {RMSE} | QLIKE: {QLIKE} | MAPE: {MAE}")

df2=forecasts.drop(columns=["cutoff", "y"]).set_index("ds")
df2=df2.pivot(columns="unique_id")
df2.to_excel("NBEATSx-2DWD forecasts.xlsx")
