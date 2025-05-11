# %%
import math
import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import warnings
warnings.filterwarnings("ignore")


# %%
IND = 'kvotzat otzar'

# %%
data = pd.read_csv("ZH_data_as_tuple.csv")
data = data.dropna(subset=['MOF_class_in'])
data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month'].astype(str), format='%Y-%m') + pd.offsets.MonthEnd(0)
time_serieses = data.groupby(['MOF_class_out', 'MOF_class_in', 'date'])['value'].sum()
data_as_frame = time_serieses.unstack(level=[0, 1]).fillna(0)

# %% [markdown]
# # models
# 

# %%
class NaiveModel:    
    def __init__(self, data):
        self.data = data

    def fit(self):
        return self

    def forecast(self, steps_to_forecast) -> pd.Series:
        return pd.Series(self.data.values[-1], index=pd.date_range(self.data.index[-1] + pd.offsets.MonthEnd(1), periods=steps_to_forecast, freq='ME'))

class MeanModel:    
    def __init__(self, data):
        self.data = data

    def fit(self):
        return self

    def forecast(self, steps_to_forecast) -> pd.Series:
        return pd.Series(self.data.mean(), index=pd.date_range(self.data.index[-1] + pd.offsets.MonthEnd(1), periods=steps_to_forecast, freq='ME'))
    
class SeasonalNaiveModel:
    def __init__(self, data, seasonality=12):
        self.data = data
        self.seasonality = seasonality

    def fit(self):
        return self

    def _forecast_h(self, h):
        p = math.ceil(h/self.seasonality)
        return self.data.values[h-self.seasonality*p-1]

    def forecast(self, steps_to_forecast) -> pd.Series:
        return pd.Series([self._forecast_h(h) for h in range(1, steps_to_forecast + 1)], index=pd.date_range(self.data.index[-1] + pd.offsets.MonthEnd(1), periods=steps_to_forecast, freq='ME'))

class TSConvergenceError(Exception):
    pass
   
class AutoArima:
    def __init__(self, data, seasonal=True, seasonality=12):
        self.data = data
        self.seasonal = seasonal
        self.seasonality = seasonality
        self.model = None
        
    def fit(self):
        if not self.model:
            try:
                self.model = auto_arima(self.data, seasonal=self.seasonal, m=self.seasonality)
            except ValueError as e:  
                raise TSConvergenceError from e
        return self

    def forecast(self, steps_to_forecast):
        try:
            return self.model.predict(steps_to_forecast)
        except ValueError as e:  
            raise TSConvergenceError from e

class TSModel4:
    def __init__(self, data_by_ozar_groups,year_to_forcast):
        self.data_by_ozar_groups = data_by_ozar_groups
        self.r2_score_values = {}
        self.testData = {}
        self.forecastData = {}
        self.tillpastYearData = {}
        self.year_to_forcast = pd.Timestamp(f"{year_to_forcast-1}-01-31")

    def fit(self, size_of_validation_data,modelType):
        for i, group in enumerate(self.data_by_ozar_groups.columns):
            group_data = self.data_by_ozar_groups[group].dropna()
            train_data, test_data = group_data[:-size_of_validation_data], group_data[-size_of_validation_data:]
            model = modelType(train_data)
            model_fit = model.fit()
            forecast = model_fit.forecast(12)
            self.testData[group] = test_data
            self.forecastData[group] = forecast
            self.tillpastYearData[group] = train_data
            r2_score_value = r2_score(test_data, forecast)
            self.r2_score_values[group] = r2_score_value
        
                
    def r2_score(self):
        return self.r2_score_values

templates = {
   'naive': NaiveModel,
   'seasonal_naive': SeasonalNaiveModel,
   'mean': MeanModel,
   'SimpleExpSmoothing' : SimpleExpSmoothing
}

# %% [markdown]
# # PREPROCESS
# 

# %% [markdown]
# # biggest sums went into kvotzot otzar in 2024 compare to 2023
#  

# %%
temp = data_as_frame.resample("YE").sum().loc['2023':'2024'].T.groupby(level=1).sum()
increment = temp['2024-12-31'] - temp['2023-12-31']
increment=increment.sort_values(ascending=False)

# %%
increment_df = pd.DataFrame(increment)
increment_df['name'] = increment_df.index
increment_df.iloc[9:, 1] = 'Others'
increment_df = increment_df.groupby('name').sum()


# %% [markdown]
# # Comparison of Sum of Top vs Bottom kvotzot otzar

# %%
sums_anomaly_by_year_data = {}
for (i,j) in data_as_frame:
    sum_2023 = data_as_frame[(i, j)].resample('YE').sum().loc["2023-12-31"]
    sum_2024 = data_as_frame[(i, j)].resample('YE').sum().loc["2024-12-31"]
    diff_between_sum = sum_2024 - sum_2023
    sums_anomaly_by_year_data[(i,j)] = diff_between_sum
increments_by_MOF_class = (pd.DataFrame(list(sums_anomaly_by_year_data.items()), columns=['kvotzat otzar', 'Value'])).sort_values(by=['Value'], ascending=False).set_index('kvotzat otzar')


# %% [markdown]
# #  data forcast 2025 year

# %%
r2_score_values_data = {}
data_split_month= 12
year_to_forcast = 2025
for key in templates:
    model = TSModel4(data_as_frame,year_to_forcast)
    model.fit(data_split_month,templates[key])
    r2_score_values_data[key] = model.r2_score()
r2_score_values_data=pd.DataFrame(r2_score_values_data)

# %%
wining_model={}
r2_of_wining_models = {}
for i in r2_score_values_data.index:
    max_mask = r2_score_values_data.loc[i].values == r2_score_values_data.loc[i].values.max()
    wining_model[i] = r2_score_values_data.columns[max_mask][0]
    r2_of_wining_models[i] = r2_score_values_data.loc[i].max()


# %%
month_to_predict = 8
forcast_data_2025 = {}
for i, kvotzat_otzar_sahar in enumerate(wining_model):
    model = templates[wining_model[kvotzat_otzar_sahar]](data_as_frame[kvotzat_otzar_sahar])
    model_fit = model.fit()
    forecast = model_fit.forecast(month_to_predict)
    forcast_data_2025[kvotzat_otzar_sahar] = forecast


# %%
data_so_far_2025 = data_as_frame[data_as_frame.index>"2024-12-31"]
forcast_ashbarot_2025 = pd.concat([data_so_far_2025, pd.DataFrame(forcast_data_2025)], axis=0)
data_so_far_2025_sum = forcast_ashbarot_2025.sum(axis=1).sum() 

# %%
data_so_far_2025 = data_as_frame[data_as_frame.index>"2024-12-31"]
forcast_ashbarot_2025 = pd.concat([data_so_far_2025, pd.DataFrame(forcast_data_2025)], axis=0)
data_so_far_2025_sum = forcast_ashbarot_2025.sum(axis=1).sum() 
forcast_ashbarot_2025=forcast_ashbarot_2025.sum(axis=0).groupby(level=0).sum()
forcast_ashbarot_2025=pd.DataFrame(forcast_ashbarot_2025).rename(columns={0:'ZH'})


# %%
forcast_ashbarot_2025.index.name = IND
forcast_ashbarot_2025.rename('ZH').to_csv('ashbarot_2025.csv')