# %%
import argparse
import math
import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
from sklearn.metrics import r2_score
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import warnings
warnings.filterwarnings("ignore")


IND  = 'kvotzat otzar'

parser = argparse.ArgumentParser(description="Forecasting script")
parser.add_argument("--past_year",   type=int, required=True, help="Year to forecast")
parser.add_argument("--curr_year",   type=int, required=True, help="Current Year")
parser.add_argument("--curr_month",  type=int, required=True, help="Current Month")
parser.add_argument("--months_back", type=int, required=False, default = -1, help="Month to train on")

year_to_predict                  = parser.parse_args().past_year
how_much_month_in_current_year_in_data   = parser.parse_args().curr_month
current_year                     = parser.parse_args().curr_year
how_much_month_back_to_use       = parser.parse_args().months_back
flag_for_using_only_part_of_data = how_much_month_back_to_use != -1
how_much_months_in_year          = 12

# %% [markdown]
# # export DATA

# %%
data = pd.read_csv(r"Data\ZH_data_as_tuple.csv")
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
    def __init__(self, data_by_ozar_groups, year_to_forcast):
        self.data_by_ozar_groups = data_by_ozar_groups
        self.r2_score_values = {}
        self.testData = {}
        self.forecastData = {}
        self.tillpastYearData = {}
        self.year_to_forecast = year_to_forcast

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

# %%
data_as_frame = time_serieses.unstack(level=[0, 1]).fillna(0)

# %%
temp = data_as_frame.resample("YE").sum().loc['2023':'2024'].T.groupby(level=1).sum().sum().sum()

# %% [markdown]
# # biggest sums went into kvotzot otzar in 2024 compare to 2023


# %% [markdown]
# # functions

# %%
def find_r2_score_values_data(how_much_months_in_year,data_by_ozar_groups,year_to_predict):
    r2_score_values_data = {}
    for key in templates:
        model = TSModel4(data_by_ozar_groups,year_to_predict)
        model.fit(how_much_months_in_year,templates[key])
        r2_score_values_data[key] = model.r2_score()
    r2_score_values_data=pd.DataFrame(r2_score_values_data)
    return r2_score_values_data

def find_wining_models(r2_score_values_data_specific_year):
    wining_model={}
    r2_of_wining_models = {}
    for i in r2_score_values_data_specific_year.index:
        wining_model[i] = r2_score_values_data_specific_year.columns[r2_score_values_data_specific_year.loc[i].values == r2_score_values_data_specific_year.loc[i].values.max()]
        r2_of_wining_models[i] = r2_score_values_data_specific_year.loc[i].max()
    return wining_model, r2_of_wining_models
    

def forcast_data(month_to_predict,wining_model_specific_year,data_we_got_to_use_in_prediction,flag_for_using_only_part_of_data,how_much_month_back_to_use):
    forcast_data_specific_year = {}
    for i, kvotzat_otzar_sahar in enumerate(wining_model_specific_year):
        if(flag_for_using_only_part_of_data):
            model = templates[wining_model_specific_year[kvotzat_otzar_sahar][0]](data_we_got_to_use_in_prediction[kvotzat_otzar_sahar][-how_much_month_back_to_use:])
        else:
            model = templates[wining_model_specific_year[kvotzat_otzar_sahar][0]](data_we_got_to_use_in_prediction[kvotzat_otzar_sahar])
        model_fit = model.fit()
        forecast = model_fit.forecast(month_to_predict)
        forcast_data_specific_year[kvotzat_otzar_sahar] = forecast
    return forcast_data_specific_year

# %% [markdown]
# # pre process for forecasting by specific year

# %%
how_many_years_look_back_to_find_specific_year = (current_year - year_to_predict) * how_much_months_in_year
actual_data_specific_year = data_as_frame.loc[str(year_to_predict)].fillna(0)
data_we_got_to_use_in_prediction_specific_year =  data_as_frame.loc[:str(year_to_predict-1)].fillna(0)
data_we_got_to_use_in_prediction_current_year_year =  pd.concat([data_as_frame.loc[:str(current_year-1)],data_as_frame.loc[str(current_year)].head(how_much_month_in_current_year_in_data)]).fillna(0)

# %% [markdown]
# # data forcaast specific year

# %%
r2_score_values_data_specific_year = find_r2_score_values_data(how_much_months_in_year,data_as_frame, year_to_predict)
wining_model_specific_year, r2_of_wining_models_specific_year = find_wining_models(r2_score_values_data_specific_year)
forcast_data_specific_year = forcast_data(how_much_months_in_year,wining_model_specific_year,data_we_got_to_use_in_prediction_specific_year,flag_for_using_only_part_of_data,how_much_month_back_to_use)

# %%
actual_data_sum_specific_year = actual_data_specific_year.sum(axis=1).resample('YE').sum() / 1e9
forcast_data_sum_specific_year = pd.DataFrame(forcast_data_specific_year).sum(axis=1).resample('YE').sum() / 1e9


# %% [markdown]
# #  data forcast current_year year

# %%
r2_score_values_data_current_year_year = find_r2_score_values_data(how_much_months_in_year,data_as_frame, current_year)
wining_model_current_year_year, r2_of_wining_models_current_year_year = find_wining_models(r2_score_values_data_current_year_year)
forcast_data_current_year_year = forcast_data(how_much_months_in_year - how_much_month_in_current_year_in_data,wining_model_current_year_year,data_we_got_to_use_in_prediction_current_year_year,flag_for_using_only_part_of_data,how_much_month_back_to_use)

data_so_far_current_year = data_as_frame[f'{current_year}-01-01':]
data_so_far_current_year = data_as_frame[data_as_frame.index>"2024-12-31"]
data_current_year = pd.concat([data_so_far_current_year, pd.DataFrame(forcast_data_current_year_year)], axis=0)
data_so_far_current_year_sum = data_current_year.sum(axis=1).sum() 

# %%
forcast_ashbarot_specific_year_in = pd.DataFrame(forcast_data_specific_year).T.groupby(level=1).sum()
forcast_ashbarot_specific_year_in.index.name = 'ZH_in'
forcast_ashbarot_specific_year_in=forcast_ashbarot_specific_year_in*(-1)
forcast_ashbarot_specific_year_out = pd.DataFrame(forcast_data_specific_year).T.groupby(level=0).sum()
forcast_ashbarot_specific_year_out.index.name = 'ZH_out'

forcast_ashbarot_bad_otzar_pairs_specific_year_out = forcast_ashbarot_specific_year_out.where(forcast_ashbarot_specific_year_out == 0, np.nan)
forcast_ashbarot_bad_otzar_pairs_specific_year_in = forcast_ashbarot_specific_year_in.where(forcast_ashbarot_specific_year_in == 0, np.nan)

data_so_far_current_year = data_as_frame[data_as_frame.index>"2024-12-31"]
forcast_ashbarot_current_year = pd.concat([data_so_far_current_year, pd.DataFrame(forcast_data_current_year_year)], axis=0)
forcast_ashbarot_current_year_in = pd.DataFrame(forcast_ashbarot_current_year).T.groupby(level=1).sum()
forcast_ashbarot_current_year_in.index.name = 'ZH_in'
forcast_ashbarot_current_year_in=forcast_ashbarot_current_year_in*(-1)
forcast_ashbarot_current_year_out = pd.DataFrame(forcast_ashbarot_current_year).T.groupby(level=0).sum()
forcast_ashbarot_current_year_out.index.name = 'ZH_out'

forcast_ashbarot_bad_otzar_pairs_current_year_out = forcast_ashbarot_current_year_out.where(forcast_ashbarot_current_year_out == 0, np.nan)
forcast_ashbarot_bad_otzar_pairs_current_year_in = forcast_ashbarot_current_year_in.where(forcast_ashbarot_current_year_in == 0, np.nan)


actual_ashbarot_specific_year_out_yearly = actual_data_specific_year.sum(axis=0).groupby(level=0).sum()
actual_ashbarot_specific_year_out_yearly = pd.DataFrame(actual_ashbarot_specific_year_out_yearly)
actual_ashbarot_specific_year_in_yearly = actual_data_specific_year.sum(axis=0).groupby(level=1).sum() *(-1)
actual_ashbarot_specific_year_in_yearly = pd.DataFrame(actual_ashbarot_specific_year_in_yearly)
# %%

for name, frame in zip(['ZH_in', 'ZH_out'], [actual_ashbarot_specific_year_in_yearly, actual_ashbarot_specific_year_out_yearly]):
    frame.columns = ['actual']
    frame.index.name = IND
    frame.reset_index().to_csv(f"full_actual_{name}_{year_to_predict}.csv")

for name, frame in zip(['ZH_in', 'ZH_out'], [forcast_ashbarot_specific_year_in, forcast_ashbarot_specific_year_out]):
    frame.insert(0, 'kvuzat sahar', f"forcast_{name}_{year_to_predict}.csv")
    frame.index.name = IND
    frame.reset_index().to_csv(f"forcast_{name}_{year_to_predict}.csv")

for name, frame in zip(['ZH_in', 'ZH_out'], [forcast_ashbarot_current_year_in, forcast_ashbarot_current_year_out]):
    frame.insert(0, 'kvuzat sahar', f'forcast_{name}_{current_year}')
    frame.index.name = IND
    frame.reset_index().to_csv(f"forcast_{name}_{current_year}.csv")

for name, frame in zip(['ZH_in', 'ZH_out'], [forcast_ashbarot_bad_otzar_pairs_specific_year_in, forcast_ashbarot_bad_otzar_pairs_specific_year_out]):
    frame.insert(0, 'kvuzat sahar', f'actual_data_{year_to_predict}_bad_otzar_only_{name}')
    frame.index.name = IND
    frame.reset_index().to_csv(f"actual_data_{year_to_predict}_bad_otzar_only_{name}.csv")

for name, frame in zip(['ZH_in', 'ZH_out'], [forcast_ashbarot_bad_otzar_pairs_current_year_in, forcast_ashbarot_bad_otzar_pairs_current_year_out]):
    frame.insert(0, 'kvuzat sahar', f'actual_data_{current_year}_bad_otzar_only_{name}')
    frame.index.name = IND
    frame.reset_index().to_csv(f"actual_data_{current_year}_bad_otzar_only_{name}.csv")