# %%
import argparse
import math
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima.arima import auto_arima
from sklearn.metrics import r2_score
from statsmodels.tsa.holtwinters import Holt
import warnings
warnings.filterwarnings("ignore")


# %% [markdown]
# # Abstract

# %% [markdown]
# # Intreduction
parser = argparse.ArgumentParser(description="Forecasting script")
parser.add_argument("--path", type=str, required=True, help="Path to the CSV file")

# %%
# %% [markdown]
# # Data

# %%
PATH    = parser.parse_args().path
TSCOL   = "IIT_INVOICE_LO_AL_SMAH_NO_EMF_AD_KO"
INDCOLS = ['OTZAR_GROUP', 'DT']


# %%
data = pd.read_csv(PATH, index_col = INDCOLS)
data.columns = [TSCOL]
data.index = data.index.set_levels(pd.to_datetime(data.index.levels[1], format="%Y-%m-%d"), level=1)
data = data[data.index.get_level_values(1) <= "28-02-2025"]


# %%
class TSPreprocessor:
    def __init__(self, data: pd.DataFrame, ts_col: str):
        self._data = data
        self._epsilon = {}
        self._sign = {}
        self.ts_col = ts_col
        self._cache = {}
        self.success = {}
        self._is_calculated = False

    def fit_transform(self) -> dict[str, pd.Series]:
        if self._is_calculated:
            return self._cache
        ts_column = self.ts_col
        transformed_data = self._data.copy()
        results = {}
        for ozar_group in transformed_data.index.get_level_values(0).unique():
            results[ozar_group] = transformed_data.loc[ozar_group, ts_column]
            results[ozar_group] = get_monthly_values(results[ozar_group])
            results[ozar_group] = results[ozar_group].asfreq("ME")
        self._is_calculated = True
        self._cache = results
        return results

def get_monthly_values(data):
    temp = data.copy()
    temp = temp.groupby(temp.index.year).diff()
    temp.loc[temp.index.month == 1] = data.loc[data.index.month == 1]  # The first element is not nan but rather the original value.
    return temp


# %%
preprocessor = TSPreprocessor(data, TSCOL)
data_by_ozar_groups = preprocessor.fit_transform()
data_by_ozar_groups = pd.DataFrame(data_by_ozar_groups)

# %%
class NaiveModel:    
    def __init__(self, data):
        self.data = data

    def fit(self):
        return self

    def forecast(self, steps_to_forecast) -> pd.Series:
        return pd.Series(self.data.values[-1], index=pd.date_range(self.data.index[-1] + pd.offsets.MonthEnd(1), periods=steps_to_forecast, freq='ME'))


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
            except ValueError as e:  # if auto_arima fails we consider it a TSConvergenceError
                raise TSConvergenceError from e
        return self

    def forecast(self, steps_to_forecast):
        try:
            return self.model.predict(steps_to_forecast)
        except ValueError as e:  # if auto_arima fails we consider it a TSConvergenceError
            raise TSConvergenceError from e

class TSModel4:
    def __init__(self, data_by_ozar_groups, year_to_forcast):
        self.data_by_ozar_groups = data_by_ozar_groups
        self.r2_score_values = {}
        self.testData = {}
        self.forecastData = {}
        self.tillpastYearData = {}
        self.bad_otzar_groups = []
        self.year_to_forecast = year_to_forcast

    def fit(self, size_of_validation_data,modelType):
        for i, group in enumerate(self.data_by_ozar_groups.columns):
            group_data = self.data_by_ozar_groups[group].dropna()
            if (group_data.count() < 2 * size_of_validation_data) or (group_data.iloc[-2 * size_of_validation_data:].sum() == 0) or (group_data.index[-1].year < self.year_to_forecast):
                self.bad_otzar_groups.append(group)
            else:
                train_data, test_data = group_data[:-size_of_validation_data], group_data[-size_of_validation_data:]
                model = modelType(train_data)
                model_fit = model.fit()
                forecast = model_fit.forecast(12)
                self.testData[group] = test_data
                self.forecastData[group] = forecast
                self.tillpastYearData[group] = train_data
                r2_score_value = r2_score(test_data, forecast)
                self.r2_score_values[group] = r2_score_value
        
    def bad_otzar(self):
        return self.bad_otzar_groups
                
    def r2_score(self):
        return self.r2_score_values





# %%
templates = {
 "holt": Holt,
  'sarima': SARIMAX,
   'naive': NaiveModel,
  'snaive': SeasonalNaiveModel,
   "ExponentialSmoothing": ExponentialSmoothing,
}

# %% [markdown]
# # experiment data forcast specific year
# 

# %%
year_to_predict = 2020
# add 2022 and 2021

how_much_months_in_year=12
how_much_month_in_2025_in_data=2
current_year = 2025
how_many_years_look_back = (current_year - year_to_predict) * how_much_months_in_year
actual_data = data_by_ozar_groups.iloc[len(data_by_ozar_groups)-how_much_month_in_2025_in_data-how_many_years_look_back : len(data_by_ozar_groups)-how_much_month_in_2025_in_data-how_many_years_look_back+how_much_months_in_year].fillna(0)
data_we_got_to_use_in_prediction = data_by_ozar_groups[:-(how_much_month_in_2025_in_data+how_many_years_look_back)].fillna(0)


r2_score_values_data = {}
bad_otzar_groups_specific_year = []
data_split_month= 12
for key in templates:
    model = TSModel4(data_by_ozar_groups,year_to_predict)
    bad_otzar_groups_specific_year= model.bad_otzar()
    model.fit(data_split_month,templates[key])
    r2_score_values_data[key] = model.r2_score()
r2_score_values_data=pd.DataFrame(r2_score_values_data)


wining_model={}
r2_of_wining_models = {}
for i in r2_score_values_data.index:
    wining_model[i] = r2_score_values_data.columns[r2_score_values_data.loc[i].values == r2_score_values_data.loc[i].values.max()]
    r2_of_wining_models[i] = r2_score_values_data.loc[i].max()


# %%
month_do_predict = 12
forcast_data_specific_year = {}
for i, kvotzat_otzar_sahar in enumerate(wining_model):
    model = templates[wining_model[kvotzat_otzar_sahar][0]](data_we_got_to_use_in_prediction[kvotzat_otzar_sahar].dropna())
    model_fit = model.fit()
    forecast = model_fit.forecast(month_do_predict)
    forcast_data_specific_year[kvotzat_otzar_sahar] = forecast


# %%
kvotzot_otzar_got_changed=[]
month_do_predict = 12
for kvotzat_otzar_sahar in forcast_data_specific_year:
    if wining_model[kvotzat_otzar_sahar][0] == 'holt' or wining_model[kvotzat_otzar_sahar][0] == 'ExponentialSmoothing':
        if forcast_data_specific_year[kvotzat_otzar_sahar].sum() < 0:
            kvotzot_otzar_got_changed.append(kvotzat_otzar_sahar)
            model = NaiveModel(data_we_got_to_use_in_prediction[kvotzat_otzar_sahar].dropna())
            model_fit = model.fit()
            forecast = model_fit.forecast(month_do_predict)
            forcast_data_specific_year[kvotzat_otzar_sahar] = forecast
    else:
        if forcast_data_specific_year[kvotzat_otzar_sahar].sum() < 0:
            kvotzot_otzar_got_changed.append(kvotzat_otzar_sahar)
            forcast_data_specific_year[kvotzat_otzar_sahar] = SeasonalNaiveModel(data_we_got_to_use_in_prediction[kvotzat_otzar_sahar].dropna()) 
            model_fit = model.fit()
            forecast = model_fit.forecast(month_do_predict)
            forcast_data_specific_year[kvotzat_otzar_sahar] = forecast

# %%
actual_data_sum_specific_year = actual_data.sum(axis=1).resample('YE').sum()
forcast_data_sum_specific_year = pd.DataFrame(forcast_data_specific_year).sum(axis=1).resample('YE').sum()

# %% [markdown]
# # Dealing with Bad Otzar Group
# ### A "Bad Otzar Group" is defined as one of the following:
# ### 1. Insufficient data: There is no data available for the last 12 months.
# ### 2. Outdated data: The most recent date in the data is not today's date (2025-02-28).
# ###
# ### These conditions need to be checked before further processing to ensure
# ### that the Otzar group has valid and up-to-date data.
# 

# %%
data_by_ozar_groups[bad_otzar_groups_specific_year].loc["2024"].sum().sum() / 1e6

# %% [markdown]
# # forcast

# %%
r2_score_values_data = {}
bad_otzar_groups_2025 = []
data_split_month= 12
year_to_forcast = 2025
for key in templates:
    model = TSModel4(data_by_ozar_groups, year_to_forcast)
    model.fit(data_split_month,templates[key])
    r2_score_values_data[key] = model.r2_score()
    bad_otzar_groups_2025= model.bad_otzar()
r2_score_values_data=pd.DataFrame(r2_score_values_data)
print(model.bad_otzar())


# %%
wining_model={}
r2_of_wining_models = {}
for i in r2_score_values_data.index:
    wining_model[i] = r2_score_values_data.columns[r2_score_values_data.loc[i].values == r2_score_values_data.loc[i].values.max()]
    r2_of_wining_models[i] = r2_score_values_data.loc[i].max()


# %%
month_do_predict = 10
forcast_data_2025 = {}
for i, kvotzat_otzar_sahar in enumerate(wining_model):
    model = templates[wining_model[kvotzat_otzar_sahar][0]](data_by_ozar_groups[kvotzat_otzar_sahar].dropna())
    model_fit = model.fit()
    forecast = model_fit.forecast(month_do_predict)
    forcast_data_2025[kvotzat_otzar_sahar] = forecast

# %%
kvotzot_otzar_got_changed=[]
month_do_predict = 10
for kvotzat_otzar_sahar in forcast_data_2025:
    if wining_model[kvotzat_otzar_sahar][0] == 'holt' or wining_model[kvotzat_otzar_sahar][0] == 'ExponentialSmoothing':
        if forcast_data_2025[kvotzat_otzar_sahar].sum() < 0:
            kvotzot_otzar_got_changed.append(kvotzat_otzar_sahar)
            model = NaiveModel(data_by_ozar_groups[kvotzat_otzar_sahar])
            model_fit = model.fit()
            forecast = model_fit.forecast(month_do_predict)
            forcast_data_2025[kvotzat_otzar_sahar] = forecast
    else:
        if forcast_data_2025[kvotzat_otzar_sahar].sum() < 0:
            kvotzot_otzar_got_changed.append(kvotzat_otzar_sahar)
            forcast_data_2025[kvotzat_otzar_sahar] = SeasonalNaiveModel(data_by_ozar_groups[kvotzat_otzar_sahar]) 
            model_fit = model.fit()
            forecast = model_fit.forecast(month_do_predict)
            forcast_data_2025[kvotzat_otzar_sahar] = forecast

# %%
data_so_far_2025 = data_by_ozar_groups['2025-01-01':]

# %%
data_so_far_2025_sum = data_so_far_2025.sum(axis=1)
forcast_data_sum = pd.DataFrame(forcast_data_2025).sum(axis=0)
(forcast_data_sum.sum() + data_so_far_2025_sum.sum()) / 1e9

# %%
# %% [markdown]
# # tO CSV
# 

# %%
expanditure_name = PATH.split('-')[1]

actual_spesific_year = actual_data.sum(axis = 0).rename('actual')
actual_spesific_year.index.name = 'kvotzat otzar'
actual_spesific_year.to_csv(f"full_actual_{expanditure_name}_{year_to_predict}.csv")

forcast_specific_year = pd.DataFrame(forcast_data_specific_year).T 
forcast_specific_year.insert(0, 'kvuzat sahar', f"forcast_{expanditure_name}_{year_to_predict}.csv")
forcast_specific_year.index.name = 'kvotzat otzar'
forcast_specific_year.to_csv(f"forcast_{expanditure_name}_{year_to_predict}.csv")

forcast_data_2025 = pd.DataFrame(forcast_data_2025).fillna(0)
data_so_far_2025=data_by_ozar_groups['2025-01-01':]
data_so_far_2025=data_so_far_2025.T[data_so_far_2025.columns.isin(forcast_data_2025.columns)].T
forcast_2025_combined = pd.concat([data_so_far_2025,forcast_data_2025.iloc[2:]]).T
forcast_2025_combined.insert(0, 'kvuzat sahar', f'forcast_{expanditure_name}_2025')
forcast_2025_combined.index.name = 'kvotzat otzar'
forcast_2025_combined.to_csv(f"forcast_{expanditure_name}_2025.csv")

actual_data_specific_year__bad_otzar_only = data_by_ozar_groups[bad_otzar_groups_specific_year].loc[str(year_to_predict)]
actual_data_specific_year__bad_otzar_only=actual_data_specific_year__bad_otzar_only.T
actual_data_specific_year__bad_otzar_only.insert(0, 'kvuzat sahar', f'actual_data_{year_to_predict}_bad_otzar_only_{expanditure_name}')
actual_data_specific_year__bad_otzar_only.index.name = 'kvotzat otzar'
actual_data_specific_year__bad_otzar_only.to_csv(f"actual_data_{year_to_predict}_bad_otzar_only_{expanditure_name}.csv")

actual_data_2025__bad_otzar_only = data_by_ozar_groups[bad_otzar_groups_2025].loc[str(current_year)]
actual_data_2025__bad_otzar_only=actual_data_2025__bad_otzar_only.T
actual_data_2025__bad_otzar_only.insert(0, 'kvuzat sahar', f'actual_data_2025_bad_otzar_only_{expanditure_name}')
actual_data_2025__bad_otzar_only.index.name = 'kvotzat otzar'
actual_data_2025__bad_otzar_only.to_csv(f"actual_data_2025_bad_otzar_only_{expanditure_name}.csv")


