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

# %%
parser = argparse.ArgumentParser(description="Forecasting script")
parser.add_argument("--path", type=str, required=True, help="Path to the CSV file")

PATH    = parser.parse_args().path
TSCOL   = "IIT_INVOICE_LO_AL_SMAH_NO_EMF_AD_KO"
INDCOLS = ['OTZAR_GROUP', 'DT']


# %%
data = pd.read_csv(PATH, index_col = INDCOLS)
data.columns = [TSCOL]
data.index = data.index.set_levels(pd.to_datetime(data.index.levels[1], format="%Y-%m-%d"), level=1)
data = data[data.index.get_level_values(1) <= "28-02-2025"]
data.head()

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
class datePreprocessor:
    def __init__(self, data:pd.DataFrame):
        self.data = data
        self.till_wanted_year = None
        self.from_wanted_year = None

    def split_data(self, split_month):
        self.till_wanted_year = self.data[:split_month]
        self.from_wanted_year = self.data[split_month:]
        return self.till_wanted_year, self.from_wanted_year


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
    def __init__(self, data_by_ozar_groups):
        self.data_by_ozar_groups = data_by_ozar_groups
        self.r2_score_values = {}
        self.testData = {}
        self.forecastData = {}
        self.tillpastYearData = {}
        self.bad_otzar_groups = []
    
    def fit(self, size_of_validation_data,modelType):
        for i, group in enumerate(self.data_by_ozar_groups.columns):
            group_data = data_by_ozar_groups[group]
            group_data = group_data.dropna()
            if (group_data.count() < 2 * size_of_validation_data) or (group_data.iloc[-2 * size_of_validation_data:] == 0) or (group_data.index[-1] < pd.Timestamp("2024-01-31"):
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

# %%
r2_score_values_data = {}
bad_otzar_groups = []
data_split_month= 12
for key in templates:
    model = TSModel4(data_by_ozar_groups)
    model.fit(data_split_month,templates[key])
    r2_score_values_data[key] = model.r2_score()
    bad_otzar_groups = model.bad_otzar_groups
r2_score_values_data=pd.DataFrame(r2_score_values_data)

# %%
wining_model={}
r2_of_wining_models = {}
for i in r2_score_values_data.index:
    wining_model[i] = r2_score_values_data.columns[r2_score_values_data.loc[i].values == r2_score_values_data.loc[i].values.max()]
    r2_of_wining_models[i] = r2_score_values_data.loc[i].max()


# %%
how_much_months_in_year=12
how_much_month_in_2025=2
actual_data = data_by_ozar_groups.iloc[len(data_by_ozar_groups)-how_much_month_in_2025-how_much_months_in_year : len(data_by_ozar_groups)-how_much_month_in_2025]
data_we_got_to_use_in_prediction = data_by_ozar_groups[:-(how_much_month_in_2025+how_much_months_in_year)]

# %%
r2_score_values_data = {}
bad_otzar_groups = []
data_split_month= 12
for key in templates:
    model = TSModel4(data_we_got_to_use_in_prediction)
    model.fit(data_split_month,templates[key])
    r2_score_values_data[key] = model.r2_score()
    bad_otzar_groups = model.bad_otzar_groups
r2_score_values_data=pd.DataFrame(r2_score_values_data)

# %%
month_do_predict = 12
forcast_data_2024 = {}
for i, kvotzat_otzar_sahar in enumerate(wining_model):
    model = templates[wining_model[kvotzat_otzar_sahar][0]](data_we_got_to_use_in_prediction[kvotzat_otzar_sahar].dropna())
    model_fit = model.fit()
    forecast = model_fit.forecast(month_do_predict)
    forcast_data_2024[kvotzat_otzar_sahar] = forecast



# %%
kvotzot_otzar_got_changed=[]
data_split_month= 12
for kvotzat_otzar_sahar in forcast_data_2024:
    if wining_model[kvotzat_otzar_sahar][0] == 'holt' or wining_model[kvotzat_otzar_sahar][0] == 'ExponentialSmoothing':
        if forcast_data_2024[kvotzat_otzar_sahar].sum() < 0:
            kvotzot_otzar_got_changed.append(kvotzat_otzar_sahar)
            model = NaiveModel(data_we_got_to_use_in_prediction[kvotzat_otzar_sahar].dropna())
            model_fit = model.fit()
            forecast = model_fit.forecast(data_split_month)
            forcast_data_2024[kvotzat_otzar_sahar] = forecast
    else:
        if forcast_data_2024[kvotzat_otzar_sahar].sum() < 0:
            kvotzot_otzar_got_changed.append(kvotzat_otzar_sahar)
            forcast_data_2024[kvotzat_otzar_sahar] = SeasonalNaiveModel(data_we_got_to_use_in_prediction[kvotzat_otzar_sahar].dropna()) 
            model_fit = model.fit()
            forecast = model_fit.forecast(data_split_month)
            forcast_data_2024[kvotzat_otzar_sahar] = forecast



# %%
actual_data_sum_2024 = actual_data.sum(axis=1).resample('YE').sum()
forcast_data_sum_2024 = pd.DataFrame(forcast_data_2024).sum(axis=1).resample('YE').sum()


# %%
month_do_predict = 10
forcast_data_2025 = {}
for i, kvotzat_otzar_sahar in enumerate(wining_model):
    model = templates[wining_model[kvotzat_otzar_sahar][0]](data_by_ozar_groups[kvotzat_otzar_sahar].dropna()[-12:])
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
            model = NaiveModel(data_by_ozar_groups[kvotzat_otzar_sahar].dropna()[-12:])
            model_fit = model.fit()
            forecast = model_fit.forecast(month_do_predict)
            forcast_data_2025[kvotzat_otzar_sahar] = forecast
    else:
        if forcast_data_2025[kvotzat_otzar_sahar].sum() < 0:
            kvotzot_otzar_got_changed.append(kvotzat_otzar_sahar)
            forcast_data_2025[kvotzat_otzar_sahar] = SeasonalNaiveModel(data_by_ozar_groups[kvotzat_otzar_sahar].dropna()[-12:]) 
            model_fit = model.fit()
            forecast = model_fit.forecast(10)
            forcast_data_2025[kvotzat_otzar_sahar] = forecast


# %%
data_so_far_2025 = data_by_ozar_groups['2025-01-01':]

# %%
data_so_far_2025_sum = data_so_far_2025.sum(axis=1)
forcast_data_sum = pd.DataFrame(forcast_data_2025).sum(axis=0)
(forcast_data_sum.sum() + data_so_far_2025_sum.sum()) / 1e9

# %%
data_sum = data_by_ozar_groups.sum(axis=1).resample('YE').sum()
data_sum[len(data_sum)-1] = (forcast_data_sum.sum() + data_so_far_2025_sum.sum())
# %%
expanditure_name = PATH.split('-')[1]

forcast_2024 = pd.DataFrame(forcast_data_2024).T
forcast_2024.insert(0, 'kvuzat sahar', f'forcast_{expanditure_name}_2024')
forcast_2024.index.name = 'kvotzat otzar'
forcast_2024.to_csv(f"forcast_{expanditure_name}_2024.csv")

forcast_data_2025 = pd.DataFrame(forcast_data_2025).fillna(0)
data_so_far_2025=data_by_ozar_groups['2025-01-01':]
data_so_far_2025=data_so_far_2025.T[data_so_far_2025.columns.isin(forcast_data_2025.columns)].T
forcast_2025 = pd.concat([data_so_far_2025,forcast_data_2025]).T
forcast_2025.insert(0, 'kvuzat sahar', f'forcast_{expanditure_name}_2025')
forcast_2025.index.name = 'kvotzat otzar'
forcast_2025.to_csv(f"forcast_{expanditure_name}_2025.csv")

mask = ~actual_data.columns.isin(forcast_data_2024)
actual_data_2024_without_bad_otzar_only = actual_data.loc[:, mask]
actual_data_2024_without_bad_otzar_only=actual_data_2024_without_bad_otzar_only.T
actual_data_2024_without_bad_otzar_only.insert(0, 'kvuzat sahar', f'actual_data_2024_without_bad_otzar_only_{expanditure_name}')
actual_data_2024_without_bad_otzar_only.index.name = 'kvotzat otzar'
actual_data_2024_without_bad_otzar_only.to_csv(f"actual_data_2024_without_bad_otzar_only_{expanditure_name}.csv")

mask = actual_data.columns.isin(forcast_data_2024)
actual_data_2024_with_bad_otzar_only = actual_data.loc[:, mask]
actual_data_2024_with_bad_otzar_only=actual_data_2024_with_bad_otzar_only.T
actual_data_2024_with_bad_otzar_only.insert(0, 'kvuzat sahar', f'actual_data_2024_with_bad_otzar_only_{expanditure_name}')
actual_data_2024_with_bad_otzar_only.index.name = 'kvotzat otzar'
actual_data_2024_with_bad_otzar_only.to_csv(f"actual_data_2024_with_bad_otzar_only_{expanditure_name}.csv")