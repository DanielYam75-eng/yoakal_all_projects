import argparse
import math
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from sklearn.metrics import r2_score
from statsmodels.tsa.holtwinters import Holt
import warnings

warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

class DummyModel:
    def __init__(self, index):
        self.index = index

    def fit(self):
        return self

    def forecast(self, steps_to_forecast) -> pd.Series:
        return pd.Series(0, index=self.index)


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
    temp.loc[temp.index.month == 1] = data.loc[
        data.index.month == 1
    ]  # The first element is not nan but rather the original value.
    return temp

class AvgFactorModel:
    def __init__(self, data):
        self.data = data

    def fit(self):
        return self

    def forecast(self, steps_to_forecast) -> pd.Series:
        last_12_month = self.data.iloc[-12:].sum(axis=0)
        last_24_month = self.data.iloc[-24:-12].sum(axis=0)
        if last_24_month != 0:
            factor = last_12_month / last_24_month
        else:
            factor = 0
        return pd.Series(
            factor * last_12_month / 12,
            index=pd.date_range(
                self.data.index[-1] + pd.offsets.MonthEnd(1),
                periods=steps_to_forecast,
                freq="ME",
            ),
        )

class NaiveModel:
    def __init__(self, data):
        self.data = data

    def fit(self):
        return self

    def forecast(self, steps_to_forecast) -> pd.Series:
        return pd.Series(
            self.data.values[-1],
            index=pd.date_range(
                self.data.index[-1] + pd.offsets.MonthEnd(1),
                periods=steps_to_forecast,
                freq="ME",
            ),
        )

class SeasonalNaiveModel:
    def __init__(self, data, seasonality=12):
        self.data = data
        self.seasonality = seasonality

    def fit(self):
        return self

    def _forecast_h(self, h):
        p = math.ceil(h / self.seasonality)
        return self.data.values[h - self.seasonality * p - 1]

    def forecast(self, steps_to_forecast) -> pd.Series:
        return pd.Series(
            [self._forecast_h(h) for h in range(1, steps_to_forecast + 1)],
            index=pd.date_range(
                self.data.index[-1] + pd.offsets.MonthEnd(1),
                periods=steps_to_forecast,
                freq="ME",
            ),
        )

class TSConvergenceError(Exception):
    pass

class TSModel4:
    def __init__(self, data_by_ozar_groups, year_to_forcast):
        self.data_by_ozar_groups = data_by_ozar_groups
        self.r2_score_values = {}
        self.testData = {}
        self.forecastData = {}
        self.tillpastYearData = {}
        self.bad_otzar_groups = []
        self.year_to_forecast = year_to_forcast

    def fit(self, size_of_validation_data, modelType):
        for i, group in enumerate(self.data_by_ozar_groups.columns):
            group_data = self.data_by_ozar_groups[group].dropna()
            if (
                (group_data.count() < 2 * size_of_validation_data)
                or (group_data.iloc[-2 * size_of_validation_data :].sum() == 0)
                or (group_data.index[-1].year < self.year_to_forecast - 1)
            ):
                self.bad_otzar_groups.append(group)
            else:
                train_data, test_data = (
                    group_data[:-size_of_validation_data],
                    group_data[-size_of_validation_data:],
                )
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

class MeanModel:
    def __init__(self, data):
        self.data = data

    def fit(self):
        return self

    def forecast(self, steps_to_forecast) -> pd.Series:
        return pd.Series(
            self.data.mean(),
            index=pd.date_range(
                self.data.index[-1] + pd.offsets.MonthEnd(1),
                periods=steps_to_forecast,
                freq="ME",
            ),
        )

class MonthlylModel:
    def __init__(self, data):
        self.data = data
        self.season = data.groupby(data.index.month).mean()

    def fit(self):
        return self

    def predict(self, index):
        prediction = pd.Series(index=index)
        for i in index:
            prediction.loc[i] = self.season.loc[i.month]
        return prediction

    def forecast(self, steps_to_forecast):
        last_date = self.data.index[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.offsets.MonthEnd(1),
            periods=steps_to_forecast,
            freq="ME",
        )
        return self.predict(forecast_index)

class SeasonalLinearModel:
    def __init__(self, data):
        self.data = data
        self.data_size = None
        self.TrendModel = LinearRegression()
        self.SeasonalModel = None

    def fit(self):
        y = self.data.rolling(12).mean()
        X = (
            np.arange(len(y)).reshape(-1, 1) - 11
        )  # Adjusting for the loss of the first 11 months due to rolling mean
        self.TrendModel.fit(np.arange(len(y.dropna())).reshape(-1, 1), y.dropna())
        self.data_size = len(y.dropna())
        self.SeasonalModel = MonthlylModel(
            self.data / pd.Series(self.TrendModel.predict(X), index=y.index)
        ).fit()
        return self

    def forecast(self, steps_to_forecast):
        last_date = self.data.index[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.offsets.MonthEnd(1),
            periods=steps_to_forecast,
            freq="ME",
        )
        forecast_tensor = np.arange(
            self.data_size, self.data_size + steps_to_forecast
        ).reshape(-1, 1)
        return pd.Series(
            self.TrendModel.predict(forecast_tensor).reshape(-1)
            * self.SeasonalModel.predict(forecast_index),
            index=forecast_index,
        )

def find_r2_score_values_data(
    how_much_months_in_year, data_by_ozar_groups, year_to_predict, templates
):
    r2_score_values_data = {}
    bad_otzar_groups_specific_year = []
    for key in templates:
        model = TSModel4(data_by_ozar_groups, year_to_predict)
        bad_otzar_groups_specific_year = model.bad_otzar()
        model.fit(how_much_months_in_year, templates[key])
        r2_score_values_data[key] = model.r2_score()
    r2_score_values_data = pd.DataFrame(r2_score_values_data)
    return r2_score_values_data, bad_otzar_groups_specific_year

def find_wining_models(r2_score_values_data_specific_year):
    wining_model = {}
    r2_of_wining_models = {}
    for i in r2_score_values_data_specific_year.index:
        wining_model[i] = r2_score_values_data_specific_year.columns[
            r2_score_values_data_specific_year.loc[i].values
            == r2_score_values_data_specific_year.loc[i].values.max()
        ]
        r2_of_wining_models[i] = r2_score_values_data_specific_year.loc[i].max()
    return wining_model, r2_of_wining_models

def forcast_data(
    month_to_predict,
    wining_model_specific_year,
    data_we_got_to_use_in_prediction,
    flag_for_using_only_part_of_data,
    how_much_month_back_to_use,
    forecast_index,
    templates
):
    forcast_data_specific_year = {}
    for i, kvotzat_otzar_sahar in enumerate(wining_model_specific_year):
        if flag_for_using_only_part_of_data:
            if data_we_got_to_use_in_prediction[kvotzat_otzar_sahar][
                -how_much_month_back_to_use:
            ].empty:
                model = DummyModel(forecast_index)
            else:
                model = templates[
                    wining_model_specific_year[kvotzat_otzar_sahar][0]
                ](
                    data_we_got_to_use_in_prediction[kvotzat_otzar_sahar][
                        -how_much_month_back_to_use:
                    ]
                )
        else:
            if data_we_got_to_use_in_prediction[kvotzat_otzar_sahar].empty:
                model = DummyModel(forecast_index)
            else:
                model = templates[
                    wining_model_specific_year[kvotzat_otzar_sahar][0]
                ](data_we_got_to_use_in_prediction[kvotzat_otzar_sahar])
        model_fit = model.fit()
        forecast = model_fit.forecast(month_to_predict)
        if not data_we_got_to_use_in_prediction[kvotzat_otzar_sahar].empty:
            forecast.index = pd.date_range(
                data_we_got_to_use_in_prediction[kvotzat_otzar_sahar].index[-1]
                + pd.offsets.MonthEnd(1),
                periods=month_to_predict,
                freq="ME",
            )

        else:
            forecast.index = forecast_index
        forcast_data_specific_year[kvotzat_otzar_sahar] = forecast
    return forcast_data_specific_year

