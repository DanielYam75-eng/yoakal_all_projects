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
from . import models
from .models import TSPreprocessor, SeasonalNaiveModel, AvgFactorModel, NaiveModel, TSModel4, find_r2_score_values_data
from .models import find_wining_models, forcast_data

class DummyModel:
    def __init__(self, index):
        self.index = index

    def fit(self):
        return self

    def forecast(self, steps_to_forecast) -> pd.Series:
        return pd.Series(0, index=self.index)


def main(path, type_, past_year, curr_year, curr_month, months_back, coin_type):
    year_to_predict = past_year
    how_much_month_in_curr_year_in_data = curr_month
    how_much_month_back_to_use = months_back
    current_year = curr_year
    coin_type = coin_type
    how_much_months_in_year = 12
    flag_for_using_only_part_of_data = how_much_month_back_to_use != -1
    PATH = path
    TSCOL = "IIT_INVOICE_LO_AL_SMAH_NO_EMF_AD_KO"
    INDCOLS = ["OTZAR_GROUP", "DT"]



    # Define the templates based on the type
    if coin_type == 1:
        if type_ == "career_salary":
            templates = {
                # "SeasonalLinear": SeasonalLinearModel,
                "avg_factor": models.AvgFactorModel,
                "naive": models.NaiveModel,
                "snaive": models.SeasonalNaiveModel,
            }

        elif type_ == "drafted_salary":

            templates = {
                "SeasonalLinear": models.SeasonalLinearModel,
                "avg_factor": models.AvgFactorModel,
                "naive": models.NaiveModel,
                "snaive": models.SeasonalNaiveModel,
            }

        elif type_ == "pensions":

            templates = {
                "SeasonalLinear": models.SeasonalLinearModel,
                #'avg_factor' : AvgFactorModel,
                # "naive": NaiveModel,
                # "snaive": SeasonalNaiveModel,
            }

        elif type_ == "idf_workers_salary":

            templates = {
                # "SeasonalLinear": SeasonalLinearModel,
                "avg_factor": models.AvgFactorModel,
                "naive": models.NaiveModel,
                "snaive": models.SeasonalNaiveModel,
            }

        elif type_ == "dd_workers_salary":

            templates = {
                # "SeasonalLinear": SeasonalLinearModel,
                "avg_factor": models.AvgFactorModel,
                "naive": models.NaiveModel,
                "snaive": models.SeasonalNaiveModel,
            }

        elif type_ == "pre_draft_salary":

            templates = {
                "SeasonalLinear": models.SeasonalLinearModel,
                "avg_factor": models.AvgFactorModel,
                # "naive": NaiveModel,
                # "snaive": SeasonalNaiveModel,
            }

        elif type_ == "additional_drafted_service_salary":

            templates = {
                # "SeasonalLinear": SeasonalLinearModel,
                "avg_factor": models.AvgFactorModel,
                # "naive": NaiveModel,
                # "snaive": SeasonalNaiveModel,
            }

        elif type_ == "commemoration":

            templates = {
                "holt": Holt,
                #'sarima': SARIMAX,
                #'naive': NaiveModel,
                #'snaive': SeasonalNaiveModel,
                "ExponentialSmoothing": models.ExponentialSmoothing,
                # "SeasonalLinear": SeasonalLinearModel,
                #'SimpleExpSmoothing' : SimpleExpSmoothing,
                #'mean': MeanModel,
                "avg_factor": models.AvgFactorModel,
            }

        elif type_ == "affilated_other":

            templates = {
                "holt": Holt,
                #'sarima': SARIMAX,
                "naive": models.NaiveModel,
                "snaive": models.SeasonalNaiveModel,
                "ExponentialSmoothing": models.ExponentialSmoothing,
            }

        elif type_ == "arnona":

            templates = {
                "holt": Holt,
                #'sarima': SARIMAX,
                #'naive': NaiveModel,
                "snaive": models.SeasonalNaiveModel,
                "ExponentialSmoothing": models.ExponentialSmoothing,
                "SimpleExpSmoothing": models.SimpleExpSmoothing,
                #'mean': MeanModel,
                "avg_factor": models.AvgFactorModel,
            }

        elif type_ == "KM":

            templates = {
                "holt": models.Holt,
                "sarima": models.SARIMAX,
                "naive": models.NaiveModel,
                "snaive": models.SeasonalNaiveModel,
                "ExponentialSmoothing": models.ExponentialSmoothing,
                # 'SimpleExpSmoothing' : SimpleExpSmoothing,
                #'mean': MeanModel,
            }

        elif type_ == "KT":

            templates = {
                "holt": models.Holt,
                "sarima": models.SARIMAX,
                "naive": models.NaiveModel,
                "snaive": models.SeasonalNaiveModel,
                "ExponentialSmoothing": models.ExponentialSmoothing,
                # 'SimpleExpSmoothing' : SimpleExpSmoothing,
                #'mean': MeanModel,
            }

        elif type_ == "electricity":

            templates = {
                "holt": models.Holt,
                "sarima": models.SARIMAX,
                "naive": models.NaiveModel,
                "snaive": models.SeasonalNaiveModel,
                "ExponentialSmoothing": models.ExponentialSmoothing,
                # 'SimpleExpSmoothing' : SimpleExpSmoothing,
                #'mean': MeanModel,
            }

        elif type_ == "water":

            templates = {
                "seasonal_linear": models.SeasonalLinearModel,
                # "holt": Holt,
                #'sarima': SARIMAX,
                #'naive': NaiveModel,
                "snaive": models.SeasonalNaiveModel,
                # "ExponentialSmoothing": ExponentialSmoothing,
                # 'SimpleExpSmoothing' : SimpleExpSmoothing,
                #'mean': MeanModel,
                "avg_factor": models.AvgFactorModel,
            }
        elif type_ == "vehicles":

            templates = {
                "holt": models.Holt,
                "sarima": models.SARIMAX,
                "naive": models.NaiveModel,
                "snaive": models.SeasonalNaiveModel,
                "ExponentialSmoothing": models.ExponentialSmoothing,
                # 'SimpleExpSmoothing' : SimpleExpSmoothing,
                #'mean': MeanModel,
            }

        elif type_ == "overseas_transportation":

            templates = {
                "holt": models.Holt,
                "sarima": models.SARIMAX,
                "naive": models.NaiveModel,
                "snaive": models.SeasonalNaiveModel,
                "ExponentialSmoothing": models.ExponentialSmoothing,
                # 'SimpleExpSmoothing' : SimpleExpSmoothing,
                #'mean': MeanModel,
            }

        elif type_ == "tariffs":

            templates = {
                "holt": models.Holt,
                "sarima": models.SARIMAX,
                "naive": models.NaiveModel,
                "snaive": models.SeasonalNaiveModel,
                "ExponentialSmoothing": models.ExponentialSmoothing,
                # 'SimpleExpSmoothing' : SimpleExpSmoothing,
                #'mean': MeanModel,
            }

        elif type_ == "insurance":

            templates = {
                "holt": models.Holt,
                "sarima": models.SARIMAX,
                "naive": models.NaiveModel,
                "snaive": models.SeasonalNaiveModel,
                "ExponentialSmoothing": models.ExponentialSmoothing,
                # 'SimpleExpSmoothing' : SimpleExpSmoothing,
                #'mean': MeanModel,
            }

        elif type_ == "special_compensation":

            templates = {
                "holt": models.Holt,
                "sarima": models.SARIMAX,
                "naive": models.NaiveModel,
                "snaive": models.SeasonalNaiveModel,
                "ExponentialSmoothing": models.ExponentialSmoothing,
                # 'SimpleExpSmoothing' : SimpleExpSmoothing,
                #'mean': MeanModel,
            }

        elif type_ == "special_research":

            templates = {
                "holt": models.Holt,
                "sarima": models.SARIMAX,
                "naive": models.NaiveModel,
                "snaive": models.SeasonalNaiveModel,
                "ExponentialSmoothing": models.ExponentialSmoothing,
                # 'SimpleExpSmoothing' : SimpleExpSmoothing,
                #'mean': MeanModel,
            }

        elif type_ == "SA":

            templates = {
                "mean": models.MeanModel,
            }

        elif type_ == "rest":

            templates = {
                # "holt": Holt,
                #'sarima': SARIMAX,
                #'naive': NaiveModel,
                "snaive": models.SeasonalNaiveModel,
                # "ExponentialSmoothing": ExponentialSmoothing,
                #'SimpleExpSmoothing' : SimpleExpSmoothing,
                #'mean': MeanModel,
                #'LinearRegression': SeasonalLinearModel,
                "avg_factor": models.AvgFactorModel,
            }

        elif type_ == "hostages":

            templates = {
                # "holt": Holt,
                #'sarima': SARIMAX,
                #'naive': NaiveModel,
                "snaive": models.SeasonalNaiveModel,
                "ExponentialSmoothing": models.ExponentialSmoothing,
                # 'SimpleExpSmoothing' : SimpleExpSmoothing,
                #'mean': MeanModel,
            }

        elif type_ == "fiancees":

            templates = {
                # "holt": Holt,
                #'sarima': SARIMAX,
                #'naive': NaiveModel,
                "snaive": SeasonalNaiveModel,
                "ExponentialSmoothing": models.ExponentialSmoothing,
                # 'SimpleExpSmoothing' : SimpleExpSmoothing,
                #'mean': MeanModel,
            }
        else:
            raise Exception(f"Type {type_} doesn't exist for path {PATH}.")
    if coin_type == 5:
        if type_ == "ZW":
            templates = {
                "holt": models.Holt,
                #  'sarima': SARIMAX,
                "naive": models.NaiveModel,
                "snaive": models.SeasonalNaiveModel,
                #  "ExponentialSmoothing": ExponentialSmoothing,
                "SimpleExpSmoothing": models.SimpleExpSmoothing,
                #    'mean': MeanModel,
                "LinearRegression": models.SeasonalLinearModel,
                "avg_factor": models.AvgFactorModel,
            }
        elif type_ == "ZC":
            templates = {
                "holt": models.Holt,
                #    'sarima': SARIMAX,
                #    'naive': NaiveModel,
                #    'snaive': SeasonalNaiveModel,
                "ExponentialSmoothing": models.ExponentialSmoothing,
                #    'SimpleExpSmoothing' : SimpleExpSmoothing,
                #    'mean': MeanModel,
                "LinearRegression": models.SeasonalLinearModel,
                "avg_factor": models.AvgFactorModel,
            }
        elif type_ == "travel-KRKG":
            templates = {
                "holt": models.Holt,
                #    'sarima': SARIMAX,
                "naive": models.NaiveModel,
                "snaive": models.SeasonalNaiveModel,
                "ExponentialSmoothing": models.ExponentialSmoothing,
                "SimpleExpSmoothing": models.SimpleExpSmoothing,
                #    'mean': MeanModel,
                "LinearRegression": models.SeasonalLinearModel,
                #    'avg_factor' : AvgFactorModel,
            }
        elif type_ == "14-KRKG":
            templates = {
                "holt": models.Holt,
                #   'sarima': SARIMAX,
                "naive": models.NaiveModel,
                "snaive": models.SeasonalNaiveModel,
                "ExponentialSmoothing": models.ExponentialSmoothing,
                "SimpleExpSmoothing": models.SimpleExpSmoothing,
                #    'mean': MeanModel,
                "LinearRegression": models.SeasonalLinearModel,
                #    'avg_factor' : AvgFactorModel,
            }
        elif type_ == "SA":
            templates = {
                "mean": models.MeanModel,
            }
        elif type_ == "rest":
            templates = {
                "holt": models.Holt,
                #'sarima': SARIMAX,
                "naive": models.NaiveModel,
                "snaive": models.SeasonalNaiveModel,
                "ExponentialSmoothing": models.ExponentialSmoothing,
                "SimpleExpSmoothing": models.SimpleExpSmoothing,
                #'mean': MeanModel,
                "LinearRegression": models.SeasonalLinearModel,
                # 'avg_factor' : AvgFactorModel,
            }
        else:
            raise Exception(f"Type {type_} doesn't exist for path {PATH}.")

    data = pd.read_csv(PATH, index_col=INDCOLS)
    data.columns = [TSCOL]
    data.index = data.index.set_levels(
        pd.to_datetime(data.index.levels[1], format="%Y-%m-%d"), level=1
    )

    preprocessor = TSPreprocessor(data, TSCOL)
    data_by_ozar_groups = preprocessor.fit_transform()
    data_by_ozar_groups = pd.DataFrame(data_by_ozar_groups)

    # data for forcast
    how_many_years_look_back_to_find_specific_year = (
        current_year - year_to_predict
    ) * how_much_months_in_year
    actual_data_specific_year = data_by_ozar_groups.loc[str(year_to_predict)].fillna(0)
    data_we_got_to_use_in_prediction_specific_year = data_by_ozar_groups.loc[
        : str(year_to_predict - 1)
    ].fillna(0)
    data_we_got_to_use_in_prediction_current_year_year = pd.concat(
        [
            data_by_ozar_groups.loc[: str(current_year - 1)],
            data_by_ozar_groups.loc[str(current_year)].head(
                how_much_month_in_curr_year_in_data
            ),
        ]
    ).fillna(0)

    # forcast by specific year

    r2_score_values_data_specific_year, bad_otzar_groups_specific_year = (
        find_r2_score_values_data(
            how_much_months_in_year, data_by_ozar_groups, year_to_predict, templates
        )
    )
    wining_model_specific_year, r2_of_wining_models_specific_year = find_wining_models(
        r2_score_values_data_specific_year
    )
    forcast_data_specific_year = forcast_data(
        how_much_months_in_year,
        wining_model_specific_year,
        data_we_got_to_use_in_prediction_specific_year,
        flag_for_using_only_part_of_data,
        how_much_month_back_to_use,
        pd.DatetimeIndex(
            pd.date_range(f"{curr_year - 1}-01-31", f"{curr_year - 1}-12-31", freq="ME")
        ),
        templates
    )
    actual_data_sum_specific_year = (
        actual_data_specific_year.sum(axis=1).resample("YE").sum()
    )
    forcast_data_sum_specific_year = (
        pd.DataFrame(forcast_data_specific_year).sum(axis=1).resample("YE").sum()
    )

    # forcast by year current_year

    r2_score_values_data_current_year_year, bad_otzar_groups_current_year_year = (
        find_r2_score_values_data(
            how_much_months_in_year, data_by_ozar_groups, current_year, templates
        )
    )
    wining_model_current_year_year, r2_of_wining_models_current_year_year = find_wining_models(
        r2_score_values_data_current_year_year
    )
    forcast_data_current_year_year = forcast_data(
        how_much_months_in_year - how_much_month_in_curr_year_in_data,
        wining_model_current_year_year,
        data_we_got_to_use_in_prediction_current_year_year,
        flag_for_using_only_part_of_data,
        how_much_month_back_to_use,
        pd.DatetimeIndex(
            pd.date_range(
                pd.Timestamp(f"{curr_year}-{curr_month}-01") + pd.offsets.MonthEnd(1),
                f"{curr_year}-12-31",
                freq="ME",
            )
        ),
        templates
    )

    data_so_far_current_year = data_by_ozar_groups.loc[str(current_year)].head(
        how_much_month_in_curr_year_in_data
    )
    forcast_current_year_combined = pd.concat(
        [pd.DataFrame(data_so_far_current_year), pd.DataFrame(forcast_data_current_year_year)]
    ).T

    # exporting data

    expanditure_name = PATH.split("-")[1]

    actual_spesific_year = actual_data_specific_year.sum(axis=0).rename(
        f"actual_{expanditure_name}"
    )
    actual_spesific_year.index.name = "kvotzat otzar"
    actual_spesific_year.to_csv(f"full_actual_{expanditure_name}_{year_to_predict}.csv")

    forcast_specific_year = pd.DataFrame(forcast_data_specific_year).T
    forcast_specific_year.insert(
        0, "kvuzat sahar", f"forcast_{expanditure_name}_{year_to_predict}.csv"
    )
    forcast_specific_year.index.name = "kvotzat otzar"
    forcast_specific_year.to_csv(f"forcast_{expanditure_name}_{year_to_predict}.csv")

    forcast_current_year_combined.insert(
        0, "kvuzat sahar", f"forcast_{expanditure_name}_{current_year}"
    )
    forcast_current_year_combined.index.name = "kvotzat otzar"
    forcast_current_year_combined.to_csv(f"forcast_{expanditure_name}_{current_year}.csv")

    actual_data_specific_year__bad_otzar_only = data_by_ozar_groups[
        bad_otzar_groups_specific_year
    ].loc[str(year_to_predict)]
    actual_data_specific_year__bad_otzar_only = (
        actual_data_specific_year__bad_otzar_only.T
    )
    actual_data_specific_year__bad_otzar_only.insert(
        0,
        "kvuzat sahar",
        f"actual_data_{year_to_predict}_bad_otzar_only_{expanditure_name}",
    )
    actual_data_specific_year__bad_otzar_only.index.name = "kvotzat otzar"
    actual_data_specific_year__bad_otzar_only.to_csv(
        f"actual_data_{year_to_predict}_bad_otzar_only_{expanditure_name}.csv"
    )

    actual_data_current_year__bad_otzar_only = data_by_ozar_groups[
        bad_otzar_groups_current_year_year
    ].loc[str(current_year)]
    actual_data_current_year__bad_otzar_only = actual_data_current_year__bad_otzar_only.T
    actual_data_current_year__bad_otzar_only.insert(
        0,
        "kvuzat sahar",
        f"actual_data_{current_year}_bad_otzar_only_{expanditure_name}",
    )
    actual_data_current_year__bad_otzar_only.index.name = "kvotzat otzar"
    actual_data_current_year__bad_otzar_only.to_csv(
        f"actual_data_{current_year}_bad_otzar_only_{expanditure_name}.csv"
    )

    print("finished for", expanditure_name)


if __name__ == "__main__":
    main()
