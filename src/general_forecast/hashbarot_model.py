import argparse
import math
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import warnings
from .get_ZH_tuples import main as get_ZH_tuples_main

warnings.filterwarnings("ignore")
from . import models
from .models import TSPreprocessor, SeasonalNaiveModel, AvgFactorModel, NaiveModel, TSModel4, find_r2_score_values_data
from .models import find_wining_models, forcast_data

def main(path, past_year, curr_year, curr_month, months_back, coin_type, bucket):
    IND = "kvotzat otzar"

    year_to_predict = past_year
    how_much_month_in_current_year_in_data = curr_month
    current_year = curr_year
    how_much_month_back_to_use = months_back
    coin_type = coin_type
    flag_for_using_only_part_of_data = how_much_month_back_to_use != -1
    how_much_months_in_year = 12

    # %% [markdown]
    # # export DATA
    # %%
    data = get_ZH_tuples_main(path, bucket)

        
    data = data.dropna(subset=["MOF_class_in"])
    data["date"] = pd.to_datetime(
        data["year"].astype(str) + "-" + data["month"].astype(str), format="%Y-%m"
    ) + pd.offsets.MonthEnd(0)
    time_serieses = data.groupby(["MOF_class_out", "MOF_class_in", "date"])[
        "value"
    ].sum()
    data_as_frame = time_serieses.unstack(level=[0, 1]).fillna(0)



    templates = {
        "naive": models.NaiveModel,
        "seasonal_naive": models.SeasonalNaiveModel,
        "mean": models.MeanModel,
        "SimpleExpSmoothing": models.SimpleExpSmoothing,
    }

    

    # %% [markdown]
    # # pre process for forecasting by specific year

    # %%
    how_many_years_look_back_to_find_specific_year = (
        current_year - year_to_predict
    ) * how_much_months_in_year
    actual_data_specific_year = data_as_frame.loc[str(year_to_predict)].fillna(0)
    data_we_got_to_use_in_prediction_specific_year = data_as_frame.loc[
        : str(year_to_predict - 1)
    ].fillna(0)
    data_we_got_to_use_in_prediction_current_year_year = pd.concat(
        [
            data_as_frame.loc[: str(current_year - 1)],
            data_as_frame.loc[str(current_year)].head(
                how_much_month_in_current_year_in_data
            ),
        ]
    ).fillna(0)

    # %% [markdown]
    # # data forcaast specific year

    # %%
    r2_score_values_data_specific_year,  bad_otzar_groups_specific_year = find_r2_score_values_data(
        how_much_months_in_year, data_as_frame, year_to_predict, templates
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
            pd.date_range(
                pd.Timestamp(f"{curr_year}-{curr_month}-01") + pd.offsets.MonthEnd(1),
                f"{curr_year}-12-31",
                freq="ME",
            )
        ),
        templates
    )

    # %%
    actual_data_sum_specific_year = (
        actual_data_specific_year.sum(axis=1).resample("YE").sum() / 1e9
    )
    forcast_data_sum_specific_year = (
        pd.DataFrame(forcast_data_specific_year).sum(axis=1).resample("YE").sum() / 1e9
    )

    # %% [markdown]
    # #  data forcast current_year year

    # %%
    r2_score_values_data_current_year_year,  bad_otzar_groups_specific_year = find_r2_score_values_data(
        how_much_months_in_year, data_as_frame, current_year, templates
    )
    wining_model_current_year_year, r2_of_wining_models_current_year_year = (
        find_wining_models(r2_score_values_data_current_year_year)
    )
    forcast_data_current_year_year = forcast_data(
        how_much_months_in_year - how_much_month_in_current_year_in_data,
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

    # data_so_far_current_year = data_as_frame[f'{current_year}-01-01':]
    # data_so_far_current_year = data_as_frame[data_as_frame.index>"2024-12-31"]
    # data_current_year = pd.concat([data_so_far_current_year, pd.DataFrame(forcast_data_current_year_year)], axis=0)
    # data_so_far_current_year_sum = data_current_year.sum(axis=1).sum()

    # %%
    forcast_ashbarot_specific_year_in = (
        pd.DataFrame(forcast_data_specific_year).T.groupby(level=1).sum()
    )
    forcast_ashbarot_specific_year_in.index.name = "ZH_in"
    forcast_ashbarot_specific_year_in = forcast_ashbarot_specific_year_in * (-1)
    forcast_ashbarot_specific_year_out = (
        pd.DataFrame(forcast_data_specific_year).T.groupby(level=0).sum()
    )
    forcast_ashbarot_specific_year_out.index.name = "ZH_out"

    forcast_ashbarot_bad_otzar_pairs_specific_year_out = (
        forcast_ashbarot_specific_year_out.where(
            forcast_ashbarot_specific_year_out == 0, np.nan
        )
    )
    forcast_ashbarot_bad_otzar_pairs_specific_year_in = (
        forcast_ashbarot_specific_year_in.where(
            forcast_ashbarot_specific_year_in == 0, np.nan
        )
    )

    data_so_far_current_year = data_as_frame[
        data_as_frame.index > f"{current_year-1}-12-31"
    ]
    # data_so_far_current_year = data_as_frame[data_as_frame.index>"2024-12-31"]
    forcast_ashbarot_current_year = pd.concat(
        [data_so_far_current_year, pd.DataFrame(forcast_data_current_year_year)], axis=0
    )
    forcast_ashbarot_current_year_in = (
        pd.DataFrame(forcast_ashbarot_current_year).T.groupby(level=1).sum()
    )
    forcast_ashbarot_current_year_in.index.name = "ZH_in"
    forcast_ashbarot_current_year_in = forcast_ashbarot_current_year_in * (-1)
    forcast_ashbarot_current_year_out = (
        pd.DataFrame(forcast_ashbarot_current_year).T.groupby(level=0).sum()
    )
    forcast_ashbarot_current_year_out.index.name = "ZH_out"

    forcast_ashbarot_bad_otzar_pairs_current_year_out = (
        forcast_ashbarot_current_year_out.where(
            forcast_ashbarot_current_year_out == 0, np.nan
        )
    )
    forcast_ashbarot_bad_otzar_pairs_current_year_in = (
        forcast_ashbarot_current_year_in.where(
            forcast_ashbarot_current_year_in == 0, np.nan
        )
    )

    actual_ashbarot_specific_year_out_yearly = (
        actual_data_specific_year.sum(axis=0).groupby(level=0).sum()
    )
    actual_ashbarot_specific_year_out_yearly = pd.DataFrame(
        actual_ashbarot_specific_year_out_yearly
    )
    actual_ashbarot_specific_year_in_yearly = actual_data_specific_year.sum(
        axis=0
    ).groupby(level=1).sum() * (-1)
    actual_ashbarot_specific_year_in_yearly = pd.DataFrame(
        actual_ashbarot_specific_year_in_yearly
    )
    # %%

    for name, frame in zip(
        ["ZH_in", "ZH_out"],
        [
            actual_ashbarot_specific_year_in_yearly,
            actual_ashbarot_specific_year_out_yearly,
        ],
    ):
        frame.columns = ["actual"]
        frame.index.name = IND
        frame.reset_index().to_csv(
            f"full_actual_{name}_{year_to_predict}.csv", index=False
        )

    for name, frame in zip(
        ["ZH_in", "ZH_out"],
        [forcast_ashbarot_specific_year_in, forcast_ashbarot_specific_year_out],
    ):
        frame.insert(0, "kvuzat sahar", f"forcast_{name}_{year_to_predict}.csv")
        frame.index.name = IND
        frame.reset_index().to_csv(f"forcast_{name}_{year_to_predict}.csv", index=False)

    for name, frame in zip(
        ["ZH_in", "ZH_out"],
        [forcast_ashbarot_current_year_in, forcast_ashbarot_current_year_out],
    ):
        frame.insert(0, "kvuzat sahar", f"forcast_{name}_{current_year}")
        frame.index.name = IND
        frame.reset_index().to_csv(f"forcast_{name}_{current_year}.csv", index=False)

    for name, frame in zip(
        ["ZH_in", "ZH_out"],
        [
            forcast_ashbarot_bad_otzar_pairs_specific_year_in,
            forcast_ashbarot_bad_otzar_pairs_specific_year_out,
        ],
    ):
        frame.insert(
            0, "kvuzat sahar", f"actual_data_{year_to_predict}_bad_otzar_only_{name}"
        )
        frame.index.name = IND
        frame.reset_index().to_csv(
            f"actual_data_{year_to_predict}_bad_otzar_only_{name}.csv", index=False
        )

    for name, frame in zip(
        ["ZH_in", "ZH_out"],
        [
            forcast_ashbarot_bad_otzar_pairs_current_year_in,
            forcast_ashbarot_bad_otzar_pairs_current_year_out,
        ],
    ):
        frame.insert(
            0, "kvuzat sahar", f"actual_data_{current_year}_bad_otzar_only_{name}"
        )
        frame.index.name = IND
        frame.reset_index().to_csv(
            f"actual_data_{current_year}_bad_otzar_only_{name}.csv", index=False
        )


if __name__ == "__main__":
    main()
