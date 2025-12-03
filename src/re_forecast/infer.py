import os
import pandas as pd
import xgboost as xgb
import pickle
import warnings
import mlflow

warnings.filterwarnings("ignore")
from . import globals as glb
from .utils import get_cumulative_portion


def forecast(
    model: xgb.XGBRFRegressor, data: pd.DataFrame, horizon: int, feature_list
) -> pd.DataFrame:

    predictions = []
    for h in range(1, horizon + 1):
        prediction = pd.Series(
            model.predict(data.loc[data["age"] >= 0, model.feature_names_in_]), index=data[data["age"] >= 0].index
        )
        prediction.name = h

        # update step
        data["age"] += 1
        data["cumulative_portion"] += prediction
        predictions.append(prediction)

    total_predictions = pd.concat(predictions, axis=1).fillna(0)
    return total_predictions


def infer(
    orders: pd.DataFrame,
    invoices: pd.DataFrame,
    past_sums: pd.DataFrame,
    curr_year: int,
    curr_month: int,
    forecast_to: int,
    model,
    categorial_features,
    floating_features,
    integer_features,
    debug,
):

    if debug:
        orders.to_csv(os.path.join('debug-output', "orders-input-to-inference.csv"))
    data = orders.merge(invoices, how="left", left_index=True, right_index=True)
    if debug:
        data.to_csv(os.path.join('debug-output', "data-after-merge.csv"))

    data["age"] = (
        (curr_year - data["order_year"]).mul(12).add(curr_month - data["order_month"])
    )

    data["cumulative_portion"] = data.apply(get_cumulative_portion, axis=1)

    # Do not infer on POs with no balance left
    data = data[data["cumulative_portion"] < 0.98]

    data[categorial_features] = data[categorial_features].astype("category")
    data[integer_features] = data[integer_features].astype("int32")
    data[floating_features] = data[floating_features].astype("float32")
    feature_list = categorial_features + integer_features + floating_features
    # We only infer for POs ages 10 years or less
    data = data[data["age"] <= 120]

    if debug:
        data.to_csv(os.path.join("debug-output", "data-being-forecasted.csv"))
    forecasted_orders = forecast(model, data, 12 - curr_month + forecast_to * 12, feature_list)
    if debug:
        forecasted_orders.to_csv(os.path.join("debug-output", "raw_output.csv"))
        forecast_only = (
            forecasted_orders.iloc[:, -12:].sum(axis=1) * data["po_net_value"]
        )
        forecast_only.to_csv(os.path.join("debug-output", "pure_forecast.csv"))
        pd.DataFrame(past_sums).merge(
            data["fingroup"], left_index=True, right_index=True, how="right"
        ).to_csv(os.path.join("debug-output", "past_sums.csv"))
        mlflow.log_artifact(
            os.path.join("debug-output", "raw_output.csv"), artifact_path="debug-output"
        )

    # If forecast_to > 0, only consider forecasted values for the last year
    if forecast_to > 0:
        sum_forecasted_orders: pd.Series = (
            forecasted_orders.iloc[:, -12:]
            .sum(axis=1)
            .mul(data["po_net_value"], axis=0)
        )
    else:
        sum_forecasted_orders: pd.Series = (
            forecasted_orders.sum(axis=1)
            .mul(data["po_net_value"], axis=0)
            .add(past_sums, fill_value=0)
        )
    if debug:
        sum_forecasted_orders.to_csv(os.path.join("debug-output", "sum-forecast-pre-merge.csv"))
        data[['fingroup']].to_csv(os.path.join("debug-output", "data-being-merged.csv"))
    sum_forecasted_orders = sum_forecasted_orders.to_frame().merge(
        data[["fingroup"]], left_index=True, right_index=True
    )
    return sum_forecasted_orders
