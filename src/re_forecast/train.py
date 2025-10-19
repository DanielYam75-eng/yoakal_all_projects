import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import warnings
import mlflow

warnings.filterwarnings("ignore")
from . import globals as glb
from .utils import get_cumulative_portion, get_target


def get_train_data(
    orders: pd.DataFrame,
    invoices: pd.DataFrame,
    edits: pd.DataFrame,
    curr_year: int,
    curr_month: int,
    sample_frac: int,
) -> pd.DataFrame:
    sample_frac = np.sqrt(
        sample_frac
    )  # The sampling is two-staged, so we need to take a square root of the sampling fractions for equivalent sample sizes.

    # n = 36
    # three_years = pd.concat([orders[(orders['order_year'] < curr_year - 3) | (orders['order_year'] == curr_year - 3) & (orders['order_month'] <= curr_month)]] * n, keys=range(n), names=["age"]).reset_index(level='age')
    # n = 24
    # two_years = pd.concat([orders[(orders['order_year'] < curr_year - 2) | (orders['order_year'] == curr_year - 2) & (orders['order_month'] <= curr_month)]] * n, keys=range(n), names=["age"]).reset_index(level='age')
    sampled_orders = orders[
        (orders["order_year"] < curr_year - 1)
        | (orders["order_year"] == curr_year - 1)
        & (orders["order_month"] <= curr_month)
    ].sample(frac=sample_frac, random_state=glb.SEED)
    training_datasets = []
    n = 12
    for years_old in range(1, 11):
        n = 12 * years_old
        sampled_orders = sampled_orders[
            (sampled_orders["order_year"] < curr_year - years_old)
            | (sampled_orders["order_year"] == curr_year - years_old)
            & (sampled_orders["order_month"] <= curr_month)
        ]
        one_year = pd.concat(
            [sampled_orders] * n,
            keys=range(n),
            names=["age"],
        ).reset_index(level="age")
        training_datasets.append(one_year)

    training_data = pd.concat(training_datasets)
    training_data["age"] = training_data["age"].astype(int)
    training_data: pd.DataFrame = training_data.sample(
        frac=sample_frac, random_state=glb.SEED
    )

    training_data["abs order date"] = (
        training_data["order_year"] * 12 + training_data["order_month"]
    )
    edits["abs edit date"] = edits["order_year"] * 12 + edits["order_month"]
    training_data = training_data.merge(
        edits[["abs edit date", "volume"]],
        how="left",
        left_index=True,
        right_index=True,
    )
    training_data = training_data[
        training_data["abs edit date"].between(
            training_data["abs order date"],
            training_data["abs order date"] + training_data["age"],
        )
    ]
    training_data["po_net_value"] = training_data.groupby(glb.KEY + ["age"])[
        "volume"
    ].transform("sum")
    training_data = training_data.drop_duplicates()
    training_data = training_data[training_data["po_net_value"] > 0]

    data = training_data.merge(invoices, how="left", left_index=True, right_index=True)

    training_data["cumulative_portion"] = data.apply(get_cumulative_portion, axis=1)
    training_data["target"] = data.apply(get_target, axis=1)

    categorial_features = [
        "po_type",
        "fingroup",
        "huka",
        "porcurment_organization",
        "expanditure_type",
        "quarter",
    ]
    floating_features = ["po_net_value", "cumulative_portion"]
    integer_features = ["age", "N"]
    training_data[categorial_features] = training_data[categorial_features].astype(
        "category"
    )
    training_data[integer_features] = training_data[integer_features].astype("int32")
    training_data[floating_features] = training_data[floating_features].astype(
        "float32"
    )
    training_data["target"] = training_data["target"].astype("float32")
    training_data = training_data[
        categorial_features + integer_features + floating_features + ["target"]
    ]

    return training_data


def train_model(
    data: pd.DataFrame,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> xgb.XGBRFRegressor:

    train_data, test_data = train_test_split(data, test_size=0.1, random_state=glb.SEED)
    X_train = train_data.drop(columns=["target"])
    y_train = train_data["target"].clip(0, 1)
    X_test = test_data.drop(columns=["target"])
    y_test = test_data["target"]

    model = xgb.XGBRFRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=glb.SEED,
        obj="reg:logistic",
        enable_categorical=True,
    )
    X_train.to_csv("X_train.csv")
    mlflow.log_artifact("X_train.csv", artifact_path="Intermediate Artifacts")
    y_train.to_csv("y_train.csv")
    mlflow.log_artifact("y_train.csv", artifact_path="Intermediate Artifacts")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    pd.Series(y_pred).to_csv("y_pred.csv")
    mlflow.log_artifact("y_pred.csv", artifact_path="Intermediate Artifacts")
    rmse = root_mean_squared_error(y_test, y_pred)
    mlflow.log_metric("rmse", rmse)
    mae = mean_absolute_error(y_test, y_pred)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r-squared", model.score(X_test, y_test))

    return model


def train(
    orders: pd.DataFrame,
    invoices: pd.DataFrame,
    order_edits: pd.DataFrame,
    curr_year: int,
    curr_month: int,
    sample_frac: float,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
):

    training_data = get_train_data(
        orders,
        invoices,
        order_edits,
        curr_year,
        curr_month,
        sample_frac,
    )
    training_data = training_data[
        (training_data["target"] >= 0) & (training_data["target"] <= 1.05)
    ]
    print("The length of the training data is " + str(len(training_data)))
    training_data.to_csv("training_data.csv")
    model = train_model(training_data, n_estimators, max_depth, learning_rate)
    with open(glb.MODEL, "wb") as f:
        pickle.dump(model, f)
