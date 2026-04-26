import os
import pandas as pd
from . import preprocess
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
import argparse
import json
from read_file import read
from . import globals as glb
import time


class NaiveBayes:
    def __init__(self):
        self.model = CategoricalNB()
        self.features_encoder = OrdinalEncoder()
        self.target_encoder = OrdinalEncoder()

    def fit(self, X, y):
        temp_X = self.features_encoder.fit_transform(X)
        temp_y = self.target_encoder.fit_transform(pd.DataFrame(y))
        self.model = self.model.fit(temp_X, temp_y)
        return self

    def predict(self, X):
        temp_X = self.features_encoder.transform(X)
        return pd.DataFrame(
            self.target_encoder.inverse_transform(
                self.model.predict(temp_X).reshape(-1, 1)
            ),
            columns=["prediction"],
        )

    def generate_random_features(self, target_class: float):
        target_class_bin = self.target_encoder.transform(
            np.array([target_class]).reshape(-1, 1)
        )[0][0]
        features = []
        for feature, log_probs in enumerate(self.model.feature_log_prob_):
            probs = np.exp(log_probs[int(target_class_bin)])
            assert np.isclose(probs.sum(), 1), "Probabilities do not sum to 1"
            feature_value = np.random.choice(
                self.features_encoder.categories_[feature], p=probs
            )
            features.append(feature_value)
        return pd.DataFrame(
            [features + [target_class]],
            columns=self.features_encoder.feature_names_in_.tolist()
            + self.target_encoder.feature_names_in_.tolist(),
        )


class Generator:
    def __init__(self):
        self.model = None
        self.bins = None
        self.mu = None
        self.sigma_squared = None

    def fit(self, data: pd.DataFrame):
        y = data["po_net_value"]
        self.mu, self.sigma_squared = self.estimate_lognormal_distrbuition(y)
        data = self.preprocess(data)
        self.model = NaiveBayes().fit(
            data.drop(columns="po_net_value"), data["po_net_value"]
        )
        return self

    def generate_synthetic_data(self, total_amount: float):
        up_to_amount = 0
        pos_rows = []
        while up_to_amount < total_amount:
            log_amount = np.random.normal(self.mu, np.sqrt(self.sigma_squared))
            amount = np.exp(log_amount)
            target_class, *_ = pd.cut([log_amount], bins=self.bins)
            if (
                not target_class in self.model.target_encoder.categories_[0]
            ):  # if target_class is not in training data
                continue
            features = self.model.generate_random_features(target_class)
            up_to_amount += amount
            features["po_net_value"] = amount
            pos_rows.append(features)
        return pd.concat(pos_rows, ignore_index=True)

    @staticmethod
    def estimate_lognormal_distrbuition(
        data: pd.Series,
    ) -> tuple[float, float]:
        y = np.log(data[data > 0])
        mu = y.mean()
        sigma_squared = y.var()
        return mu, sigma_squared

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[data["po_net_value"] > 0]
        data = data[~data.index.duplicated(keep="first")]
        data["po_net_value"] = pd.cut(np.log(data["po_net_value"]), bins=100)
        self.bins = data["po_net_value"].cat.categories
        data = data[
            [
                "po_type",
                "huka",
                "porcurment_organization",
                "expanditure_type",
                "fingroup",
                "po_net_value",
            ]
        ]
        for col_name in data.columns:
            data[col_name] = data[col_name].astype("category")
        return data


def get_simulated_index(start_id, length, fund_year):
    return pd.MultiIndex.from_product(
        [
            [
                str(doc_number) + "N"
                for doc_number in range(start_id, start_id + length)
            ],
            [fund_year],
            [10],
        ],
        names=glb.KEY,
    )


def augmentation_by_sum_per_month(data, month_dict):
    time1 = time.time()
    all_predictions = []
    all_dates = []
    generator = Generator()
    generator = generator.fit(data)
    time2 = time.time()
    current_index = 0
    for year in month_dict:
        for month in month_dict[year]:
            for fund_year in month_dict[year][month]:
                specific_month_orders = generator.generate_synthetic_data(
                    month_dict[year][month][fund_year]
                )
                index = get_simulated_index(
                    current_index, len(specific_month_orders), fund_year
                )
                specific_month_orders.index = index
                current_index += len(specific_month_orders)
                specific_month_dates = pd.DataFrame(index=index, columns=["order_date"])
                specific_month_dates["order_date"] = f"01.{month.zfill(2)}.{year}"
                all_dates.append(specific_month_dates)
                all_predictions.append(specific_month_orders)
    time3 = time.time()
    if len(all_predictions) == 0:
        columns = [col for col in data.columns if col != "order_date"]
        dtypes = data.dtypes.to_dict()
        del dtypes["order_date"]
        concatenated_predictions = pd.DataFrame(
            columns=columns, index=data.index[:0]
        ).astype(dtypes)
    else:
        concatenated_predictions = pd.concat(all_predictions)
    if len(all_dates) == 0:
        concatenated_dates = pd.DataFrame(
            columns=["order_date"],
            index=data.index[:0],
        ).astype(
            {
                "order_date": "datetime64[ns]",
            }
        )
    else:
        concatenated_dates = pd.concat(all_dates)
    time4 = time.time()
    times = {
        "augmentation_by_sum_per_month::train-generative-model": time2 - time1,
        "augmentation_by_sum_per_month::generate-simulated-POs": time3 - time2,
        "augmentation_by_sum_per_month::finalize-simulated-table": time4 - time3,
    }
    return concatenated_predictions, concatenated_dates, times


def main():
    parser = argparse.ArgumentParser(description="Main script")
    parser.add_argument("-c", "--config", type=str, help="Path to config file")

    args = parser.parse_args()
    config_path = args.config

    using_user_path = config_path is not None
    default_config_path = "config.json"
    if using_user_path and not os.path.exists(config_path):
        print(f"\033[0;33mWarning\033[0m: {config_path} doesn't exist.")
        using_user_path = False
    if not using_user_path and os.path.exists(default_config_path):
        print(
            f"Info: Reading configuration from the default path {default_config_path}."
        )
        config_path = default_config_path

    config = {}
    if config_path is not None and os.path.exists(config_path):
        with open(config_path) as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                print(
                    f"\033[0;33mWarning\033[0m: {config_path} is not a valid JSON file. Ignoring it."
                )
    if "orders" not in config:
        config["orders"] = input(
            "Please enter the key name for the uploaded file orders: "
        )
    if "augmentation_dict" not in config:
        augmentation_dict = {}
        print("Info: Augmentation isn't executed,")
    else:
        augmentation_dict = config["augmentation_dict"]

    key_orders = config["orders"]

    orders = read(key_orders)
    orders.set_index(glb.KEY, inplace=True)

    simulated_orders, simulated_dates = augmentation_by_sum_per_month(
        orders, augmentation_dict
    )

    simulated_orders.to_csv("simulated_orders.csv")


if __name__ == "__main__":
    main()
