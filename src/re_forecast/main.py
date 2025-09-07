from .train import train
from .preprocess import preprocess
from .infer import infer
import argparse
import pandas as pd
from read_file import read
import dagshub
import mlflow
from importlib_metadata import version
import os
import json


def main():
    parser = argparse.ArgumentParser(description="Main script")
    parser.add_argument(
        "-o", "--output_path", type=str, required=True, help="Path to the data file"
    )
    parser.add_argument("-c", "--config", type=str, help="Path to config file")

    args = parser.parse_args()
    output_path = args.output_path
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
    if "invoices" not in config:
        config["invoices"] = input(
            "Please enter the key name for the uploaded file invoices: "
        )
    if "orders_dates" not in config:
        config["orders_dates"] = input(
            "Please enter the key name for the uploaded file orders_dates: "
        )
    if "order_edits" not in config:
        config["order_edits"] = input(
            "Please enter the key name for the uploaded file order_edits: "
        )
    if "curr_year" not in config:
        while True:
            config["curr_year"] = input("Please enter the current year (YYYY): ")
            try:
                cy = int(config["curr_year"])
                if 2000 <= cy <= 2100:
                    config["curr_year"] = cy
                    break
                else:
                    print(
                        f"\033[0;33mWarning\033[0m: {config['curr_year']} is not a valid year. Please enter a valid year between 2000 and 2100."
                    )
            except ValueError:
                print(
                    f"\033[0;33mWarning\033[0m: {config['curr_year']} is not a valid integer. Please enter a valid year between 2000 and 2100."
                )
    if "curr_month" not in config:
        while True:
            config["curr_month"] = input("Please enter the current month (MM): ")
            try:
                cm = int(config["curr_month"])
                if 1 <= cm <= 12:
                    config["curr_month"] = cm
                    break
                else:
                    print(
                        f"\033[0;33mWarning\033[0m: {config['curr_month']} is not a valid month. Please enter a valid month between 1 and 12."
                    )
            except ValueError:
                print(
                    f"\033[0;33mWarning\033[0m: {config['curr_month']} is not a valid integer. Please enter a valid month between 1 and 12."
                )
    if "sample_frac" not in config:
        while True:
            config["sample_frac"] = input(
                "Please enter the fraction of data to be used for training (between 0 and 1): "
            )
            try:
                sf = float(config["sample_frac"])
                if 0.0 < sf <= 1.0:
                    config["sample_frac"] = sf
                    break
                else:
                    print(
                        f"\033[0;33mWarning\033[0m: {config['sample_frac']} is not in the valid range (0.0, 1.0]. Please enter a valid fraction."
                    )
            except ValueError:
                print(
                    f"\033[0;33mWarning\033[0m: {config['sample_frac']} is not a valid float. Please enter a valid fraction."
                )
    if "n_estimators" not in config:
        while True:
            config["n_estimators"] = input("Please enter the number of estimators: ")
            try:
                ne = int(config["n_estimators"])
                if ne > 0:
                    config["n_estimators"] = ne
                    break
                else:
                    print(
                        f"\033[0;33mWarning\033[0m: {config['n_estimators']} is not a valid positive integer. Please enter a valid positive integer."
                    )
            except ValueError:
                print(
                    f"\033[0;33mWarning\033[0m: {config['n_estimators']} is not a valid integer. Please enter a valid positive integer."
                )
    if "max_depth" not in config:
        while True:
            config["max_depth"] = input("Please enter the maximum depth: ")
            try:
                md = int(config["max_depth"])
                if md > 0:
                    config["max_depth"] = md
                    break
                else:
                    print(
                        f"\033[0;33mWarning\033[0m: {config['max_depth']} is not a valid positive integer. Please enter a valid positive integer."
                    )
            except ValueError:
                print(
                    f"\033[0;33mWarning\033[0m: {config['max_depth']} is not a valid integer. Please enter a valid positive integer."
                )
    if "learning_rate" not in config:
        while True:
            config["learning_rate"] = input("Please enter the learning rate: ")
            try:
                lr = float(config["learning_rate"])
                if 0.0 < lr <= 1.0:
                    config["learning_rate"] = lr
                    break
                else:
                    print(
                        f"\033[0;33mWarning\033[0m: {config['learning_rate']} is not in the valid range (0.0, 1.0]. Please enter a valid learning rate."
                    )
            except ValueError:
                print(
                    f"\033[0;33mWarning\033[0m: {config['learning_rate']} is not a valid float. Please enter a valid learning rate."
                )

    key_orders = config["orders"]
    key_invoices = config["invoices"]
    key_orders_dates = config["orders_dates"]
    key_order_edits = config["order_edits"]
    curr_year = config["curr_year"]
    curr_month = config["curr_month"]
    sample_frac = config["sample_frac"]
    n_estimators = config["n_estimators"]
    max_depth = config["max_depth"]
    learning_rate = config["learning_rate"]
    mode = config["mode"] if "mode" in config else ""

    orders = read(key_orders)
    orders_dates = read(key_orders_dates)
    order_edits = read(key_order_edits, dtype={"order_date": str})
    invoices = read(key_invoices)

    dagshub.init(repo_owner="yoacal.data.science", repo_name="exp-repo", mlflow=True)
    mlflow.set_experiment("re_forecast")
    with mlflow.start_run():
        package_version = version("mof-class-forecaster")
        mlflow.set_tags({"mlflow.source.name": "mof-class-forecaster"})
        mlflow.log_param("curr_year", curr_year)
        mlflow.log_param("curr_month", curr_month)
        mlflow.log_param("sample_frac", sample_frac)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("mode", mode)
        mlflow.set_tags(
            {
                "mlflow.source.git.commit": package_version.split("+")[1][1:].split(
                    "."
                )[0]
            }
        )

        mlflow.log_input(
            mlflow.data.from_pandas(
                orders,
                source="s3://bucket/" + key_orders,
                name="data for orders",
            )
        )
        mlflow.log_input(
            mlflow.data.from_pandas(
                orders_dates,
                source="s3://bucket/" + key_orders_dates,
                name="data for orders dates",
            )
        )
        mlflow.log_input(
            mlflow.data.from_pandas(
                order_edits,
                source="s3://bucket/" + key_order_edits,
                name="data for order edits",
            )
        )
        mlflow.log_input(
            mlflow.data.from_pandas(
                invoices,
                source="s3://bucket/" + key_invoices,
                name="data for invoices",
            )
        )
        orders, invoices, past_sums, order_edits = preprocess(
            orders,
            invoices,
            orders_dates,
            order_edits,
            curr_year,
            curr_month,
        )
        train(
            orders,
            invoices,
            order_edits,
            curr_year,
            curr_month,
            sample_frac,
            n_estimators,
            max_depth,
            learning_rate,
        )
        if mode == "train":
            pass
        else:
            infer(
                orders,
                invoices,
                past_sums,
                curr_year,
                curr_month,
                output_path,
            )

if __name__ == "__main__":
    main()
