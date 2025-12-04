import pandas as pd
import numpy as np


def get_cumulative_portion(data: pd.DataFrame):
    age = data["age"].values
    po_net_value = data["po_net_value"]

    matrix = data.loc[:, 0:]

    vector_length = len(matrix.columns)
    logical_age_matrix = (np.arange(vector_length) < age[:, None]).astype(int)

    return (matrix * logical_age_matrix).sum(axis=1) / po_net_value


def get_target(data: pd.DataFrame):

    # The age should be a column because the data was built to contain all ages up to the orders.

    age = data["age"].values
    po_net_value = data["po_net_value"]

    matrix = data.loc[:, 0:]

    vector_length = len(matrix.columns)
    logical_age_matrix = (np.arange(vector_length) == age[:, None]).astype(int)
    return (matrix * logical_age_matrix).sum(axis=1) / data["po_net_value"]


def cast_to_best_dtype(x):
    result = x
    try:
        result = int(x)
        return result
    except:
        pass
    try:
        result = float(x)
        return result
    except:
        pass
    return result


def load_configuration(f):
    result = {}
    augmentation_dict = {}
    for line in f:
        line = line.split("#", 1)[0].strip()
        if line and ":" not in line and "|" not in line:
            raise DecodingExcpetion(f"Invalid configuration line: {line}")
        elif ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if ',' in value:
                value = value.split(',')
            if not key:
                raise DecodingException(f"Invalid configuration line: {line}")
            result[key] = cast_to_best_dtype(value)
        else:
            year, month, fund_year, amount = line.split("|", 4)
            year = year.strip()
            month = month.strip()
            fund_year = fund_year.strip()
            amount = float(amount.strip())
            if year not in augmentation_dict:
                augmentation_dict[year] = {}
            if month not in augmentation_dict[year]:
                augmentation_dict[year][month] = {}
            if fund_year not in augmentation_dict[year][month]:
                augmentation_dict[year][month][fund_year] = 0
            augmentation_dict[year][month][fund_year] += amount
    result["augmentation_dict"] = augmentation_dict
    return result


class DecodingException(Exception):
    pass
