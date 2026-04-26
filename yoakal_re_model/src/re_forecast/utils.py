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
    rval = data.copy()
    age = data["age"].values
    for col in data.loc[:, 0:].columns:
        if col == 0:
            rval[col] = rval[col] / rval["po_net_value"]
        else:
            past = rval.loc[:, 0:(col - 1)]
            past = 1 / (1 - past).prod(axis=1)
            rval[col] = rval[col] * past / rval["po_net_value"]
    vector_length = len(data.loc[:, 0:].columns)
    # This is a logical mxn matrix that is 1 only when the columns is equal to the row's PO's age
    logical_age_matrix = (np.arange(vector_length) == age[:, None])
    result_matrix = rval.loc[:, 0:].to_numpy()
    return result_matrix.sum(axis=1, where=logical_age_matrix)


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
