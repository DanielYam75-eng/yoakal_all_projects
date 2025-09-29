import pandas as pd


def get_cumulative_portion(row: pd.Series):
    age = row["age"]
    po_net_value = row["po_net_value"]
    row = row.loc[0:]
    row = row.iloc[:-1]
    so_far = row.loc[row.index < age].sum()
    so_far_prc = so_far / po_net_value

    return so_far_prc


def get_target(row: pd.Series):

    if row["po_net_value"] == 0:
        return 0

    # The age should be a column because the data was built to contain all ages up to the orders.
    return row.loc[row["age"]] / row["po_net_value"]
