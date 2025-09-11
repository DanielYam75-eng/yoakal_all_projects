import pytest
import dagshub
import pandas as pd
import re_forecast.globals as glb
from re_forecast.preprocess import (
    preprocess,
    combine_dates,
    prepare_index,
)


def fake_get_repo_bucket_client(*args, **kwargs):
    pass


# Fixture for current year
@pytest.fixture
def curr_year():
    return 2025


# Fixture for current month
@pytest.fixture
def curr_month():
    return 9


# Fixture for the 'orders' DataFrame
@pytest.fixture
def orders():
    data = {
        "doc_id": ["1001", "1002", "1003"],
        "item": ["1", "2", "3"],
        "fund_year": [2025, 2025, 2025],
        "po_type": ["Standard", "Urgent", "Standard"],
        "huka": ["A1", "B2", "C3"],
        "porcurment_organization": ["Org1", "Org2", "Org3"],
        "expanditure_type": ["CapEx", "OpEx", "CapEx"],
        "fingroup": ["Group1", "Group2", "Group1"],
        "po_net_value": [1000.50, 2500.75, 1500.00],
    }
    return pd.DataFrame(data)


# Fixture for the 'invoices' DataFrame
@pytest.fixture
def invoices():
    data = {
        "doc_id": ["1001", "1002", "1003"],
        "item": ["1", "2", "3"],
        "fund_year": [2025, 2025, 2025],
        "invoice_year": [2025, 2025, 2025],
        "invoice_month": [8, 9, 9],
        "mof_class": ["ClassA", "ClassB", "ClassC"],
        "RE": [100.0, 200.0, 150.0],
        "ZY": [50.0, 60.0, 70.0],
        "ZF": [30.0, 40.0, 50.0],
    }
    return pd.DataFrame(data)


# Fixture for the 'order_edits' DataFrame
@pytest.fixture
def order_edits():
    data = {
        "doc_id": ["1001", "1002", "1003"],
        "item": ["1", "2", "3"],
        "fund_year": [2025, 2025, 2025],
        "order_date": ["20250901", "20250905", "20250830"],
        "volume": [10, 20, 15],
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def dates():
    data = {
        "doc_id": [1001, 1002, 1003],
        "item": [1, 2, 3],
        "fund_year": [2025, 2025, 2025],
        "order_date": ["2025-09-01", "2025-09-05", "2025-08-30"],
    }
    df = pd.DataFrame(data)
    df["order_date"] = pd.to_datetime(
        df["order_date"]
    )  # Ensure datetime dtype
    return df


def test_preprocess(orders, invoices, order_edits, dates, curr_year, curr_month):
    
    orders = prepare_index(orders)
    invoices = prepare_index(invoices)
    order_edits = prepare_index(order_edits)
    dates = prepare_index(dates)

    orders = combine_dates(orders, dates)

    orders, invoices, order_edits, past_sums = preprocess(orders, invoices, order_edits, curr_year, curr_month)

    # dtypes
    assert type(orders) is pd.DataFrame
    assert type(invoices) is pd.DataFrame
    assert type(order_edits) is pd.Series
    assert type(past_sums) is pd.DataFrame

    # index
    assert orders.index.names == glb.KEY
    assert invoices.index.names == glb.KEY
    assert order_edits.index.names == glb.KEY
    assert past_sums.index.names == glb.KEY

    # columns
    assert set(["N", "quarter"]).issubset(orders.columns)

    # range
    assert not orders.empty
    assert not invoices.empty
    assert not order_edits.empty
    assert not past_sums.empty

    assert orders["order_year"].max() <= curr_year
    assert orders["order_month"].max() <= curr_month