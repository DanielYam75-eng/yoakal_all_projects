import pytest
import dagshub
import pandas as pd


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
    df["order_date"] = pd.to_datetime(df["order_date"])  # Ensure datetime dtype
    return df


def test_prepare_index1(orders, monkeypatch):
    from re_forecast.preprocess import prepare_index

    monkeypatch.setattr(dagshub, "get_repo_bucket_client", fake_get_repo_bucket_client)
    prepared = prepare_index(orders)
    assert list(prepared.index.names) == ["doc_id", "fund_year", "item"]
    assert set(prepared.columns) == set(orders.columns) - {
        "doc_id",
        "item",
        "fund_year",
    }


def test_prepare_index2(order_edits, monkeypatch):
    from re_forecast.preprocess import prepare_index

    monkeypatch.setattr(dagshub, "get_repo_bucket_client", fake_get_repo_bucket_client)
    prepared = prepare_index(order_edits)
    assert list(prepared.index.names) == ["doc_id", "fund_year", "item"]
    assert set(prepared.columns) == set(order_edits.columns) - {
        "doc_id",
        "item",
        "fund_year",
    }


def test_prepare_index3(invoices, monkeypatch):
    from re_forecast.preprocess import prepare_index

    monkeypatch.setattr(dagshub, "get_repo_bucket_client", fake_get_repo_bucket_client)
    prepared = prepare_index(invoices)
    assert list(prepared.index.names) == ["doc_id", "fund_year", "item"]
    assert set(prepared.columns) == set(invoices.columns) - {
        "doc_id",
        "item",
        "fund_year",
    }


def test_combine_dates(orders, dates, monkeypatch):
    from re_forecast.preprocess import combine_dates

    monkeypatch.setattr(dagshub, "get_repo_bucket_client", fake_get_repo_bucket_client)
    combined = combine_dates(orders, dates)
    assert "order_date" in combined.columns
    assert "order_year" in combined.columns
    assert "order_month" in combined.columns
    assert pd.api.types.is_datetime64_any_dtype(combined["order_date"])
    assert combined.shape[0] == orders.shape[0]
