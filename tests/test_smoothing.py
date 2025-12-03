import pytest
from re_forecast.train import smooth_labels
import pandas as pd
import numpy as np


@pytest.fixture
def invoices():
    x = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [
                ("11", "10", "2025"),
                ("12", "10", "2025"),
                ("13", "10", "2025"),
                ("14", "10", "2025"),
            ],
            names=["doc_id", "doc_item", "fund_year"],
        ),
        columns=[1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        data=[
            [4.0, -1.5, 4.0, -1.0, -4.5, 1.0, 0.0, -1.5, 1.0, -0.5, 0.5, 1.5, 0.5],
            [-1.5, -0.5, -4.5, 3.0, -4.5, 0.5, 2.5, 2.5, 0.0, -1.5, 0.5, 0.0, -3.5],
            [3.0, -1.5, -3.5, -1.0, 0.0, -3.0, 2.0, -0.5, 0.5, -3.0, -0.5, 1.0, 4.5],
            [-2.0, -0.5, -2.5, 4.0, -4.0, 0.0, 3.0, 0.0, -4.0, -0.5, -4.5, 2.0, -3.0],
        ],
    )
    return x


def test_smooth_labels(invoices):
    result = smooth_labels(invoices, 4)
    assert np.all(result.index == invoices.index)
    assert np.all(result.columns == invoices.columns)
    assert np.all(
        np.isclose(
            result.to_numpy(),
            [
                [
                    1.375,
                    1.375,
                    1.375,
                    1.375,
                    -1.25,
                    -1.25,
                    -1.25,
                    -1.25,
                    0.625,
                    0.625,
                    0.625,
                    0.625,
                    0.5,
                ],
                [
                    -0.875,
                    -0.875,
                    -0.875,
                    -0.875,
                    0.25,
                    0.25,
                    0.25,
                    0.25,
                    -0.25,
                    -0.25,
                    -0.25,
                    -0.25,
                    -3.5,
                ],
                [
                    -0.75,
                    -0.75,
                    -0.75,
                    -0.75,
                    -0.375,
                    -0.375,
                    -0.375,
                    -0.375,
                    -0.5,
                    -0.5,
                    -0.5,
                    -0.5,
                    4.5,
                ],
                [
                    -0.25,
                    -0.25,
                    -0.25,
                    -0.25,
                    -0.25,
                    -0.25,
                    -0.25,
                    -0.25,
                    -1.75,
                    -1.75,
                    -1.75,
                    -1.75,
                    -3,
                ],
            ],
        )
    )
