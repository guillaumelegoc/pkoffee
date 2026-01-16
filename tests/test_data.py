"""Tests for the data module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_validate_wrong_names():
    from pkoffee.data import MissingColumnsError, validate

    df = pd.DataFrame({"not_cups": [0, 1, 2], "prod": [0, 1, 2]})

    with pytest.raises(MissingColumnsError):
        validate(df)


def test_validate_good_name():
    from pkoffee.data import validate

    df = pd.DataFrame({"cups": [0, 1, 2], "productivity": [0, 1, 2]})

    validate(df)  # should pass
    assert True


def test_validate_wrong_dtype():
    from pkoffee.data import validate, ColumnTypeError

    df = pd.DataFrame({"cups": ["not", "numeric"], "productivity": [0, 1]})

    with pytest.raises(ColumnTypeError):
        validate(df)


def test_curate_with_nan():
    from pkoffee.data import curate

    df = pd.DataFrame(
        {"cups": [0.0, 1.0, 2.0, np.nan], "productivity": [0.0, 1.0, 2.0, 3.0]}
    )
    expected = pd.DataFrame({"cups": [0.0, 1.0, 2.0], "productivity": [0.0, 1.0, 2.0]})

    res = curate(df)
    pd.testing.assert_frame_equal(res, expected)


def test_curate_without_nan():
    from pkoffee.data import curate

    df = pd.DataFrame(
        {"cups": [0.0, 1.0, 2.0, 3.0], "productivity": [0.0, 1.0, 2.0, 3.0]}
    )
    expected = df.copy()

    res = curate(df)
    pd.testing.assert_frame_equal(res, expected)


def test_load_csv_file_missing():
    from pkoffee.data import load_csv

    with pytest.raises(FileNotFoundError):
        load_csv(Path("/definitely/not/a/file"))


def test_load_csv_not_csv():
    from pkoffee.data import load_csv, CSVReadError

    with pytest.raises(CSVReadError):
        load_csv(Path(__file__))
