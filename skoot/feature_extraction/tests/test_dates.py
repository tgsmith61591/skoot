# -*- coding: utf-8 -*-

from __future__ import absolute_import

from skoot.utils.testing import assert_raises
from skoot.feature_extraction import DateFactorizer

from datetime import datetime as dt
import pandas as pd

data = [
    [1, dt.strptime("06-01-2018", "%m-%d-%Y")],
    [2, dt.strptime("06-02-2018", "%m-%d-%Y")],
    [3, dt.strptime("06-03-2018", "%m-%d-%Y")],
    [4, dt.strptime("06-04-2018", "%m-%d-%Y")],
    [5, None]
]

df = pd.DataFrame.from_records(data, columns=["a", "b"])


def test_factorize():
    trans = DateFactorizer(
        cols=['b'], features=("year", "month")).fit_transform(df)
    assert trans.columns.tolist() == ['a', 'b_year', 'b_month']


def test_non_date_factorize():
    # Fails since not a date time
    assert_raises(ValueError, DateFactorizer(cols=["a", "b"]).fit, df)


def test_factorize_preserve_original():
    # keep the original columns
    trans = DateFactorizer(
        cols=['b'], features=("year", "month"),
        drop_original=False).fit_transform(df)
    assert trans.columns.tolist() == ['a', 'b', 'b_year', 'b_month']


def test_factorize_attribute_error():
    # also show we can handle a non-iterable in features
    factorizer = DateFactorizer(cols=['b'], features="yr")
    assert_raises(AttributeError, factorizer.fit, df)
