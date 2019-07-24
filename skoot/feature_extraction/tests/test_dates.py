# -*- coding: utf-8 -*-

from skoot.feature_extraction import DateFactorizer, TimeDeltaFeatures
from skoot.utils.testing import (assert_raises, assert_transformer_asdf,
                                 assert_persistable)

from datetime import datetime as dt
import pandas as pd

import numpy as np
from numpy.testing import assert_array_equal

strp = dt.strptime
data = [
    [1, strp("06-01-2018", "%m-%d-%Y")],
    [2, strp("06-02-2018", "%m-%d-%Y")],
    [3, strp("06-03-2018", "%m-%d-%Y")],
    [4, strp("06-04-2018", "%m-%d-%Y")],
    [5, None]
]

df = pd.DataFrame.from_records(data, columns=["a", "b"])

data2 = [
    [1, strp("06-01-2018", "%m-%d-%Y"), strp("06-02-2018", "%m-%d-%Y")],
    [2, strp("06-02-2018", "%m-%d-%Y"), strp("06-03-2018", "%m-%d-%Y")],
    [3, strp("06-03-2018", "%m-%d-%Y"), strp("06-04-2018", "%m-%d-%Y")],
    [4, strp("06-04-2018", "%m-%d-%Y"), strp("06-05-2018", "%m-%d-%Y")],
    [5, None, strp("06-04-2018", "%m-%d-%Y")]
]

df2 = pd.DataFrame.from_records(data2, columns=['a', 'b', 'c'])


def test_factorize():
    fact = DateFactorizer(cols=['b'], features=("year", "month"))
    trans = fact.fit_transform(df)
    assert trans.columns.tolist() == ['a', 'b_year', 'b_month']

    # Assert that the transform function works independent of fit_transform
    assert trans.equals(fact.transform(df))


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


def test_time_deltas():
    # Days
    tbe = TimeDeltaFeatures(cols=['b', 'c'], units='days')
    trans = tbe.fit_transform(df2)
    assert trans.columns.tolist() == ['a', 'b', 'c', 'b_c_delta'], \
        trans.columns
    assert_array_equal(trans.b_c_delta.values, [-1, -1, -1, -1, np.nan])

    # Assert that the transform function works independent of fit_transform
    assert trans.equals(tbe.transform(df2))

    # Hours
    tbe = TimeDeltaFeatures(cols=['b', 'c'], units='hours')
    trans = tbe.fit_transform(df2)
    assert_array_equal(trans.b_c_delta.values, [-24, -24, -24, -24, np.nan])

    # Minutes
    tbe = TimeDeltaFeatures(cols=['b', 'c'], units='minutes')
    trans = tbe.fit_transform(df2)
    assert_array_equal(trans.b_c_delta.values,
                       [-1440, -1440, -1440, -1440, np.nan])

    # Seconds
    tbe = TimeDeltaFeatures(cols=['b', 'c'], units='seconds')
    trans = tbe.fit_transform(df2)
    assert_array_equal(trans.b_c_delta.values,
                       [-86400, -86400, -86400, -86400, np.nan])


# Test the as_df functionality
def test_date_factorizer_asdf():
    assert_transformer_asdf(DateFactorizer(cols=['b']), df)


# Test the as_df functionality
def test_time_deltas_asdf():
    assert_transformer_asdf(TimeDeltaFeatures(cols=['b', 'c']), df2)


def test_date_factorizer_persistable():
    assert_persistable(DateFactorizer(cols=['b']), location="loc.pkl", X=df)


def test_time_deltas_persistable():
    assert_persistable(TimeDeltaFeatures(cols=['b', 'c']),
                       location='loc.pkl', X=df2)
