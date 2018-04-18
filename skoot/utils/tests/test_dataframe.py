# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import absolute_import

from skoot.utils.dataframe import get_numeric_columns, safe_vstack
from skoot.datasets import load_iris_df

from numpy.testing import assert_array_equal

import numpy as np
import pandas as pd

# get iris loaded
iris = load_iris_df(names=['a', 'b', 'c', 'd'], tgt_name='e')


def test_get_numeric():
    subset = get_numeric_columns(iris)
    assert subset.equals(iris)
    assert subset is not iris


def test_get_numeric_subset():
    df = iris.copy()
    df['e'] = df['e'].astype(str)
    subset = get_numeric_columns(df)
    assert subset.shape != df.shape


def test_safe_vstack_df():
    # take the first five rows
    first5 = iris.iloc[:5]

    # show we can vstack the same (pd) to get a dataframe
    df = safe_vstack(first5, first5)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 10
    assert_array_equal(df.values, np.vstack([first5.values, first5.values]))


def test_safe_vstack_mix():
    # take the first five rows
    first5 = iris.iloc[:5]

    # show we can vstack the same array in mixed types to get an array
    arr = safe_vstack(first5, first5.values)
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == 10
    assert_array_equal(arr, np.vstack([first5.values, first5.values]))


def test_safe_vstack_array():
    # take the first five rows, get the array values
    first5 = iris.iloc[:5].values

    # show we can vstack the same array to get an array
    arr = safe_vstack(first5, first5)
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == 10
    assert_array_equal(arr, np.vstack([first5, first5]))
