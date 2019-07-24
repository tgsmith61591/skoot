# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from skoot.datasets import load_iris_df
from skoot.utils.testing import assert_raises
from skoot.utils.dataframe import (get_numeric_columns, safe_vstack,
                                   safe_drop_samples, safe_mask_samples,
                                   dataframe_or_array, get_categorical_columns)

from numpy.testing import assert_array_equal

import numpy as np
import pandas as pd

# get iris loaded
iris = load_iris_df(names=['a', 'b', 'c', 'd'], tgt_name='e')


def test_dataframe_or_array():
    X = dataframe_or_array(iris, True)
    assert X is iris
    X_np = dataframe_or_array(iris, False)
    assert isinstance(X_np, np.ndarray)
    assert_raises(AssertionError, dataframe_or_array, X_np, True)


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


def test_safe_drop():
    df = pd.DataFrame.from_records(np.random.rand(5, 5))

    # drop first 3 rows
    df2 = safe_drop_samples(df, np.arange(3))
    assert_array_equal(df2.index.values, [3, 4])

    # do the same for an array
    arr = safe_drop_samples(df.values, np.arange(3))
    assert arr.shape[0] == 2, arr
    assert_array_equal(arr, df.values[3:, :])


def test_safe_mask():
    df = pd.DataFrame.from_records(np.random.rand(5, 5))

    # drop first 3 rows
    mask = [False, False, False, True, True]
    df2 = safe_mask_samples(df, mask)
    assert_array_equal(df2.index.values, [3, 4])

    # do the same for an array
    arr = safe_mask_samples(df.values, mask)
    assert arr.shape[0] == 2, arr
    assert_array_equal(arr, df.values[3:, :])


def test_get_categorical():
    irs_copy = iris.copy()
    irs_copy['cat'] = ['a' if x == 0 else 'b' if x == 1 else 'c'
                       for x in iris['e']]
    assert get_categorical_columns(irs_copy).columns.tolist() == ['cat']
