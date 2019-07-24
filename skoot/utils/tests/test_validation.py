# -*- coding: utf-8 -*-

from skoot.utils.testing import assert_raises
from skoot.utils.validation import (check_dataframe,
                                    validate_test_set_columns,
                                    validate_multiple_rows,
                                    type_or_iterable_to_col_mapping)
from sklearn.utils.validation import check_random_state

import pandas as pd
import numpy as np

# single random state used throughout the tests here
random_state = check_random_state(42)

# some DFs we'll re-use
array = random_state.rand(150, 5)
cols = ['col_%i' % i for i in range(array.shape[1])]
X = pd.DataFrame.from_records(array, columns=cols)


# test valid check dataframe with all cols provided
def test_check_dataframe_all_cols():
    # a check with all columns present
    X_copy, cols_copy = check_dataframe(X, cols=cols)

    # neither copy should not share the same reference
    assert X_copy is not X
    assert cols_copy is not cols

    # X_copy should equal X
    assert X.equals(X_copy)
    assert cols == cols_copy, (cols, cols_copy)
    assert isinstance(cols_copy, list)
    assert X_copy.columns.tolist() == cols


# test valid assert_all_finite
def test_check_dataframe_assert_all_finite():
    # a check with all columns present
    X_copy, cols_copy = check_dataframe(X, assert_all_finite=True)
    assert X.equals(X_copy)
    assert X_copy is not X

    # X_copy should equal X
    assert cols == cols_copy, (cols, cols_copy)


# test bad columns
def test_check_dataframe_bad_cols():
    # a check with all columns present
    assert_raises(ValueError, check_dataframe,
                  X, cols=['bad', 'cols'])


# test bad dataframe
def test_check_dataframe_bad_X():
    assert_raises(TypeError, check_dataframe, 'string')


# test valid check dataframe with no cols provided
def test_check_dataframe_no_cols():
    # a check with all columns present
    X_copy, cols_copy = check_dataframe(X, cols=None)
    assert X.equals(X_copy)

    # assert cols is a list that equals ALL cols
    assert cols_copy is not cols
    assert cols == cols_copy, (cols, cols_copy)
    assert isinstance(cols_copy, list)
    assert X_copy.columns.tolist() == cols


# test valid check dataframe scalar col
def test_check_dataframe_scalar_col():
    # a check with all columns present
    X_copy, cols_copy = check_dataframe(X, cols='col_0')
    assert X.equals(X_copy)

    # cols_copy should NOT equal cols
    assert cols_copy != cols
    assert isinstance(cols_copy, list)
    assert len(cols_copy) == 1
    assert cols_copy[0] == 'col_0'


# test valid check dataframe with subset of cols provided
def test_check_dataframe_some_cols():
    # a check with all columns present
    X_copy, cols_copy = check_dataframe(X, cols=cols[:3])
    assert X.equals(X_copy)

    # cols_copy should NOT equal cols
    assert cols_copy != cols
    assert isinstance(cols_copy, list)


def test_check_dataframe_with_diff():
    # a check with all columns present
    X_copy, cols_copy, diff = check_dataframe(X, cols=cols, column_diff=True)

    # neither copy should not share the same reference (still)
    assert X_copy is not X
    assert cols_copy is not cols

    # assert equalities
    assert X.equals(X_copy)
    assert cols == cols_copy, (cols, cols_copy)
    assert not diff


# test works on array
def test_check_dataframe_array():
    X_copy, cols_copy = check_dataframe(array, cols=None)
    assert isinstance(X_copy, pd.DataFrame)
    assert cols_copy == list(range(5))


# test does NOT work if non-dataframe input and cols provided
def test_check_dataframe_array_cols():
    assert_raises(ValueError, check_dataframe, array,
                  cols=[1, 2, 3, 4, 5])


# test df with infinite values
def test_check_dataframe_infinite():
    X_nan = X.mask(X < 0.3)

    # should not raise initially
    X_copy, _ = check_dataframe(X_nan)
    assert X_copy.equals(X_nan)

    # this will raise, since assert_all_finite is True
    assert_raises(ValueError, check_dataframe, X_nan,
                  assert_all_finite=True)


def test_validate_test_cols():
    fit = ['a', 'b', 'c']
    test = ['a', 'b', 'c']

    # this will pass; all fit are in test
    validate_test_set_columns(fit, test)

    # this will also pass; all fit are in test
    test.append('d')
    validate_test_set_columns(fit, test)

    # this will NOT pass; one is now missing from test
    test = test[1:]
    assert_raises(ValueError, validate_test_set_columns, fit, test)


def test_validate_multiple_rows():
    X_copy = np.random.rand(2, 2)
    assert_raises(ValueError, validate_multiple_rows, "cls", X_copy[:1, :])

    # this works
    validate_multiple_rows("cls", X_copy)


def test_validate_np_array_with_provided_cols():
    # test check_dataframe with a np.ndarray and provided cols
    x, cols = check_dataframe(np.random.rand(5, 5), cols=[0, 1, 3])
    assert isinstance(x, pd.DataFrame)
    assert isinstance(cols, list)
    assert cols == [0, 1, 3]
    assert x.columns.tolist() == [0, 1, 2, 3, 4]


def test_type_or_iterable():
    c = ["a", "c"]
    x = type_or_iterable_to_col_mapping(c, 0.5, "n_components", (float, int))
    assert x == {"a": 0.5, "c": 0.5}, x

    y = type_or_iterable_to_col_mapping(c, "uniform", "strategy", str)
    assert y == {"a": "uniform", "c": "uniform"}, y

    z = type_or_iterable_to_col_mapping(c, [3, 5], "q", int)
    assert z == {"a": 3, "c": 5}, z
    assert z == type_or_iterable_to_col_mapping(c, {"a": 3, "c": 5}, "q", int)
