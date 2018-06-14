# -*- coding: utf-8 -*-

from __future__ import absolute_import

from skoot.exploration.multivariate import summarize
from skoot.datasets import load_iris_df

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

# used throughout
nan = np.nan
float_fields = ["a", "b", "c", "d"]

# load iris and add a string field
iris = load_iris_df(include_tgt=True, names=float_fields,
                    tgt_name="species")
iris["cls"] = ["A" if x == 0 else "B" if x == 1 else "C"
               for x in iris["species"]]


def test_summarize_all_continuous():
    cont = iris[float_fields]
    summary = summarize(cont)
    # show we get the stats we expect
    expected = [
        [5.843333, 3.054000, 3.758667, 1.198667],    # mean
        [5.800000, 3.000000, 4.350000, 1.300000],    # median
        [7.900000, 4.400000, 6.900000, 2.500000],    # max
        [4.300000, 2.000000, 1.000000, 0.100000],    # min
        [0.685694, 0.188004, 3.113179, 0.582414],    # variance
        [0.311753, 0.330703, -0.271712, -0.103944],  # skewness
        [-0.57357, 0.241443, -1.395359, -1.335246],  # kurtosis
        [nan, nan, nan, nan],                        # least freq
        [nan, nan, nan, nan],                        # most freq
        [nan, nan, nan, nan],                        # class balance
        [nan, nan, nan, nan],                        # n_levels
        [nan, nan, nan, nan],                        # arity
        [0.000000, 0.000000, 0.000000, 0.000000]     # n_missing
    ]
    assert_array_almost_equal(summary.values, expected, decimal=4)


def test_summarize_all_categorical():
    cat = iris[[c for c in iris.columns if c not in float_fields]]
    assert cat.shape[1] == 2
    summary = summarize(cat)
    expected = [
        [nan, nan],                    # mean
        [nan, nan],                    # median
        [nan, nan],                    # max
        [nan, nan],                    # min
        [nan, nan],                    # variance
        [nan, nan],                    # skewness
        [nan, nan],                    # kurtosis
        [1, 1],                        # class balance
        [3, 3],                        # n_levels
        [0.02, 0.02],                  # arity
        [0, 0],                        # n_missing
    ]

    remove = [7, 8]
    assert_array_almost_equal(
        np.delete(summary.values, remove, axis=0).astype(float), expected)

    # remove the least/most freq rows for the comparison
    expected_levels = [
        [(0, 1, 2), ("A", "B", "C")],  # least freq
        [(0, 1, 2), ("A", "B", "C")],  # most freq
    ]

    levels = summary.values[remove, :].tolist()
    levels = map(lambda row: [sorted(row[0]), sorted(row[1])], levels)
    assert_array_equal(levels, expected_levels)


def test_summary_with_all():
    # add a cat variable with only one level
    X = iris.copy()
    X["one"] = "1"
    # just show we can get it to work...
    summarize(X)
