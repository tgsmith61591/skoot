# -*- coding: utf-8 -*-

from skoot.exploration.multivariate import summarize
from skoot.datasets import load_iris_df, load_adult_df

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

adult_fields = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                'capital-loss', 'hours-per-week']
adult = load_adult_df(include_tgt=True)


def test_summarize_all_continuous():
    # This is a bit strange since the whole purpose of the summarize func
    # is to discern between floats and ints. However, in version 0.20+ of
    # sklearn, the iris dataset was changed, so we don't want to assert on
    # the attributes of the Iris dataset anymore. The only static dataset
    # is the one native to skoot, which is the adult dataset.
    cont = adult[adult_fields].astype(float)
    summary = summarize(cont)

    # show we get the stats we expect
    expected = [
        [3.85816468e+01, 1.89778367e+05, 1.00806793e+01,
         1.07764884e+03, 8.73038297e+01, 4.04374559e+01],  # mean
        [3.70000e+01, 1.78356e+05, 1.00000e+01,
         0.00000e+00, 0.00000e+00, 4.00000e+01],           # median
        [90., 1484705., 16., 99999., 4356., 99.],          # max
        [17., 12285., 1., 0., 0., 1.],                     # min
        [1.86061400e+02, 1.11407978e+10, 6.61888991e+00,
         5.45425392e+07, 1.62376938e+05, 1.52458995e+02],  # variance
        [5.58717629e-01, 1.44691344e+00, -3.11661510e-01,
         1.19532970e+01, 4.59441746e+00, 2.27632050e-01],  # skewness
        [-1.66286214e-01, 6.21767181e+00, 6.23164080e-01,
         1.54775484e+02, 2.03734886e+01, 2.91605467e+00],  # kurtosis
        [nan, nan, nan, nan, nan, nan],                    # least freq
        [nan, nan, nan, nan, nan, nan],                    # most freq
        [nan, nan, nan, nan, nan, nan],                    # class balance
        [nan, nan, nan, nan, nan, nan],                    # n_levels
        [nan, nan, nan, nan, nan, nan],                    # arity
        [0., 0., 0., 0., 0., 0.]     # n_missing
    ]
    assert np.allclose(summary.values, expected, equal_nan=True)


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
    levels = list(map(lambda row: [sorted(row[0]), sorted(row[1])], levels))
    assert_array_equal(levels, expected_levels)


def test_summary_with_all():
    # add a cat variable with only one level
    X = iris.copy()
    X["one"] = "1"
    # just show we can get it to work...
    summarize(X)
