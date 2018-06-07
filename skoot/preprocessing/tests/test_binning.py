# -*- coding: utf-8 -*-

from __future__ import absolute_import

from skoot.preprocessing import BinningTransformer
from skoot.datasets import load_iris_df
from skoot.utils.testing import assert_raises

# for testing the entropy function
from skoot.preprocessing._binning import C_entropy

import numpy as np
from numpy.testing import assert_array_equal

iris = load_iris_df(include_tgt=False, names=["a", "b", "c", "d"])


def test_binning_simple():
    binner = BinningTransformer(cols=["a"], n_bins=3, strategy="uniform",
                                return_bin_label=True)
    binner.fit(iris)
    trans = binner.transform(iris)

    # show the dfs are not the same
    assert trans is not iris

    # show the columns stayed the same, though
    assert trans.columns.tolist() == iris.columns.tolist()

    # show we have a string datatype now
    assert trans.dtypes['a'].name == 'object'

    # if we set the return_bin_label to false and then transform again
    # show we actually get an integer back
    binner.return_bin_label = False
    trans2 = binner.transform(iris)
    assert trans2.dtypes['a'].name.startswith("int")

    # show there are three levels
    assert_array_equal(np.unique(trans2.a.values), [0, 1, 2])


def test_binning_complex():
    # Test with complex n_bins
    binner = BinningTransformer(cols=["a", "b"], n_bins=[2, 3],
                                strategy="uniform",
                                return_bin_label=False)

    binner.fit(iris)
    trans = binner.transform(iris)

    # show the columns stayed the same
    assert trans.columns.tolist() == iris.columns.tolist()

    # assert the different levels of integers
    assert_array_equal(np.unique(trans.a.values), [0, 1])
    assert_array_equal(np.unique(trans.b.values), [0, 1, 2])

    # show both types are now int
    assert trans.dtypes['a'].name.startswith("int")
    assert trans.dtypes['b'].name.startswith("int")


def test_binning_corners():

    # assertion function to assert fails
    def f(binner, exc):
        assert_raises(exc, binner.fit, iris)

    # this one will fail since n_bins contains a non-specified column
    f(BinningTransformer(cols=["a"], n_bins={"b": 2}), ValueError)

    # this one will fail for the same reason
    f(BinningTransformer(cols=["a", "c"], n_bins={"a": 3, "b": 2}), ValueError)

    # this one will fail for a bad integer
    f(BinningTransformer(cols=["a", "c"], n_bins=[2, 1]), ValueError)

    # this one will fail for a dim mismatch
    f(BinningTransformer(cols=["a", "c"], n_bins=[2]), ValueError)

    # this one will fail since n_bins is illegal
    f(BinningTransformer(cols=["a", "c"], n_bins=None), TypeError)

    # this one will fail since strategy is illegal
    f(BinningTransformer(cols=["a"], n_bins=3, strategy="illegal"), ValueError)


def test_entropy():
    events = np.asarray(9 * [0] + 5 * [1])  # 9/14, 5/14
    _, cts = np.unique(events, return_counts=True)
    ent = C_entropy(events.astype(np.float64), cts.astype(np.float32))
    assert round(ent, 2) == 0.94, round(ent, 2)
