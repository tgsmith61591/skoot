# -*- coding: utf-8 -*-

from skoot.preprocessing import BinningTransformer
from skoot.datasets import load_iris_df
from skoot.utils.testing import (assert_raises, assert_transformer_asdf,
                                 assert_persistable)

import numpy as np
from numpy.testing import assert_array_equal

iris = load_iris_df(include_tgt=False, names=["a", "b", "c", "d"])


def test_binning_simple():
    binner = BinningTransformer(cols=["a"], n_bins=3, strategy="uniform",
                                return_bin_label=True, overwrite=True)
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
                                return_bin_label=False,
                                overwrite=True,
                                n_jobs=2)

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

    # Test with overwrite = False
    binner.overwrite = False
    trans2 = binner.transform(iris)
    assert trans2.shape[1] == 6
    assert trans2.columns.tolist() == ["a", "b", "c", "d",
                                       "a_binned", "b_binned"], trans2.columns


def test_binning_pctile():
    binner = BinningTransformer(cols=["a"], n_bins=3,
                                strategy="percentile",
                                return_bin_label=True,
                                overwrite=False)

    binner.fit(iris)
    trans = binner.transform(iris)
    unq = np.unique(trans["a_binned"].values).tolist()
    assert unq == ["(-Inf, 5.40]", "(5.40, 6.30]", "(6.30, Inf]"], unq


def test_binning_corners():

    # assertion function to assert fails. 'f' for code golf...
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


def test_binning_asdf():
    assert_transformer_asdf(BinningTransformer(), iris)


def test_binning_persistable():
    assert_persistable(BinningTransformer(), "location.pkl", iris)
