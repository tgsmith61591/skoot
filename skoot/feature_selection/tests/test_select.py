# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

import numpy as np
import pandas as pd

from skoot.datasets import load_iris_df
from skoot.utils.testing import (assert_raises, assert_transformer_asdf,
                                 assert_persistable)
from skoot.feature_selection import (FeatureFilter, SparseFeatureFilter,
                                     MultiCorrFilter, NearZeroVarianceFilter)

from numpy.testing import assert_array_equal, assert_array_almost_equal

# get some datasets defined for use later
iris = load_iris_df(include_tgt=False,
                    names=['a', 'b', 'c', 'd'])

sparse = pd.DataFrame.from_records(
    data=[[1., 2., np.nan],
          [2., 3., np.nan],
          [np.nan, 4., 5.]],
    columns=['a', 'b', 'c'])


def test_nzv_constant_col():
    X = pd.DataFrame.from_records(
        data=np.array([[1, 2, 3], [4, 5, 3], [6, 7, 3], [8, 9, 3]]),
        columns=['a', 'b', 'c'])

    flt = NearZeroVarianceFilter(freq_cut=25)
    trans = flt.fit_transform(X)

    # show that the output is one column shorter
    assert trans.shape[1] == 2
    assert flt.drop_ == ['c']

    # show the ratios are expected
    assert_array_equal(flt.ratios_, np.array([1., 1., np.inf]))


def test_nzv_non_constant():
    X = pd.DataFrame.from_records(
        data=np.array([[1, 2, 3], [4, 5, 3], [6, 7, 5]]),
        columns=['a', 'b', 'c'])

    nzv = NearZeroVarianceFilter(freq_cut=2)  # show passes with an int
    trans = nzv.fit_transform(X)

    # show the output is down a column
    assert trans.shape[1] == 2
    assert nzv.drop_ == ['c']

    # show the ratios are expected
    assert_array_equal(nzv.ratios_, np.array([1., 1., 2.]))


def test_nzv_bad_freq_cut():
    X = pd.DataFrame.from_records(
        data=np.array([[1, 2, 3], [4, 5, 3], [6, 7, 5]]),
        columns=['a', 'b', 'c'])

    # show fails with a bad float value
    nzv_float = NearZeroVarianceFilter(freq_cut=1.)
    assert_raises(ValueError, nzv_float.fit, X)

    # show fails with a non-float/int
    nzv_str = NearZeroVarianceFilter(freq_cut='1.')
    assert_raises(ValueError, nzv_str.fit, X)


def test_nzf_asdf():
    assert_transformer_asdf(NearZeroVarianceFilter(), iris)


def test_mcf_iris_high_thresh():
    mcf = MultiCorrFilter(threshold=0.85)
    trans = mcf.fit_transform(iris)

    # there should be 3 features left
    assert trans.shape[1] == 3
    assert 'c' not in trans.columns
    assert mcf.drop_ == ['c'], mcf.drop_


def test_mcf_iris_medium_thresh():
    mcf = MultiCorrFilter(threshold=0.8, method="pearson")
    trans = mcf.fit_transform(iris)

    # there should be 2 features left
    assert trans.shape[1] == 2
    assert 'c' not in trans.columns
    assert 'd' not in trans.columns
    assert mcf.drop_ == ['c', 'd'], mcf.drop_

    # assert on the correlations
    assert np.allclose(
        mcf.mean_abs_correlations_,
        np.array([0.69976926, 0.47160736, 0.81375684, 0.78431371]),
        atol=1e-2)


def test_mcf_non_finite():
    mcf = MultiCorrFilter(threshold=0.75)
    assert_raises(ValueError, mcf.fit, sparse)


def test_mcf_asdf():
    assert_transformer_asdf(MultiCorrFilter(), iris)


def test_feature_filter_none():
    dpr = FeatureFilter(cols=None)

    # none should be dropped
    trans = dpr.fit_transform(iris)  # type: pd.DataFrame
    assert trans.equals(iris)
    assert trans is not iris

    # assert empty drop list
    assert dpr.drop_ == []


def test_feature_filter_some():
    dpr = FeatureFilter(cols=['a', 'b'])
    trans = dpr.fit_transform(iris)

    # only two should have been dropped
    assert 'a' not in trans.columns
    assert 'b' not in trans.columns

    # should be two left
    assert trans.shape[1] == 2
    assert trans.equals(iris[['c', 'd']])


def test_filter_asdf():
    assert_transformer_asdf(FeatureFilter(), iris)


def test_sparse_filter():
    sps_filter = SparseFeatureFilter(threshold=0.5)
    trans = sps_filter.fit_transform(sparse)

    # we should have filtered out the third col
    assert trans.shape[1] == 2
    assert 'c' not in trans.columns


def test_sparse_filter_dense_data():
    sps_filter = SparseFeatureFilter(threshold=0.25)
    trans = sps_filter.fit_transform(iris)

    # nothing should have changed
    assert trans.equals(iris)

    # assert on values
    assert_array_almost_equal(sps_filter.sparsity_, np.zeros(4))


def test_sparse_asdf():
    assert_transformer_asdf(SparseFeatureFilter(), iris)


def test_all_persistable():
    for est in (FeatureFilter, SparseFeatureFilter,
                MultiCorrFilter, NearZeroVarianceFilter):
        assert_persistable(est(), location="loc.pkl", X=iris)
