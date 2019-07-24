# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from sklearn.preprocessing import RobustScaler

from skoot.datasets import load_iris_df
from skoot.utils.testing import assert_transformer_asdf, assert_persistable
from skoot.preprocessing import (SelectiveStandardScaler,
                                 SelectiveRobustScaler,
                                 SelectiveMinMaxScaler,
                                 SelectiveMaxAbsScaler)

from numpy.testing import assert_array_almost_equal
import numpy as np

from pkg_resources import parse_version
import sklearn

X = load_iris_df(include_tgt=False)

# Sklearn 0.20+ changed the iris dataset, so we have to account for that when
# asserting on the means or standard deviations...
if parse_version(sklearn.__version__) >= parse_version("0.20.0"):
    expected_means = np.array([-2.77555756e-16,
                               3.05733333e+00,
                               3.75800000e+00,
                               1.19933333e+00])

    expected_std = np.array([1., 0.43441097, 1.75940407, 0.75969263])

else:
    expected_means = np.array([0., 3.054, 3.75866667, 1.19866667])
    expected_std = np.array([1., 0.43214658, 1.75852918, 0.76061262])


def test_selective_scale():
    original = X
    cols = [original.columns[0]]  # Only perform on first...

    # original_means = np.mean(X, axis=0)
    #  array([5.84333333, 3.05733333, 3.758     , 1.19933333])

    # original_std = np.std(X, axis=0)
    #  array([0.82530129, 0.43441097, 1.75940407, 0.75969263])

    transformer = SelectiveStandardScaler(
        cols=cols, trans_col_name=[cols[0]]).fit(original)
    transformed = transformer.transform(original)[original.columns]

    # expected: array([ 0.,  3.057 ,  3.75866667,  1.19866667])
    new_means = np.array(
        np.mean(transformed, axis=0).tolist())

    # expected: array([1. , 0.43441097, 1.75940407, 0.75969263])
    new_std = np.array(
        np.std(transformed, axis=0).tolist())

    assert_array_almost_equal(new_means, expected_means)
    assert_array_almost_equal(new_std, expected_std)


def test_selective_scale_robust():
    # test the ref for a provided estimator
    rb_scale = RobustScaler().fit(X)
    trans = SelectiveRobustScaler().fit(X)

    assert_array_almost_equal(rb_scale.fit_transform(X),
                              trans.transform(X).values)


def test_standard_scaler_asdf():
    assert_transformer_asdf(SelectiveStandardScaler(), X)


def test_robust_scaler_asdf():
    assert_transformer_asdf(SelectiveRobustScaler(), X)


def test_minmax_scaler_asdf():
    assert_transformer_asdf(SelectiveMinMaxScaler(), X)


def test_maxabs_scaler_asdf():
    assert_transformer_asdf(SelectiveMaxAbsScaler(), X)


def test_all_persistable():
    for est in (SelectiveStandardScaler,
                SelectiveRobustScaler,
                SelectiveMinMaxScaler,
                SelectiveMaxAbsScaler):
        assert_persistable(est(), "location.pkl", X)
