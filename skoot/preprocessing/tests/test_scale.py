# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import absolute_import

from sklearn.preprocessing import RobustScaler

from skoot.preprocessing import SelectiveScaler
from skoot.datasets import load_iris_df

from numpy.testing import assert_array_almost_equal
import numpy as np

X = load_iris_df(include_tgt=False)


def test_selective_scale():
    original = X
    cols = [original.columns[0]]  # Only perform on first...

    # original_means = np.mean(X, axis=0)
    #  array([ 5.84333333,  3.054     ,  3.75866667,  1.19866667])

    # original_std = np.std(X, axis=0)
    #  array([ 0.82530129,  0.43214658,  1.75852918,  0.76061262])

    transformer = SelectiveScaler(cols=cols).fit(original)
    transformed = transformer.transform(original)

    # expected: array([ 0.  ,  3.054     ,  3.75866667,  1.19866667])
    new_means = np.array(
        np.mean(transformed, axis=0).tolist())

    # expected: array([ 1.  ,  0.43214658,  1.75852918,  0.76061262])
    new_std = np.array(
        np.std(transformed, axis=0).tolist())

    assert_array_almost_equal(new_means,
                              np.array([0., 3.054, 3.75866667, 1.19866667]))

    assert_array_almost_equal(new_std,
                              np.array([1., 0.43214658,
                                        1.75852918, 0.76061262]))


def test_selective_scale_robust():
    # test the ref for a provided estimator
    rb_scale = RobustScaler()
    trans = SelectiveScaler(scaler=rb_scale)
    trans.fit(X)

    assert rb_scale is not trans.scaler_
