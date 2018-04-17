# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import absolute_import, division

from sklearn.utils.validation import check_random_state
from numpy.testing import assert_array_almost_equal
import numpy as np
import pandas as pd

from skoot.preprocessing import BoxCoxTransformer, YeoJohnsonTransformer
from skoot.preprocessing.skewness import _yj_transform_y
from skoot.datasets import load_iris_df

y = np.arange(5).astype(np.float) - 2.  # [-2, -1, 0, 1, 2]
X = load_iris_df()


def test_yj_transform_0lam():
    lam = 0

    # test for lambda == 0
    res = _yj_transform_y(np.array(y), lam=lam)  # test a copy...
    assert res is not y

    # denom = 2.0 - lam
    # numer = np.power((-x + 1), (2.0 - lam)) - 1.0
    # return -numer / denom
    c3 = (lambda x: -(np.power((-x + 1), (2.0 - lam)) - 1.0) / (2. - lam))
    expected = np.array([
        # case 3
        c3(-2.), c3(-1.),

        # case 2
        np.log(0 + 1.), np.log(1 + 1.), np.log(2 + 1.)
    ])

    assert_array_almost_equal(expected, res)


def test_yj_transform_2lam():
    lam = 2

    # test for lambda == 2
    res = _yj_transform_y(np.array(y), lam=lam)  # test a copy...
    assert res is not y

    c1 = (lambda x: (np.power(x + 1, lam) - 1.0) / lam)
    c4 = (lambda x: -np.log(-x + 1))
    expected = np.array([
        # case 4
        c4(-2.), c4(-1.),

        # case 1
        c1(0), c1(1), c1(2)
    ])

    assert_array_almost_equal(expected, res)


def test_bc_fit_transform():
    bc = BoxCoxTransformer(cols=X.columns[:2])  # just first two cols
    trans = bc.fit_transform(X)

    assert isinstance(trans, pd.DataFrame)
    assert_array_almost_equal(bc.lambda_,
                              np.array([
                                  -0.14475082666963388,
                                  0.26165380763371671
                              ]),
                              decimal=3)  # who knows how far they'll be off..


def test_yj_fit_transform():
    yj = YeoJohnsonTransformer(cols=X.columns[:2])  # just first two cols
    trans = yj.fit_transform(X)

    assert isinstance(trans, pd.DataFrame)

    # Test it on a random...
    m, n = 1000, 5
    random_state = check_random_state(42)
    x = random_state.rand(m, n)

    # make some random
    mask = random_state.rand(m, n) % 2 < 0.5
    signs = np.ones((m, n))
    signs[~mask] = -1
    x *= signs

    YeoJohnsonTransformer().fit(x)
