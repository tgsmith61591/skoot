# -*- coding: utf-8 -*-

from skoot.exploration import fisher_pearson_skewness, kurtosis
from skoot.datasets import load_iris_df

from numpy.testing import assert_array_almost_equal

X = load_iris_df(include_tgt=False, names=['a', 'b', 'c', 'd'])
x = X['a']


def test_skewness():
    skew = fisher_pearson_skewness(x)
    assert_array_almost_equal(skew, 0.31175306)


def test_kurtosis():
    kurt = kurtosis(x)
    assert_array_almost_equal(kurt, -0.5735679, decimal=6)
