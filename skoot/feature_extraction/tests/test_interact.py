# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from numpy.testing import assert_array_equal

import pandas as pd
import numpy as np

from skoot.feature_extraction import InteractionTermTransformer
from skoot.utils.testing import (assert_raises, assert_transformer_asdf,
                                 assert_persistable)

x_dict = {
    'a': [0, 0, 0, 1],
    'b': [1, 0, 0, 1],
    'c': [0, 1, 0, 1],
    'd': [1, 1, 1, 0]
}

X_pd = pd.DataFrame.from_dict(x_dict)[['a', 'b', 'c', 'd']]  # ordering


def test_interaction_default():

    # try with no cols arg
    trans = InteractionTermTransformer()
    X_trans = trans.fit_transform(X_pd)
    expected_names = ['a', 'b', 'c', 'd',
                      'a_b_I', 'a_c_I', 'a_d_I',
                      'b_c_I', 'b_d_I', 'c_d_I']

    assert all([i == j
                for i, j in zip(X_trans.columns.tolist(),
                                expected_names)])  # assert col names equal

    assert_array_equal(X_trans.values, np.array([
        [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 1, 0, 0]
    ]))


def test_interaction_asdf():
    assert_transformer_asdf(InteractionTermTransformer(), X_pd)


def test_interaction_custom():
    # try with a custom function...
    def cust_add(a, b):
        return (a + b).values

    trans = InteractionTermTransformer(interaction_function=cust_add,
                                       as_df=False)

    X_trans = trans.fit_transform(X_pd)
    assert_array_equal(X_trans, np.array([
        [0, 1, 0, 1, 1, 0, 1, 1, 2, 1],
        [0, 0, 1, 1, 0, 1, 1, 1, 1, 2],
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
        [1, 1, 1, 0, 2, 2, 1, 2, 1, 1]
    ]))


def test_interaction_corners():
    # assert fails with a non-function arg
    assert_raises(TypeError,
                  InteractionTermTransformer(interaction_function='a').fit,
                  X_pd)

    # test with just two cols
    # try with no cols arg
    trans = InteractionTermTransformer(cols=['a', 'b'])
    X_trans = trans.fit_transform(X_pd)
    expected_names = ['a', 'b', 'c', 'd', 'a_b_I']
    assert all([i == j
                for i, j in zip(X_trans.columns.tolist(),
                                expected_names)])  # assert col names equal

    assert_array_equal(X_trans.as_matrix(), np.array([
        [0, 1, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 1, 0, 1]
    ]))

    # test diff columns on test set to force value error
    X_test = X_pd.drop(['a'], axis=1)
    assert_raises(ValueError, trans.transform, X_test)


def test_interaction_persistable():
    assert_persistable(InteractionTermTransformer(cols=['a', 'b']),
                       location='loc.pkl', X=X_pd)
