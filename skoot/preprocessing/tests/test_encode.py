# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import absolute_import

from skoot.preprocessing import DummyEncoder
from skoot.preprocessing.encode import _le_transform
from skoot.datasets import load_iris_df
from skoot.utils.testing import assert_raises

from sklearn.preprocessing import LabelEncoder

from numpy.testing import assert_array_equal

import numpy as np
import pandas as pd

iris = load_iris_df()


def test_dummy_encoder():
    encoder = DummyEncoder(cols=['species'], drop_one_level=True)
    encoder.fit(iris)

    # assert on fit params
    assert 'species' in encoder.le_
    assert len(encoder.le_) == 1
    assert encoder.fit_cols_ == ['species']

    # transform and assert
    trans = encoder.transform(iris)
    assert trans is not iris
    assert trans.columns.tolist() == \
        iris.columns[:4].tolist() + ['species_0', 'species_1'], trans.columns


def test_le_encode_ignore():
    le = LabelEncoder()
    vec1 = np.array(["a", "b", "c", "d"])
    le.fit(vec1)

    # test where all present
    trans = _le_transform(vec1, le, "error")
    assert_array_equal(trans, [0, 1, 2, 3])

    # now test where we have a new level and we ignore
    vec2 = np.array(["a", "b", "c", "d", "e", "f"])
    trans2 = _le_transform(vec2, le, "ignore")
    assert_array_equal(trans2, [0, 1, 2, 3, 4, 4])

    # test where we have a new level and we do NOT ignore
    assert_raises(ValueError, _le_transform, vec2, le, "error")


def test_dummy_encoder_ignore():
    encoder = DummyEncoder(cols=['species'], drop_one_level=True,
                           handle_unknown="warn")
    encoder.fit(iris)

    # get a test sample, make it different
    test_sample = iris.iloc[0].copy()
    test_sample.set_value("species", 99)

    # show this does not break
    df = pd.DataFrame([test_sample])
    trans = encoder.transform(df)

    # show the sum of the "species" columns is zero
    species_cols = trans[trans.columns[trans.columns.str.contains("species")]]
    assert species_cols.sum().sum() == 0
