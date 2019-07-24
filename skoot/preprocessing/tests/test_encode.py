# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from skoot.preprocessing import DummyEncoder
from skoot.preprocessing.encode import _le_transform
from skoot.datasets import load_iris_df
from skoot.utils.testing import (assert_raises, assert_transformer_asdf,
                                 assert_persistable)

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

    # Prove we have the "timed" attribute
    assert hasattr(encoder, "fit_time_")

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
    col, trans, clz = _le_transform(col="C", vec=vec1, le=le,
                                    handle="error", sep="_")

    assert col == "C", col
    assert_array_equal(trans, [0, 1, 2, 3])
    assert clz == ["C_a", "C_b", "C_c", "C_d"], clz

    # now test where we have a new level and we ignore
    vec2 = np.array(["a", "b", "c", "d", "e", "f"])
    col2, trans2, cls2 = _le_transform(
        col="C2", vec=vec2, le=le,
        handle="ignore", sep="_")

    assert col2 == "C2", col2
    assert_array_equal(trans2, [0, 1, 2, 3, 4, 4])
    assert cls2 == ["C2_a", "C2_b", "C2_c", "C2_d"], cls2

    # test where we have a new level and we do NOT ignore
    assert_raises(ValueError, _le_transform,
                  col="C", vec=vec2, le=le, handle="error", sep="_")


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


def test_dummy_asdf():
    assert_transformer_asdf(DummyEncoder(cols=iris.columns.tolist()), iris)


def test_dummy_persistable():
    assert_persistable(DummyEncoder(cols=iris.columns.tolist()),
                       "location.pkl", iris)
