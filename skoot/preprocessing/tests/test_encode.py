# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import absolute_import

from skoot.preprocessing import DummyEncoder
from skoot.datasets import load_iris_df

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
