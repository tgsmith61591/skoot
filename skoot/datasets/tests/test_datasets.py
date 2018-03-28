# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pandas as pd
from skoot.datasets import load_boston_df, load_iris_df, load_breast_cancer_df


def test_load_iris():
    iris = load_iris_df(include_tgt=False, names=['a', 'b', 'c', 'd'])
    assert isinstance(iris, pd.DataFrame)
    assert 'species' not in iris.columns
    assert iris.shape == (150, 4)

    # assert on the names
    assert 'a' in iris


def test_load_breast_cancer():
    bc = load_breast_cancer_df(tgt_name="target")
    assert isinstance(bc, pd.DataFrame)
    assert 'target' in bc.columns


def test_load_boston():
    bo = load_boston_df(tgt_name="price")
    assert isinstance(bo, pd.DataFrame)
    assert 'price' in bo.columns
