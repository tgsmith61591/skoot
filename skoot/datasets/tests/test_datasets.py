# -*- coding: utf-8 -*-

import pandas as pd
from skoot.datasets import (load_boston_df, load_iris_df,
                            load_breast_cancer_df, load_adult_df)


def test_load_adult():
    adult = load_adult_df(include_tgt=False)
    assert "target" not in adult.columns

    adult = load_adult_df(include_tgt=True)
    assert adult.columns.tolist() == \
        ["age", "workclass", "fnlwgt", "education",
         "education-num", "marital-status", "occupation",
         "relationship", "race", "sex", "capital-gain",
         "capital-loss", "hours-per-week", "native-country",
         "target"]


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
