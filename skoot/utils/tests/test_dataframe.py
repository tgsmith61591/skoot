# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import absolute_import

from skoot.utils.dataframe import get_numeric_columns
from skoot.datasets import load_iris_df

# get iris loaded
iris = load_iris_df(names=['a', 'b', 'c', 'd'], tgt_name='e')


def test_get_numeric():
    subset = get_numeric_columns(iris)
    assert subset.equals(iris)
    assert subset is not iris


def test_get_numeric_subset():
    df = iris.copy()
    df['e'] = df['e'].astype(str)
    subset = get_numeric_columns(df)
    assert subset.shape != df.shape
