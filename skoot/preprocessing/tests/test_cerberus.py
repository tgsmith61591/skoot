# -*- coding: utf-8 -*-

from __future__ import absolute_import

from skoot.datasets import load_iris_df
from skoot.preprocessing.cerberus import SchemaNormalizer

X = load_iris_df()


def test_normalizer():
    schema = {'petal width (cm)': {'coerce': int}}
    norm = SchemaNormalizer(schema).fit(X)
    trans = norm.transform(X)
    types = trans.dtypes

    assert types['petal width (cm)'].name.startswith("int"), types
