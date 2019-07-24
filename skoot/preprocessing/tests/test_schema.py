# -*- coding: utf-8 -*-

from skoot.datasets import load_iris_df
from skoot.preprocessing.schema import SchemaNormalizer
from skoot.utils.testing import assert_persistable

X = load_iris_df()
schema = {'petal width (cm)': int}


def test_normalizer():
    norm = SchemaNormalizer(schema).fit(X)
    trans = norm.transform(X)
    types = trans.dtypes

    assert types['petal width (cm)'].name.startswith("int"), types


def test_schema_persistable():
    assert_persistable(SchemaNormalizer(schema), "location.pkl", X)
