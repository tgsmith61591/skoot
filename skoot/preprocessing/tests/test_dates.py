# -*- coding: utf-8 -*-

from skoot.preprocessing import DateTransformer
from skoot.utils.testing import assert_raises, assert_persistable

import pandas as pd
from datetime import datetime as dt

_6_1 = dt.strptime("06-01-2018", "%m-%d-%Y")
_6_2 = dt.strptime("06-02-2018", "%m-%d-%Y")
_6_3 = dt.strptime("06-03-2018", "%m-%d-%Y")
_6_4 = dt.strptime("06-04-2018", "%m-%d-%Y")
_6_5 = dt.strptime("06-05-2018", "%m-%d-%Y")

data = [
    # N/A, Specified, Pre-datetype, Infer
    [1, "06/01/2018", _6_1, "06/01/2018"],
    [2, "06/02/2018", _6_2, "06/02/2018"],
    [3, "06/03/2018", _6_3, "06/03/2018"],
    [4, None, _6_4, None],
    [4, "06/05/2018", None, "06/05/2018"]
]

df = pd.DataFrame.from_records(data, columns=["a", "b", "c", "d"])


def test_date_trans():
    converter = DateTransformer(cols=["b", "c", "d"],
                                date_format=["%m/%d/%Y", None, None])

    trans = converter.fit_transform(df)
    b = trans["b"].tolist()
    c = trans["c"].tolist()
    d = trans["d"].tolist()

    # assert which are null
    assert b[3] is pd.NaT
    assert c[4] is pd.NaT
    assert d[3] is pd.NaT

    # assert on the dt values
    assert b[0] == d[0] == c[0] == _6_1
    assert b[1] == d[1] == c[1] == _6_2
    assert b[2] == d[2] == c[2] == _6_3
    assert c[3] == _6_4
    assert b[4] == d[4] == _6_5

    # Test that we fail on column A
    assert_raises(ValueError, DateTransformer(cols=["a"]).fit, df)

    # But if we allow int64, show it will pass.
    DateTransformer(cols=["a"], allowed_types=("int64")).fit(df)


def test_date_transformer_persistable():
    assert_persistable(DateTransformer(cols=["b", "c", "d"],
                                       date_format=["%m/%d/%Y", None, None]),
                       "location.pkl", df)
