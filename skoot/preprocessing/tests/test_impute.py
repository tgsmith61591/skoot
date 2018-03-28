from __future__ import print_function
import pandas as pd
import numpy as np
from numpy.random import choice
from sklearn.datasets import load_iris
from skutil.preprocessing import *
from skutil.utils import shuffle_dataframe
from skutil.testing import assert_fails
from sklearn.ensemble import RandomForestClassifier


def _random_X(m, n, cols):
    return pd.DataFrame.from_records(
        data=np.random.rand(m, n),
        columns=cols)


def test_bagged_imputer():
    nms = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    X = _random_X(500, 10, nms)
    Z = _random_X(200, 10, nms)

    # test works on 100 full...
    imputer = BaggedImputer()
    imputed = imputer.fit_transform(X)
    null_ct = imputed.isnull().sum().sum()
    assert null_ct == 0, 'expected no nulls but got %i' % null_ct

    # operates in place
    def fill_with_nas(x):
        # make some of them NaN - only 10% tops
        ceil = int(x.shape[0] * 0.1)
        for col in nms:
            n_missing = max(1, choice(ceil))  # at least one in each
            missing_idcs = choice(range(x.shape[0]), n_missing)

            # fill with some NAs
            x.loc[missing_idcs, col] = np.nan

    # throw some NAs in
    fill_with_nas(X)
    fill_with_nas(Z)

    # ensure there are NAs now
    null_ct = X.isnull().sum().sum()
    assert null_ct > 0, 'expected some missing values but got %i' % null_ct

    # now fit the imputer on ALL with some missing:
    imputed = imputer.fit_transform(X)
    null_ct = imputed.isnull().sum().sum()
    assert null_ct == 0, 'expected no nulls but got %i' % null_ct

    # test the transform method on new data
    z = imputer.transform(Z)
    null_ct = z.isnull().sum().sum()
    assert null_ct == 0, 'expected no nulls but got %i' % null_ct


def test_bagged_imputer_classification():
    iris = load_iris()

    # make DF, add species col
    X = pd.DataFrame.from_records(data=iris.data, columns=iris.feature_names)
    X['species'] = iris.target

    # shuffle...
    X = shuffle_dataframe(X)

    # set random indices to be null.. 15% should be good
    rands = np.random.rand(X.shape[0])
    mask = rands > 0.85
    X['species'].iloc[mask] = np.nan

    # define imputer, assert no missing
    imputer = BaggedCategoricalImputer(cols=['species'])
    y = imputer.fit_transform(X)
    assert y['species'].isnull().sum() == 0, 'expected no null...'

    # now test with a different estimator
    imputer = BaggedCategoricalImputer(cols=['species'], base_estimator=RandomForestClassifier())
    y = imputer.fit_transform(X)
    assert y['species'].isnull().sum() == 0, 'expected no null...'


def test_selective_imputer():
    a = pd.DataFrame.from_records([
        [1, 2, 3],
        [np.nan, 2, 2],
        [2, np.nan, np.nan]
    ], columns=['a', 'b', 'c'])

    # first, use an int
    imputer = SelectiveImputer(fill=-1)
    y = imputer.fit_transform(a)
    assert imputer.fills_ == -1
    assert y.isnull().sum().sum() == 0, ('expected no nulls but got:\n', y)
    assert all([x == -1 for x in (y.iloc[1, 0], y.iloc[2, 1], y.iloc[2, 2])])

    # now try with a string...
    imputer = SelectiveImputer(fill='mode')
    y = imputer.fit_transform(a)
    assert y.isnull().sum().sum() == 0, ('expected no nulls but got:\n', y)
    assert all([y.iloc[1, 0] in (1, 2), y.iloc[2, 1] == 2, y.iloc[2, 2] in (3, 2)])

    # now try with a string...
    imputer = SelectiveImputer(fill='mean')
    y = imputer.fit_transform(a)
    assert y.isnull().sum().sum() == 0, ('expected no nulls but got:\n', y)
    assert all([y.iloc[1, 0] == 1.5, y.iloc[2, 1] == 2.0, y.iloc[2, 2] == 2.5])

    # now try with a string...
    imputer = SelectiveImputer(fill='median')
    y = imputer.fit_transform(a)
    assert y.isnull().sum().sum() == 0, ('expected no nulls but got:\n', y)
    assert all([y.iloc[1, 0] == 1.5, y.iloc[2, 1] == 2, y.iloc[2, 2] == 2.5])

    # now test with an iterable
    imputer = SelectiveImputer(fill=[5, 6, 7])
    y = imputer.fit_transform(a)
    assert y.isnull().sum().sum() == 0, ('expected no nulls but got:\n', y)
    assert all([y.iloc[1, 0] == 5, y.iloc[2, 1] == 6, y.iloc[2, 2] == 7])

    # test with a mixed iterable
    imputer = SelectiveImputer(fill=[5, 'mode', 'mean'])
    y = imputer.fit_transform(a)
    assert y.isnull().sum().sum() == 0, ('expected no nulls but got:\n', y)
    assert all([y.iloc[1, 0] == 5, y.iloc[2, 1] == 2, y.iloc[2, 2] == 2.5])

    # test with a mixed iterable -- again
    imputer = SelectiveImputer(fill=['median', 3, 'mean'])
    y = imputer.fit_transform(a)
    assert y.isnull().sum().sum() == 0, ('expected no nulls but got:\n', y)
    assert all([y.iloc[1, 0] == 1.5, y.iloc[2, 1] == 3, y.iloc[2, 2] == 2.5])

    # test with a dict
    imputer = SelectiveImputer(fill={'a': 'median', 'b': 3, 'c': 'mean'})
    y = imputer.fit_transform(a)
    assert y.isnull().sum().sum() == 0, ('expected no nulls but got:\n', y)
    assert all([y.iloc[1, 0] == 1.5, y.iloc[2, 1] == 3, y.iloc[2, 2] == 2.5])

    # test failures now...
    assert_fails(SelectiveImputer(fill='blah').fit, TypeError, a)
    assert_fails(SelectiveImputer(fill=[1, 2]).fit, ValueError, a)
    assert_fails(SelectiveImputer(fill=['a', 'b', 'c']).fit, TypeError, a)
    assert_fails(SelectiveImputer(fill='a').fit, TypeError, a)
    assert_fails(SelectiveImputer(fill=[1, 2, 'a']).fit, TypeError, a)

    # generate anonymous class for test...
    class SomeObject(object):
        def __init__(self):
            pass

    assert_fails(SelectiveImputer(fill=SomeObject()).fit, TypeError, a)


def test_bagged_imputer_errors():
    nms = ['a', 'b', 'c', 'd', 'e']
    X = _random_X(500, 5, nms)

    # ensure works on just fit
    BaggedImputer().fit(X)

    # make all of a NaN
    X.a = np.nan

    # test that all nan will fail
    failed = False
    try:
        BaggedImputer().fit(X)
    except ValueError:
        failed = True
    assert failed, 'Expected imputation on fully missing feature to fail'

    # test on just one col
    failed = False
    try:
        u = pd.DataFrame()
        u['b'] = X.b
        BaggedImputer().fit(u)
    except ValueError:
        failed = True
    assert failed, 'Expected fitting on one col to fail'

    # test with a categorical column
    f = ['a' if choice(4) % 2 == 0 else 'b' for i in range(X.shape[0])]
    X['f'] = f
    failed = False
    try:
        BaggedImputer().fit(X[['d', 'e', 'f']])
    except ValueError:
        failed = True
    assert failed, 'Expected imputation with categorical feature to fail'
