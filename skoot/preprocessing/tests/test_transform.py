from __future__ import print_function
import numpy as np
import pandas as pd
from numpy.testing import (assert_array_equal, assert_array_almost_equal)
from sklearn.datasets import load_iris
from skutil.preprocessing import *
from skutil.decomposition import *
from skutil.utils import validate_is_pd
from skutil.utils.fixes import dict_values
from skutil.testing import assert_fails

# Def data for testing
iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)


def test_boxcox():
    transformer = BoxCoxTransformer().fit(X)  # Will fit on all cols

    # Assert similar lambdas
    assert_array_almost_equal(sorted(dict_values(transformer.lambda_)),
                              np.array([
                                -0.14475082666963388, 
                                0.26165380763371671, 
                                0.64441777772515185,
                                0.93129521538860016
                              ]))

    # Assert exact shifts
    assert_array_equal(dict_values(transformer.shift_), np.array([0., 0., 0., 0.]))

    # Now subtract out some fixed amt from X, assert we get different values:
    x = X - 10
    transformer = BoxCoxTransformer().fit(x)

    # Assert similar lambdas
    assert_array_almost_equal(sorted(dict_values(transformer.lambda_)),
                              np.array([
                                0.42501980692063013, 
                                0.5928185584100969, 
                                0.59843688208993162, 
                                0.69983717204250795
                              ]))

    # Assert exact shifts
    assert_array_equal(sorted(dict_values(transformer.shift_)), np.array([5.700001, 8.000001, 9.000001, 9.900001]))

    # assert transform works
    transformed = transformer.transform(X)
    assert isinstance(transformed, pd.DataFrame)

    # assert as df false yields array
    assert isinstance(BoxCoxTransformer(as_df=False).fit_transform(X), np.ndarray)

    # test the selective mixin
    assert transformer.cols is None

    # Test on only one row...
    assert_fails(BoxCoxTransformer().fit, ValueError, X.iloc[0])
    assert_fails(BoxCoxTransformer().fit, ValueError, np.random.rand(1, 5))


def test_function_mapper():
    Y = np.array([['USA', 'RED', 'a'],
                  ['MEX', 'GRN', 'b'],
                  ['FRA', 'RED', 'b']])
    y = pd.DataFrame.from_records(data=Y, columns=['A', 'B', 'C'])
    # Tack on a pseudo-numeric col
    y['D'] = np.array(['$5,000', '$6,000', '$7'])
    y['E'] = np.array(['8%', '52%', '0.3%'])

    def fun(i):
        return i.replace('[\$,%]', '', regex=True).astype(float)

    transformer = FunctionMapper(cols=['D', 'E'], fun=fun).fit(y)
    transformed = transformer.transform(y)
    assert transformed['D'].dtype == float

    # test on all, assert all columns captured
    x = y[['D', 'E']]
    t = FunctionMapper(fun=fun).fit_transform(x)
    assert t['D'].dtype == float and t['E'].dtype == float

    # Try on just one column
    t = FunctionMapper(cols='D', fun=fun).fit_transform(x)
    assert t['D'].dtype == float and t['E'].dtype == object

    # Try on no function
    assert x.equals(FunctionMapper().fit_transform(x))

    # Test on non-function
    assert_fails(FunctionMapper(fun='woo-hoo').fit, ValueError, x)


def test_interactions():
    x_dict = {
        'a': [0, 0, 0, 1],
        'b': [1, 0, 0, 1],
        'c': [0, 1, 0, 1],
        'd': [1, 1, 1, 0]
    }

    X_pd = pd.DataFrame.from_dict(x_dict)[['a', 'b', 'c', 'd']]  # ordering

    # try with no cols arg
    trans = InteractionTermTransformer()
    X_trans = trans.fit_transform(X_pd)
    expected_names = ['a', 'b', 'c', 'd', 'a_b_I', 'a_c_I', 'a_d_I', 'b_c_I', 'b_d_I', 'c_d_I']
    assert all([i == j for i, j in zip(X_trans.columns.tolist(), expected_names)])  # assert col names equal
    assert_array_equal(X_trans.as_matrix(), np.array([
        [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 1, 0, 0]
    ]))

    # try with a custom function...
    def cust_add(a, b):
        return (a + b).values

    trans = InteractionTermTransformer(interaction_function=cust_add, as_df=False)
    X_trans = trans.fit_transform(X_pd)
    assert_array_equal(X_trans, np.array([
        [0, 1, 0, 1, 1, 0, 1, 1, 2, 1],
        [0, 0, 1, 1, 0, 1, 1, 1, 1, 2],
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
        [1, 1, 1, 0, 2, 2, 1, 2, 1, 1]
    ]))

    # assert fails with a non-function arg
    assert_fails(InteractionTermTransformer(interaction_function='a').fit, TypeError, X_pd)

    # test with just two cols
    # try with no cols arg
    trans = InteractionTermTransformer(cols=['a', 'b'])
    X_trans = trans.fit_transform(X_pd)
    expected_names = ['a', 'b', 'c', 'd', 'a_b_I']
    assert all([i == j for i, j in zip(X_trans.columns.tolist(), expected_names)])  # assert col names equal
    assert_array_equal(X_trans.as_matrix(), np.array([
        [0, 1, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 1, 0, 1]
    ]))

    # test on only_return_interactions...
    trans = InteractionTermTransformer(cols=['a', 'b'], only_return_interactions=True)
    X_trans = trans.fit_transform(X_pd)
    expected_names = sorted(['a', 'b', 'a_b_I'])
    actual_names = sorted(X_trans.columns.tolist())
    assert all([expected_names[i] == actual_names[i] for i in range(len(expected_names))])


def test_yeo_johnson():
    transformer = YeoJohnsonTransformer().fit(X)  # will fit on all cols

    # Assert transform works...
    transformed = transformer.transform(X)
    assert isinstance(transformed, pd.DataFrame)

    # assert as df false yields array
    assert isinstance(YeoJohnsonTransformer(as_df=False).fit_transform(X), np.ndarray)
    assert transformer.cols is None

    # Test on only one row...
    assert_fails(YeoJohnsonTransformer().fit, ValueError, X.iloc[0])

    # Test it on a random...
    m, n = 1000, 5
    x = np.random.rand(m, n)

    # make some random
    mask = np.random.rand(m, n) % 2 < 0.5
    signs = np.ones((m, n))
    signs[~mask] = -1
    x *= signs

    YeoJohnsonTransformer().fit(x)


# TODO: more


def test_spatial_sign():
    transformer = SpatialSignTransformer().fit(X)  # will fit to all cols

    # Assert transform works
    transformer.transform(X)

    vals = np.array(dict_values(transformer.sq_nms_))
    l = len(vals[vals == np.inf])
    assert l == 0, 'expected len == 0, but got %i' % l

    # Force inf as the sq norm
    x = np.zeros((5, 5))
    xdf = pd.DataFrame.from_records(data=x)
    transformer = SpatialSignTransformer().fit(xdf)

    # Assert transform works
    transformed = transformer.transform(xdf)
    assert isinstance(transformed, pd.DataFrame)

    # Assert all Inf
    vals = np.array(dict_values(transformer.sq_nms_))
    l = len(vals[vals == np.inf])
    assert l == 5, 'expected len == 5, but got %i' % l

    # assert as df false yields array
    assert isinstance(SpatialSignTransformer(as_df=False).fit_transform(X), np.ndarray)
    assert transformer.cols is None


def test_strange_input():
    # test numpy array input with numeric cols
    x = iris.data
    cols = [0, 2]

    SelectiveScaler(cols=cols).fit_transform(x)
    SelectiveScaler(cols=[]).fit_transform(x)

    SelectivePCA(cols=cols).fit_transform(x)
    SelectivePCA(cols=[]).fit_transform(x)

    # test bad input
    assert_fails(validate_is_pd, TypeError, "bad", None)


def test_selective_scale():
    original = X
    cols = [original.columns[0]]  # Only perform on first...

    # original_means = np.mean(X, axis=0)  # array([ 5.84333333,  3.054     ,  3.75866667,  1.19866667])
    # original_std = np.std(X, axis=0)  # array([ 0.82530129,  0.43214658,  1.75852918,  0.76061262])

    transformer = SelectiveScaler(cols=cols).fit(original)
    transformed = transformer.transform(original)

    new_means = np.array(
        np.mean(transformed, axis=0).tolist())  # expected: array([ 0.  ,  3.054     ,  3.75866667,  1.19866667])
    new_std = np.array(
        np.std(transformed, axis=0).tolist())  # expected: array([ 1.  ,  0.43214658,  1.75852918,  0.76061262])

    assert_array_almost_equal(new_means, np.array([0., 3.054, 3.75866667, 1.19866667]))
    assert_array_almost_equal(new_std, np.array([1., 0.43214658, 1.75852918, 0.76061262]))

    # test the selective mixin
    assert isinstance(transformer.cols, list)
