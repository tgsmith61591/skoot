# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from skoot.utils.testing import assert_raises, assert_persistable, \
    assert_transformer_asdf
from skoot.impute import SelectiveImputer, _get_callable, \
    _get_present_values, _mean, _median, _most_frequent, \
    BaggedClassifierImputer, BaggedRegressorImputer

from numpy.testing import assert_array_equal, assert_array_almost_equal

nan = np.nan
X = pd.DataFrame.from_records(
    data=np.array([[1.0, nan, 3.1],
                   [nan, 2.3, nan],
                   [2.1, 2.1, 3.1]]),
    columns=['a', 'b', 'c'])

Y = X.copy()  # type: pd.DataFrame
Y['label'] = [1, 2, nan]


def test_regressor_imputers_persistable():
    for est in (SelectiveImputer, BaggedRegressorImputer):
        assert_persistable(est(), "location.pkl", X)


def test_classifier_imputers_persistable():
    for est in (BaggedClassifierImputer,):
        assert_persistable(
            est(cols=['label'], random_state=42),
            "location.pkl", Y)


def test_regressor_imputers_asdf():
    for est in (SelectiveImputer, BaggedRegressorImputer):
        assert_transformer_asdf(est(), X)


def test_classifier_imputers_asdf():
    for est in (BaggedClassifierImputer,):
        assert_transformer_asdf(
            est(cols=['label'], random_state=42), Y)


def test_get_valid_callable_from_string():
    strat = "strategy"
    valid = {strat: (lambda: None)}
    assert _get_callable(strat, valid)() is None


def test_get_valid_callable_from_callable():
    strat = (lambda: None)
    valid = {"doesn't matter": (lambda: None)}  # not even touched interally
    assert _get_callable(strat, valid)() is None


def test_get_callable_key_error():
    strat = "some strategy"
    valid = {"some other strategy": (lambda: None)}
    assert_raises(ValueError, _get_callable, strat, valid)


def test_get_callable_type_error():
    strat = 123  # not a string or callable
    valid = {"some strategy": (lambda: None)}
    assert_raises(TypeError, _get_callable, strat, valid)


def test_get_present_values():
    series = pd.Series([1., np.nan, 3.])
    mask = pd.isnull(series)
    present = _get_present_values(series, mask)
    assert_array_equal(present, np.array([1., 3.]))


def test_none_present():
    series = pd.Series(np.ones(5) * np.nan)
    mask = pd.isnull(series)
    assert_raises(ValueError, _get_present_values, series, mask)


def test_mean():
    series = pd.Series([1., np.nan, 3., 3., 4.])
    mask = pd.isnull(series)
    assert _mean(series, mask) == 2.75


def test_median():
    series = pd.Series([1., np.nan, 3., 3., 4.])
    mask = pd.isnull(series)
    assert _median(series, mask) == 3.


def test_most_frequent():
    series = pd.Series([1., np.nan, 3., 3., 4.])
    mask = pd.isnull(series)
    assert _most_frequent(series, mask) == 3.


def test_selective_imputer_multi_strategy():
    imputer = SelectiveImputer(
        strategy=('mean', (lambda *args: -999.),
                  'most_frequent'))

    trans = imputer.fit_transform(X)
    assert trans is not X
    values = trans.values
    assert_array_almost_equal(values,
                              np.array([[1., -999., 3.1],
                                        [1.55, 2.3, 3.1],
                                        [2.1, 2.1, 3.1]]))


def test_selective_imputer_bad_strategies():
    # raises for a bad strategy string
    imputer = SelectiveImputer(strategy="bad strategy")
    assert_raises(ValueError, imputer.fit, X)

    # raises for a dim mismatch in cols and strategy
    imputer = SelectiveImputer(cols=['a'], strategy=['mean', 'mean'])
    assert_raises(ValueError, imputer.fit, X)

    # test type error for bad strategy
    imputer = SelectiveImputer(strategy=1)
    assert_raises(TypeError, imputer.fit, X)

    # test dict input that does not match dim-wise
    imputer = SelectiveImputer(cols=['a'],
                               strategy={'a': 'mean', 'b': 'median'})
    assert_raises(ValueError, imputer.fit, X)

    # test a dict input with bad columns breaks
    imputer = SelectiveImputer(strategy={'a': 'mean', 'D': 'median'})
    assert_raises(ValueError, imputer.fit, X)


def test_selective_imputer_dict_strategies():
    # case where cols not specified, and strategy has fewer cols than present
    imputer = SelectiveImputer(strategy={'a': 'mean'})
    trans = imputer.fit_transform(X)

    # show it worked...
    assert trans is not X  # it's a copy
    assert_array_almost_equal(trans.values,
                              np.array([[1., nan, 3.1],
                                        [1.55, 2.3, nan],
                                        [2.1, 2.1, 3.1]]))

    # assert on the strategies stored
    assert 'a' in imputer.strategy_
    assert len(imputer.strategy_) == 1


def test_bagged_regressor_imputer():
    imputer = BaggedRegressorImputer(random_state=42)
    trans = imputer.fit_transform(X)

    assert trans is not X
    assert pd.isnull(trans).values.sum() == 0

    assert_array_almost_equal(trans.values,
                              np.array([[1., 2.16, 3.1],
                                        [1.77, 2.3, 3.1],
                                        [2.1, 2.1, 3.1]]))


def test_bagged_regressor_single_predictor_corner():
    # fails because only one predictor, and it's in cols
    imputer = BaggedRegressorImputer(predictors=['a'])
    assert_raises(ValueError, imputer.fit, X)


def test_bagged_regressor_single_predictor_valid():
    # works because even though only 1 predictor, it's not in cols
    imputer = BaggedRegressorImputer(predictors=['a'], cols=['b', 'c'],
                                     random_state=42)
    trans = imputer.fit_transform(X)
    assert trans is not X

    # still a nan in there because we didn't impute A
    assert_array_almost_equal(trans.values,
                              np.array([[1., 2.16, 3.1],
                                        [nan, 2.3, 3.1],
                                        [2.1, 2.1, 3.1]]))


def test_bagged_classifier():
    imputer = BaggedClassifierImputer(cols=['label'], random_state=42)
    trans = imputer.fit_transform(Y)
    assert trans is not Y
    assert_array_almost_equal(trans.values,
                              np.array([[1., nan, 3.1, 1.],
                                        [nan, 2.3, nan, 2.],
                                        [2.1, 2.1, 3.1, 1.]]))


def test_bagged_classifier_continuous():
    imputer = BaggedClassifierImputer()

    # fails on continuous data!
    assert_raises(ValueError, imputer.fit, X)
