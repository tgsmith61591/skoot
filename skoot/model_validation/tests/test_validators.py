# -*- coding: utf-8 -*-

from skoot.model_validation import DistHypothesisValidator, CustomValidator
from skoot.utils.testing import assert_raises

import numpy as np


X = np.random.rand(100, 5)
X2 = X * 5.


def test_hypothesis_validator():
    # show that our validator passes for similar data
    DistHypothesisValidator().fit_transform(X)
    DistHypothesisValidator().fit_transform(
        X + (np.random.rand(*X.shape) * 0.001))

    # and show that we fail for different data
    assert_raises(ValueError,
                  DistHypothesisValidator(action="raise").fit(X).transform,
                  X2)


def test_custom_validator():
    CustomValidator().fit_transform(X)  # works when func=None

    # will work with these custom funcs
    sub_2 = (lambda v: np.max(v) < 2.)
    CustomValidator(func=sub_2).fit_transform(X)

    # won't necessarily work on the X2, though
    assert_raises(ValueError,
                  CustomValidator(action="raise",
                                  func=sub_2).fit(X).transform,
                  X2)


def test_chi2_validator():
    data = np.random.RandomState(42).rand(10000, 4)

    # these will all become categorical
    x = (data > 0.4).astype(int)
    x[data > 0.75] = 2

    # Now split and test
    train = x[:9000, :]
    test = x[9000:, :]

    # show the validator will work initially since they're all
    # roughly the same number of occurrences
    val = DistHypothesisValidator(action="raise")
    val.fit(train).transform(test)

    # Make some adjustments to force this to fail
    # test set col 0 should have nothing but 2s
    t2 = test.copy()
    t2[:, 0] = 2
    assert_raises(ValueError, val.transform, t2)

    # now show that if the strategy for categorical vars were not
    # ratio, we would pass
    val.categorical_strategy = None
    val.transform(t2)
