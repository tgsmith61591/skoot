# -*- coding: utf-8 -*-

from __future__ import absolute_import

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
