# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import absolute_import

from sklearn.base import clone
from numpy.testing import assert_array_equal

import pandas as pd
import numpy as np

__all__ = [
    'assert_raises',
    'assert_transformer_asdf'
]


def assert_raises(exception_type, func, *args, **kwargs):
    """Assert that a function raises an exception.

    This function is a testing utility that asserts that a function
    will raise a given exception. If it does not, it will raise an
    AssertionError. Alternatively, if it raises a separate exception,
    it will raise *that* exception.

    Parameters
    ----------
    exception_type : BaseException or BaseError
        The exception type

    func : callable
        The function that is expected to raise

    Notes
    -----
    This is roughly equivalent to the ``nose.tools.assert_raises`` utility,
    but since the nose package has been deprecated and we favor pytest,
    we provide this here to avoid another dependency.

    Examples
    --------
    >>> def function_that_raises():
    ...     raise ValueError("boo!")
    >>> assert_raises(ValueError, function_that_raises)
    """
    try:
        func(*args, **kwargs)
    # except only the prescribed type
    except exception_type:
        pass
    # anything else raises
    except Exception:
        raise
    # otherwise we got nothing
    else:
        raise AssertionError("%s did not raise %r"
                             % (func.__name__, exception_type))


def assert_transformer_asdf(estimator, X, y=None, **fit_kwargs):
    r"""Assert that the ``as_df`` arg works on a transformer.

    This will fit and transform an estimator twice, asserting the
    expected types:

        * Once with ``as_df=True``, asserting output as a DataFrame
        * Once with ``as_df=False``, asserting the output as a numpy array

    Parameters
    ----------
    estimator : BasePDTransformer
        The estimator to fit and from which to transform.

    X : array-like, shape=(n_samples, n_features)
        The training array

    y : array-like, shape=(n_features,) or None, optional (default=None)
        The training labels, optional.

    **fit_kwargs : dict or keyword args, optional
        Any keyword args to pass to the estimator's ``fit`` method.
    """
    # Clone it so as not to impact the estimator in place
    est = clone(estimator)

    # First assert where as_df=False
    est.as_df = False
    est.fit(X, y)
    array_transform = est.transform(X)
    assert isinstance(array_transform, np.ndarray), \
        "Expected numpy nparray when as_df=False, but got %s" \
        % type(array_transform)

    # Now reset the as_df param and re-transform, asserting a DF output
    est.as_df = True
    df_transform = est.transform(X)
    assert isinstance(df_transform, pd.DataFrame), \
        "Expected pandas DataFrame when as_df=True, but got %s" \
        % type(df_transform)

    # Assert the values are equal. This is tricky since we might have values
    # in columns that are string, datetime, or other complex, non-float
    # dtype. We should only resort to the numpy assert_array_equal where all
    # values are float.
    if array_transform.dtype == "O":
        # FIXME: Any better method out there?
        assert str(array_transform) == str(df_transform.values), \
            "Arrays do not match:\n\nArray:\n%r\n\nDataframe:\n%r" \
            % (array_transform, df_transform)

    # Else it's float and can use the easy way
    else:
        assert_array_equal(array_transform, df_transform.values)
