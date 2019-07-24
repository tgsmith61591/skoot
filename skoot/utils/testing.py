# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from sklearn.base import clone
import joblib
from numpy.testing import assert_array_almost_equal

import pandas as pd
import numpy as np

import os

from ..decorators import suppress_warnings

__all__ = [
    'assert_persistable',
    'assert_raises',
    'assert_transformer_asdf'
]


def _assert_equal(a, b):
    if a.dtype != b.dtype:
        raise AssertionError("dtypes do not match!")

    # Assert the values are equal. This is tricky since we might have values
    # in columns that are string, datetime, or other complex, non-float
    # dtype. We should only resort to the numpy assert_array_equal where all
    # values are float.
    if a.dtype == "O":
        # FIXME: Any better method out there?
        assert str(a) == str(b), \
            "Arrays do not match:\n\nArray:\n%r\n\nDataframe:\n%r" \
            % (a, b)

    # Else it's float and can use the easy way
    else:
        assert_array_almost_equal(a, b, decimal=6)


@suppress_warnings
def _suppressed_clone(estimator):
    return clone(estimator)


def assert_persistable(estimator, location, X, y=None, **fit_kwargs):
    r"""Assert that the estimator can be persisted.

    This will fit the estimator, pickle it with ``joblib`` and assert it
    can be read back and produce transformations (or predictions).

    Parameters
    ----------
    estimator : BasePDTransformer
        The estimator to fit and persist.

    location : str or unicode
        The location to store the pickle. If it already exists, an exception
        will be raised.

    X : array-like, shape=(n_samples, n_features)
        The training array

    y : array-like, shape=(n_features,) or None, optional (default=None)
        The training labels, optional.

    **fit_kwargs : dict or keyword args, optional
        Any keyword args to pass to the estimator's ``fit`` method.
    """
    if os.path.exists(location):
        raise OSError("Pickle location already exists: %s" % location)

    # Clone it
    est = _suppressed_clone(estimator)
    try:
        # Fit and persist
        est.fit(X, y, **fit_kwargs)
        trans1 = est.transform(X) if hasattr(est, "transform") \
            else est.predict(X)
        joblib.dump(est, location, compress=3)

        # Load and transform
        est_loaded = joblib.load(location)
        trans2 = est_loaded.transform(X) if hasattr(est_loaded, "transform") \
            else est_loaded.predict(X)

        # make them numpy arrays if they are not
        trans1, trans2 = map(
            lambda x: x.values if isinstance(x, pd.DataFrame) else x,
            (trans1, trans2))
        _assert_equal(trans1, trans2)

    # Always remove the pickle
    finally:
        os.unlink(location)


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
    est = _suppressed_clone(estimator)

    # First assert where as_df=False
    est.as_df = False
    est.fit(X, y, **fit_kwargs)
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

    _assert_equal(array_transform, df_transform.values)
