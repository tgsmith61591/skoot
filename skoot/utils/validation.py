# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

import pandas as pd
import numpy as np
import copy

from .iterables import is_iterable

__all__ = [
    'check_dataframe',
    'type_or_iterable_to_col_mapping',
    'validate_multiple_cols',
    'validate_multiple_rows',
    'validate_test_set_columns'
]


def check_dataframe(X, cols=None, assert_all_finite=False, column_diff=False):
    r"""Check an input dataframe.

    Determine whether an input frame is a Pandas dataframe or whether it can
    be coerced as one, and raise a TypeError if not. Also check for finite
    values if specified. If columns are provided, checks that all columns
    are present within the dataframe and raises an assertion error if not.

    **Note**: if ``X`` is not a dataframe (i.e., a list of lists or a numpy
    array), the columns will not be specified when creating a pandas dataframe
    and will thus be indices. Any columns provided should account for this
    behavior.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        The input frame. Should be a pandas DataFrame, numpy ``ndarray`` or
        a similar array-like structure. Any non-pandas structure will be
        attempted to be cast to pandas; if it cannot be cast, it will fail
        with a TypeError.

    cols : list, iterable or None
        Any columns to check for. If this is provided, all columns will
        be checked for presence in the ``X.columns`` index. If any are not
        present, a ValueError will be raised.

    assert_all_finite : bool, optional (default=False)
        Whether to assert that all values within the ``X`` frame are
        finite. Note that if ``cols`` is specified, this will only assert
        all values in the specified columns are finite.

    column_diff : bool, optional (default=False)
        Whether to also get the columns present in ``X`` that are not present
        in ``cols``. This is returned as the third element in the output if
        ``column_diff`` is True.

    Examples
    --------
    When providing a dataframe and columns, the columns should be present:

    >>> from skoot.datasets import load_iris_df
    >>> df = load_iris_df(include_tgt=False, names=['a', 'b', 'c', 'd'])
    >>> df, cols = check_dataframe(df, cols=('a', 'c'))
    >>> assert cols == ['a', 'c']
    >>> df.head()
         a    b    c    d
    0  5.1  3.5  1.4  0.2
    1  4.9  3.0  1.4  0.2
    2  4.7  3.2  1.3  0.2
    3  4.6  3.1  1.5  0.2
    4  5.0  3.6  1.4  0.2

    When passing numpy arrays, account for the fact that the columns cannot
    be specified when creating the pandas dataframe:

    >>> df2, cols = check_dataframe(df.values, cols=[0, 2])
    >>> cols
    [0, 2]
    >>> df2.columns.tolist()
    [0, 1, 2, 3]
    >>> df2.head()
         0    1    2    3
    0  5.1  3.5  1.4  0.2
    1  4.9  3.0  1.4  0.2
    2  4.7  3.2  1.3  0.2
    3  4.6  3.1  1.5  0.2
    4  5.0  3.6  1.4  0.2

    If you want to get the ``column_diff``, or the left-out columns, this will
    be returned as a third element in the tuple when specifed:

    >>> df2, cols, diff = check_dataframe(df.values, [0, 2], column_diff=True)
    >>> cols
    [0, 2]
    >>> df2.columns.tolist()
    [0, 1, 2, 3]
    >>> diff
    [1, 3]

    Returns
    -------
    X_copy : DataFrame
        A copy of the ``X`` dataframe.

    cols : list
        The list of columns on which to apply a function to this dataframe.
        if ``cols`` was specified in the function, this is equal to ``cols``
        as a list. Else, it's the ``X.columns`` index.

    diff : list
        If ``column_diff`` is True, will return as the third position in the
        tuple the columns that are within ``X`` but NOT present in ``cols``.
    """
    # determine if it's currently a DF or if it needs to be cast as one.
    if not isinstance(X, pd.DataFrame):
        if not is_iterable(X):
            raise TypeError("X must be a DataFrame, iterable or np.ndarray, "
                            "but got type=%s" % type(X))

        # Old behavior:
        # if cols is not None:
        #     raise ValueError("When X is not a DataFrame, cols cannot be "
        #                      "defined. Either pre-cast your data to Pandas, "
        #                      "or pass cols=None.")

        # Discussion (feel free to add below):
        #   * Skoot is intended to speed things up and make life easier.
        #     Unnecessary constraints like this make life more difficult and
        #     add work for the user. I vote we do away with this constraint.
        #     This will allow users to pipe a sklearn transformer into a skoot
        #     transformer and use numeric columns as indices rather than having
        #     to pipe into a DF transformer FIRST. ALSO the next stage makes
        #     sure the columns they pass are valid, so as long as they pass
        #     integers, this should be totally fine.

        X = pd.DataFrame.from_records(X)

    # if columns are provided, check...
    present_columns = set(X.columns)
    if cols is not None:
        # ensure iterable, or copy if not
        cols = copy.deepcopy(cols) if is_iterable(cols) else [cols]

        # better to use "any" since it will short circuit!
        if any(c not in present_columns for c in cols):
            raise ValueError("All columns in `cols` must be present in X. "
                             "X columns=%r" % present_columns)

    # otherwise, if not specified, make sure we define it since we
    # end up returning cols (this is converted to a list in the next step)
    else:
        cols = X.columns

    # cols might have been a np.array or might be an Index -- make it a list
    if hasattr(cols, 'tolist'):
        cols = cols.tolist()
    elif not isinstance(cols, list):
        cols = list(cols)

    # if specified, check that all values are finite
    if assert_all_finite and \
            X[cols].apply(lambda x: (~np.isfinite(x)).sum()).sum() > 0:
        raise ValueError('Expected all entries in specified columns '
                         'to be finite')

    # get the copy of X to return
    X_copy = X.copy()

    # if column diff is defined, we need to get it...
    if column_diff:
        colset = set(cols)
        # make sure to iter X.columns and not present_columns to preserve order
        diff = [c for c in X.columns if c not in colset]  # O(1) lookup (set)
        return X_copy, cols, diff

    return X_copy, cols


def type_or_iterable_to_col_mapping(cols, param, param_name,
                                    permitted_scalar_types):
    """Map a parameter to various columns in a dict.

    Many estimators accept either scalar values or iterables as parameters to
    allow for different values across different features. This function creates
    a dictionary mapping column names to parameter values and validates scalars
    within a tuple of permitted scalar types.

    Note: this is primarily intended to be an internal method.

    Parameters
    ----------
    cols : list
        The list of columns against which to map some function or parameters.

    param : int, float, str, iterable or object
        The parameter value.

    param_name : str or unicode
        The name of the parameter

    permitted_scalar_types : type or iterable
        The permitted types.

    Examples
    --------
    >>> cols = ["a", "c"]
    >>> ticm = type_or_iterable_to_col_mapping  # too many characters...
    >>> assert ticm(cols, 0.5, "n_components", float) == {'a': 0.5, 'c': 0.5}
    >>> assert ticm(cols, "uniform", "strategy", str) == {'a': 'uniform',
    ...                                                   'c': 'uniform'}
    >>> assert ticm(cols, [3, 5], "q", int) == {'a': 3, 'c': 5}
    >>> assert ticm(cols, {"a": 3, "c": 5}, "q", int) == {'a': 3, 'c': 5}

    Returns
    -------
    param : dict
        The param dictionary.
    """
    # we need permitted scalar types to be a tuple of allowed values
    # (an instance, not the class)
    if not isinstance(permitted_scalar_types, tuple):
        permitted_scalar_types = (permitted_scalar_types,)

    # validate the parameter
    if is_iterable(param):
        # first smoke test is easy -- if the length of the number of
        # bins does not match the number of columns prescribed, raise
        if len(param) != len(cols):
            raise ValueError("Dim mismatch between cols and %s" % param_name)

        # next, we're concerned with whether the param iterable is a dict
        # and if it is, we have to validate the keys are all there...
        if isinstance(param, dict):

            # get sets of the columns and keys so we can easily compare
            scols = set(cols)
            skeys = set(param.keys())

            # if there are extra keys (skeys - scols) or missing keys
            # from the prescribed columns (scols - skeys) we have to raise
            if scols - skeys or skeys - scols:
                raise ValueError("When %s is provided as a dictionary "
                                 "its keys must match the provided cols."
                                 % param_name)

        # otherwise it's a non-dict iterable, and what we ultimately
        # want IS a dictionary
        else:
            param = dict(zip(cols, param))

    else:
        if not isinstance(param, permitted_scalar_types):
            raise TypeError("Permitted types for %s if not iterable: %s"
                            % (param_name, str(permitted_scalar_types)))

        # make it into a dictionary mapping cols to n_bins
        param = {c: param for c in cols}

    return param


def validate_multiple_cols(clsname, cols):
    """Validate that there are at least two columns to evaluate.

    This is used for various feature selection techniques, as well as
    in several feature extraction techniques.

    Parameters
    ----------
    clsname : str or unicode
        The name of the class that is calling the function.
        Used for more clear error messages.

    cols : array-like, shape=(n_features,)
        The columns to evaluate. If ``cols`` is not None
        and the length is less than 2, will raise a
        ``ValueError``.
    """
    if len(cols) < 2:
        raise ValueError('%s requires at least two features. Your data '
                         '(or the passed ``cols`` parameter) includes too '
                         'few features (%i)' % (clsname, len(cols)))


def validate_multiple_rows(clsname, frame):
    """Validate that there are at least two samples to evaluate.

    This is used for various feature transformation techniques, such as
    box-cox and yeo-johnson transformations.

    Parameters
    ----------
    clsname : str or unicode
        The name of the class that is calling the function.
        Used for more clear error messages.

    frame : array-like or pd.DataFrame, shape=(n_features, n_features)
        The samples to evaluate. If contains less than two samples,
        will raise a ValueError.
    """
    n_samples = frame.shape[0]
    if n_samples < 2:
        raise ValueError('%s requires at least two samples. Your data '
                         'includes too few samples (%i)'
                         % (clsname, n_samples))


def validate_test_set_columns(fit_columns, test_columns):
    """Validate that the test set columns will work.

    This function checks that the ``fit_columns`` are present in the
    ``test_columns`` set and raises a ValueError if not.

    Parameters
    ----------
    fit_columns : list or iterable
        The column names the estimator was fit on.

    test_columns : list or iterable
        The column names the test set contains.
    """
    present_cols = set(test_columns)  # O(1) lookup
    if not all(t in present_cols for t in fit_columns):
        raise ValueError("Not all fit columns present in test data! "
                         "(expected=%r, present=%r)"
                         % (fit_columns, test_columns))


# nosetest pb:
validate_test_set_columns.__test__ = False
