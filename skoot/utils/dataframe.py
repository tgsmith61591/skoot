# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pandas as pd
import numpy as np

__all__ = [
    'get_numeric_columns',
    'safe_vstack'
]


def get_numeric_columns(X):
    """Get all numeric columns from a pandas DataFrame.

    This function selects all numeric columns from a pandas
    DataFrame. A numeric column is defined as a column whose
    ``dtype`` is a ``np.number``.

    Parameters
    ----------
    X : pd.DataFrame
        The input dataframe.
    """
    return X.select_dtypes(include=[np.number])


def safe_vstack(a, b):
    """Stack two arrays on top of one another.

    Safely handle vertical stacking of arrays. This works for
    either np.ndarrays or pd.DataFrames.

    Parameters
    ----------
    a : array-like, shape=(n_samples, n_features)
        The array that will be stacked on the top vertically.

    b : array-like, shape=(n_samples, n_features)
        The array that will be stacked below the other vertically.
    """
    # we can only pd.concat if they BOTH are DataFrames
    if all(isinstance(x, pd.DataFrame) for x in (a, b)):
        return pd.concat([a, b], axis=0)

    # otherwise, at least one of them is a numpy array (we think)
    return np.vstack([a, b])
