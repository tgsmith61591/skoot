# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pandas as pd
import numpy as np

__all__ = [
    'get_numeric_columns'
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
