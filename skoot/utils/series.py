# -*- coding: utf-8 -*-

__all__ = [
    'is_datetime_type'
]


def is_datetime_type(series):
    """Determine whether a series is a datetime.

    Unfortunately, Pandas doesn't check series dtypes in a very canonical
    manner. We have to rely on string equality... this function is essentially
    a placeholder for checking for whether a series is a datetime until such
    a method is added by the Pandas developers.

    Parameters
    ----------
    series : pd.Series

    References
    ----------
    .. [1] Pandas Issue 8814, "API: preferred way to check if column/Series
           has Categorical dtype" (or any dtype for that matter)
           https://github.com/pandas-dev/pandas/issues/8814
    """
    return series.dtype.name == "datetime64[ns]"
