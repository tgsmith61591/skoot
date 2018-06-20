# -*- coding: utf-8 -*-
#
# Feature engineering for dates

from __future__ import absolute_import

from ..base import BasePDTransformer
from ..utils.validation import check_dataframe, validate_test_set_columns
from ..utils.iterables import ensure_iterable
from ..utils.series import is_datetime_type
from ..utils.dataframe import dataframe_or_array

from sklearn.utils.validation import check_is_fitted

import numpy as np
import pandas as pd

__all__ = [
    "DateFactorizer"
]


def _factorize(X, cols, feature_names, sep):
    right_side = None
    for col in cols:
        series = X[col]

        # Now the real challenge here is that some of the columns passed
        # may not be datetimes, which is required for this transformer.
        if not is_datetime_type(series):
            raise ValueError("The DateFactorizer requires passed columns "
                             "to be DateTime types. Consider using the "
                             "skoot.preprocessing.DateTransformer first.")

        # First, just extract each individual component from the date
        feat = np.asarray(
            series.apply(
                lambda d: [getattr(d, f)
                           for f in feature_names]).values.tolist())

        pd_features = pd.DataFrame.from_records(
            feat, columns=["%s%s%s" % (col, sep, feature)
                           for feature in feature_names])

        # Our single feature has just become a matrix. We'll make it into
        # a pandas frame that keeps getting concatenated together
        if right_side is None:  # first pass
            right_side = pd_features
        else:
            # No need to reset index here since right_side will be 0, 1, .., N
            right_side = pd.concat([right_side, pd_features], axis=1)

    # concat to the original df. we DO need to reset index of right_side here.
    right_side.index = X.index
    X = pd.concat([X, right_side], axis=1)
    return X


class DateFactorizer(BasePDTransformer):
    """Extract new features from datetime features.

    Automatically extract new features from datetime features. This class
    operates on datetime series objects and extracts features such as "year",
    "month", etc. These can then be expanded via one-hot encoding or further
    processed via other pre-processing techniques.

    Parameters
    ----------
    cols : array-like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        Will apply to all columns if None specified. Note that in this class,
        the columns applied-to must be DateTime types or this will raise a
        ValueError.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.

    drop_original : bool, optional (default=True)
        Whether to drop the original features from the dataframe prior to
        returning from the ``transform`` method.

    sep : str or unicode, optional (default="_")
        The string separator between the existing feature name and the
        extracted feature. E.g., for a feature named "Transaction" and for
        ``features=("year", "month")``, the original variable will be split
        into two new ones: "Transaction_year" and "Transaction_month".

    features : iterable, optional (default=("year", "month", "day", "hour"))
        The features to extract. These are attributes of the DateTime class
        and will raise an AttributeError if an invalid feature is passed.

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime as dt
    >>> data = [
    ...     [1, dt.strptime("06-01-2018", "%m-%d-%Y")],
    ...     [2, dt.strptime("06-02-2018", "%m-%d-%Y")],
    ...     [3, dt.strptime("06-03-2018", "%m-%d-%Y")],
    ...     [4, dt.strptime("06-04-2018", "%m-%d-%Y")],
    ...     [5, None]
    ... ]
    >>> df = pd.DataFrame.from_records(data, columns=["a", "b"])
    >>> DateFactorizer(cols=['b']).fit_transform(df)
       a  b_year  b_month  b_day  b_hour
    0  1  2018.0      6.0    1.0     0.0
    1  2  2018.0      6.0    2.0     0.0
    2  3  2018.0      6.0    3.0     0.0
    3  4  2018.0      6.0    4.0     0.0
    4  5     NaN      NaN    NaN     NaN

    Attributes
    ----------
    fit_cols_ : list
        The columns the transformer was fit on.
    """
    def __init__(self, cols=None, as_df=True, drop_original=True, sep="_",
                 features=("year", "month", "day", "hour")):
        super(DateFactorizer, self).__init__(
            cols=cols, as_df=as_df)

        self.drop_original = drop_original
        self.sep = sep
        self.features = features

    def fit(self, X, y=None):
        """Fit the date factorizer.

        This is a tricky class because the "fit" isn't super necessary...
        But we use it as a validation stage to ensure the defined cols
        genuinely are datetime columns. That's the only reason this all
        happens in the fit portion.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : array-like or None, shape=(n_samples,), optional (default=None)
            Pass-through for ``sklearn.pipeline.Pipeline``.
        """
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None, **kwargs):
        """Fit the estimator and apply the date factorization to a dataframe.

        This is a tricky class because the "fit" isn't super necessary...
        But we use it as a validation stage to ensure the defined cols
        genuinely are datetime types. That's the only reason this all
        happens in the fit portion.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to fit. The operation will
            be applied to a copy of the input data, and the result
            will be returned.

        y : array-like or None, shape=(n_samples,), optional (default=None)
            Pass-through for ``sklearn.pipeline.Pipeline``.

        Returns
        -------
        X : pd.DataFrame or np.ndarray, shape=(n_samples, n_features)
            The operation is applied to a copy of ``X``,
            and the result set is returned.
        """
        X, cols = check_dataframe(X, cols=self.cols, assert_all_finite=False)

        # compute the factorized features and unify with the original DF
        features = ensure_iterable(self.features)
        X = _factorize(X, cols, features, self.sep)  # type: pd.DataFrame

        # remove the original columns if necessary
        if self.drop_original:
            X = X.drop(cols, axis=1)

        # set the self params
        self.fit_cols_ = cols
        return dataframe_or_array(X, self.as_df)

    def transform(self, X):
        """Apply the date transformation to a dataframe.

        This method will extract features from datetime features as
        specified by the ``features`` arg.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to transform. The operation will
            be applied to a copy of the input data, and the result
            will be returned.

        Returns
        -------
        X : pd.DataFrame or np.ndarray, shape=(n_samples, n_features)
            The operation is applied to a copy of ``X``,
            and the result set is returned.
        """
        check_is_fitted(self, "fit_cols_")
        X, _ = check_dataframe(X, cols=self.cols)

        # validate that fit cols in test set
        cols = self.fit_cols_
        validate_test_set_columns(cols, X.columns)

        # compute the factorized features and unify with the original DF
        X = _factorize(X, cols, ensure_iterable(self.features),
                       self.sep)  # type: pd.DataFrame

        # remove the original columns if necessary
        if self.drop_original:
            X = X.drop(cols, axis=1)
        return dataframe_or_array(X, self.as_df)
