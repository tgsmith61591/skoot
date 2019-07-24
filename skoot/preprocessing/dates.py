# -*- coding: utf-8 -*-
#
# Coerce string fields to dates

import six
from sklearn.utils.validation import check_is_fitted

from ..base import BasePDTransformer
from ..utils.validation import type_or_iterable_to_col_mapping
from ..utils.validation import check_dataframe, validate_test_set_columns
from ..utils.metaestimators import timed_instance_method
from ..utils.compat import NoneType
from ..utils.series import is_datetime_type

import pandas as pd

__all__ = [
    "DateTransformer"
]


def _cast_to_datetime(X, cols, formats, allowed_types):
    # Now the real challenge here is that some of the columns passed
    # may not be date-parseable... we'll duck type it. If it fails, it
    # cannot be parsed, and we will let Pandas raise for that. No sense
    # policing it if they are already doing that.
    def cast(f):
        fmt = formats[f.name]

        # First make sure the type is in allowed types
        dtype = f.dtype.name
        if dtype not in allowed_types:
            raise ValueError("dtype '%s' not in `allowed_types` (%r)"
                             % (dtype, allowed_types))

        # Now if the format is already a datetime, we can return early.
        # If the format isn't defined we can infer it, otherwise we can
        # parse it explicitly
        if is_datetime_type(f):
            return f
        elif fmt is None:
            return pd.to_datetime(f, infer_datetime_format=True,
                                  errors='raise')
        # otherwise the fmt is defined so we'll let it fail out on its own
        # if it cannot cast it
        return pd.to_datetime(f, format=fmt)

    casted = X[cols].apply(cast)
    X[cols] = casted
    return X


class DateTransformer(BasePDTransformer):
    """Cast features to datetime.

    Convert multiple features with potentially differing formats to datetime
    with specified formats or by inferring the formats. Note that unlike most
    other Skoot transformers, this one requires that the output be a DataFrame
    (note the lack of the ``as_df`` constructor arg).

    Parameters
    ----------
    cols : array-like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        Will apply to all columns if None specified

    date_format : str, iterable or None, optional (default=None)
        The date format. If None, will infer. If a string, will be used to
        parse the datetime. If an iterable, should contain strings or None
        positionally corresponding to ``cols`` (or a dict mapping columns
        to formats).

    allowed_types : iterable, optional (default=("object", "datetime64[ns]"))
        Permitted Series types. This is used to prevent accidentally casting
        Series of unexpected types to DateTime. For instance, integer types
        can be cast to DateTime even though the behavior may be unexpected.

    Notes
    -----
    The ``fit`` method here is only used for validation that the columns can
    be cast to datetime.

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime as dt
    >>> data = [
    ...     [1, "06/01/2018", dt.strptime("06-01-2018", "%m-%d-%Y")],
    ...     [2, "06/02/2018", dt.strptime("06-02-2018", "%m-%d-%Y")],
    ...     [3, "06/03/2018", dt.strptime("06-03-2018", "%m-%d-%Y")],
    ...     [4, None, dt.strptime("06-04-2018", "%m-%d-%Y")],
    ...     [5, "06/05/2018", None]
    ... ]
    >>> df = pd.DataFrame.from_records(data, columns=["a", "b", "c"])
    >>> converter = DateTransformer(cols=["b", "c"],
    ...                             date_format=["%m/%d/%Y", None])
    >>> converter.fit_transform(df)
       a          b          c
    0  1 2018-06-01 2018-06-01
    1  2 2018-06-02 2018-06-02
    2  3 2018-06-03 2018-06-03
    3  4        NaT 2018-06-04
    4  5 2018-06-05        NaT

    Attributes
    ----------
    DEFAULT_PERMITTED_DTYPES : tuple
        This is a static attribute that controls what types can be cast.
        These are the default permitted pandas dtypes that are allowed.
        If a column is not one of these types, a ValueError will be raised.
        To cast an int to datetime, for instance, the ``allowed_types`` arg
        will need to include "int64"::

            allowed_types=DateTransformer.DEFAULT_PERMITTED_DTYPES + ('int64',)

    fit_cols_ : list
        The columns the transformer was fit on.

    formats_ : dict
        Maps column name to date format, in case of varying date formats
        passed in the ``date_format`` parameter.
    """
    DEFAULT_PERMITTED_DTYPES = ("object", "datetime64[ns]")

    def __init__(self, cols=None, date_format=None,
                 allowed_types=DEFAULT_PERMITTED_DTYPES):
        super(DateTransformer, self).__init__(
            cols=cols, as_df=True)

        self.date_format = date_format
        self.allowed_types = allowed_types

    # Don't decorate 'fit' since it calls fit_transform
    def fit(self, X, y=None):
        """Fit the date transformer.

        This is a tricky class because the "fit" isn't super necessary...
        But we use it as a validation stage to ensure the defined cols
        genuinely can be cast to datetime. That's the only reason this all
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

    @timed_instance_method(attribute_name="fit_time_")
    def fit_transform(self, X, y=None, **kwargs):
        """Fit the estimator and apply the date transformation
        to a dataframe.

        This is a tricky class because the "fit" isn't super necessary...
        But we use it as a validation stage to ensure the defined cols
        genuinely can be cast to datetime. That's the only reason this all
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

        # Different fields may have different formats, so we have to
        # allow a number of different formats to be passed.
        formats = type_or_iterable_to_col_mapping(
            cols=cols, param=self.date_format,
            param_name="date_format",
            permitted_scalar_types=six.string_types + (NoneType,))

        X = _cast_to_datetime(X, cols, formats, self.allowed_types)

        self.fit_cols_ = cols
        self.formats_ = formats
        return X

    def transform(self, X):
        """Apply the date transformation to a dataframe.

        This method will cast string features to datetimes as specified by
        the ``date_format`` arg.

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

        # transform
        return _cast_to_datetime(X, cols, self.formats_, self.allowed_types)
