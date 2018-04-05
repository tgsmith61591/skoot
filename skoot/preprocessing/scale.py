# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Scaling predictors

from __future__ import absolute_import

from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone

from ..base import BasePDTransformer
from ..utils.validation import check_dataframe, validate_test_set_columns

__all__ = [
    'SelectiveScaler'
]


class SelectiveScaler(BasePDTransformer):
    """Scale selected features.

    A class that will apply scaling only to a select group
    of columns. Useful for data that may contain features that should not
    be scaled, such as those that have been dummied, or for any already-in-
    scale features. Perhaps, even, there are some features you'd like to scale
    in a different manner than others. This, then, allows two back-to-back
    ``SelectiveScaler`` instances with different columns & strategies in a
    pipeline object.

    Parameters
    ----------
    cols : array-like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the transformer will be ``fit``
        on the entire frame. Note that the transformation will also only
        apply to the specified columns, and any other non-specified
        columns will still be present after transformation. Note that since
        this transformer can only operate on numeric columns, not explicitly
        setting the ``cols`` parameter may result in errors for categorical
        data.

    scaler : sklearn.preprocessing.BaseScaler, optional (default=None)
        The scaler to fit against ``cols``. Must be an instance of
        ``sklearn.preprocessing.BaseScaler``. If None, will default to
        ``StandardScaler``. If provided, the fit estimator will be a clone
        of the input estimator.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.
        Since most skoot transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.

    Attributes
    ----------
    scaler_ : BaseScaler
        The fit scaler. This is different from the reference passed in the
        constructor, even if an estimator was provided. This will be a clone
        of the input scaler.

    fit_cols_ : list
        The list of column names on which the transformer was fit. This
        is used to validate the presence of the features in the test set
        during the ``transform`` stage.

    Examples
    --------
    The following example will scale only the first two features
    in the iris dataset using the default, a StandardScaler:

    >>> from skoot.preprocessing import SelectiveScaler
    >>> from skoot.datasets import load_iris_df
    >>>
    >>> X = load_iris_df(include_tgt=False)
    >>> trans = SelectiveScaler(cols=X.columns[:2])
    >>> X_transform = trans.fit_transform(X)
    >>>
    >>> X_transform.head()
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
    0          -0.900681          1.032057                1.4               0.2
    1          -1.143017         -0.124958                1.4               0.2
    2          -1.385353          0.337848                1.3               0.2
    3          -1.506521          0.106445                1.5               0.2
    4          -1.021849          1.263460                1.4               0.2

    This example shows how to use the selective scaler with an already-
    initialized scaler estimator:

    >>> from sklearn.preprocessing import RobustScaler
    >>> rb_scale = RobustScaler()  # pre-initialized
    >>> trans = SelectiveScaler(cols=X.columns[:2], scaler=rb_scale)
    >>> X_transform = trans.fit_transform(X)
    >>>
    >>> X_transform.head()
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
    0          -0.538462               1.0                1.4               0.2
    1          -0.692308               0.0                1.4               0.2
    2          -0.846154               0.4                1.3               0.2
    3          -0.923077               0.2                1.5               0.2
    4          -0.615385               1.2                1.4               0.2
    >>> assert trans.scaler_ is not rb_scale  # not the same ref; a clone
    """
    def __init__(self, cols=None, scaler=None, as_df=True):
        super(SelectiveScaler, self).__init__(cols=cols, as_df=as_df)
        self.scaler = scaler

    def fit(self, X, y=None):
        """Fit the scaler.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : array-like or None, shape=(n_samples,), optional (default=None)
            Pass-through for ``sklearn.pipeline.Pipeline``.
        """
        # check on state of X and cols
        X, cols = check_dataframe(X, cols=self.cols)

        # if scaler does not exist, create one
        scaler = self.scaler
        if scaler is None:
            scaler = StandardScaler()

        # otherwise try to clone it
        else:
            scaler = clone(scaler)

        # throws exception if the cols don't exist
        scaler.fit(X[cols])

        # set the scaler and transform columns as fit params
        self.scaler_ = scaler
        self.fit_cols_ = cols

        # this is our fit param
        self.is_fit_ = True
        return self

    def transform(self, X):
        """Scale a test dataframe.

        This method will scale selected values within a test
        dataframe with the statistics computed in ``fit``.

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
        check_is_fitted(self, 'scaler_')

        # check on state of X and cols
        X, _ = check_dataframe(X, cols=self.cols)

        # validate test set cols
        cols = self.fit_cols_
        validate_test_set_columns(cols, test_columns=X.columns.tolist())
        X[cols] = self.scaler_.transform(X[cols])

        return X if self.as_df else X.values
