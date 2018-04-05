# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import division, print_function, absolute_import

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.validation import check_is_fitted

import pandas as pd

from ..base import BasePDTransformer
from ..utils.validation import check_dataframe, validate_test_set_columns

__all__ = [
    'DummyEncoder'
]


class DummyEncoder(BasePDTransformer):
    """

    A custom one-hot encoding class that handles previously unseen
    levels and automatically drops one level from each categorical
    feature to avoid the dummy variable trap.

    Parameters
    ----------
    cols : array-like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the transformer will be ``fit``
        on the entire frame. Note that the transformation will also only
        apply to the specified columns, and any other non-specified
        columns will still be present after transformation. Specified
        columns will be dropped after they are expanded.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.
        Since most skoot transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.

    sep : str or unicode, optional (default='_')
        The string separator between the categorical feature name
        and the level name.

    drop_one_level : bool, optional (default=True)
        Whether to drop one level for each categorical variable.
        This helps avoid the dummy variable trap.

    Attributes
    ----------
    ohe_ : OneHotEncoder
        The one hot encoder

    le_ : dict[str: LabelEncoder]
        A dictionary mapping column names to their respective LabelEncoder
        instances.

    fit_cols_ : list
        The list of column names on which the transformer was fit. This
        is used to validate the presence of the features in the test set
        during the ``transform`` stage.
    """
    def __init__(self, cols, as_df=True, sep='_', drop_one_level=True):

        super(DummyEncoder, self).__init__(
            cols=cols, as_df=as_df)

        self.sep = sep
        self.drop_one_level = drop_one_level

    def fit(self, X, y=None):
        """Fit the dummy encoder.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : array-like or None, shape=(n_samples,), optional (default=None)
            Pass-through for ``sklearn.pipeline.Pipeline``.
        """
        # validate the input, and get a copy of it
        X, cols = check_dataframe(X, cols=self.cols,
                                  assert_all_finite=True)

        # begin fit
        # for each column, fit a label encoder
        lab_encoders = {}
        for col in cols:
            # get the vec, fit the label encoder
            vec = X[col].values
            le = LabelEncoder()
            lab_encoders[col] = le.fit(vec)

            # transform the column, re-assign
            X[col] = le.transform(vec)

        # fit a single OHE on the transformed columns
        ohe = OneHotEncoder(sparse=False).fit(X[cols])

        # assign fit params
        self.ohe_ = ohe
        self.le_ = lab_encoders
        self.fit_cols_ = cols

        return self

    def transform(self, X):
        """Apply the imputation to a dataframe.

        This method will fill in the missing values within a test
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
        check_is_fitted(self, 'ohe_')
        X, _ = check_dataframe(X, cols=self.cols)

        # validate that fit cols in test set
        cols = self.fit_cols_
        validate_test_set_columns(cols, X.columns)

        # fit params that we need
        ohe = self.ohe_
        lenc = self.le_
        sep = self.sep
        drop = self.drop_one_level
        col_order = []
        drops = []

        for col in cols:
            # get the vec, transform via the label encoder
            vec = X[col].values

            le = lenc[col]
            vec_trans = le.transform(vec)  # Union[str, int] -> int
            X[col] = vec_trans

            # get the column names (levels) so we can predict the
            # order of the output cols
            le_clz = le.classes_.tolist()
            classes = ["%s%s%s" % (col, sep, clz) for clz in le_clz]
            col_order.extend(classes)

            # if we want to drop one, just drop the last
            if drop and len(le_clz) > 1:
                drops.append(classes[-1])

        # now we can get the transformed OHE
        ohe_trans = pd.DataFrame.from_records(data=ohe.transform(X[cols]),
                                              columns=col_order)

        # set the index to be equal to X's for a smooth concat
        ohe_trans.index = X.index

        # if we're dropping one level, do so now
        if drops:
            ohe_trans = ohe_trans.drop(drops, axis=1)

        # drop the original columns from X
        X = X.drop(cols, axis=1)

        # concat the new columns
        X = pd.concat([X, ohe_trans], axis=1)  # type: pd.DataFrame
        return X if self.as_df else X.values
