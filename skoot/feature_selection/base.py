# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import print_function, division, absolute_import

from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six
from abc import ABCMeta

from ..base import BasePDTransformer
from ..utils.validation import check_dataframe

import warnings

__all__ = [
    'BaseFeatureSelector'
]


def validate_multiple_cols(clsname, cols):
    """Validate that there are at least two columns to evaluate.

    This is used for the MulticollinearityFilterer, the
    LinearCombinationsFilterer, and potentially more,
    as they require there be at least two columns.

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


class BaseFeatureSelector(six.with_metaclass(ABCMeta, BasePDTransformer)):
    """Base class for feature selectors.

    The base class for all skoot feature selectors, the _BaseFeatureSelector
    should adhere to the following behavior:

        * The ``fit`` method should only fit the specified columns
          (since it's also a ``SelectiveMixin``), fitting all columns
          only when ``cols`` is None.

        * The ``fit`` method should not change the state of the training frame.

        * The transform method should return a copy of the test frame,
          dropping the columns identified as "bad" in the ``fit`` method.

    Parameters
    ----------
    cols : array-like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the transformer will be ``fit``
        on the entire frame. Note that the transformation will also only
        apply to the specified columns, and any other non-specified
        columns will still be present after transformation.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead. 
        Since most skoot transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.
    """
    def __init__(self, cols=None, as_df=True):
        # simple pass-through for the super constructor call
        super(BaseFeatureSelector, self).__init__(
            cols=cols, as_df=as_df)

    def transform(self, X):
        """Transform a test dataframe.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to transform. The operation will
            be applied to a copy of the input data, and the result
            will be returned.

        Returns
        -------
        X_select : pd.DataFrame, shape=(n_samples, n_features)
            The selected columns from ``X``.
        """
        check_is_fitted(self, 'drop_')

        # check on state of X and cols
        X, cols = check_dataframe(X, self.cols)

        # if there's nothing to drop
        drop_columns = self.drop_  # type: list
        if not drop_columns:
            return X if self.as_df else X.as_matrix()

        # otherwise, there's something to drop
        else:
            # what if we don't want to throw this key error for a non-existent
            # column that we hope to drop anyways? We need to at least inform
            # the user...
            colset = set(X.columns)
            drops = [x for x in drop_columns if x in colset]

            # for length mismatch, we know there's a missing column
            if len(drops) != len(drop_columns):
                warnings.warn('one or more features to drop not contained '
                              'in input data feature names (drop=%r)'
                              % drop_columns, UserWarning)

            dropped = X.drop(drops, axis=1)
            return dropped if self.as_df else dropped.as_matrix()
