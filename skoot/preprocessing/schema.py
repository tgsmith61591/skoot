# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from sklearn.utils.validation import check_is_fitted
import six

from ..base import BasePDTransformer
from ..utils.validation import check_dataframe, validate_test_set_columns
from ..utils.metaestimators import timed_instance_method

__all__ = [
    'SchemaNormalizer'
]


class SchemaNormalizer(BasePDTransformer):
    r"""Enforce a schema on an input dataframe.

    The SchemaNormalizer enforces a schema across incoming train and
    test data. This ensures that all data matches the expected schema.
    Note that unlike most other Skoot transformers, this one requires
    that the output be a DataFrame (note the lack of the ``as_df``
    constructor arg).

    Parameters
    ----------
    schema : dict
        The schema. This dictionary maps column names to actions. For
        instance the following schema will cast the iris dataset
        "petal widtch (cm)" column to integer::

            >>> schema = {'petal width (cm)': int}

    Attributes
    ----------
    fit_cols_ : list
        The list of column names on which the transformer was fit. This
        is used to validate the presence of the features in the test set
        during the ``transform`` stage.
    """

    def __init__(self, schema):

        super(SchemaNormalizer, self).__init__(
            as_df=True,  # Does not really matter, it always returns one
            cols=None)

        self.schema = schema

    @timed_instance_method(attribute_name="fit_time_")
    def fit(self, X, y=None):
        """Fit the transformer.

        Default behavior is not to fit any parameters and return self.
        This is useful for transformers which do not require
        parameterization, but need to fit into a pipeline.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to fit.

        y : array-like or None, shape=(n_samples,), optional (default=None)
            Pass-through for ``sklearn.pipeline.Pipeline``.
        """
        _, self.fit_cols_ = check_dataframe(X, cols=self.cols)
        return self

    def transform(self, X):
        """Apply the schema normalization.

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

        # normalize
        for k, v in six.iteritems(self.schema):
            X[k] = X[k].astype(v)

        return X  # DataFrame
