# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import absolute_import

from sklearn.utils.validation import check_is_fitted
from cerberus import Validator

from skoot.base import BasePDTransformer
from skoot.utils.validation import check_dataframe, validate_test_set_columns

import pandas as pd

__all__ = [
    'Normalizer'
]


class Normalizer(BasePDTransformer):
    def __init__(self, schema, as_df=True):

        super(Normalizer, self).__init__(
            as_df=as_df, cols=None)

        self.schema = schema

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
        self.validator_ = Validator(self.schema)
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
        check_is_fitted(self, "validator_")
        X, _ = check_dataframe(X, cols=self.cols)

        # make the document, normalize
        v = self.validator_
        X = pd.DataFrame.from_records([
            v.normalized(record)
            for record in X.to_dict(orient='records')
        ])

        return X if self.as_df else X.values
