# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import absolute_import

from sklearn.utils.validation import check_is_fitted
from cerberus import Validator

from ..base import BasePDTransformer
from ..utils.validation import check_dataframe
from ..utils.dataframe import dataframe_or_array

import pandas as pd

__all__ = [
    'SchemaNormalizer'
]


class SchemaNormalizer(BasePDTransformer):
    r"""Enforce a schema on an input dataframe.

    The SchemaNormalizer fits a cerberus Validator class on training data,
    enforcing that schema across incoming test data. This ensures that new
    test data matches the expected schema.

    Parameters
    ----------
    schema : dict
        The schema. This dictionary maps column names to actions. For
        instance the following schema will cast the iris dataset
        "petal widtch (cm)" column to integer::

            >>> schema = {'petal width (cm)': {'coerce': int}}

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.
        Since most skoot transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.

    Attributes
    ----------
    validator_ : cerberus.Validator
        The cerberus validator object. Used to enforce schemas on
        input test data.
    """

    def __init__(self, schema, as_df=True):

        super(SchemaNormalizer, self).__init__(
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

        return dataframe_or_array(X, self.as_df)
