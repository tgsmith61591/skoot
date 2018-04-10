# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import absolute_import, division, print_function

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import six
from abc import ABCMeta

import copy

__all__ = [
    'BasePDTransformer'
]


class BasePDTransformer(six.with_metaclass(ABCMeta, BaseEstimator,
                                           TransformerMixin)):
    """The base class for all Pandas frame transformers.

    Provides the base class for all skoot transformers that require
    Pandas dataframes as input.

    Parameters
    ----------
    cols : array_like, shape=(n_features,), optional (default=None)
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

    Examples
    --------
    The following is an example of how to subclass a BasePDTransformer:

        >>> from skoot.base import BasePDTransformer
        >>> class A(BasePDTransformer):
        ...     def __init__(self, cols=None, as_df=None):
        ...             super(A, self).__init__(cols, as_df)
        ...
        >>> A()
        A(as_df=None, cols=None)
    """
    def __init__(self, cols=None, as_df=True):
        self.cols = copy.deepcopy(cols)  # do not let be mutable!
        self.as_df = as_df

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
        return self
