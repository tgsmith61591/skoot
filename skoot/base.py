# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import absolute_import, division, print_function

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import six

from abc import ABCMeta
import pandas as pd

from .exceptions import DeveloperError
from .utils.validation import check_dataframe, validate_test_set_columns
from .utils.iterables import is_iterable
from .utils.compat import xrange
from .utils.dataframe import dataframe_or_array

# namespace import to avoid explicitly protected imports in global namespace
from .utils import _docstr as dsutils

import warnings
import copy

__all__ = [
    'BasePDTransformer'
]


class BasePDTransformer(six.with_metaclass(ABCMeta, BaseEstimator,
                                           TransformerMixin)):
    __doc__ = """The base class for all Pandas frame transformers.

    Provides the base class for all skoot transformers that require
    Pandas dataframes as input.

    Parameters
    ----------
    {_cols_doc}
    
    {_as_df_doc}

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
    """.format(_cols_doc=dsutils._cols_doc, _as_df_doc=dsutils._as_df_doc)

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


class _SelectiveTransformerWrapper(six.with_metaclass(dsutils._WritableDoc,
                                                      BasePDTransformer)):
    # non-estimator parameters only used for the wrapper and not in set_params
    _p_names = ('cols', 'as_df', 'trans_col_name')

    # Build a selective transformer on the fly.
    #
    # This is a private method used at the head of submodules to wrap
    # sklearn modules in the selective interface. Do not include in __all__
    # since this method is for power-users/package developers. Do not
    # include in __all__ since this class is for power-users/package
    # developers.
    def __init__(self, cols=None, as_df=True, trans_col_name=None, **kwargs):

        super(_SelectiveTransformerWrapper, self).__init__(
            cols=cols, as_df=as_df)

        # this is a STATIC attribute of subclasses
        try:
            cls = self._cls
        except AttributeError:
            raise DeveloperError("_SelectiveTransformerWrapper subclasses "
                                 "must contain a static _cls attribute that "
                                 "maps to a sklearn type!")

        # get the (default) parameters for the estimator in question
        # and initialize to default
        self.estimator_ = cls()
        default_est_parms = self.estimator_.get_params(deep=True)

        # set the attributes in the estimator AND in the constructor so this
        # class behaves like sklearn in grid search
        self.estimator_.set_params(**kwargs)

        # set the kwargs here to behave like sklearn
        for k, v in six.iteritems(default_est_parms):
            if kwargs:
                v = kwargs.get(k, v)  # try get from kwargs, fail w def. value
            setattr(self, k, v)

        self.trans_col_name = trans_col_name

    def fit(self, X, y=None, **fit_kwargs):
        """Fit the wrapped transformer.

        This method will fit the wrapped sklearn transformer on the
        selected columns, leaving other columns alone.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None. Furthermore, ``X`` will
            not be altered in the process of the fit.

        y : array-like or None, shape=(n_samples,), optional (default=None)
            Pass-through for ``sklearn.pipeline.Pipeline``. Even
            if explicitly set, will not change behavior of ``fit``.
        """
        # check on state of X and cols
        X, cols = check_dataframe(X, self.cols)

        # fit the estimator in place
        self.estimator_.fit(X[cols], **fit_kwargs)

        # the columns we fit on
        self.fit_cols_ = cols

        return self

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
        X : pd.DataFrame, shape=(n_samples, n_features)
            The operation is applied to a copy of ``X``,
            and the result set is returned.
        """
        check_is_fitted(self, 'fit_cols_')

        # check on state of X and cols
        X, _, other_nms = check_dataframe(X, cols=self.cols,
                                          column_diff=True)

        # validate that the test set columns exist in the fit columns
        cols = self.fit_cols_
        validate_test_set_columns(cols, X.columns)

        # get the transformer
        est = self.estimator_
        transform = est.transform(X[cols])

        # get the transformed column names
        trans = self.trans_col_name
        n_trans_cols = transform.shape[1]
        if is_iterable(trans):
            if len(trans) != n_trans_cols:
                raise ValueError("dim mismatch in transformed column names "
                                 "and transformed column shape! (%i!=%i)"
                                 % (len(trans), n_trans_cols))
        # else it's some scalar
        else:
            if trans is None:  # default to class name
                trans = self.estimator_.__class__.__name__
            # this gets caught if it's None as well:
            trans = ["%s%i" % (trans, i + 1) for i in xrange(n_trans_cols)]

        # stack the transformed variables onto the RIGHT side
        right = pd.DataFrame.from_records(
            data=transform,
            columns=trans)

        # set the index of right to be equal to that of the input so
        # we can concat seamlessly
        right.index = X.index

        # concat if needed
        x = pd.concat([X[other_nms], right], axis=1) if other_nms else right
        return dataframe_or_array(x, self.as_df)

    @classmethod
    def _get_param_names(cls):
        # so we can get constructor args for grid search
        # (this is a closure)
        return list(cls._p_names) + \
               cls._cls._get_param_names()  # must have _cls!


class _AnonymousPDTransformer(BasePDTransformer):
    """General transformer wrapper used to make a commutative function
    into a Pipeline-able function.

    Parameters
    ----------
    func : callable
        The commutative function used to transform the train or test set.
    """
    def __init__(self, func):
        super(_AnonymousPDTransformer, self).__init__(
            cols=None, as_df=True)

        self.func = func

    def transform(self, X):
        """Apply the commutative function to the train or test set.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to transform.
        """
        return self.func(X)


def make_transformer(func):
    """Make a function into a scikit-learn TransformerMixin.

    Wraps a commutative function as an anonymous BasePDTransformer in order to
    fit into a Pipeline. The returned transformer class methods adhere to the
    standard BasePDTransformer ``fit`` and ``transform`` signatures.

    This is useful when a transforming function that does not fit any
    parameters is used to pre-process data at a point that might split a
    pipeline.

    Parameters
    ----------
    func : callable
        The function that will be used to transform a dataset.
    """
    # first, if it's a lambda function, warn the user.
    lam = lambda: None
    if isinstance(func, type(lam)) and func.__name__ == lam.__name__:
        warnings.warn("A lambda function was passed to the make_transformer "
                      "function. While not explicitly unsupported, this will "
                      "complicate transformer persistence. To persist "
                      "dynamically-created transformers, use def-style "
                      "functions.", UserWarning)

    return _AnonymousPDTransformer(func)
