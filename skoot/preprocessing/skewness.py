# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Classes and functions for rectifying skewness in transformers.

import numpy as np
from abc import ABCMeta, abstractmethod

from scipy import optimize
from scipy.stats import boxcox

import six
from joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted

from ..base import BasePDTransformer
from ..decorators import suppress_warnings as suppress
from ..utils.validation import (check_dataframe, validate_multiple_rows,
                                validate_test_set_columns)
from ..utils.dataframe import dataframe_or_array
from ..utils.metaestimators import timed_instance_method

__all__ = [
    'BoxCoxTransformer',
    'YeoJohnsonTransformer'
]

ZERO = 1E-16


def _bc_est_lam(y, min_value, dtype, suppress_warnings):
    """Estimate the lambda param for box-cox transformations.

    Estimate lambda for a single y, given a range of lambdas
    through which to search. No validation performed.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
       The vector from which lambda is being estimated
    """
    # ensure is array, floor at min_value
    y = np.maximum(np.asarray(y).astype(dtype), min_value)

    # Use scipy's log-likelihood estimator (suppress the inner optimization
    # routine otherwise it gets pretty annoyingly verbose)
    @suppress
    def _boxcox_inner():
        return boxcox(y, lmbda=None)

    # if we want to suppress, decorate now
    if suppress_warnings:
        _boxcox_inner = suppress(_boxcox_inner)
    b = _boxcox_inner()

    # Return lambda corresponding to maximum P
    return b[1]


def _yj_est_lam(y, brack, dtype=np.float32):
    y = np.asarray(y).astype(dtype)

    # Use MLE to compute the optimal YJ parameter
    def _mle_opt(i, brck):
        def _eval_mle(lmb, data):
            # Function to minimize
            return -_yj_llf(data, lmb)

        # Suppress the invalid scalar warnings we might get in the
        # optimization routine.
        @suppress
        def brent_optimize():
            return optimize.brent(_eval_mle, brack=brck, args=(i,))

        # suppressed version:
        return brent_optimize()

    return _mle_opt(y, brack)  # _mle(x, brack)


def _yj_llf(data, lmb):
    """YJ-transform a vector.

    Transform a y vector given a single lambda value,
    and compute the log-likelihood function. No validation
    is applied to the input.

    Parameters
    ----------
    data : array-like
       The vector to transform

    lmb : scalar
       The lambda value
    """
    # make into a numpy array, if not already one
    data = np.asarray(data)
    n = data.shape[0]

    # transform the vector
    y_trans = _yj_transform_y(data, lmb)

    # We can't take the canonical log of data, as there could be
    # zeros or negatives. Thus, we need to shift both distributions
    # up by some arbitrary factor just for the LLF computation
    min_d, min_y = np.min(data), np.min(y_trans)
    if min_d < ZERO:
        shift = np.abs(min_d) + 1
        data += shift

    # Same goes for Y
    if min_y < ZERO:
        shift = np.abs(min_y) + 1
        y_trans += shift

    # Compute mean on potentially shifted data
    y_mean = np.mean(y_trans, axis=0)
    var = np.sum((y_trans - y_mean) ** 2. / n, axis=0)

    # If var is 0.0, we'll get a warning. Means all the
    # values were nearly identical in y, so we will return
    # NaN so we don't optimize for this value of lam
    if 0 == var:
        return np.nan

    llf = (lmb - 1) * np.sum(np.log(data), axis=0)
    llf -= n / 2.0 * np.log(var)

    return llf


def _yj_transform_y(y, lam, dtype=np.float32):
    # should already be a vec, but just gotta be sure
    y = np.asarray(y).astype(dtype)

    # need some different masks...
    gte_zero_mask = y >= 0
    lt_zero_mask = ~gte_zero_mask  # negative number

    # lambda "masks" (just scalar booleans...)
    lam_gt_zero = lam > ZERO
    lam_eq_zero = not lam_gt_zero  # because bound in (0, 2)
    lam_eq_two = lam == 2.  # max, because bound in (0, 2)
    lam_not_two = not lam_eq_two

    # Case 1: x >= 0 and lambda is not 0
    c1_mask = gte_zero_mask & lam_gt_zero
    y[c1_mask] = (((y[c1_mask] + 1.) ** lam) - 1.0) / lam

    # Case 2: x >= 0 and lambda IS 0
    c2_mask = gte_zero_mask & lam_eq_zero
    y[c2_mask] = np.log(y[c2_mask] + 1.)

    # Case 3: x < 0 and lambda is not two
    c3_mask = lt_zero_mask & lam_not_two
    two_min_lam = (2. - lam)
    y[c3_mask] = -(((-y[c3_mask] + 1.) ** two_min_lam) - 1.0) / two_min_lam

    # Case 4: x < 0 and lam == 2.
    c4_mask = lt_zero_mask & lam_eq_two
    y[c4_mask] = -np.log(-y[c4_mask] + 1.)

    # Old method of mapping over single elements (super slow)
    # def _yj_trans_single_x(x):
    #     if x >= 0:
    #         # Case 1: x >= 0 and lambda is not 0
    #         if not _eqls(lam, ZERO):
    #             return (np.power(x + 1, lam) - 1.0) / lam
    #
    #         # Case 2: x >= 0 and lambda is zero
    #         return log(x + 1)
    #     else:
    #         # Case 3: x < 0 and lambda is not two
    #         if not lam == 2.0:
    #             denom = 2.0 - lam
    #             numer = np.power((-x + 1), (2.0 - lam)) - 1.0
    #             return -numer / denom
    #
    #         # Case 4: x < 0 and lambda is two
    #         return -log(-x + 1)
    #
    # return np.array([_yj_trans_single_x(x) for x in y])

    return y


class _BaseSkewnessTransformer(six.with_metaclass(ABCMeta, BasePDTransformer)):
    def __init__(self, cols, n_jobs, as_df, dtype):

        super(_BaseSkewnessTransformer, self).__init__(
            cols=cols, as_df=as_df)

        self.n_jobs = n_jobs
        self.dtype = dtype

    def _fit(self, X, estimation_function):
        # check on state of X and cols (all cols need to be finite!)
        X, cols = check_dataframe(X, cols=self.cols, assert_all_finite=True)

        # ensure enough rows
        validate_multiple_rows(self.__class__.__name__, X)

        # Now estimate the lambdas in parallel
        n_jobs = self.n_jobs
        dtype = self.dtype
        kwargs = self._estimator_kwargs()

        self.lambda_ = list(
            Parallel(n_jobs=n_jobs)(
                delayed(estimation_function)(X[i], dtype, **kwargs)
                for i in cols))

        # set the fit cols
        self.fit_cols_ = cols

        return self

    @abstractmethod
    def _transform_vector(self, y, lam):
        """An abstract function for box-cox and YJ transformers.
        This function should transform a vector given the pre-estimated
        lambda value.
        """

    def _estimator_kwargs(self):
        """Get the **kwargs for the estimator functions.

        BoxCox and YeoJohnson estimators take different args for their
        estimating functions, so this allows us to pass the kwargs to the
        ``estimate`` func. Default is an empty dict but can be overridden
        by subclasses that need it.
        """
        return dict()

    def transform(self, X):
        """Apply the transformation to a dataframe.

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
        check_is_fitted(self, 'lambda_')

        # check on state of X and cols
        X, _ = check_dataframe(X, cols=self.cols, assert_all_finite=True)

        # validate the test columns
        cols = self.fit_cols_
        validate_test_set_columns(cols, X.columns)

        # we don't care how many samples are in the test set... just need
        # > 1 for the fit/estimation procedure, but not the test set.
        _, n_features = X.shape
        lambdas_ = self.lambda_

        # do transformations
        for nm, lam in zip(cols, lambdas_):
            X[nm] = self._transform_vector(X[nm], lam)

        return dataframe_or_array(X, self.as_df)


# A dumb hack bc we cannot pickle functions or instancemethods...
# so these estimator wrappers simply call the appropriate estimator
# function while allowing us to abstract out the fit/transform code.
class _BCEstimator(object):
    def __init__(self, min_val):
        self.min_val = min_val

    def __call__(self, y, dtype, suppress_warnings, **kwargs):
        return _bc_est_lam(y, self.min_val, dtype,
                           suppress_warnings)


class _YJEstimator(object):
    def __init__(self, brack):
        self.brack = brack

    def __call__(self, y, dtype, **kwargs):
        return _yj_est_lam(y, self.brack, dtype)


class BoxCoxTransformer(_BaseSkewnessTransformer):
    r"""Apply the Box-Cox transformation to select features in a dataframe.

    Estimate a lambda parameter for each feature, and transform it to a
    distribution more-closely resembling a Gaussian bell using the Box-Cox
    transformation.

    The Box-Cox transformation cannot handle zeros or negative values in
    :math:`y`. Skoot attempts to deal with this scenario by imposing a ceiling
    function of ``min_value`` for any values that are <= 0. The transformation
    is defined as:

        :math:`y_{i} = \left\{\begin{matrix}
        \frac{y_{i}^\lambda - 1}{\lambda} & \textup{if } \lambda \neq 0, \\
        ln(y_{i}) & \textup{if } \lambda = 0
        \end{matrix}\right.`

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

    n_jobs : int, 1 by default
       The number of jobs to use for the computation. This works by
       estimating each of the feature lambdas in parallel.

       If -1 all CPUs are used. If 1 is given, no parallel computing code
       is used at all, which is useful for debugging. For n_jobs below -1,
       (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but
       one are used.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.
        Since most skutil transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.

    min_value : float, optional (default=1e-12)
        The minimum value as a ceiling function for values in prescribed
        features. Values below this amount will be set to ``min_value``.

    dtype : type, optional (default=np.float32)
        The type of float to which to cast the vector. Default is float32
        to avoid overflows.

    suppress_warnings : bool, optional (default=False)
        Whether to suppress warnings in the scipy.stats.boxcox function.
        Default is False.

    Attributes
    ----------
    lambda_ : list
       The lambda values corresponding to each feature

    fit_cols_ : list
        The list of column names on which the transformer was fit. This
        is used to validate the presence of the features in the test set
        during the ``transform`` stage.
    """

    def __init__(self, cols=None, n_jobs=1, as_df=True, min_value=1e-12,
                 dtype=np.float32, suppress_warnings=False):

        super(BoxCoxTransformer, self).__init__(
            cols=cols, as_df=as_df, n_jobs=n_jobs, dtype=dtype)

        self.min_value = min_value
        self.suppress_warnings = suppress_warnings

    @timed_instance_method(attribute_name="fit_time_")
    def fit(self, X, y=None):
        """Fit the transformer.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : array-like or None, shape=(n_samples,), optional (default=None)
            Pass-through for ``sklearn.pipeline.Pipeline``.
        """
        min_value = self.min_value
        return self._fit(X, estimation_function=_BCEstimator(min_value))

    def _transform_vector(self, vec, lam):
        # make a np array, make sure we've floored
        y = np.maximum(np.asarray(vec).astype(self.dtype), self.min_value)

        # if lam is not "zero", y gets the power treatment:
        if lam > ZERO:
            return (y ** lam - 1.) / lam

        # otherwise, it gets logged
        return np.log(y)

    def _estimator_kwargs(self):
        return dict(suppress_warnings=self.suppress_warnings)


class YeoJohnsonTransformer(_BaseSkewnessTransformer):
    r"""Apply the Yeo-Johnson transformation to a dataset.

    Estimate a lambda parameter for each feature, and transform
    it to a distribution more-closely resembling a Gaussian bell
    using the Yeo-Johnson transformation.

    The Yeo-Johnson transformation, unlike the :class:`BoxCoxTransformer`,
    allows for zero and negative values of :math:`y` and as defined as:

        :math:`y_{i} = \left\{\begin{matrix}
        ((y_{i} + 1)^\lambda - 1)/\lambda & \textup{if } \lambda
        \neq 0, y \geq  0 \\
        log(y_{i} + 1) & \textup{if } \lambda = 0, y \geq 0 \\
        -[(-y_{i} + 1)^{(2 - \lambda)} - 1]/(2 - \lambda) &
        \textup{if } \lambda \neq 2, y < 0 \\
        -log(-y_{i} + 1) &  \textup{if } \lambda = 2, y < 0 \\
        \end{matrix}\right.`

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

    n_jobs : int, 1 by default
       The number of jobs to use for the computation. This works by
       estimating each of the feature lambdas in parallel.

       If -1 all CPUs are used. If 1 is given, no parallel computing code
       is used at all, which is useful for debugging. For n_jobs below -1,
       (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but
       one are used.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.
        Since most skutil transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.

    brack : tuple, optional (default=(-2, 2))
        Either a triple (xa, xb, xc) where xa < xb < xc and func(xb) <
        func(xa), func(xc) or a pair (xa, xb) which are used as a
        starting interval for a downhill bracket search. Providing the
        pair (xa, xb) does not always mean the obtained solution will
        satisfy xa <= x <= xb.

    dtype : type, optional (default=np.float32)
        The type of float to which to cast the vector. Default is float32
        to avoid overflows.

    Attributes
    ----------
    lambda_ : list
       The lambda values corresponding to each feature

    fit_cols_ : list
        The list of column names on which the transformer was fit. This
        is used to validate the presence of the features in the test set
        during the ``transform`` stage.
    """
    def __init__(self, cols=None, n_jobs=1, as_df=True, brack=(-2, 2),
                 dtype=np.float32):

        super(YeoJohnsonTransformer, self).__init__(
            cols=cols, as_df=as_df, n_jobs=n_jobs,
            dtype=dtype)

        self.brack = brack

    @timed_instance_method(attribute_name="fit_time_")
    def fit(self, X, y=None):
        """Fit the transformer.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : array-like or None, shape=(n_samples,), optional (default=None)
            Pass-through for ``sklearn.pipeline.Pipeline``.
        """
        brack = self.brack
        return self._fit(X, estimation_function=_YJEstimator(brack))

    def _transform_vector(self, y, lam):
        return _yj_transform_y(y, lam, self.dtype)
