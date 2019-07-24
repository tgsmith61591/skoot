# -*- coding: utf-8 -*-
#
# Author: Taylor G Smith <taylor.smith@alkaline-ml.com>
#
# Bin your continuous features.

import six
from joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted

import numpy as np
import pandas as pd

from ..base import BasePDTransformer
from ..utils.iterables import chunk
from ..utils.dataframe import dataframe_or_array
from ..utils.validation import (check_dataframe, validate_test_set_columns,
                                type_or_iterable_to_col_mapping)
from ..utils.metaestimators import timed_instance_method

__all__ = [
    'BinningTransformer'
]


def _validate_n_bins(x, n):
    # get unique values
    unique, cts = np.unique(x, return_counts=True)
    if unique.shape[0] < n:
        raise ValueError("Fewer unique values than bins!")
    return unique, cts


def _uniform(x, n):
    # get unique and cut it at the uniform points
    unique, _ = _validate_n_bins(x, n)
    chunks = list(chunk(unique, n))

    # So now our chunks may resemble:
    # >>> list(chunk(np.arange(10), 4))
    # [array([0, 1, 2]), array([3, 4, 5]), array([6, 7]), array([8, 9])]
    # Transform them to bins
    return _Bins(chunks)


def _percentile(x, n):
    # bin by quartiles, quantiles, deciles, etc. This is really
    # easy to delegate to pandas...
    bins = pd.qcut(x, q=n, retbins=True)[1]

    # we can use the returned bins to create our own intervals
    return _Bins(list(zip(bins[:-1], bins[1:])))


_STRATEGIES = {"uniform": _uniform,
               "percentile": _percentile}


class _Bins(object):
    """Binning class that keeps track of upper and lower bounds of bins.
    The algorithm for assigning bins to a test vector is as follows:

        1. Initialize all bins as the highest bin
        2. For each lower bound in bin levels, determine which values in ``x``
           are >= to the bound. Invert the mask and decrement those bins (in
           other words, decrement the indices where the value is < the lower
           bound for the bin in question).
        3. Continue until there is no mask to invert (lowest bin).
    """
    def __init__(self, chunks):
        # chunks is a list of bin arrays
        self.n_bins = len(chunks)

        # create the repr for each bin and create the mins/maxes arrays
        upper_bounds = []
        lower_bounds = []
        reprs = []
        for i, (this_chunk, next_chunk) in \
                enumerate(zip(chunks[:-1], chunks[1:])):

            # If it's the first one, it's just less than
            # the next chunk's min.
            upper_bound = next_chunk[0]
            if i == 0:
                lower_bound = -np.inf
                rep = "(-Inf, %.2f]" % upper_bound

            # Otherwise we know it's a middle one (not the last since we
            # lagged with the zip function and handle that at the end)
            else:
                lower_bound = this_chunk[0]
                rep = "(%.2f, %.2f]" % (lower_bound, upper_bound)

            upper_bounds.append(upper_bound)
            lower_bounds.append(lower_bound)
            reprs.append(rep)

        # since we missed the last chunk due to the lag, get the last one
        lower_bounds.append(chunks[-1][0])
        upper_bounds.append(np.inf)
        reprs.append("(%.2f, Inf]" % lower_bounds[-1])

        # set the attributes
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds
        self.reprs = reprs

    def assign(self, v, as_str):
        # given some vector of values, assign the appropriate bins. We can
        # do this in one pass, really. Just pass over one of the bounds arrays
        # and keep track of the level at which the elements in V are no longer
        # within the boundaries

        # Initialize by setting all to the highest bin
        bins = (np.ones(v.shape[0]) * (self.n_bins - 1)).astype(int)

        # now progress backwards
        for boundary in self.lower_bounds[::-1]:

            # figure out which are >= to the lower boundary. They should NOT
            # be changed. The ones that are FALSE, however, should be
            # decremented by 1. On the first pass, anything that actually
            # belongs in the top bin will not be adjusted, but everything
            # else will drop by one. Next, everything that is still below the
            # lower boundary will decrement again, etc., until the lowest bin
            # where the lower_bound is -np.inf. Since everything is >= that,
            # there will be no anti mask and nothing will change
            mask = v >= boundary
            anti_mask = ~mask  # type: np.ndarray

            if anti_mask.shape[0] > 0:
                bins[anti_mask] -= 1

        # now we have bin indices, get the reprs to return...
        if as_str:
            return np.array([self.reprs[i] for i in bins])
        # otherwise user just wants the bin level
        return bins


# Executed in parallel:
def _make_bin(binner, vec, c, n):
    # Parallelize the bin operation over columns
    return c, binner(vec, n)


# Executed in parallel:
def _assign_bin(binner, vec, c, return_label):
    # Parallelize bin assignment
    return c, binner.assign(vec, return_label)


class BinningTransformer(BasePDTransformer):
    r"""Bin continuous variables.

    The BinningTransformer will create buckets for continuous variables,
    effectively transforming continuous features into categorical features.

    Pros of binning:

      * Particularly useful in the case of very skewed data where an
        algorithm may make assumptions on the underlying distribution of the
        variables
      * Quick and easy way to take curvature into account

    There are absolutely some negatives to binning:
    
      * You can tend to throw away information from continuous variables
      * You might end up fitting "wiggles" rather than a linear
        relationship itself
      * You use up a lot of degrees of freedom

    For a more exhaustive list of detrimental effects of binning, take a look
    at [1].

    Parameters
    ----------
    cols : array-like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        Optional. If None, will be applied to all features (which could
        prove to be expensive)

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.
        Since most skoot transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.

    n_bins : int or iterable, optional (default=10)
        The number of bins into which to separate each specified feature.
        Default is 20, but can also be an iterable or dict of the same length
        as ``cols``, where positional integers indicate a different bin size
        for that feature.

    strategy : str or unicode, optional (default="uniform")
        The strategy for binning. Default is "uniform", which uniformly
        segments a feature. Alternatives include "percentile" which uses
        ``n_bins`` to compute quantiles (for ``n_bins=5``), quartiles
        (for ``n_bins=4``), etc. Note that for percentile binning, the
        outer bin boundaries (low boundary of lowest bin and high
        boundary of the highest bin) will be set to -inf and inf,
        respectively, to behave similar to other binning strategies.

    return_bin_label : bool, optional (default=True)
        Whether to return the string representation of the bin (i.e., "<25.2")
        rather than the bin level, an integer.

    overwrite : bool, optional (default=True)
        Whether to overwrite the original feature with the binned feature.
        Default is True so that the output names match the input names. If
        False, the output columns will be appended to the right side of
        the frame with "_binned" appended.

    n_jobs : int, 1 by default
       The number of jobs to use for the encoding. This works by
       fitting each incremental LabelEncoder in parallel.

       If -1 all CPUs are used. If 1 is given, no parallel computing code
       is used at all, which is useful for debugging. For n_jobs below -1,
       (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but
       one are used.

    Notes
    -----
    If a feature has fewer than ``n_bins`` unique values, it will raise a
    ValueError in the fit procedure.

    Examples
    --------
    Bin two features in iris:

    >>> from skoot.datasets import load_iris_df
    >>> iris = load_iris_df(include_tgt=False, names=['a', 'b', 'c', 'd'])
    >>> binner = BinningTransformer(cols=["a", "b"], strategy="uniform")
    >>> trans = binner.fit_transform(iris)
    >>> trans.head()
                  a             b    c    d
    0  (5.10, 5.50]  (3.40, 3.60]  1.4  0.2
    1  (4.70, 5.10]  (3.00, 3.20]  1.4  0.2
    2  (4.70, 5.10]  (3.20, 3.40]  1.3  0.2
    3  (-Inf, 4.70]  (3.00, 3.20]  1.5  0.2
    4  (4.70, 5.10]  (3.60, 3.80]  1.4  0.2
    >>> trans.dtypes
    a     object
    b     object
    c    float64
    d    float64
    dtype: object

    Attributes
    ----------
    bins_ : dict
        A dictionary mapping the column names to the corresponding bins,
        which are internal _Bin objects that store data on upper and lower
        bounds.

    fit_cols_ : list
        The list of column names on which the transformer was fit. This
        is used to validate the presence of the features in the test set
        during the ``transform`` stage.

    References
    ----------
    .. [1] "Problems Caused by Categorizing Continuous Variables"
           http://biostat.mc.vanderbilt.edu/wiki/Main/CatContinuous
    """
    def __init__(self, cols=None, as_df=True, n_bins=10, strategy="uniform",
                 return_bin_label=True, overwrite=True, n_jobs=1):

        super(BinningTransformer, self).__init__(
            cols=cols, as_df=as_df)

        self.n_bins = n_bins
        self.strategy = strategy
        self.return_bin_label = return_bin_label
        self.overwrite = overwrite
        self.n_jobs = n_jobs

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
        # validate the input, and get a copy of it
        X, cols = check_dataframe(X, cols=self.cols,
                                  assert_all_finite=True)

        # validate n_bins...
        n_bins = type_or_iterable_to_col_mapping(cols=cols, param=self.n_bins,
                                                 param_name="n_bins",
                                                 permitted_scalar_types=int)

        # now that we have a dictionary, we can assess the actual integer
        for _, v in six.iteritems(n_bins):
            if not (isinstance(v, int) and v > 1):
                raise ValueError("Each n_bin value must be an integer > 1")

        # get and validate the strategy
        strategy = self.strategy
        try:
            binner = _STRATEGIES[strategy]
        except KeyError:
            raise ValueError("strategy must be one of %r, but got %r"
                             % (str(list(_STRATEGIES.keys())), strategy))

        # compute the bins for each feature
        bins = dict(Parallel(n_jobs=self.n_jobs)(
            delayed(_make_bin)(binner, vec=X[c].values, c=c, n=n)
            for c, n in six.iteritems(n_bins)))

        # set the instance attribute
        self.bins_ = bins
        self.fit_cols_ = cols
        return self

    def transform(self, X):
        """Apply the transformation to a dataframe.

        This method will bin the continuous values in the test frame with the
        bins designated in the ``fit`` stage.

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
        check_is_fitted(self, 'bins_')
        X, _ = check_dataframe(X, cols=self.cols)  # X is a copy now

        # validate that fit cols in test set
        cols = self.fit_cols_
        validate_test_set_columns(cols, X.columns)

        # the bins
        bins = self.bins_

        # now apply the binning. Rather that use iteritems, iterate the cols
        # themselves so we get the order prescribed by the user
        bin_assignments = dict(Parallel(n_jobs=self.n_jobs)(
            delayed(_assign_bin)(
                bins[col], vec=X[col].values, c=col,
                return_label=self.return_bin_label)
            for col in cols))

        # Simple pass of O(N) to assign to dataframes. Lightweight, no
        # actual computations here. That all happened in parallel
        for c in cols:
            binned = bin_assignments[c]
            # if we overwrite, it's easy
            if self.overwrite:
                X[c] = binned
            # otherwise create a new feature
            else:
                X["%s_binned" % c] = binned

        return dataframe_or_array(X, self.as_df)
