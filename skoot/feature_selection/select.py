# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

import numpy as np
import pandas as pd

from .base import BaseFeatureSelector
from ..utils.validation import check_dataframe, validate_multiple_cols
from ..utils.metaestimators import timed_instance_method

__all__ = [
    'FeatureFilter',
    'MultiCorrFilter',
    'NearZeroVarianceFilter',
    'SparseFeatureFilter'
]


# TODO: add functionality for a dummy column indicating present value?
class SparseFeatureFilter(BaseFeatureSelector):
    """Drop overly sparse features.

    Retains features that are less sparse (NaN) than the provided
    threshold. Useful in situations where matrices are too sparse to
    impute reliably.

    Parameters
    ----------
    cols : array-like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the transformer will be ``fit``
        on the entire frame. Note that the transformation will also only
        apply to the specified columns, and any other non-specified
        columns will still be present after transformation.

    threshold : float, optional (default=0.5)
        The threshold of sparsity above which features will be
        deemed "too sparse" and will be dropped.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.
        Since most skutil transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>>
    >>> nan = np.nan
    >>> X = np.array([
    ...     [1.0, 2.0, nan],
    ...     [2.0, 3.0, nan],
    ...     [3.0, nan, 1.0],
    ...     [4.0, 5.0, nan]
    ... ])
    >>>
    >>> X = pd.DataFrame.from_records(data=X, columns=['a','b','c'])
    >>> dropper = SparseFeatureFilter(threshold=0.5)
    >>> X_transform = dropper.fit_transform(X)
    >>> assert X_transform.shape[1] == 2 # drop out last column

    Attributes
    ----------
    sparsity_ : array-like, shape=(n_features,)
        The array of sparsity values
    
    drop_ : array-like, shape=(n_features,)
        Assigned after calling ``fit``. These are the features that
        are designated as "bad" and will be dropped in the ``transform``
        method.

    Notes
    -----
    Sometimes the presence of a value in an overly sparse column can be highly
    informative. If you're using the sparse filter, consider creating a new
    (dummy) feature indicating whether there was a value present.
    """
    def __init__(self, cols=None, threshold=0.5, as_df=True):

        super(SparseFeatureFilter, self).__init__(
            cols=cols, as_df=as_df)

        self.threshold = threshold

    @timed_instance_method(attribute_name="fit_time_")
    def fit(self, X, y=None):
        """Fit the transformer.

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
        X, cols = check_dataframe(X, cols=self.cols)
        thresh = self.threshold

        # validate the threshold
        if not (isinstance(thresh, float) and (0.0 <= thresh < 1.0)):
            raise ValueError('thresh must be a float between '
                             '0 (inclusive) and 1. Got %s' % str(thresh))

        # assess sparsity
        subset = X[cols]
        self.sparsity_ = subset.apply(
            lambda x: x.isnull().sum() / x.shape[0]).values  # type: np.ndarray

        mask = self.sparsity_ > thresh  # numpy boolean array
        self.drop_ = subset.columns[mask].tolist()
        return self


class FeatureFilter(BaseFeatureSelector):
    """A simple feature-dropping transformer class.

    A very simple class to be used at the beginning or any stage of a
    Pipeline that will drop the given features from the remainder of the pipe.
    This is useful if a transformer or encoder creates variables that you're
    disinterested in and would like to exclude from your modeling process.

    Parameters
    ----------
    cols : array-like, shape=(n_features,), optional (default=None)
        The features to drop. Note that ``FeatureFilter`` behaves slightly
        differently from all other ``BaseFeatureSelector`` classes in the sense
        that it will drop all of the features prescribed in this parameter.
        However, if ``cols`` is None, it will not drop any (which is counter
        to other classes, which will operate on all columns in the absence
        of an explicit ``cols`` parameter).

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.
        Since most skoot transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.

    Examples
    --------
    An example using the FeatureFilter:

    >>> import numpy as np
    >>> import pandas as pd
    >>>
    >>> X = pd.DataFrame.from_records(data=np.random.rand(3,3),
    ...                               columns=['a','b','c'])
    >>> dropper = FeatureFilter(cols=['a','b'])
    >>> X_transform = dropper.fit_transform(X)
    >>> assert X_transform.shape[1] == 1 # drop out first two columns

    Attributes
    ----------
    drop_ : array-like, shape=(n_features,)
        Assigned after calling ``fit``. These are the features that
        are designated as "bad" and will be dropped in the ``transform``
        method.
    """
    def __init__(self, cols=None, as_df=True):
        # just a pass-through for super constructor
        super(FeatureFilter, self).__init__(
            cols=cols, as_df=as_df)

    def fit(self, X, y=None):
        # check on state of X and cols
        X, cols = check_dataframe(X, cols=self.cols)

        # if the provided self.cols was None, we drop nothing. otherwise
        # we drop the specified columns
        self.drop_ = [] if not self.cols else cols
        return self


class MultiCorrFilter(BaseFeatureSelector):
    """Remove highly correlated features.

    Multi-collinear data (features which are not independent from one another)
    can pose problems in coefficient stability for parametric models, or
    feature importance scores for non-parametric models.

    This class filters out features with a correlation greater than the
    provided threshold. When a pair of correlated features is identified, the
    mean absolute correlation (MAC) of each feature is considered, and the
    feature with the highest MAC is discarded.

    Parameters
    ----------
    cols : array-like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the transformer will be ``fit``
        on the entire frame. Note that the transformation will also only
        apply to the specified columns, and any other non-specified
        columns will still be present after transformation.

    threshold : float, optional (default=0.85)
        The threshold above which to filter correlated features

    method : str, optional (default='pearson')
        The method used to compute the correlation,
        one of ('pearson', 'kendall', 'spearman').

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.
        Since most skoot transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.

    Examples
    --------
    The following demonstrates a simple multi-correlation filter
    applied to the iris dataset.

    >>> from skoot.datasets import load_iris_df
    >>>
    >>> X = load_iris_df(include_tgt=False)
    >>> mcf = MultiCorrFilter(threshold=0.85)
    >>> mcf.fit_transform(X).head()
       sepal length (cm)  sepal width (cm)  petal width (cm)
    0                5.1               3.5               0.2
    1                4.9               3.0               0.2
    2                4.7               3.2               0.2
    3                4.6               3.1               0.2
    4                5.0               3.6               0.2

    Attributes
    ----------
    drop_ : array-like, shape=(n_features,)
        Assigned after calling ``fit``. These are the features that
        are designated as "bad" and will be dropped in the ``transform``
        method.

    mean_abs_correlations_ : list, float
        The corresponding mean absolute correlations of each ``drop_`` name
    """

    def __init__(self, cols=None, threshold=0.85,
                 method='pearson', as_df=True):

        super(MultiCorrFilter, self).__init__(
            cols=cols, as_df=as_df)

        self.threshold = threshold
        self.method = method

    def fit(self, X, y=None):
        """Fit the multi-collinearity filter.

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
        # check on state of X and cols. Also need all columns to be finite!
        X, cols = check_dataframe(X, cols=self.cols, assert_all_finite=True)

        # we need to make sure there's more than 1 column!
        validate_multiple_cols(self.__class__.__name__, cols)

        # Generate correlation matrix
        c = X[cols].corr(method=self.method)

        # get drops list
        # TODO: write a _find_correlations_exact for smaller matrices
        self.drop_, self.mean_abs_correlations_ = \
            self._find_correlations_fast(c, self.threshold)

        return self

    @staticmethod
    def _find_correlations_fast(c, threshold):
        """Filter highly correlated features.

        This function identifies the correlations between features that
        are greater than the provided threshold, and identifies which
        to drop on the basis of mean absolute correlation. This function is
        based on Caret's ``findCorrelation_fast`` method [1], which is very
        efficient for large matrices.

        Parameters
        ----------
        c : pd.DataFrame
            The pre-computed correlation matrix.

        threshold : float
            The threshold above which to filter features which
            are multi-collinear in nature.

        Returns
        -------
        drop_names : list, shape=(n_features,)
            The feature names that should be dropped

        average_corr : array-like, shape=(n_features,)
            The mean absolute correlations between
            the features.

        References
        ----------
        .. [1] Caret findCorrelations.R (findCorrelation_fast)
               https://bit.ly/2E1AMcJ
        """
        # get the average absolute column correlations
        c_abs = c.abs()  # type: pd.DataFrame
        average_corr = c_abs.mean().values  # type: np.ndarray

        # get the sort order
        average_corr_order = np.argsort(average_corr)

        # set the lower tri to NaN so we don't consider anymore
        # first, get the lower tri indices and then zip them
        lower_tri = np.tril_indices(n=c.shape[0], k=0)
        c.values[lower_tri] = np.nan

        # get those above cutoff (re-compute abs on amended array)
        c_abs = c.abs()  # type: pd.DataFrame
        combs_above_thresh = np.where(c_abs > threshold)
        rows_to_check, cols_to_check = combs_above_thresh

        cols_to_discard = (average_corr_order[cols_to_check] >
                           average_corr_order[rows_to_check])
        rows_to_discard = ~cols_to_discard

        # append each set of discard rows/cols, get the distinct
        drop_cols = np.unique(
            np.concatenate([cols_to_check[cols_to_discard],
                            rows_to_check[rows_to_discard]]))

        # the names to drop
        drop_names = c.columns[drop_cols].tolist()
        return drop_names, average_corr


class NearZeroVarianceFilter(BaseFeatureSelector):
    r"""Identify near zero variance predictors.

    Diagnose and remove any features that have one unique value
    (i.e., are zero variance predictors) or that are have both of the
    following characteristics: they have very few unique values relative
    to the number of samples and the ratio of the frequency of the most
    common value to the frequency of the second most common value is large.

    A note of caution: if you attempt to run this over large, continuous data,
    it might take a long time. Since for each column in ``cols`` it will
    compute ``value_counts``, applying to continuous data could be a bad idea.

    Parameters
    ----------
    cols : array-like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the transformer will be ``fit``
        on the entire frame. Note that the transformation will also only
        apply to the specified columns, and any other non-specified
        columns will still be present after transformation.

    freq_cut : float, optional (default=95/5)
        The cutoff for the ratio of the most common value to the second most
        common value. That is, if the frequency of the most common value is
        >= ``freq_cut`` times the frequency of the second most, the feature
        will be dropped.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.
        Since most skutil transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.

    Examples
    --------
    An example of the near zero variance filter on a completely
    constant column:

    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> X = pd.DataFrame.from_records(
    ...     data=np.array([[1,2,3], [4,5,3], [6,7,3], [8,9,3]]),
    ...     columns=['a','b','c'])
    >>> flt = NearZeroVarianceFilter(freq_cut=25)
    >>> flt.fit_transform(X)
       a  b
    0  1  2
    1  4  5
    2  6  7
    3  8  9

    An example on a column with two unique values represented at 2:1. Also
    shows how we can extract the fitted ratios and drop names:

    >>> X = pd.DataFrame.from_records(
    ...     data=np.array([[1,2,3], [4,5,3], [6,7,5]]),
    ...     columns=['a','b','c'])
    >>> nzv = NearZeroVarianceFilter(freq_cut=2.)
    >>> nzv.fit_transform(X)
       a  b
    0  1  2
    1  4  5
    2  6  7
    >>> nzv.ratios_
    array([ 1.,  1.,  2.])
    >>> nzv.drop_
    ['c']

    Attributes
    ----------
    drop_ : array-like, shape=(n_features,)
        Assigned after calling ``fit``. These are the features that
        are designated as "bad" and will be dropped in the ``transform``
        method.

    ratios_ : array-like, shape=(n_features,)
        The ratios of the counts of the most populous classes to the second
        most populated classes for each column in ``cols``.

    References
    ----------
    .. [1] Kuhn, M. & Johnson, K. "Applied Predictive
           Modeling" (2013). New York, NY: Springer.

    .. [2] Caret (R package) nearZeroVariance R code
           https://bit.ly/2J0ozbM
    """
    def __init__(self, cols=None, freq_cut=95. / 5., as_df=True):

        super(NearZeroVarianceFilter, self).__init__(
            cols=cols, as_df=as_df)

        self.freq_cut = freq_cut

    def fit(self, X, y=None):
        """Fit the near-zero variance filter.

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

        # get the freq cut and validate it is an appropriate value...
        freq_cut = self.freq_cut
        if not (isinstance(freq_cut, (int, float)) and 1. < freq_cut):
            raise ValueError("freq_cut must be a float > 1.0")

        # make sure it's cast to a float if not already
        freq_cut = float(freq_cut)

        # get a mask of which should be dropped
        subset = X[cols]
        ratios = subset.apply(self._filter_freq_cut).values
        self.drop_ = subset.columns[ratios >= freq_cut].tolist()
        self.ratios_ = ratios

        return self

    @staticmethod
    def _filter_freq_cut(series):
        """Filter above a frequency cut.

        Parameters
        ----------
        series : pd.Series
            One series in the specified column list.

        Returns
        -------
        ratio : float
            The ratio of the count of the most populated class to the second
            most populated class. If there is only one class, will return
            infinity.
        """
        vc = series.value_counts()
        n_levels = vc.shape[0]

        # base case 1: vc len is 1 (single value, no variance at all)
        if n_levels == 1:
            return np.inf

        # get the first two levels and counts
        first_two = vc.values[:2].astype(float)
        return first_two[0] / first_two[1]
