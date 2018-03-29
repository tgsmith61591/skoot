# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import print_function, division, absolute_import

from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.linalg import matrix_rank
import pandas as pd

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.externals import six

from ..base import BasePDTransformer
from ..decorators import overrides
from ..utils.validation import check_dataframe

# local submodule funcs that use Fortran subroutines
from ._dqrutl import (qr_decomposition, _call_dqrcf,
                      _validate_matrix_size, _qr_R)

__all__ = [
    'SelectivePCA',
    'SelectiveTruncatedSVD',
    'QRDecomposition'
]


class _BaseSelectiveDecomposer(six.with_metaclass(ABCMeta, BasePDTransformer)):
    def __init__(self, component_prefix, cols=None,
                 n_components=None, as_df=True):

        super(_BaseSelectiveDecomposer, self).__init__(
            cols=cols, as_df=as_df)

        self.component_prefix = component_prefix
        self.n_components = n_components

    def get_decomposition(self):
        """Get the decomposition from a fitted instance.

        Retrieves the fitted decomposition class from a fitted instance.
        If the transformer has not yet been fit, returns None.
        """
        attrname = self._decomposition_name()  # type: str or unicode
        return getattr(self, attrname) if hasattr(self, attrname) else None

    @abstractmethod
    def _decomposition_name(self):
        """To be overridden by subclasses.

        Get the name of the fitted decomposition attribute. This is an
        internal method used as a getter to lookup the fitted decomposition
        if it exists.

        Returns
        -------
        attrname : str or unicode
            The name of the fitted decomposition attribute.
        """


class SelectivePCA(_BaseSelectiveDecomposer):
    """Apply PCA only to a select group of columns.

    This wraps the ``sklearn.decomposition.PCA`` and is useful for data that
    may contain a mix of columns that we do and don't want to decompose.

    Parameters
    ----------
    cols : array_like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the transformer will be ``fit``
        on the entire frame. Note that the transformation will also only
        apply to the specified columns, and any other non-specified
        columns will still be present after transformation.

    n_components : int, float, None or string, optional (default=None)
        The number of components to keep, per sklearn:

        * if n_components is not set, all components are kept:

            n_components == min(n_samples, n_features)

        * if n_components == 'mle' and svd_solver == 'full', Minka's MLE
          is used to guess the dimension.

        * if ``0 < n_components < 1`` and svd_solver == 'full', select the
          number of components such that the amount of variance that needs
          to be explained is greater than the percentage specified by
          ``n_components``

        * ``n_components`` cannot be equal to ``n_features`` for
          ``svd_solver`` == 'arpack'.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead. 
        Since most skoot transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by
        default.

    whiten : bool, optional (default False)
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.
        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    component_prefix : str or unicode, optional (default='PC')
        The prefix for transformed components. This string is appended before
        the component number when creating columns for the output dataframe.

    do_weight : bool, optional (default False)
        When True (False by default) the `explained_variance_` vector is used
        to weight the features post-transformation. This is especially useful
        in clustering contexts, where features are all implicitly assigned the
        same importance, even though PCA by nature orders the features by
        importance (i.e., not all components are created equally). When True,
        weighting will subtract the median variance from the weighting vector,
        and add one (so as not to down sample or upsample everything), then
        multiply the weights across the transformed features.

    Examples
    --------
    An example decomposition:

    >>> from skoot.decomposition import SelectivePCA
    >>> from skoot.datasets import load_iris_df
    >>>
    >>> X = load_iris_df(include_tgt=False)
    >>> pca = SelectivePCA(n_components=2)
    >>> X_transform = pca.fit_transform(X)
    >>> assert X_transform.shape[1] == 2

    Attributes
    ----------
    pca_ : PCA
        The fitted ``sklearn.decomposition.PCA`` instance.
    """
    def __init__(self, cols=None, n_components=None, whiten=False,
                 component_prefix='PC', do_weight=False, as_df=True):

        super(SelectivePCA, self).__init__(
            component_prefix=component_prefix,
            cols=cols, n_components=n_components,
            as_df=as_df)

        self.whiten = whiten
        self.do_weight = do_weight

    def fit(self, X, y=None):
        """Fit the transformer.

        This method will fit a ``sklearn.decomposition.PCA`` instance on the
        provided dataframe, and for the ``cols`` passed in the constructor.

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

        # fails thru if names don't exist:
        self.pca_ = PCA(n_components=self.n_components,
                        whiten=self.whiten)\
            .fit(X[cols])

        return self

    def transform(self, X):
        """Transform a test dataframe.

        This method will apply the PCA transformation to the columns
        provided in the ``cols`` parameter.

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
        check_is_fitted(self, 'pca_')

        # check on state of X and cols
        X, cols, other_nms = check_dataframe(X, cols=self.cols,
                                             column_diff=True)

        # get the transformation
        pca = self.pca_  # type: PCA
        transform = pca.transform(X[cols])

        # do weighting if necessary
        if self.do_weight:
            # get the weight vals
            weights = pca.explained_variance_ratio_
            weights = 1 + (weights - np.median(weights))

            # now add to the transformed features
            transform *= weights

        # stack the transformed variables onto the RIGHT side
        right = pd.DataFrame.from_records(
            data=transform,
            columns=[('%s%i' % (self.component_prefix, i + 1))
                     for i in range(transform.shape[1])])

        # concat if needed
        x = pd.concat([X[other_nms], right], axis=1) if other_nms else right
        return x if self.as_df else x.values

    @overrides(_BaseSelectiveDecomposer)
    def _decomposition_name(self):
        return "pca_"

    def score(self, X, y=None):
        """Return the average log-likelihood of all samples.

        This calls ``sklearn.decomposition.PCA``'s score method
        on the specified columns [1]. Note that if the transformer has not
        yet been fitted, this will fail.

        Parameters
        ----------
        X: pd.DataFrame, shape=(n_samples, n_features)
            The dataframe to score.

        y : array-like or None, shape=(n_samples,), optional (default=None)
            Pass-through for pipeline

        Returns
        -------
        ll: float
            Average log-likelihood of the samples under the fit
            PCA model (``self.pca_``)

        References
        ----------
        .. [1] Bishop, C.  "Pattern Recognition and Machine Learning"
               12.2.1 p. 574 http://www.miketipping.com/papers/met-mppca.pdf
        """
        check_is_fitted(self, 'pca_')

        X, cols = check_dataframe(X, self.cols)
        pca = self.pca_  # type: PCA
        return pca.score(X[cols], y)


class SelectiveTruncatedSVD(_BaseSelectiveDecomposer):
    """Apply TruncatedSVD only to a select group of columns.

    This wraps the ``sklearn.decomposition.TruncatedSVD`` and is useful
    for data that may contain a mix of columns that we do and don't want
    to decompose. TruncatedSVD is the equivalent of Latent Semantic Analysis;
    it returns the "concept space" of the decomposed features.

    Parameters
    ----------
    cols : array_like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the transformer will be ``fit``
        on the entire frame. Note that the transformation will also only
        apply to the specified columns, and any other non-specified
        columns will still be present after transformation.

    n_components : int, (default=2)
        Desired dimensionality of output data.
        Must be strictly less than the number of features.
        The default value is useful for visualisation. For LSA, a value of
        100 is recommended.

    algorithm : string, (default="randomized")
        SVD solver to use. Either "arpack" for the ARPACK wrapper in SciPy
        (scipy.sparse.linalg.svds), or "randomized" for the randomized
        algorithm due to Halko (2009).

    n_iter : int, optional (default=5)
        Number of iterations for randomized SVD solver. Not used by ARPACK.
        The default is larger than the default in `randomized_svd` to handle
        sparse matrices that may have large slowly decaying spectrum.

    component_prefix : str or unicode, optional (default='Concept')
        The prefix for transformed components. This string is appended before
        the component number when creating columns for the output dataframe.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead. 
        Since most skoot transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True
        by default.

    Examples
    --------
    An example decomposition:

    >>> from skoot.decomposition import SelectiveTruncatedSVD
    >>> from skoot.datasets import load_iris_df
    >>>
    >>> X = load_iris_df(include_tgt=False)
    >>> svd = SelectiveTruncatedSVD(n_components=2)
    >>> X_transform = svd.fit_transform(X)
    >>> assert X_transform.shape[1] == 2

    Attributes
    ----------
    svd_ : TruncatedSVD
        The fitted ``sklearn.decomposition.TruncatedSVD`` instance.
    """

    def __init__(self, cols=None, n_components=2, algorithm='randomized',
                 n_iter=5, component_prefix='Concept', as_df=True):

        super(SelectiveTruncatedSVD, self).__init__(
            cols=cols, n_components=n_components,
            component_prefix=component_prefix,
            as_df=as_df)

        self.algorithm = algorithm
        self.n_iter = n_iter

    def fit(self, X, y=None):
        """Fit the transformer.

        This method will fit a ``sklearn.decomposition.TruncatedSVD``
        instance on the provided dataframe, and for the ``cols`` passed
        in the constructor.

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
        X, cols = check_dataframe(X, cols=self.cols)

        # fails thru if names don't exist:
        self.svd_ = TruncatedSVD(n_components=self.n_components,
                                 algorithm=self.algorithm,
                                 n_iter=self.n_iter)\
            .fit(X[cols])

        return self

    def transform(self, X):
        """Transform a test dataframe.

        This method will apply the TruncatedSVD transformation to the columns
        provided in the ``cols`` parameter.

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
        check_is_fitted(self, 'svd_')

        # check on state of X and cols
        X, cols, other_nms = check_dataframe(X, cols=self.cols,
                                             column_diff=True)

        svd = self.svd_  # type: TruncatedSVD
        transform = svd.transform(X[cols])

        # make into a dataframe we'll append to the RIGHT side
        right = pd.DataFrame.from_records(
            data=transform,
            columns=[('%s%i' % (self.component_prefix, i + 1))
                     for i in range(transform.shape[1])])

        # concat if needed
        x = pd.concat([X[other_nms], right], axis=1) if other_nms else right
        return x if self.as_df else x.values

    @overrides(_BaseSelectiveDecomposer)
    def _decomposition_name(self):
        return "svd_"


class QRDecomposition(object):
    """Perform the QR decomposition on a matrix.

    Performs the QR decomposition using LINPACK, BLAS and LAPACK
    Fortran subroutines, and provides an interface for other useful
    QR utility methods.

    Unlike most other classes in skoot, the QRDecomposition does not
    conform to the sklearn interface, and is fit immediately upon
    instantiation.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The matrix to decompose. Unlike many other classes in skoot,
        this one does not require a Pandas frame, and can be applied
        directly to numpy arrays.

    pivot : bool, optional (default=True)
        Whether to perform pivoting. Default is True.

    Examples
    --------
    The following example applies the QRDecomposition to the Iris dataset:

    >>> from skoot.datasets import load_iris_df
    >>> iris = load_iris_df(include_tgt=False)
    >>> qr = QRDecomposition(iris)
    >>> iris.qr[:5]
    array([[ -7.22762063e+01,  -3.69551770e+01,  -4.82074278e+01,
             -1.56019534e+01],
           [  6.77954786e-02,  -7.83357455e+00,   1.37362617e+01,
              5.74998657e+00],
           [  6.50283162e-02,   9.48052595e-02,   8.38802821e+00,
              4.51317500e+00],
           [  6.36447350e-02,   8.87140100e-02,   1.77374548e-02,
              2.33408594e+00],
           [  6.91790598e-02,   1.25844572e-01,  -4.65297710e-03,
              2.63036986e-02]])

    Attributes
    ----------
    qr : array-like, shape (n_samples, n_features)
        The decomposed matrix

    qraux : array-like, shape (n_features,)
        Contains further information required to recover
        the orthogonal part of the decomposition.

    pivot : array-like, shape (n_features,)
        The pivots, if pivot was set to 1, else None

    rank : int
        The rank of the input matrix
    """
    def __init__(self, X, pivot=True):
        self.job_ = 0 if not pivot else 1
        self._decompose(X)

    def _decompose(self, X):
        """Decomposes the matrix"""
        # perform the decomposition
        self.qr, self.rank, self.qraux, self.pivot = \
            qr_decomposition(X, self.job_)

    def get_coef(self, X):
        qr, qraux = self.qr, self.qraux
        n, p = qr.shape

        # sanity check
        assert isinstance(qr, np.ndarray), \
            'internal error: QR should be a np.ndarray but got %s' % type(qr)
        assert isinstance(qraux, np.ndarray), \
            'internal error: qraux should be a np.ndarray but got %s' \
            % type(qraux)

        # validate input array
        X = check_array(X, dtype='numeric', copy=True,
                        order='F')  # type: np.ndarray
        nx, ny = X.shape

        if nx != n:
            raise ValueError('qr and X must have same number of rows')

        # check on size
        _validate_matrix_size(n, p)

        # get the rank of the decomposition
        k = self.rank

        # get ix vector
        # if p > n:
        #   ix = np.ones(n + (p - n)) * np.nan
        #   ix[:n] = np.arange(n) # i.e., array([0,1,2,nan,nan,nan])
        # else:
        #   ix = np.arange(n)

        # set up the structures to alter
        coef, info = (np.zeros((k, ny), dtype=np.double, order='F'),
                      np.zeros(1, dtype=np.int, order='F'))

        # call the fortran module IN PLACE
        _call_dqrcf(qr, n, k, qraux, X, ny, coef)

        # post-processing
        # if k < p:
        #   cf = np.ones((p,ny)) * np.nan
        #   cf[self.pivot[np.arange(k)], :] = coef
        return coef if not k < p else coef[self.pivot[np.arange(k)], :]

    def get_rank(self):
        """Get the rank of the decomposition.

        Returns
        -------
        self.rank : int
            The rank of the decomposition
        """
        return self.rank

    def get_R(self):
        """Get the R matrix from the decomposition.

        Returns
        -------
        r : np.ndarray
            The R portion of the decomposed matrix.
        """
        r = _qr_R(self.qr)
        return r

    def get_R_rank(self):
        """Get the rank of the R matrix.

        Returns
        -------
        rank : int
            The rank of the R matrix
        """
        return matrix_rank(self.get_R())
