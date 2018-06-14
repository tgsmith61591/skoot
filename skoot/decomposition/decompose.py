# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import print_function, division, absolute_import

import numpy as np
from numpy.linalg import matrix_rank

from sklearn.utils.validation import check_array
from sklearn.decomposition import (PCA, TruncatedSVD, KernelPCA, NMF,
                                   IncrementalPCA)

from ..base import _SelectiveTransformerWrapper

# namespace import to avoid explicitly protected imports in global namespace
from ..utils import _docstr as dsutils

# local submodule funcs that use Fortran subroutines
from ._dqrutl import (qr_decomposition, _call_dqrcf,
                      _validate_matrix_size, _qr_R)

__all__ = [
    'SelectiveIncrementalPCA',
    'SelectiveKernelPCA',
    'SelectiveNMF',
    'SelectivePCA',
    'SelectiveTruncatedSVD',
    'QRDecomposition'
]


# Selective decomposition classes
@dsutils.wraps_estimator(IncrementalPCA,
                         add_sections=[(
                                 'See also', ['SelectiveKernelPCA',
                                              'SelectiveNMF',
                                              'SelectivePCA',
                                              'SelectiveTruncatedSVD'], True)])
class SelectiveIncrementalPCA(_SelectiveTransformerWrapper):
    pass


@dsutils.wraps_estimator(KernelPCA,
                         add_sections=[(
                                 'See also', ['SelectiveIncrementalPCA',
                                              'SelectiveNMF',
                                              'SelectivePCA',
                                              'SelectiveTruncatedSVD'], True)])
class SelectiveKernelPCA(_SelectiveTransformerWrapper):
    pass


@dsutils.wraps_estimator(NMF,
                         add_sections=[(
                                 'See also', ['SelectiveIncrementalPCA',
                                              'SelectiveKernalPCA',
                                              'SelectivePCA',
                                              'SelectiveTruncatedSVD'], True)])
class SelectiveNMF(_SelectiveTransformerWrapper):
    pass


@dsutils.wraps_estimator(PCA,
                         add_sections=[(
                                 'See also', ['SelectiveIncrementalPCA',
                                              'SelectiveKernalPCA',
                                              'SelectiveNMF',
                                              'SelectiveTruncatedSVD'], True)])
class SelectivePCA(_SelectiveTransformerWrapper):
    pass


@dsutils.wraps_estimator(TruncatedSVD,
                         add_sections=[(
                                 'See also', ['SelectiveIncrementalPCA',
                                              'SelectiveKernalPCA',
                                              'SelectiveNMF',
                                              'SelectivePCA'], True)])
class SelectiveTruncatedSVD(_SelectiveTransformerWrapper):
    pass


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
    >>> qr.qr[:5]
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
