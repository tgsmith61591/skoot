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
    The following example applies the QRDecomposition to the diabetes dataset:

    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=20,
    ...                            n_informative=12, random_state=1)
    >>> qr = QRDecomposition(X)
    >>> qr.qr[:3]
    array([[ 6.62984626e+01, -2.09133094e+00, -1.09276829e+00,
             5.58014214e+00, -8.32668709e+00, -1.97547759e+01,
            -6.85313166e+00, -6.42853241e+00,  5.94403138e+00,
            -2.15967470e+01, -2.35917991e-01, -5.20261414e+00,
             2.57589906e+00,  1.16805385e+01, -3.04018942e-01,
             7.90088801e-01,  3.02117704e-01, -1.09919010e+01,
             4.41783544e-01,  4.46781544e+00],
           [ 6.18367459e-02, -7.72568215e+01, -1.55131934e+00,
            -1.16888104e+01, -5.94188765e+00, -4.20310720e+01,
            -6.79982237e+00, -1.16643515e+00,  1.23441742e+01,
             5.68140358e+01,  1.48759893e+00, -3.07980793e+00,
            -1.30638396e-01, -1.40662087e+00,  4.72221164e-03,
            -2.67913340e-01,  1.08518423e+00,  6.48536112e+00,
            -3.61589065e+00, -8.54657339e+00],
           [-1.46203358e-03, -2.82718061e-03, -3.13247668e+01,
            -4.42956256e-03, -2.27949848e+00, -2.37512023e+00,
            -1.50550170e+00,  2.39909438e+00, -5.01917157e+00,
            -5.84909738e+00,  5.47610545e-01, -9.82967076e-01,
             8.36013852e-01, -3.06521652e+00, -6.12860254e-01,
            -3.57806556e-01, -1.64002608e-01,  9.76526585e-01,
             5.15293669e-01,  1.78207627e+00]])

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
        coef, _ = (np.zeros((k, ny), dtype=np.double, order='F'),  # noqa: F841
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
