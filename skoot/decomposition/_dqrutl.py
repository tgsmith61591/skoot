# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import print_function, division, absolute_import

import numpy as np

from sklearn.utils import check_array
from numpy.linalg import matrix_rank

# Fortran module import: make this absolute
from skoot.decomposition import _dqrsl as dqrsl

# WARNING: there is little-to-no validation of input in these functions,
# and crashes may be caused by inappropriate usage. Use with care...

__all__ = [
    'qr_decomposition'
]


def _validate_matrix_size(n, p):
    if n * p > 2147483647:
        raise ValueError('too many elements for Fortran LINPACK routine')


def _safecall(fun, *args, **kwargs):
    """A method to call a LAPACK or LINPACK subroutine internally. This
    is a bit of a misnomer... it's not really that safe...
    """
    fun(*args, **kwargs)


def qr_decomposition(X, job=1):
    """Perform the QR decomposition on a matrix.

    Performs the QR decomposition using LINPACK, BLAS and LAPACK
    Fortran subroutines.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The matrix to decompose

    job : int, optional (default=1)
        Whether to perform pivoting. 0 is False, any other value
        will be coerced to 1 (True).

    Returns
    -------
    X : np.ndarray, shape=(n_samples, n_features)
        The matrix

    rank : int
        The rank of the matrix

    qraux : np.ndarray, shape=(n_features,)
        Contains further information required to recover
        the orthogonal part of the decomposition.

    pivot : np.ndarray, shape=(n_features,)
        The pivot array, or None if not ``job``
    """

    X = check_array(X, dtype='numeric', order='F',
                    copy=True)  # type: np.ndarray
    n, p = X.shape

    # check on size
    _validate_matrix_size(n, p)
    rank = matrix_rank(X)

    # validate job:
    job_ = 0 if not job else 1

    qraux, pivot, work = (np.zeros(p, dtype=np.double, order='F'),
                          # can't use arange, because need fortran
                          # order ('order' not kw in arange)
                          np.array([i for i in range(1, p + 1)],
                                   dtype=np.int, order='F'),
                          np.zeros(p, dtype=np.double, order='F'))

    # sanity checks
    assert qraux.shape[0] == p, 'expected qraux to be of length %i' % p
    assert pivot.shape[0] == p, 'expected pivot to be of length %i' % p
    assert work.shape[0] == p, 'expected work to be of length %i' % p

    # call the fortran module IN PLACE
    _safecall(dqrsl.dqrdc, X, n, n, p, qraux, pivot, work, job_)

    # do returns
    return (X,
            rank,
            qraux,

            # subtract one because pivot started at 1 for the fortran
            (pivot - 1) if job_ else None)


def _qr_R(qr):
    """Extract the R matrix from a QR decomposition"""
    min_dim = min(qr.shape)
    return qr[:min_dim + 1, :]


def _call_dqrcf(qr, n, k, qraux, X, ny, coef):
    """Call the dqrcf Fortran subroutine"""
    _safecall(dqrsl.dqrcf, qr, n, k, qraux, X, ny, coef, 0)
