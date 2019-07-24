# -*- coding: utf-8 -*-

import numpy as np

__all__ = [
    'fisher_pearson_skewness',
    'kurtosis'
]


def _skew_inner(x, power):
    x = np.asarray(x)
    mean = x.mean()
    std = x.std()
    n_samples = x.shape[0]

    skew_numer = (((x - mean) ** power) / n_samples).sum()
    return skew_numer / (std ** power)


def fisher_pearson_skewness(x):
    """Compute the skewness of a variable.

    Parameters
    ----------
    x : array-like or iterable, shape=(n_samples,)
        The vector of samples.

    Returns
    -------
    skewness : float
        The Fisher-Pearson skewness of a variable.
    """
    return _skew_inner(x, power=3.)


def kurtosis(x):
    """Compute the kurtosis of a variable.

    Note that skoot uses the definition of "excess" kurtosis, meaning
    we subtract 3 from the typical computation. This is to account for the
    fact that the kurtosis of a normal distribution is 3. For more info,
    see [1].

    Parameters
    ----------
    x : array-like or iterable, shape=(n_samples,)
        The vector of samples.

    Returns
    -------
    kurt : float
        The kurtosis of a variable.

    References
    ----------
    .. [1] Measures of Skewness and Kurtosis
           https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm
    """
    return _skew_inner(x, power=4.) - 3.
