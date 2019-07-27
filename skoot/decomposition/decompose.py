# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import print_function, division, absolute_import

from sklearn.decomposition import (PCA, TruncatedSVD, KernelPCA, NMF,
                                   IncrementalPCA)

from ..base import _SelectiveTransformerWrapper

# namespace import to avoid explicitly protected imports in global namespace
from ..utils import _docstr as dsutils

__all__ = [
    'SelectiveIncrementalPCA',
    'SelectiveKernelPCA',
    'SelectiveNMF',
    'SelectivePCA',
    'SelectiveTruncatedSVD',
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
