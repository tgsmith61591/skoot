# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Scaling predictors

from __future__ import absolute_import

from sklearn.preprocessing import (StandardScaler, RobustScaler, MaxAbsScaler,
                                   MinMaxScaler)

from ..base import _selective_copy_doc_for, _SelectiveTransformerWrapper

__all__ = [
    'SelectiveStandardScaler',
    'SelectiveMaxAbsScaler',
    'SelectiveMinMaxScaler',
    'SelectiveRobustScaler'
]


# Selective decomposition classes
@_selective_copy_doc_for(StandardScaler)
class SelectiveStandardScaler(_SelectiveTransformerWrapper):
    _cls = StandardScaler

@_selective_copy_doc_for(RobustScaler)
class SelectiveRobustScaler(_SelectiveTransformerWrapper):
    _cls = RobustScaler

@_selective_copy_doc_for(MaxAbsScaler)
class SelectiveMaxAbsScaler(_SelectiveTransformerWrapper):
    _cls = MaxAbsScaler

@_selective_copy_doc_for(MinMaxScaler)
class SelectiveMinMaxScaler(_SelectiveTransformerWrapper):
    _cls = MinMaxScaler
