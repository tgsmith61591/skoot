# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Scaling predictors

from sklearn.preprocessing import StandardScaler, RobustScaler, \
    MaxAbsScaler, MinMaxScaler

from ..base import _SelectiveTransformerWrapper

# namespace import to avoid explicitly protected imports in global namespace
from ..utils import _docstr as dsutils

__all__ = [
    'SelectiveStandardScaler',
    'SelectiveMaxAbsScaler',
    'SelectiveMinMaxScaler',
    'SelectiveRobustScaler'
]


# Selective decomposition classes
@dsutils.wraps_estimator(StandardScaler, remove_sections=['Notes'])
class SelectiveStandardScaler(_SelectiveTransformerWrapper):
    pass


@dsutils.wraps_estimator(RobustScaler, remove_sections=['Notes'])
class SelectiveRobustScaler(_SelectiveTransformerWrapper):
    pass


@dsutils.wraps_estimator(MaxAbsScaler, remove_sections=['Notes'])
class SelectiveMaxAbsScaler(_SelectiveTransformerWrapper):
    pass


@dsutils.wraps_estimator(MinMaxScaler, remove_sections=['Notes'])
class SelectiveMinMaxScaler(_SelectiveTransformerWrapper):
    pass
