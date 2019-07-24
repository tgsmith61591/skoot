# -*- coding: utf-8 -*-
#
# Compatibility utilities

# TODO: deprecate this since we always use python 3+ now
# Python <3
try:
    from types import NoneType
    xrange = xrange

# Python 3+
except (NameError, ImportError):
    NoneType = type(None)
    xrange = range
