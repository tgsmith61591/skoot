# -*- coding: utf-8 -*-
#
# Compatibility utilities

# Python <3
try:
    from types import NoneType
    xrange = xrange

# Python 3+
except (NameError, ImportError):
    NoneType = type(None)
    xrange = range
