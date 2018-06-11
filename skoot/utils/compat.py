# -*- coding: utf-8 -*-
#
# Compatibility utilities

# Python <3
try:
    xrange = xrange

# Python 3+
except NameError:
    xrange = range
