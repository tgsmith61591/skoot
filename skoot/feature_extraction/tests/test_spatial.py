# -*- coding: utf-8 -*-

from __future__ import absolute_import

from skoot.feature_extraction.spatial import haversine_distance
from skoot.utils.testing import assert_raises


def test_bad_units():
    assert_raises(ValueError, haversine_distance,
                  None, None, None, None, "radians")


# TODO:
def test_haversine():
    pass
