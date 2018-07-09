# -*- coding: utf-8 -*-

from __future__ import absolute_import

from skoot.utils.testing import (assert_raises, assert_transformer_asdf,
                                 assert_persistable)
from skoot.feature_extraction.spatial import (haversine_distance,
                                              HaversineFeatures)


def test_bad_units():
    assert_raises(ValueError, haversine_distance,
                  None, None, None, None, "radians")


# TODO:
def test_haversine():
    pass


def test_haversine_transformer():
    pass


def test_haversine_asdf():
    pass


def test_haversine_persistable():
    pass
