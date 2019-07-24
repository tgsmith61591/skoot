# -*- coding: utf-8 -*-

import pandas as pd
from numpy.testing import assert_almost_equal

from skoot.utils.testing import (assert_raises, assert_transformer_asdf,
                                 assert_persistable)
from skoot.feature_extraction.spatial import (haversine_distance,
                                              HaversineFeatures)


X = pd.DataFrame.from_records([
    [10001, 30.2672, -97.7431, 32.7767, -96.7970],
    [10011, 40.8781, -87.6298, 40.7128, -74.0060]
], columns=['id', 'from_lat', 'from_lon', 'to_lat', 'to_lon'])

# Expected values
expected_mi = [182.132066, 712.034570]
expected_km = [293.095073, 1145.83790]


def test_bad_units():
    assert_raises(ValueError, haversine_distance,
                  None, None, None, None, "radians")


def test_haversine():
    mi = haversine_distance(X.from_lat, X.from_lon,
                            X.to_lat, X.to_lon, units='mi').values

    km = haversine_distance(X.from_lat, X.from_lon,
                            X.to_lat, X.to_lon, units='km').values

    assert_almost_equal(mi, expected_mi, decimal=5)
    assert_almost_equal(km, expected_km, decimal=5)


def test_haversine_transformer_mi():
    est = HaversineFeatures(cols=[('from_lat', 'from_lon'),
                                  ('to_lat', 'to_lon')], units='mi')

    # Assert the columns
    trans = est.fit_transform(X)
    assert trans.columns.tolist() == \
        ['id', '(from_lat,from_lon)_(to_lat,to_lon)_mi']

    dist = trans['(from_lat,from_lon)_(to_lat,to_lon)_mi']
    assert_almost_equal(dist, expected_mi, decimal=5)


def test_haversine_transformer_km():
    est = HaversineFeatures(cols=[('from_lat', 'from_lon'),
                                  ('to_lat', 'to_lon')], units='km')

    # Assert the columns
    trans = est.fit(X).transform(X)  # make sure the fit and transform work
    assert trans.columns.tolist() == \
        ['id', '(from_lat,from_lon)_(to_lat,to_lon)_km']

    dist = trans['(from_lat,from_lon)_(to_lat,to_lon)_km']
    assert_almost_equal(dist, expected_km, decimal=5)


# Test where name_suffix is not None
def test_haversine_transformer_suffix():
    est = HaversineFeatures(cols=[('from_lat', 'from_lon'),
                                  ('to_lat', 'to_lon')],
                            name_suffix="DIST")

    # Assert the columns
    trans = est.fit_transform(X)
    assert trans.columns.tolist() == \
        ['id', '(from_lat,from_lon)_(to_lat,to_lon)_DIST']


def test_haversine_asdf():
    assert_transformer_asdf(
        HaversineFeatures(cols=[('from_lat', 'from_lon'),
                                ('to_lat', 'to_lon')]), X)


def test_haversine_persistable():
    assert_persistable(
        HaversineFeatures(cols=[('from_lat', 'from_lon'),
                                ('to_lat', 'to_lon')]),
        "location.pkl", X)


def test_haversine_bad_cols():
    est = HaversineFeatures(cols=None)
    assert_raises(TypeError, est.fit, X)
