# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np

__all__ = [
    'haversine_distance'
]


_units = {'mi': 3959.,
          'km': 6371.}


def haversine_distance(lat1, lon1, lat2, lon2, units='mi'):
    """Compute the Haversine distance between two points.

    Calculates the Great Circle distance between two lat/lon points.
    Can be applied to scalars or array-like types (np.ndarray or pd.Series).

    Parameters
    -----------
    lat1 : float or array-like, shape=(n_samples,)
        The first latitude

    lon1 : float or array-like, shape=(n_samples,)
        The first longitude

    lat2 : float or array-like, shape=(n_samples,)
        The second latitude

    lon2 : float or array-like, shape=(n_samples,)
        The second longitude

    units : str or unicode, optional (default='mi')
        The units to return. One of ('mi', 'km')
    """
    # Evaluate the units, raise if illegal unit
    try:
        r = _units[units]  # radius of the Earth in given units
    except KeyError:
        raise ValueError("'units' must be one of %r. Got %s"
                         % (list(_units.keys()), units))

    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * r
