# -*- coding: utf-8 -*-

from joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted
import numpy as np

from itertools import combinations

from .base import BaseCompoundFeatureDeriver
from ..utils import is_iterable, flatten_all, dataframe_or_array
from ..utils.validation import check_dataframe, validate_test_set_columns

__all__ = [
    'haversine_distance',
    'HaversineFeatures'
]


_units = {'mi': 3959.,
          'km': 6371.}


def haversine_distance(lat1, lon1, lat2, lon2, units='mi'):
    """Compute the Haversine distance between two points.

    Calculates the Great Circle distance between two lat/lon points.
    Can be applied to scalars or array-like types (np.ndarray or pd.Series),
    and can compute distance in either KM ('km') or miles ('mi').

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

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * r


def _validate_spatial_cols(X, cols):
    # First thing: cols cannot be None
    if cols is None:
        raise TypeError("cols must be non-None")

    # Next, it must be iterable, and each individual element must be iterables
    # of length 2, all of which are column names that are present in X.
    if not (is_iterable(cols) and
            all(is_iterable(t) and len(t) == 2 for t in cols)):
        raise ValueError("For spatial distance transformers, columns must "
                         "be specified as an iterable of length-2 tuples or "
                         "lists containing lat/long column names. I.e.:\n\n"
                         "[('lat_1', 'lon_1'), ..., ('lat_n', 'lon_n')]")

    # We need to ensure the field names all exist in X
    cols_unpacked = set(flatten_all(cols))
    X, _ = check_dataframe(X, cols=cols_unpacked, assert_all_finite=False)
    return X, cols, cols_unpacked


def _haversine(pair_1, pair_2, lat1, lon1, lat2, lon2, units):
    return (pair_1, pair_2), \
        haversine_distance(lat1=lat1, lon1=lon1,
                           lat2=lat2, lon2=lon2,
                           units=units)


class HaversineFeatures(BaseCompoundFeatureDeriver):
    """Derive distance features between lat/long features.

    Calculates the Great Circle distance between lat/lon features.
    This is a valuable way to derive meaningful features from otherwise
    hard-to-leverage geo-spatial features.

    Parameters
    ----------
    cols : array-like, shape=(n_features,)
        The names of the columns on which to apply the transformation.
        In spatial transformers, the columns must be an iterable of length-2
        iterables resembling the following::

            [('lat_1', 'lon_1'), ..., ('lat_n', 'lon_n')]

        The distance between each unique pair will be computed.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.

    n_jobs : int, 1 by default
       The number of jobs to use for the encoding. This works by
       fitting each incremental LabelEncoder in parallel.

       If -1 all CPUs are used. If 1 is given, no parallel computing code
       is used at all, which is useful for debugging. For n_jobs below -1,
       (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but
       one are used.

    sep : str or unicode (optional, default="_")
        The separator between the new feature names. The names will be in the
        form of::

            <left><sep><right><sep><suffix>

        For examples, for columns 'a' and 'b', ``sep="_"`` and
        ``name_suffix="delta"``, the new column name would be::

            a_b_delta

    name_suffix : str or None, optional (default=None)
        The suffix to add to the new feature name in the form of::

            <feature_x>_<feature_y>_<suffix>

        If None, will be equal to ``units`` (i.e., 'km' or 'mi').
        See ``sep`` for more details about how new column names are formed.

    units : str or unicode, optional (default='mi')
        The units to return. One of ('mi', 'km')

    drop_original : bool, optional (default=True)
        Whether to drop the original features from the dataframe prior to
        returning from the ``transform`` method.

    Examples
    --------
    >>> import pandas as pd
    >>> X = pd.DataFrame.from_records([
    ...     [10001, 30.2672, -97.7431, 32.7767, -96.7970],
    ...     [10011, 40.8781, -87.6298, 40.7128, -74.0060]
    ... ], columns=['id', 'from_lat', 'from_lon', 'to_lat', 'to_lon'])
    >>> est = HaversineFeatures(cols=[('from_lat', 'from_lon'),
    ...                               ('to_lat', 'to_lon')])
    >>> est.fit_transform(X)
          id  (from_lat,from_lon)_(to_lat,to_lon)_mi
    0  10001                              182.132066
    1  10011                              712.034570
    """
    def __init__(self, cols, as_df=True, n_jobs=1, sep="_",
                 name_suffix=None, units='mi', drop_original=True):

        super(HaversineFeatures, self).__init__(
            cols=cols, as_df=as_df,
            sep=sep, name_suffix=name_suffix)

        self.n_jobs = n_jobs
        self.units = units
        self.drop_original = drop_original

    def fit(self, X, y=None):
        """Fit the Haversine transformer.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : array-like or None, shape=(n_samples,), optional (default=None)
            Pass-through for ``sklearn.pipeline.Pipeline``.
        """
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None, **kwargs):
        """Fit the Haversine transformer and transform a new dataset.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : array-like or None, shape=(n_samples,), optional (default=None)
            Pass-through for ``sklearn.pipeline.Pipeline``.
        """
        X, cols, original_cols = _validate_spatial_cols(X, self.cols)
        X = self._transform(X, cols, original_cols)

        self.fit_cols_ = cols
        self.original_cols_ = original_cols
        return dataframe_or_array(X, self.as_df)

    def transform(self, X):
        """Transform a new dataset, computing the haversine dists.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to transform.
        """
        check_is_fitted(self, "fit_cols_")
        X, _ = check_dataframe(X, cols=self.original_cols_)

        # validate that fit cols in test set
        cols = self.fit_cols_
        validate_test_set_columns(self.original_cols_, X.columns)

        return dataframe_or_array(
            self._transform(X, cols, self.original_cols_),
            as_df=self.as_df)

    def _transform(self, X, cols, original_cols):
        dists = list(Parallel(n_jobs=self.n_jobs)(
            delayed(_haversine)(pair_1=(lat1, lon1),
                                pair_2=(lat2, lon2),
                                lat1=X[lat1], lon1=X[lon1],
                                lat2=X[lat2], lon2=X[lon2],
                                units=self.units)
            for (lat1, lon1), (lat2, lon2) in combinations(cols, 2)))

        # Assign to the dataframe
        sep = self.sep
        suff = (lambda v: v if v else self.units)(self.name_suffix)
        for ((lat_1, lon_1), (lat_2, lon_2)), dist in dists:
            nm = "(%s,%s)%s(%s,%s)%s%s" \
                 % (lat_1, lon_1, sep, lat_2,
                    lon_2, sep, suff)
            X[nm] = dist

        # Drop if needed
        if self.drop_original:
            X.drop(original_cols, axis=1, inplace=True)
        return X
