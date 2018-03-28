# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import print_function, absolute_import

import pandas as pd

__all__ = [
    'load_boston_df',
    'load_breast_cancer_df',
    'load_iris_df'
]


def _load_from_bunch(bunch, include_tgt, tgt_name, names):
    # internal loading method
    X = pd.DataFrame.from_records(
        data=bunch.data,
        columns=bunch.feature_names if not names else names)

    if include_tgt:
        X[tgt_name] = bunch.target
    return X


def load_boston_df(include_tgt=True, tgt_name="target", names=None):
    """Get the Boston housing dataset.

    Loads the boston housing dataset into a dataframe with the
    target set as the "target" feature or whatever name
    is specified in ``tgt_name``.

    Parameters
    ----------
    include_tgt : bool, optional (default=True)
        Whether to include the target

    tgt_name : str, optional (default="target")
        The name of the target feature

    names : iterable or None
        The column names for the dataframe. If not
        defined, will default to the ``feature_names``
        attribute in the sklearn bunch instance.

    Returns
    -------
    X : pd.DataFrame, shape=(n_samples, n_features)
        The loaded boston dataset
    """
    from sklearn.datasets import load_boston
    return _load_from_bunch(load_boston(), include_tgt,
                            tgt_name, names)


def load_breast_cancer_df(include_tgt=True, tgt_name="target", names=None):
    """Get the breast cancer dataset.

    Loads the breast cancer dataset into a dataframe with the
    target set as the "target" feature or whatever name
    is specified in ``tgt_name``.

    Parameters
    ----------
    include_tgt : bool, optional (default=True)
        Whether to include the target

    tgt_name : str, optional (default="target")
        The name of the target feature

    names : iterable or None
        The column names for the dataframe. If not
        defined, will default to the ``feature_names``
        attribute in the sklearn bunch instance.

    Returns
    -------
    X : pd.DataFrame, shape=(n_samples, n_features)
        The loaded breast cancer dataset
    """
    from sklearn.datasets import load_breast_cancer
    return _load_from_bunch(load_breast_cancer(), include_tgt,
                            tgt_name, names)


def load_iris_df(include_tgt=True, tgt_name="species", names=None):
    """Get the iris dataset.

    Loads the iris dataset into a dataframe with the
    target set as the "species" feature or whatever name
    is specified in ``tgt_name``.

    Parameters
    ----------
    include_tgt : bool, optional (default=True)
        Whether to include the target

    tgt_name : str or unicode, optional (default="species")
        The name of the target feature

    names : iterable or None
        The column names for the dataframe. If not
        defined, will default to the ``feature_names``
        attribute in the sklearn bunch instance.

    Returns
    -------
    X : pd.DataFrame, shape=(n_samples, n_features)
        The loaded iris dataset.
    """
    from sklearn.datasets import load_iris
    return _load_from_bunch(load_iris(), include_tgt, tgt_name, names)
