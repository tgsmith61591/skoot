# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import print_function, absolute_import

import pandas as pd

from os.path import dirname, join

__all__ = [
    'load_adult_df',
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


def load_adult_df(include_tgt=True, tgt_name="target", names=None):
    """Load and return the adult dataset (classification).

    The adult dataset is a classic binary classification problem requiring
    pre-processing prior to being model-ready.

    =================   ============================
    Classes                                        2
    Samples per class       <=50k: 24720; >50k: 7841
    Samples total                              32561
    Dimensionality                                15
    Features             real, positive, categorical
    =================   ============================

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    include_tgt : bool, optional (default=True)
        Whether to include the target

    tgt_name : str, optional (default="target")
        The name of the target feature

    names : iterable or None
        The column names for the dataframe. If not
        defined, will default to the canonical feature names.

    Returns
    -------
    X : pd.DataFrame, shape=(n_samples, n_features)
        The loaded adult dataset

    References
    ----------
    .. [1] Ronny Kohavi and Barry Becker, "Data Mining and Visualization"
           Silicon Graphics. https://archive.ics.uci.edu/ml/datasets/Adult
    """
    # if names isn't defined, use the canonical names
    if names is None:
        names = ["age", "workclass", "fnlwgt", "education",
                 "education-num", "marital-status", "occupation",
                 "relationship", "race", "sex", "capital-gain",
                 "capital-loss", "hours-per-week", "native-country"]

    module_path = dirname(__file__)
    df = pd.read_csv(join(module_path, 'data', 'adult.csv'), header=None,
                     names=names + [tgt_name])

    # if we want to drop the target, do so now
    if not include_tgt:
        df.pop(tgt_name)
    return df


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
