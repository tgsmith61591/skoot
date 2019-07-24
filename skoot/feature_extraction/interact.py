# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Interaction between variables.

from sklearn.utils.validation import check_is_fitted
from itertools import combinations

from .base import BaseCompoundFeatureDeriver
from ..utils.validation import (check_dataframe, validate_multiple_cols,
                                validate_test_set_columns)
from ..utils.dataframe import dataframe_or_array
from ..utils.metaestimators import timed_instance_method

__all__ = [
    'InteractionTermTransformer'
]


# Just here so that we don't try to pickle a lambda later
def _mult(a, b):
    return a * b


class InteractionTermTransformer(BaseCompoundFeatureDeriver):
    """Create interaction terms between predictors.

    This class will compute interaction terms between selected columns.
    An interaction captures some relationship between two independent
    variables in the form of :math:`I_{ij} = x_{i} \times x_{j}`.

    Parameters
    ----------
    cols : array-like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the transformer will be ``fit``
        on the entire frame. Note that the transformation will also only
        apply to the specified columns, and any other non-specified
        columns will still be present after transformation.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.
        Since most skoot transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.

    interaction_function : callable, optional (default=None)
        A callable for interactions. Default None will result in
        multiplication of two Series objects. Use caution when passing
        a ``lambda`` expression, since they cannot be persisted via pickle!

    sep : str or unicode (optional, default="_")
        The separator between the new feature names. The names will be in the
        form of::

            <left><sep><right><sep><suffix>

        For examples, for columns 'a' and 'b', ``sep="_"`` and
        ``name_suffix="delta"``, the new column name would be::

            a_b_delta

    name_suffix : str, optional (default='I')
        The suffix to add to the new feature name in the form of
        <feature_x>_<feature_y>_<suffix>

    Attributes
    ----------
    fun_ : callable
        The interaction term function

    fit_cols_ : list
        The list of column names on which the transformer was fit. This
        is used to validate the presence of the features in the test set
        during the ``transform`` stage.

    Examples
    --------
    The following example interacts the first two columns of the iris
    dataset using the default ``_mul`` function (product).

    >>> from skoot.feature_extraction import InteractionTermTransformer
    >>> from skoot.datasets import load_iris_df
    >>>
    >>> X = load_iris_df(include_tgt=False)
    >>>
    >>> trans = InteractionTermTransformer(cols=X.columns[:2])
    >>> X_transform = trans.fit_transform(X)
    >>>
    >>> assert X_transform.shape[1] == X.shape[1] + 1 # only added 1 col
    >>> X_transform[X_transform.columns[-1]].head()
    0    17.85
    1    14.70
    2    15.04
    3    14.26
    4    18.00
    Name: sepal length (cm)_sepal width (cm)_I, dtype: float64
    """
    def __init__(self, cols=None, as_df=True, interaction_function=None,
                 sep="_", name_suffix='I'):

        super(InteractionTermTransformer, self).__init__(
            cols=cols, as_df=as_df,
            sep=sep, name_suffix=name_suffix)

        self.interaction_function = interaction_function

    @timed_instance_method(attribute_name="fit_time_")
    def fit(self, X, y=None):
        """Fit the interaction term transformer.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : array-like or None, shape=(n_samples,), optional (default=None)
            Pass-through for ``sklearn.pipeline.Pipeline``.
        """
        X, cols = check_dataframe(X, cols=self.cols)

        # validate multiple columns present
        validate_multiple_cols(self.__class__.__name__, cols)

        # if not provided default to multiplication
        self.fun_ = self.interaction_function \
            if self.interaction_function is not None else _mult

        # need to store the transform columns since they may differ in the
        # transform call
        self.fit_cols_ = cols

        # validate function
        if not hasattr(self.fun_, '__call__'):
            raise TypeError('require callable for interaction_function')

        return self

    def transform(self, X):
        """Transform a test matrix given the already-fit transformer.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to transform. The operation will
            be applied to a copy of the input data, and the result
            will be returned.

        Returns
        -------
        X : Pandas ``DataFrame``
            The operation is applied to a copy of ``X``,
            and the result set is returned.
        """
        check_is_fitted(self, 'fun_')
        X, _ = check_dataframe(X, cols=self.cols)

        # get the ones we need to transform, and which are present
        transform_cols = self.fit_cols_
        validate_test_set_columns(transform_cols, X.columns.tolist())

        suff = self.name_suffix
        fun = self.fun_

        # create a generator of names/features that we'll use the itertools
        # combinations to iterate and map out
        features = [(t, X[t]) for t in transform_cols]  # (name, feature)
        for (name_a, feat_a), (name_b, feat_b) in combinations(features, 2):
            new_nm = '%s%s%s%s%s' % (name_a, self.sep, name_b, self.sep, suff)

            # assign the new column to X
            X[new_nm] = fun(feat_a, feat_b)

        # return matrix if needed
        return dataframe_or_array(X, self.as_df)
