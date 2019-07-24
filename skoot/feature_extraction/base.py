# -*- coding: utf-8 -*-

from ..base import BasePDTransformer


class BaseCompoundFeatureDeriver(BasePDTransformer):
    """Base class for feature extractors.

    Parameters
    ----------
    cols : array-like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.

    sep : str or unicode (optional, default="_")
        The separator between the new feature names. The names will be in the
        form of::

            <left><sep><right><sep><suffix>

        For examples, for columns 'a' and 'b', ``sep="_"`` and
        ``name_suffix="delta"``, the new column name would be::

            a_b_delta

    name_suffix : str, optional (default='delta')
        The suffix to add to the new feature name in the form of::

            <feature_x>_<feature_y>_<suffix>

        See ``sep`` for more details about how new column names are formed.
    """
    def __init__(self, cols, as_df, sep, name_suffix):

        super(BaseCompoundFeatureDeriver, self).__init__(
            cols=cols, as_df=as_df)

        self.sep = sep
        self.name_suffix = name_suffix
