# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel, delayed

import pandas as pd
import numpy as np

from ..base import BasePDTransformer
from ..utils.validation import check_dataframe, validate_test_set_columns
from ..utils.dataframe import dataframe_or_array, get_continuous_columns
from ..utils.metaestimators import timed_instance_method

import warnings

__all__ = [
    'DummyEncoder'
]


# Computed in parallel
def _le_transform(col, vec, le, handle, sep):
    # if "ignore" and there are unknown labels in the array,
    # we need to handle it...
    missing_mask = ~np.in1d(vec, le.classes_)  # type: np.ndarray
    any_missing = missing_mask.any()
    ignore = handle in ("ignore", "warn")

    if ignore and any_missing:
        # if we want to warn, do so now
        if handle == "warn":
            warnings.warn("Previously unseen level(s) found in data! %r"
                          % set(vec[missing_mask].tolist()))

        # initialize vec_trans as zeros
        vec_trans = np.zeros(vec.shape[0]).astype(int)

        # where the labels are present, transform
        vec_trans[~missing_mask] = \
            le.transform(vec[~missing_mask])

        # XXX: Where the labels are NOT present, set them to n_classes + 1.
        # (is there a better dummy class for this? We know, for instance,
        # the labels are constrained to be positive, so we could theoretically
        # use -1, however we also know the OHE does not discriminate, and any
        # unknown levels are treated similarly, so it shouldn't matter, right?
        # In a sense, we're setting k+1 to be a class indicating "other")
        vec_trans[missing_mask] = le.classes_.shape[0]

    # Otherwise take our chances that they're all there
    else:
        vec_trans = le.transform(vec)  # Union[str, int] -> int

    # Get the column names (levels) so we can transform in the
    # same order of the output cols. Note that we do not need to consider
    # new levels here, since the OHE will ignore them in its transformation
    # step
    le_clz = le.classes_.tolist()
    classes = ["%s%s%s" % (col, sep, clz) for clz in le_clz]
    return col, vec_trans, classes


# Computed in parallel
def _fit_transform_one_encoder(col, vec):
    # Fit/transform a label encoder. This is run in parallel
    # using the joblib library
    le = LabelEncoder()
    trans = le.fit_transform(vec)
    return col, le, trans


class DummyEncoder(BasePDTransformer):
    """Dummy encode categorical data.

    A custom one-hot encoding class that is capable of handling previously
    unseen levels and automatically dropping one level from each categorical
    feature in order to avoid the dummy variable trap.

    Parameters
    ----------
    cols : array-like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        Unlike other BasePDTransformer instances, this should not be left
        as the default None, since dummying the entire frame could prove
        very expensive.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.
        Since most skoot transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.

    sep : str or unicode, optional (default='_')
        The string separator between the categorical feature name
        and the level name.

    drop_one_level : bool, optional (default=True)
        Whether to drop one level for each categorical variable.
        This helps avoid the dummy variable trap.

    handle_unknown : str or unicode, optional (default='ignore')
        How to handle the unknown levels. "ignore" will not raise an error
        for unknown test set levels, but "error" will. "warn" will produce
        a warning.

    n_jobs : int, 1 by default
       The number of jobs to use for the encoding. This works by
       fitting each incremental LabelEncoder in parallel.

       If -1 all CPUs are used. If 1 is given, no parallel computing code
       is used at all, which is useful for debugging. For n_jobs below -1,
       (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but
       one are used.

    Attributes
    ----------
    ohe_ : OneHotEncoder
        The one hot encoder

    le_ : dict[str: LabelEncoder]
        A dictionary mapping column names to their respective LabelEncoder
        instances.

    fit_cols_ : list
        The list of column names on which the transformer was fit. This
        is used to validate the presence of the features in the test set
        during the ``transform`` stage.
    """
    def __init__(self, cols=None, as_df=True, sep='_', drop_one_level=True,
                 handle_unknown="ignore", n_jobs=1):

        super(DummyEncoder, self).__init__(
            cols=cols, as_df=as_df)

        self.sep = sep
        self.drop_one_level = drop_one_level
        self.handle_unknown = handle_unknown
        self.n_jobs = n_jobs

    @timed_instance_method(attribute_name="fit_time_")
    def fit(self, X, y=None):
        """Fit the dummy encoder.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : array-like or None, shape=(n_samples,), optional (default=None)
            Pass-through for ``sklearn.pipeline.Pipeline``.
        """
        # validate the input, and get a copy of it
        X, cols = check_dataframe(X, cols=self.cols,
                                  assert_all_finite=False)

        # Warn if the columns were passed as default and any of the dtypes
        # in the frame are numeric
        if self.cols is None and \
                len(get_continuous_columns(X).columns.tolist()) > 0:
            warnings.warn("Continuous features detected in DummyEncoder. "
                          "This can increase runtime and dimensionality "
                          "drastically. This warning only appears when "
                          "`cols` is left as the default None, and there are "
                          "continuous features present.")

        # for each column, fit a label encoder, get the transformation
        encoded = list(Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one_encoder)(col, X[col].values)
            for col in cols))

        # Quickly run over the encoded, set the columns in the dataframe
        # pre-OHE fit and then create a dict of the encoders. This is a
        # cheap pass of N over the columns, where we simply assign columns
        # or values in a dict to the pre-computed values
        lab_encoders = {}
        for col, le, trans in encoded:
            X[col] = trans
            lab_encoders[col] = le

        # Fit a single OHE on the transformed columns. Note that the sklearn
        # OHE class does not discriminate in "warn" or "ignore", so we just
        # pass "ignore" since we already warned above.
        handle = "ignore" if self.handle_unknown in ("warn", "ignore") \
            else "error"
        ohe = OneHotEncoder(
            sparse=False,  # TODO: Is there a way to sparsify with Pandas?
            handle_unknown=handle).fit(X[cols])

        # assign fit params
        self.ohe_ = ohe
        self.le_ = lab_encoders
        self.fit_cols_ = cols

        return self

    def transform(self, X):
        """Apply the encoding to a dataframe.

        This method will encode the features in the test frame with the
        levels discovered in the ``fit`` computation.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to transform. The operation will
            be applied to a copy of the input data, and the result
            will be returned.

        Returns
        -------
        X : pd.DataFrame or np.ndarray, shape=(n_samples, n_features)
            The operation is applied to a copy of ``X``,
            and the result set is returned.
        """
        check_is_fitted(self, 'ohe_')
        X, _ = check_dataframe(X, cols=self.cols)

        # validate that fit cols in test set
        cols = self.fit_cols_
        validate_test_set_columns(cols, X.columns)

        # fit params that we need
        ohe = self.ohe_
        lenc = self.le_
        sep = self.sep
        drop = self.drop_one_level

        # Do transformations in parallel
        transformations = list(Parallel(n_jobs=self.n_jobs)(
            delayed(_le_transform)(
                col=col, vec=X[col].values, le=lenc[col],
                handle=self.handle_unknown, sep=sep)
            for col in cols))

        # This is another pass of O(N), but it's not performing any incremental
        # transformations of any sort. It just traverses the list of affected
        # columns, extending the column order list and tracking the columns to
        # drop. All of the heavy lifting for the transformations was handled
        # in parallel above.
        col_order = []
        drops = []
        for col, vec_trans, classes in transformations:
            X[col] = vec_trans
            col_order.extend(classes)

            # if we want to drop one, just drop the last
            if drop and len(classes) > 1:
                drops.append(classes[-1])

        # now we can get the transformed OHE
        ohe_trans = pd.DataFrame.from_records(data=ohe.transform(X[cols]),
                                              columns=col_order)

        # set the index to be equal to X's for a smooth concat
        ohe_trans.index = X.index

        # if we're dropping one level, do so now
        if drops:
            ohe_trans = ohe_trans.drop(drops, axis=1)

        # drop the original columns from X
        X = X.drop(cols, axis=1)

        # We might have dropped ALL columns from X. And if that's the case, we
        # can just return the encoded columns
        if not X.columns.tolist():
            return dataframe_or_array(ohe_trans, self.as_df)

        # otherwise concat the new columns
        X = pd.concat([X, ohe_trans], axis=1)  # type: pd.DataFrame
        return dataframe_or_array(X, self.as_df)
