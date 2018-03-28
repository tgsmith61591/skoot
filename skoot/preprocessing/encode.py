from __future__ import print_function, division, absolute_import
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import column_or_1d
from sklearn.preprocessing.label import _check_numpy_unicode_bug
import numpy as np
import pandas as pd
from skutil.base import BaseSkutil
from skutil.utils import validate_is_pd

__all__ = [
    'SafeLabelEncoder',
    'OneHotCategoricalEncoder'
]


def _get_unseen():
    """Basically just a static method
    instead of a class attribute to avoid
    someone accidentally changing it."""
    return 99999


class SafeLabelEncoder(LabelEncoder):
    """An extension of LabelEncoder that will
    not throw an exception for unseen data, but will
    instead return a default value of 99999

    Attributes
    ----------

    classes_ : the classes that are encoded
    """

    def transform(self, y):
        """Perform encoding if already fit.

        Parameters
        ----------

        y : array_like, shape=(n_samples,)
            The array to encode

        Returns
        -------

        e : array_like, shape=(n_samples,)
            The encoded array
        """
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        classes = np.unique(y)
        _check_numpy_unicode_bug(classes)

        # Check not too many:
        unseen = _get_unseen()
        if len(classes) >= unseen:
            raise ValueError('Too many factor levels in feature. Max is %i' % unseen)

        e = np.array([
                         np.searchsorted(self.classes_, x) if x in self.classes_ else unseen
                         for x in y
                         ])

        return e


class OneHotCategoricalEncoder(BaseSkutil, TransformerMixin):
    """This class achieves three things: first, it will fill in 
    any NaN values with a provided surrogate (if desired). Second,
    it will dummy out any categorical features using OneHotEncoding
    with a safety feature that can handle previously unseen values,
    and in the transform method will re-append the dummified features
    to the dataframe. Finally, it will return a numpy ndarray.
    
    Parameters
    ----------

    fill : str, optional (default = 'Missing')
        The value that will fill the missing values in the column

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead. 
        Since most skutil transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.


    Examples
    --------

        >>> import pandas as pd
        >>> import numpy as np
        >>> from skutil.preprocessing import OneHotCategoricalEncoder
        >>>
        >>> X = pd.DataFrame.from_records(data=np.array([
        ...                                  ['USA','RED','a'],
        ...                                  ['MEX','GRN','b'],
        ...                                  ['FRA','RED','b']]), 
        ...                               columns=['A','B','C'])
        >>>
        >>> o = OneHotCategoricalEncoder(as_df=True)
        >>> o.fit_transform(X)
           A.FRA  A.MEX  A.USA  A.NA  B.GRN  B.RED  B.NA  C.a  C.b  C.NA
        0    0.0    0.0    1.0   0.0    0.0    1.0   0.0  1.0  0.0   0.0
        1    0.0    1.0    0.0   0.0    1.0    0.0   0.0  0.0  1.0   0.0
        2    1.0    0.0    0.0   0.0    0.0    1.0   0.0  0.0  1.0   0.0

        
    Attributes
    ----------
    
    obj_cols_ : array_like
        The list of object-type (categorical) features

    lab_encoders_ : array_like
        The label encoders

    one_hot_ : an instance of a OneHotEncoder

    trans_nms_ : the dummified names
    """

    def __init__(self, fill='Missing', as_df=True):
        super(OneHotCategoricalEncoder, self).__init__(cols=None, as_df=as_df)
        self.fill = fill

    def fit(self, X, y=None):
        """Fit the encoder.

        Parameters
        ----------

        X : Pandas ``DataFrame``, shape=(n_samples, n_features)
            The Pandas frame to fit. The frame will only
            be fit on the object columns of the dataframe.

        y : None
            Passthrough for ``sklearn.pipeline.Pipeline``. Even
            if explicitly set, will not change behavior of ``fit``.

        Returns
        -------

        self
        """
        # check on state of X, don't care about cols or the warning
        X, _ = validate_is_pd(X, None)

        # Extract the object columns
        obj_cols_ = X.select_dtypes(include=['object']).columns.values

        # If we need to fill in the NAs, take care of it
        if self.fill is not None:
            X[obj_cols_] = X[obj_cols_].fillna(self.fill)

        # Set an array of uninitialized label encoders
        # Then use fit_transform for effiency purposes
        # We can also set the dummy-level feature names in the same pass
        lab_encoders_ = []
        trans_array = []
        tnms = []

        unseen = _get_unseen()
        for nm in obj_cols_:
            encoder = SafeLabelEncoder()
            lab_encoders_.append(encoder)

            # This fits the reference to the encoder, and gets
            # the transformation. We then append a single unseen
            # value to the end as a safety for the transform method.
            # After the transpose, this is tantamount to appending a row
            # of unseen values so each feature can handle the 99999
            # This will expand the matrix by N columns, but if there's
            # no new values, they will be entirely zero and can be dropped later.
            encoded_array = np.append(encoder.fit_transform(X[nm]), unseen)

            # Add the transformed row
            trans_array.append(encoded_array)  # Updates in array

            # Update the names
            n_classes = len(encoder.classes_)
            sequential_nms = ['%s.%s' % (nm, str(encoder.classes_[i])) for i in range(n_classes)]

            # Remember to append the NA col
            sequential_nms.append('%s.NA' % nm)
            tnms.append(sequential_nms)

        # Get the transpose
        trans = np.array(trans_array).transpose()

        # flatten the name array, append numeric names prior
        num_nms = [n for n in X.columns.values if n not in obj_cols_]
        trans_nms_ = [item for sublist in tnms for item in sublist]
        self.trans_nms_ = num_nms + trans_nms_

        # we might get an empty set of object cols
        shape_tup = trans.shape
        is_empty = len(shape_tup) < 2 or shape_tup[1] == 0  # zero cols

        # Now we can do the actual one hot encoding, set internal state
        self.one_hot_ = None if is_empty else OneHotEncoder().fit(trans)
        self.obj_cols_ = obj_cols_
        self.lab_encoders_ = lab_encoders_

        return self

    def transform(self, X):
        """Transform X, a DataFrame, by stripping
        out the object columns, dummifying them, and
        re-appending them to the end.
        
        Parameters
        ----------

        X : Pandas ``DataFrame``, shape=(n_samples, n_features)
            The Pandas frame to transform.

        Returns
        -------

        x : Pandas ``DataFrame`` or np.ndarray, shape=(n_samples, n_features)
            The encoded dataframe or array
        """
        check_is_fitted(self, 'obj_cols_')
        # check on state of X, don't care about cols or warning
        X, _ = validate_is_pd(X, None)

        # if there is no encoder to speak of, just bail early
        if not self.one_hot_:
            return X if self.as_df else X.as_matrix()

        # Retain just the numers
        numers = X[[nm for nm in X.columns.values if nm not in self.obj_cols_]]
        objs = X[self.obj_cols_]

        # If we need to fill in the NAs, take care of it
        if self.fill is not None:
            objs = objs.fillna(self.fill)

        # Do label encoding using the safe label encoders
        trans = np.array([v.transform(objs[self.obj_cols_[i]]) for
                          i, v in enumerate(self.lab_encoders_)]).transpose()

        # Finally, get the one-hot encoding...
        oh = self.one_hot_.transform(trans).todense()
        x = np.array(np.hstack((numers, oh)))

        return x if not self.as_df else pd.DataFrame.from_records(data=x, columns=self.trans_nms_)
