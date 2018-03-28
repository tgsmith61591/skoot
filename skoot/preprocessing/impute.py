# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, is_classifier
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.externals import six
from sklearn.utils.validation import check_is_fitted
from abc import ABCMeta
from skutil.base import SelectiveMixin, BaseSkutil
from ..utils import is_entirely_numeric, get_numeric, validate_is_pd, is_numeric
from ..utils.fixes import is_iterable

__all__ = [
    'BaggedImputer',
    'BaggedCategoricalImputer',
    'ImputerMixin',
    'SelectiveImputer'
]


def _validate_all_numeric(X):
    """Validate that all columns in X
    are numeric types. If not, raises a
    ``ValueError``

    Parameters
    ----------

    X : Pandas ``DataFrame``, shape=(n_samples, n_features)
        The dataframe to validate

    Raises
    ------

    ``ValueError`` if not all columns are numeric
    """
    if not is_entirely_numeric(X):
        raise ValueError('provided columns must be of only numeric columns')


def _col_mode(col):
    """Get the mode from a series.

    Returns
    -------

    com : int, float
        The column's most common value.
    """
    vals = col.value_counts()
    com = vals.index[0] if not np.isnan(vals.index[0]) else vals.index[1]
    return com


def _val_values(vals):
    """Validate that all values in the iterable
    are either numeric, or in ('mode', 'median', 'mean').
    If not, will raise a TypeError

    Raises
    ------

    ``TypeError`` if not all values are numeric or
    in valid values.
    """
    if not all([
                   (is_numeric(i) or (isinstance(i, six.string_types)) and i in ('mode', 'mean', 'median'))
                   for i in vals
               ]):
        raise TypeError('All values in self.fill must be numeric or in ("mode", "mean", "median"). '
                        'Got: %s' % ', '.join(vals))


class ImputerMixin:
    """A mixin for all imputer classes. Contains the default fill value.
    This mixin is used for the H2O imputer, as well.

    Attributes
    ----------

    _def_fill : int (default=-999999)
        The default fill value for NaN values
    """
    _def_fill = -999999


class _BaseImputer(six.with_metaclass(ABCMeta, BaseSkutil, TransformerMixin, ImputerMixin)):
    """A base class for all imputers. Handles assignment of the fill value.

    Parameters
    ----------

    cols : array_like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the transformer will be ``fit``
        on the entire frame. Note that the transformation will also only
        apply to the specified columns, and any other non-specified
        columns will still be present after transformation. Note that since 
        this transformer can only operate on numeric columns, not explicitly 
        setting the ``cols`` parameter may result in errors for categorical data.

    as_df : bool, optional (default=True)
        Whether to return a Pandas DataFrame in the ``transform``
        method. If False, will return a NumPy ndarray instead. 
        Since most skutil transformers depend on explicitly-named
        DataFrame features, the ``as_df`` parameter is True by default.

    fill : int, float, string or array_like, optional (default=None)
        The fill values to use for missing values in columns

    Attributes
    ----------

    fill : float, int, None or str
        The fill
    """

    def __init__(self, cols=None, as_df=True, fill=None):
        super(_BaseImputer, self).__init__(cols=cols, as_df=as_df)
        self.fill = fill if fill is not None else self._def_fill


class SelectiveImputer(_BaseImputer):
    """A more customizable form on sklearn's ``Imputer`` class. This class
    can handle more than mean, median or most common... it will also take
    numeric values. Moreover, it will take a vector of strategies or values
    with which to impute corresponding columns.

    Parameters
    ----------

    cols : array_like, optional (default=None)
        The columns on which the transformer will be ``fit``. In
        the case that ``cols`` is None, the transformer will be fit
        on all columns. Note that since this transformer can only operate
        on numeric columns, not explicitly setting the ``cols`` parameter
        may result in errors for categorical data.

    as_df : bool, optional (default=True)
        Whether to return a Pandas DataFrame in the ``transform``
        method. If False, will return a NumPy ndarray instead. 
        Since most skutil transformers depend on explicitly-named
        DataFrame features, the ``as_df`` parameter is True by default.

    fill : int, float, string or array_like, optional (default=None)
        the fill to use for missing values in the training matrix
        when fitting a ``SelectiveImputer``. If None, will default to 'mean'


    Examples
    --------

        >>> import numpy as np
        >>> import pandas as pd
        >>> from skutil.preprocessing import SelectiveImputer
        >>>
        >>> nan = np.nan
        >>> X = pd.DataFrame.from_records(data=np.array([
        ...                                 [1.0,  nan,  3.1],
        ...                                 [nan,  2.3,  nan],
        ...                                 [2.1,  2.1,  3.1]]), 
        ...                               columns=['a','b','c'])
        >>> imputer = SelectiveImputer(fill=['mean', -999, 'mode'])
        >>> imputer.fit_transform(X)
              a      b    c
        0  1.00 -999.0  3.1
        1  1.55    2.3  3.1
        2  2.10    2.1  3.1


    Attributes
    ----------

    fills_ : iterable, int or float
        The imputer fill-values
    """

    def __init__(self, cols=None, as_df=True, fill='mean'):
        super(SelectiveImputer, self).__init__(cols, as_df, fill)

    def fit(self, X, y=None):
        """Fit the imputer and return the
        transformed matrix or frame.

        Parameters
        ----------

        X : Pandas ``DataFrame``, shape=(n_samples, n_features)
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : None
            Passthrough for ``sklearn.pipeline.Pipeline``. Even
            if explicitly set, will not change behavior of ``fit``.

        Returns
        -------

        self
        """

        # check on state of X and cols
        X, self.cols = validate_is_pd(X, self.cols)
        cols = self.cols if self.cols is not None else X.columns.values

        # validate the fill, do fit
        fill = self.fill
        if isinstance(fill, six.string_types):
            fill = str(fill)
            if fill not in ('mode', 'mean', 'median'):
                raise TypeError('self.fill must be either "mode", "mean", "median", None, '
                                'a number, or an iterable. Got %s' % fill)

            if fill == 'mode':
                # for each column to impute, we go through and get the value counts
                # of each, sorting by the max...
                self.fills_ = dict(zip(cols, X[cols].apply(lambda x: _col_mode(x))))

            elif fill == 'median':
                self.fills_ = dict(zip(cols, X[cols].apply(lambda x: np.nanmedian(x.values))))

            else:
                self.fills_ = dict(zip(cols, X[cols].apply(lambda x: np.nanmean(x.values))))

        # if the fill is an iterable, we have to get a bit more stringent on our validation
        elif is_iterable(fill):

            # if fill is a dictionary
            if isinstance(fill, dict):
                # if it's a dict, we can assume that these are the cols...
                cols, fill = zip(*fill.items())
                self.cols = cols  # we reset self.cols in this case!!!

            # we need to get the length of the iterable,
            # make sure it matches the len of cols
            if not len(fill) == len(cols):
                raise ValueError('len of fill does not match that of cols')

            # make sure they're all ints
            _val_values(fill)
            d = {}
            for ind, c in enumerate(cols):
                f = fill[ind]

                if is_numeric(f):
                    d[c] = f
                else:
                    the_col = X[c]
                    if f == 'mode':
                        d[c] = _col_mode(the_col)
                    elif f == 'median':
                        d[c] = np.nanmedian(the_col.values)
                    else:
                        d[c] = np.nanmean(the_col.values)

            self.fills_ = d

        else:
            if not is_numeric(fill):
                raise TypeError('self.fill must be either "mode", "mean", "median", None, '
                                'a number, or an iterable. Got %s' % str(fill))

            # either the fill is an int, or it's something the user provided...
            # if it's not an int or float, we'll let it go and not catch it because
            # the it's their fault they were dumb.
            self.fills_ = fill

        return self

    def transform(self, X):
        """Transform a dataframe given the fit imputer.

        Parameters
        ----------

        X : Pandas ``DataFrame``, shape=(n_samples, n_features)
            The Pandas frame to transform.

        Returns
        -------

        X : pd.DataFrame or np.ndarray
            The imputed matrix
        """

        check_is_fitted(self, 'fills_')
        # check on state of X and cols
        X, _ = validate_is_pd(X, self.cols)
        cols = self.cols if self.cols is not None else X.columns.values

        # get the fills
        modes = self.fills_

        # if it's a single int, easy:
        if isinstance(modes, int):
            X[cols] = X[cols].fillna(modes)
        else:
            # it's a dict
            for nm in cols:
                X[nm] = X[nm].fillna(modes[nm])

        return X if self.as_df else X.as_matrix()


class _BaseBaggedImputer(_BaseImputer):
    """Base class for all bagged imputers. See subclasses
    ``BaggedCategoricalImputer`` and ``BaggedImputer`` for specifics.
    """

    def __init__(self, cols=None, base_estimator=None, n_estimators=10,
                 max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=True,
                 oob_score=False, n_jobs=1, random_state=None, verbose=0, as_df=True,
                 fill=None, is_classification=False):

        super(_BaseBaggedImputer, self).__init__(cols=cols, as_df=as_df, fill=fill)

        # set self attributes
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.is_classification = is_classification

    def fit(self, X, y=None):
        """Fit the bagged imputer.

        Parameters
        ----------

        X : Pandas ``DataFrame``, shape=(n_samples, n_features)
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : None
            Passthrough for ``sklearn.pipeline.Pipeline``. Even
            if explicitly set, will not change behavior of ``fit``.

        Returns
        -------

        self
        """
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        """Fit the bagged imputer and return the
        transformed (imputed) matrix.

        Parameters
        ----------

        X : Pandas ``DataFrame``, shape=(n_samples, n_features)
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : None
            Passthrough for ``sklearn.pipeline.Pipeline``. Even
            if explicitly set, will not change behavior of ``fit``.

        Returns
        -------

        X : pd.DataFrame or np.ndarray
            The imputed matrix.
        """
        # check on state of X and cols
        X, self.cols = validate_is_pd(X, self.cols)
        cols = self.cols if self.cols is not None else X.columns.values

        # subset, validate
        # we have to validate that all of the columns we're going to impute
        # are numeric (this could be float, or int...).
        _validate_all_numeric(X[cols])

        # we need to get all of the numerics out of X, because these are
        # the features we'll be modeling on.
        numeric_cols = get_numeric(X)
        numerics = X[numeric_cols]

        # if is_classification and our estimator is NOT, then we need to raise
        if self.base_estimator is not None:
            if self.is_classification and not is_classifier(self.base_estimator):
                raise TypeError('self.is_classification=True, '
                                'but base_estimator is not a classifier')

        # set which estimator type to fit:
        _model = BaggingRegressor if not self.is_classification else BaggingClassifier

        # if there's only one numeric, we know at this point it's the one
        # we're imputing. In that case, there's too few cols on which to model
        if numerics.shape[1] == 1:
            raise ValueError('too few numeric columns on which to model')

        # the core algorithm:
        # - for each col to impute
        #   - subset to all numeric columns except the col to impute
        #   - retain only the complete observations, separate the missing observations
        #   - build a bagging regressor model to predict for observations with missing values
        #   - fill in missing values in a copy of the dataframe

        models = {}
        for col in cols:
            x = numerics.copy()  # get copy of numerics for this model iteration
            y_missing = pd.isnull(x[col])  # boolean vector of which are missing in the current y
            y = x.pop(col)  # pop off the y vector from the matrix

            # if y_missing is all of the rows, we need to bail
            if y_missing.sum() == x.shape[0]:
                raise ValueError('%s has all missing values, cannot train model' % col)

            # at this point we've identified which y values we need to predict, however, we still
            # need to prep our x matrix... There are a few corner cases we need to account for:
            #
            # 1. there are no complete rows in the X matrix
            #   - we can eliminate some columns to model on in this case, but there's no silver bullet
            # 2. the cols selected for model building are missing in the rows needed to impute.
            #   - this is a hard solution that requires even more NA imputation...
            #
            # the most "catch-all" solution is going to be to fill all missing values with some val, say -999999

            x = x.fillna(self.fill)
            X_train = x[~y_missing]  # the rows that don't correspond to missing y values
            X_test = x[y_missing]  # the rows to "predict" on
            y_train = y[~y_missing]  # the training y vector

            # define the model
            model = _model(
                base_estimator=self.base_estimator,
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose)

            # fit the model
            model.fit(X_train, y_train)

            # predict on the missing values, stash the model and the features used to train it
            if X_test.shape[0] != 0:  # only do this step if there are actually any missing
                y_pred = model.predict(X_test)
                X.loc[y_missing, col] = y_pred  # fill the y vector missing slots and reassign back to X

            models[col] = {
                'model': model,
                'feature_names': X_train.columns.values
            }

        # assign the model dict to self -- this is the "fit" portion
        self.models_ = models
        return X if self.as_df else X.as_matrix()

    def transform(self, X):
        """Impute the test data after fit.

        Parameters
        ----------

        X : Pandas ``DataFrame``, shape=(n_samples, n_features)
            The Pandas frame to transform.

        Returns
        -------

        dropped : Pandas DataFrame or NumPy ndarray
            The test frame sans "bad" columns
        """
        check_is_fitted(self, 'models_')
        # check on state of X and cols
        X, _ = validate_is_pd(X, self.cols)

        # perform the transformations for missing vals
        models = self.models_
        for col, kv in six.iteritems(models):
            features, model = kv['feature_names'], kv['model']
            y = X[col]  # the y we're predicting

            # this will throw a key error if one of the features isn't there
            X_test = X[features]  # we need another copy

            # if col is in the features, there's something wrong internally
            assert col not in features, 'predictive column should not be in fit features (%s)' % col

            # since this is a copy, we can add the missing vals where needed
            X_test = X_test.fillna(self.fill)

            # generate predictions, subset where y was null
            y_null = pd.isnull(y)
            pred_y = model.predict(X_test.loc[y_null])

            # fill where necessary:
            if y_null.sum() > 0:
                y[y_null] = pred_y  # fill where null
                X[col] = y  # set back to X

        return X if self.as_df else X.as_matrix()


class BaggedCategoricalImputer(_BaseBaggedImputer):
    """Performs imputation on select columns by using BaggingRegressors
    on the provided columns.

    cols : array_like, optional (default=None)
        The columns on which the transformer will be ``fit``. In
        the case that ``cols`` is None, the transformer will be fit
        on all columns. Note that since this transformer can only operate
        on numeric columns, not explicitly setting the ``cols`` parameter
        may result in errors for categorical data.

    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
        If int, then draw max_samples samples.
        If float, then draw max_samples * X.shape[0] samples.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
        If int, then draw max_features features.
        If float, then draw max_features * X.shape[1] features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    oob_score : bool, optional (default=False)
        Whether to use out-of-bag samples to estimate the generalization error.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both fit and predict. If -1,
        then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator; If
        RandomState instance, random_state is the random number generator; If None,
        the random number generator is the RandomState instance used by np.random.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    as_df : bool, optional (default=True)
        Whether to return a Pandas DataFrame in the ``transform``
        method. If False, will return a NumPy ndarray instead. 
        Since most skutil transformers depend on explicitly-named
        DataFrame features, the ``as_df`` parameter is True by default.

    fill : int, optional (default=None)
        the fill to use for missing values in the training matrix
        when fitting a BaggingClassifier. If None, will default to -999999


    Examples
    --------

        >>> import numpy as np
        >>> import pandas as pd
        >>> from skutil.preprocessing import BaggedCategoricalImputer
        >>>
        >>> nan = np.nan
        >>> X = pd.DataFrame.from_records(data=np.array([
        ...                                 [1.0,  nan,  4.0],
        ...                                 [nan,  1.0,  nan],
        ...                                 [2.0,  2.0,  3.0]]), 
        ...                               columns=['a','b','c'])
        >>> imputer = BaggedCategoricalImputer(random_state=42)
        >>> imputer.fit_transform(X)
             a    b    c
        0  1.0  2.0  4.0
        1  2.0  1.0  4.0
        2  2.0  2.0  3.0


    Attributes
    ----------

    models_ : dict, (string : ``sklearn.base.BaseEstimator``)
        A dictionary mapping column names to the fit
        bagged estimator.
    """

    def __init__(self, cols=None, base_estimator=None, n_estimators=10,
                 max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=True,
                 oob_score=False, n_jobs=1, random_state=None, verbose=0, as_df=True, fill=None):

        # categorical imputer needs to be classification
        super(BaggedCategoricalImputer, self).__init__(
            cols=cols, as_df=as_df, fill=fill,
            base_estimator=base_estimator, n_estimators=n_estimators,
            max_samples=max_samples, max_features=max_features, bootstrap=bootstrap,
            bootstrap_features=bootstrap_features, oob_score=oob_score,
            n_jobs=n_jobs, random_state=random_state, verbose=verbose,
            is_classification=True)


class BaggedImputer(_BaseBaggedImputer):
    """Performs imputation on select columns by using BaggingRegressors
    on the provided columns.

    cols : array_like, optional (default=None)
        The columns on which the transformer will be ``fit``. In
        the case that ``cols`` is None, the transformer will be fit
        on all columns. Note that since this transformer can only operate
        on numeric columns, not explicitly setting the ``cols`` parameter
        may result in errors for categorical data.

    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
        If int, then draw max_samples samples.
        If float, then draw max_samples * X.shape[0] samples.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
        If int, then draw max_features features.
        If float, then draw max_features * X.shape[1] features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    oob_score : bool, optional (default=False)
        Whether to use out-of-bag samples to estimate the generalization error.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both fit and predict. If -1,
        then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator; If
        RandomState instance, random_state is the random number generator; If None,
        the random number generator is the RandomState instance used by np.random.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    as_df : bool, optional (default=True)
        Whether to return a Pandas DataFrame in the ``transform``
        method. If False, will return a NumPy ndarray instead. 
        Since most skutil transformers depend on explicitly-named
        DataFrame features, the ``as_df`` parameter is True by default.

    fill : int, optional (default=None)
        the fill to use for missing values in the training matrix
        when fitting a BaggingRegressor. If None, will default to -999999


    Examples
    --------

        >>> import numpy as np
        >>> import pandas as pd
        >>> from skutil.preprocessing import BaggedImputer
        >>>
        >>> nan = np.nan
        >>> X = pd.DataFrame.from_records(data=np.array([
        ...                                 [1.0,  nan,  3.1],
        ...                                 [nan,  2.3,  nan],
        ...                                 [2.1,  2.1,  3.1]]), 
        ...                               columns=['a','b','c'])
        >>> imputer = BaggedImputer(random_state=42)
        >>> imputer.fit_transform(X)
               a     b    c
        0  1.000  2.16  3.1
        1  1.715  2.30  3.1
        2  2.100  2.10  3.1


    Attributes
    ----------

    models_ : dict, (string : ``sklearn.base.BaseEstimator``)
        A dictionary mapping column names to the fit
        bagged estimator.
    """

    def __init__(self, cols=None, base_estimator=None, n_estimators=10,
                 max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=True,
                 oob_score=False, n_jobs=1, random_state=None, verbose=0, as_df=True, fill=None):
        # invoke super constructor
        super(BaggedImputer, self).__init__(
            cols=cols, as_df=as_df, fill=fill,
            base_estimator=base_estimator, n_estimators=n_estimators,
            max_samples=max_samples, max_features=max_features, bootstrap=bootstrap,
            bootstrap_features=bootstrap_features, oob_score=oob_score,
            n_jobs=n_jobs, random_state=random_state, verbose=verbose,
            is_classification=False)
