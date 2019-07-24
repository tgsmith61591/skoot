# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

import pandas as pd

from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.utils.validation import check_is_fitted
import six

from .base import BasePDTransformer
from .utils.validation import check_dataframe
from .utils.iterables import is_iterable
from .utils.dataframe import dataframe_or_array

__all__ = [
    'BaggedRegressorImputer',
    'BaggedClassifierImputer',
    'SelectiveImputer'
]


def _get_mask(X, value_to_mask):
    # Get the boolean mask X == missing_values
    if value_to_mask == 'NaN':
        return pd.isnull(X)
    else:
        return X == value_to_mask


def _get_callable(strat, valid_strats):
    # Handle lookup of callable for a strategy in valid strategies
    if isinstance(strat, six.string_types):
        try:
            return valid_strats[strat]
        except KeyError:
            raise ValueError("%s is not a valid strategy! Valid "
                             "strategies are: %r" % (strat, valid_strats))
    elif hasattr(strat, "__call__"):
        return strat
    raise TypeError("Each strategy must be a callable or string, "
                    "but got %r (type=%s)" % (repr(strat), type(strat)))


def _get_present_values(series, missing_mask):
    present_values = series[~missing_mask]
    if not present_values.shape[0]:
        raise ValueError("All values in column are missing!")
    return present_values


def _mean(series, missing_mask):
    # compute the mean of non missing elements in a pd.Series
    present_values = _get_present_values(series, missing_mask)
    return present_values.mean()


def _median(series, missing_mask):
    # compute the median of non missing elements in a pd.Series
    present_values = _get_present_values(series, missing_mask)
    return present_values.median()


def _most_frequent(series, missing_mask):
    # compute the mode of non missing elements in a pd.Series
    present_values = _get_present_values(series, missing_mask)
    return present_values.mode()[0]


class SelectiveImputer(BasePDTransformer):
    """Imputation transformer for completing missing values.

    The selective imputer applies column value imputation for the
    columns specified in ``cols`` at a more granular level than scikit-learn.
    Most notably, different strategies can be specified for each column.

    Note that if if no columns are specified, it will impute the entire
    matrix subject to ``strategy``.

    Parameters
    ----------
    cols : array-like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the transformer will be ``fit``
        on the entire frame. Note that the transformation will also only
        apply to the specified columns, and any other non-specified
        columns will still be present after transformation.

    strategy : str, iterable, dict or callable, optional (default="mean")
        The strategy to use for imputation.

        - If "mean", then replace missing values using the mean along
          the axis.
        - If "median", then replace missing values using the median along
          the axis.
        - If "most_frequent", then replace missing using the most frequent
          value along the axis.
        - If an iterable, must match the length of the ``cols`` parameter,
          and may contain strings or respective callables. This allows
          various columns to be imputed with differing strategies.
        - If a dict, the keys must be the column names specifed in ``cols``
          and must map to a string or callable.
        - If a callable, cannot be a bound method or it may cause issues when
          pickling for persistence. If the strategy is a callable, it must
          adhere to the function signature::

              function(pd.Series, is_missing_mask) -> float

    missing_values : integer or "NaN", optional (default="NaN")
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For missing values encoded as np.nan,
        use the string value "NaN".

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.
        Since most skoot transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.

    Examples
    --------
    A simple imputation example with varying strategies:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from skoot.impute import SelectiveImputer
    >>>
    >>> nan = np.nan
    >>> X = pd.DataFrame.from_records(
    ...     data=np.array([[1.0,  nan,  3.1],
    ...                    [nan,  2.3,  nan],
    ...                    [2.1,  2.1,  3.1]]),
    ...     columns=['a','b','c'])
    >>> imputer = SelectiveImputer(
    ...     strategy=('mean', (lambda *args: -999.), 'most_frequent'))
    >>> imputer.fit_transform(X)
          a      b    c
    0  1.00 -999.0  3.1
    1  1.55    2.3  3.1
    2  2.10    2.1  3.1
    >>> imputer.statistics_ # doctest: +SKIP
    {'a': 1.55, 'b': -999., 'c': 3.1}

    Attributes
    ----------
    statistics_ : dict
        A dictionary of statistics. The keys are the column names, and the
        values are the float results of the ``strategy`` callables.
    """
    def __init__(self, cols=None, strategy="mean", missing_values="NaN",
                 as_df=True):

        super(SelectiveImputer, self).__init__(
            cols=cols, as_df=as_df)

        self.missing_values = missing_values
        self.strategy = strategy

    def fit(self, X, y=None):
        """Fit the imputer.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : array-like or None, shape=(n_samples,), optional (default=None)
            Pass-through for ``sklearn.pipeline.Pipeline``.
        """
        missing_values = self.missing_values
        X, cols = check_dataframe(X, cols=self.cols)

        # validate the strategy
        strategy = self.strategy
        valid_strategies = {"mean": _mean,
                            "median": _median,
                            "most_frequent": _most_frequent}

        if isinstance(strategy, six.string_types):
            # if it's a string, map it into a dictionary for each col
            if strategy not in valid_strategies:
                raise ValueError("If strategy is a string, it must be one of "
                                 "%r. Received strategy=%s"
                                 % (valid_strategies, strategy))

            # map the corresponding value to their callable
            cllble = valid_strategies[strategy]
            strategy = {col: cllble for col in cols}

        # it it's a list, dict, etc. Handle the mapping
        if is_iterable(strategy):
            # list or tuple or something else
            if not isinstance(strategy, dict):
                # if the lengths don't match up, that's a problem...
                if len(strategy) != len(cols):
                    raise ValueError("Length of strategy iterable doesn't "
                                     "match provided columns! (%i!=%i)"
                                     % (len(strategy), len(cols)))

                strategy = {col: _get_callable(strat, valid_strategies)
                            for col, strat in zip(cols, strategy)}

            # otherwise it IS a dictionary, but we still don't want it
            # to be mutable, so go ahead and clone to change the ref
            else:
                keys = list(strategy.keys())
                scols = set(cols)  # O(1) lookup for all cols in X

                # make sure no bad keys in strategy
                if not all(k in scols for k in keys):
                    raise ValueError("Non-existant column names "
                                     "found in strategy!")

                # if it's a dictionary, the user may not have provided cols
                # specifically. If that's the case, we'll redefine cols here.
                # alternatively, if the user DID provide cols, the must match
                # which we check later
                if not self.cols:
                    cols = keys

                # also we need callables as values! Not strings!
                strategy = {col: _get_callable(v, valid_strategies)
                            for col, v in six.iteritems(strategy)}

        elif hasattr(strategy, "__call__"):
            # map the callable in a dictionary
            strategy = {col: strategy for col in cols}

        else:
            raise TypeError("strategy must be a string, callable, or "
                            "iterable. %r (type=%s) is not a valid strategy."
                            % (strategy, type(strategy)))

        # now we can actually fit!
        mask = _get_mask(X[cols], missing_values)
        self.statistics_ = {
            colname: strat(X[colname], mask[colname])
            for colname, strat in six.iteritems(strategy)}

        # another fit param we'll want is the amended strategy dict
        # (although we don't really use this...)
        self.strategy_ = strategy

        return self

    def transform(self, X):
        """Apply the imputation to a dataframe.

        This method will fill in the missing values within a test
        dataframe with the statistics computed in ``fit``.

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
        check_is_fitted(self, 'statistics_')

        # when validating columns here, we do it a bit differently... we just
        # want to ensure the columns present in 'statistics_' are present
        # in X.
        stats = self.statistics_
        cols = list(stats.keys())
        X, _ = check_dataframe(X, cols=cols)

        # now apply the stats to the X
        mask = _get_mask(X[cols], self.missing_values)
        for colname in cols:
            X.loc[mask[colname], colname] = stats[colname]

        return dataframe_or_array(X, self.as_df)


class _BaseBaggedImputer(BasePDTransformer):
    def __init__(self, imputer_class, cols, predictors, base_estimator,
                 n_estimators, max_samples, max_features, bootstrap,
                 bootstrap_features, n_jobs, random_state, verbose,
                 tmp_fill, as_df):

        super(_BaseBaggedImputer, self).__init__(
            cols=cols, as_df=as_df)

        self.predictors = predictors
        self.imputer_class = imputer_class
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.tmp_fill = tmp_fill

    def fit(self, X, y=None, **fit_params):
        """Fit the bagging imputer.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : array-like or None, shape=(n_samples,), optional (default=None)
            Pass-through for ``sklearn.pipeline.Pipeline``.
        """
        # validate that the input is a dataframe, get the columns
        X, cols = check_dataframe(X, cols=self.cols)

        # create predictors
        predictors = self.predictors
        if predictors is None:
            # if cols was defined, don't want to assign 1:1,
            # so check original ref
            predictors = cols if self.cols is None else X.columns.tolist()

        # if there were one predictor, and it's also the target, we'll
        # have an empty list at one of our fits...
        if len(predictors) == 1 and predictors[0] in cols:
            raise ValueError("Predictor list is only length 1, and the column "
                             "exists within `cols` (columns to impute). This "
                             "means one of our fits will have no predictors, "
                             "which is obviously not possible. Make sure you "
                             "A) use more than one predictor or B) do not "
                             "include the single predictor within `cols`. "
                             "(predictors=%r, cols=%r)" % (predictors, cols))

        # validate tmpfill
        tmpfill = self.tmp_fill
        if not isinstance(tmpfill, (int, float)):
            raise TypeError("tmp_fill must be a float or an int")

        # this dictionary will hold the models
        models = {}

        # this dictionary maps the impute column name(s) to the vecs
        targets = {c: X[c] for c in cols}

        # iterate the column names and the target columns
        for k, target in six.iteritems(targets):
            # separate out the predictor columns from the target column
            k_predictors = [p for p in predictors if p != k]
            subset = X[k_predictors]

            # split X row-wise into train/test where test is the missing
            # rows in the target
            test_mask = pd.isnull(target)

            # make sure to fill in values in the train set where there may
            # be missing values with the tmp_fill
            subset = subset.where(~pd.isnull(subset), tmpfill)

            train = subset.loc[~test_mask]  # only includes k_predictors
            train_y = target[~test_mask]

            # what if there are no trainable rows??
            if not train.shape[0]:
                raise ValueError("No trainable rows for target=%s, "
                                 "predictors=%r. Missing values exist in "
                                 "all predictor rows"
                                 % (k, k_predictors))

            # fit the regressor
            models[k] = self.imputer_class(
                base_estimator=self.base_estimator,
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose, oob_score=False,
                warm_start=False, **fit_params).fit(train, train_y)

        # assign fit params
        self.models_ = models
        self.predictors_ = predictors  # will need these to score on later!
        return self

    def transform(self, X):
        """Apply the imputation to a dataframe.

        This method will fill in the missing values within a test
        dataframe with the statistics computed in ``fit``.

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
        check_is_fitted(self, 'models_')
        X, _ = check_dataframe(X, cols=self.cols)
        predictors = self.predictors_

        # fill in the missing
        models = self.models_
        tmpfill = self.tmp_fill
        for k, model in six.iteritems(models):
            target = X[k]

            # separate out the predictor columns from the target column
            k_predictors = [p for p in predictors if p != k]
            subset = X[k_predictors]

            # split X row-wise into train/test where test is the missing
            # rows in the target
            test_mask = pd.isnull(target)

            # if there's nothing missing in the test set for this feature, skip
            if test_mask.sum() == 0:
                continue
            test = subset.loc[test_mask]

            # make sure to fill in values in the scoring set where there may
            # be missing values with the tmp_fill
            test = test.where(~pd.isnull(test), tmpfill)

            # generate predictions
            preds = model.predict(test)

            # impute!
            X.loc[test_mask, k] = preds

        return dataframe_or_array(X, self.as_df)


class BaggedRegressorImputer(_BaseBaggedImputer):
    """Impute a dataset using BaggingRegressor models.

    Fit bagged regressor models for each of the impute columns in order
    to impute the missing values.

    Parameters
    ----------
    cols : array_like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the transformer will be ``fit``
        on the entire frame. Note that the transformation will also only
        apply to the specified columns, and any other non-specified
        columns will still be present after transformation.

    predictors : array_like, shape=(n_features,), optional (default=None)
        The names of the columns on which to build the bagging models.
        If not specified, the models will be built on all predictors.

    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    tmp_fill : int, float, optional (default=-999.)
        Some predictors may have missing values in them. This is the value
        to use as a placeholder for training and scoring bagging models for
        the fit and score procedure.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.
        Since most skoot transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.
    """
    def __init__(self, cols=None, predictors=None, base_estimator=None,
                 n_estimators=10, max_samples=1.0, max_features=1.0,
                 bootstrap=True, bootstrap_features=False, n_jobs=1,
                 random_state=None, verbose=0, tmp_fill=-999., as_df=True):

        super(BaggedRegressorImputer, self).__init__(
            imputer_class=BaggingRegressor, cols=cols, predictors=predictors,
            base_estimator=base_estimator, n_estimators=n_estimators,
            max_samples=max_samples, max_features=max_features,
            bootstrap=bootstrap, bootstrap_features=bootstrap_features,
            n_jobs=n_jobs, random_state=random_state, verbose=verbose,
            tmp_fill=tmp_fill, as_df=as_df)


class BaggedClassifierImputer(_BaseBaggedImputer):
    """Impute a dataset using BaggingClassifier models.

    Fit bagged classifier models for each of the impute columns in order
    to impute the missing values.

    Parameters
    ----------
    cols : array_like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the transformer will be ``fit``
        on the entire frame. Note that the transformation will also only
        apply to the specified columns, and any other non-specified
        columns will still be present after transformation.

    predictors : array_like, shape=(n_features,), optional (default=None)
        The names of the columns on which to build the bagging models.
        If not specified, the models will be built on all predictors.

    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    tmp_fill : int, float, optional (default=-999.)
        Some predictors may have missing values in them. This is the value
        to use as a placeholder for training and scoring bagging models for
        the fit and score procedure.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.
        Since most skoot transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.
    """
    def __init__(self, cols=None, predictors=None, base_estimator=None,
                 n_estimators=10, max_samples=1.0, max_features=1.0,
                 bootstrap=True, bootstrap_features=False, n_jobs=1,
                 random_state=None, verbose=0, tmp_fill=-999., as_df=True):

        super(BaggedClassifierImputer, self).__init__(
            imputer_class=BaggingClassifier, cols=cols, predictors=predictors,
            base_estimator=base_estimator, n_estimators=n_estimators,
            max_samples=max_samples, max_features=max_features,
            bootstrap=bootstrap, bootstrap_features=bootstrap_features,
            n_jobs=n_jobs, random_state=random_state, verbose=verbose,
            tmp_fill=tmp_fill, as_df=as_df)
