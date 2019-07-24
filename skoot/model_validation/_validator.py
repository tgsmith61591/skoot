# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Validator classes for model monitoring. These classes can be stacked after
# transformers in a pipeline in order to ensure the test data distributions
# resemble those of the training data and can provide early warnings for
# covariate shift.

from ..base import BasePDTransformer
from ..utils.validation import check_dataframe, type_or_iterable_to_col_mapping
from ..utils.dataframe import get_continuous_columns, dataframe_or_array
from ..utils.metaestimators import timed_instance_method
from ..exceptions import ValidationWarning

import six
from sklearn.utils.validation import check_is_fitted

from scipy.stats import ttest_ind_from_stats

import numpy as np
import warnings
import collections

from abc import abstractmethod, ABCMeta

__all__ = [
    "CustomValidator",
    "DistHypothesisValidator"
]


def _passthrough(_):
    # Default for when user does not set a function in the custom validator
    # (design choice: should we FORCE a value there?)
    return True


def _compute_stats(v, continuous):
    # determine if this needs a T-test or freq test
    if continuous:
        # Compute the t-test stats
        mean = np.nanmean(v)
        std = np.nanstd(v)
        return mean, std, v.shape[0]

    # otherwise it's categorical or integer (probably ordinal)
    else:
        unique_levels, counts = np.unique(v, return_counts=True)
        return unique_levels, counts, v.shape[0]


class _BaseValidator(six.with_metaclass(ABCMeta, BasePDTransformer)):
    """Base validator class."""
    def __init__(self, cols, as_df, action):
        super(_BaseValidator, self).__init__(
            cols=cols, as_df=as_df)

        self.action = action

    @abstractmethod
    def _is_as_expected(self, index, feature_name, feature):
        """Validate the test feature.

        Abstract method to compute the validation statistic. Should
        return a boolean indicating whether the feature adheres to the
        expectation.
        """

    def transform(self, X):
        """Validate the features in the test dataframe.

        This method will apply the validation test over each prescribed
        feature, and raise or warn appropriately.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to validate. The operation will
            be applied to a copy of the input data, and the result
            will be returned.

        Returns
        -------
        X : pd.DataFrame or np.ndarray, shape=(n_samples, n_features)
            The operation is applied to a copy of ``X``,
            and the result set is returned.
        """
        check_is_fitted(self, "fit_cols_")
        X, _ = check_dataframe(X, cols=self.cols)  # X is a copy now
        cols = self.fit_cols_  # assigned in the "fit" method

        for i, c in enumerate(cols):
            v = X[c].values  # get the feature

            # determine whether it's valid
            if not self._is_as_expected(i, c, v):
                msg = "Feature %s does not match expectation as set by %s" \
                      % (c, self.__class__.__name__)

                # if it's error or warn, we alert the user otherwise we do not.
                if self.action == "raise":
                    raise ValueError(msg)
                elif self.action == "warn":
                    warnings.warn(msg, ValidationWarning)

        # just return X if we get here
        return dataframe_or_array(X, self.as_df)


class CustomValidator(_BaseValidator):
    """Validate test features given custom functions.

    Apply test set validator behavior over custom functions. This can be
    especially useful in cases where a feature should never exhibit values
    within a certain range (i.e., sensor data).

    Parameters
    ----------
    cols : array-like, shape=(n_features,)
        The names of the columns on which to apply the transformation.
        If ``cols`` is None, will apply to the entire dataframe.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.
        Since most skoot transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.

    func : callable or iterable, optional (default=None)
        The function used to validate the feature. Can be as complex or as
        simple as needed, but must adhere to the following criteria:

          * The signature must accept a single vector
          * The output must be a boolean

        Note also that providing a lambda expression as a function can prove
        to be problematic when it comes time to serialize your class, as
        lambda expressions cannot be serialized via pickle. It's best to
        provide a ``def``-style function or closure.

    action : str or unicode, optional (default="warn")
        The default action for handling validation mismatches. Options include
        "warn", "raise" or "ignore". If ``action`` is "raise", will raise a
        ValueError if mismatched.

    Attributes
    ----------
    func_dict_ : dict
        A dictionary mapping the column names to their respective
        validation function.

    fit_cols_ : list
        The list of column names on which the transformer was fit. This
        is used to validate the presence of the features in the test set
        during the ``transform`` stage.
    """
    def __init__(self, cols=None, as_df=True, func=None, action="warn"):
        super(CustomValidator, self).__init__(
            cols=cols, as_df=as_df, action=action)

        self.func = func

    @timed_instance_method(attribute_name="fit_time_")
    def fit(self, X, y=None):
        """Fit the transformer.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all if ``cols`` is None.

        y : array-like or None, shape=(n_samples,), optional (default=None)
            Pass-through for ``sklearn.pipeline.Pipeline``.
        """
        X, cols = check_dataframe(X, cols=self.cols, assert_all_finite=False)

        # validate the func(tions):
        func = self.func

        # if it's None, make it an identity function of sorts
        if func is None:
            func = _passthrough

        f = type_or_iterable_to_col_mapping(
            cols=cols, param=func, param_name="func",
            permitted_scalar_types=collections.Callable)  # type: dict

        # save the function mapping as the fit param
        self.func_dict_ = f
        self.fit_cols_ = cols

        return self

    def _is_as_expected(self, index, feature_name, feature):
        """Validate the test feature.

        Apply the user-defined custom validation function to the test feature.
        """
        return self.func_dict_[feature_name](feature)


class DistHypothesisValidator(_BaseValidator):
    r"""Validate test distributions using various hypothesis tests.

    The distribution validator learns statistics from the training set and then
    validates that the test set features match their expected distributions.
    This can be useful for model validation tasks where model monitoring needs
    to take place.

    For continuous (float) features, a two-tailed T-test will be applied to
    the test data to ensure it matches the distribution of the training data.
    For categorical (int, object) features, we compare the frequencies of
    different categorical levels within a tolerance of ``alpha``.

    **Note**: this class is NaN-safe, meaning if it is used early in your
    pipeline when you still have NaN values in your features, it will still
    function!

    Parameters
    ----------
    cols : array-like, shape=(n_features,)
        The names of the columns on which to apply the transformation.
        Unlike other BasePDTransformer instances, if ``cols`` is None, it will
        only fit the numerical columns, since statistics such as standard
        deviation cannot be computed on categorical features. For column
        types that are integers or objects, the ratio of frequency for each
        class level will be compared to the expected ratio within a tolerance
        of ``alpha``.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.
        Since most skoot transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.

    alpha : float, optional (default=0.05)
        The :math:`\alpha` value for the T-test or level ratio comparison.
        If the resulting p-value is LESS than ``alpha``, it means that
        we would reject the null hypothesis, and that the variable likely
        follows a different distribution from the training set.

    action : str or unicode, optional (default="warn")
        The default action for handling validation mismatches. Options include
        "warn", "raise" or "ignore". If ``action`` is "raise", will raise a
        ValueError if mismatched.

    categorical_strategy : str, unicode or None, optional (default="ratio")
        How to validate categorical features. Default is "ratio", which will
        compare the ratio of each level's frequency to the overall count of
        samples in the feature within an absolute tolerance of ``alpha``.
        If None, will not perform validation on categorical features.

    Notes
    -----
    This class is NaN-safe, meaning if it is used early in your pipeline
    when you still have NaN values in your features, it will still function.
    This is a double-edge sword, since computing the ``np.nanmean`` on a
    feature of mostly-NaN values will not be very meaningful.

    Attributes
    ----------
    statistics_ : list, shape=(n_features,)
        A list of tuples over the training features. For continuous features::

            (mean, standard_dev, n_obs)

        For categorical features:

            (present_levels, present_counts, n_obs)

    fit_cols_ : list
        The list of column names on which the transformer was fit. This
        is used to validate the presence of the features in the test set
        during the ``transform`` stage.
    """
    def __init__(self, cols=None, as_df=True, alpha=0.05, action="warn",
                 categorical_strategy="ratio"):
        super(DistHypothesisValidator, self).__init__(
            cols=cols, as_df=as_df, action=action)

        self.alpha = alpha
        self.categorical_strategy = categorical_strategy

    @timed_instance_method(attribute_name="fit_time_")
    def fit(self, X, y=None):
        """Fit the transformer.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all if ``cols`` is None.

        y : array-like or None, shape=(n_samples,), optional (default=None)
            Pass-through for ``sklearn.pipeline.Pipeline``.
        """
        X, cols = check_dataframe(X, cols=self.cols, assert_all_finite=False)

        # if self.cols is None, the user might have tried to apply this to
        # every column and some may contain categorical features. So we need
        # to control for that...
        float_cols = set(get_continuous_columns(X).columns.tolist())

        # fit the test statistics over each column
        self.statistics_ = [
            _compute_stats(X[col].values, continuous=col in float_cols)
            for col in cols
        ]

        self.fit_cols_ = cols
        self.continuous_ = float_cols

        return self

    def _is_as_expected(self, index, feature_name, feature):
        """Validate the test feature.

        Compute the test statistic and, for continuous covariates,
        return whether the P-value is GREATER THAN OR EQUAL TO the specified
        alpha value (less than indicates that we would reject the null,
        so >= means it's likely from the same distribution).

        For categorical features, compare the frequencies of each level within
        ``alpha`` tolerance.
        """
        # if it's a continuous feature, we use the T-test
        if feature_name in self.continuous_:
            mean1, std1, nobs1 = self.statistics_[index]
            mean2, std2, nobs2 = _compute_stats(feature, continuous=True)
            _, pval = ttest_ind_from_stats(
                mean1=mean1, std1=std1, nobs1=nobs1,
                mean2=mean2, std2=std2, nobs2=nobs2,
                equal_var=True)  # we expect them to be the same

            return pval >= self.alpha

        # otherwise, we are dealing with categorical features
        else:

            # if we want to use the ratio strategy, do so here:
            if self.categorical_strategy == "ratio":
                exp_levels, exp_counts, n_obs = self.statistics_[index]
                prst_levels, prst_counts, n_test = \
                    _compute_stats(feature, continuous=False)

                # Get the expected ratios & present ratios
                exp_ratios = exp_counts / float(n_obs)
                prst_ratios = prst_counts / float(n_test)
                abs_diff = np.abs(exp_ratios - prst_ratios)

                # if there are any levels in the test set that are NOT in the
                # training set, we have to handle the action. Don't fail since
                # this may be before a user has encoded all levels, so we can
                # just let the transform method handle the action.
                new_lvls = ~np.in1d(prst_levels,
                                    exp_levels)  # type: np.ndarray
                valid = not new_lvls.any()  # dont want new lvls, new = invalid
                abs_mask = abs_diff <= self.alpha  # type: np.ndarray

                return valid and abs_mask.all()

            # if we add more strategies, here's where they'll go...
            return True


# in case nose has an issue here (since "test" is used all over)...
DistHypothesisValidator.__test__ = False
