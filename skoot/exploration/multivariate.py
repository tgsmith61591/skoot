# -*- coding: utf-8 -*-

from __future__ import absolute_import

from ..utils.validation import check_dataframe
from ..utils.dataframe import get_continuous_columns

import pandas as pd
import numpy as np

__all__ = [
    'summarize'
]


def summarize(X):
    """Summarize a dataframe.

    Create a more in-depth summary of a dataframe than ``pd.describe`` will
    give you. This includes details on skewness, arity (for categorical
    features) and more. For continuous features (floats), this summary
    computes:

        * Mean
        * Median
        * Max
        * Min
        * Variance
        * Skewness

    For categorical features:

        * Least frequent class
        * Most frequent class
        * Class balance (n_least_freq / n_most_freq; higher is better)
        * Num Levels
        * Arity (n_unique_classes / n_samples; lower is better)

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        The input data. Can be comprised of categorical or continuous data,
        and will be cast to pandas DataFrame for the computations.

    Returns
    -------
    stats : DataFrame
        The summarized dataframe

    Examples
    --------
    >>> import skoot
    >>> from skoot.datasets import load_iris_df
    >>> skoot.summarize(load_iris_df())
                   sepal length (cm)  sepal width (cm)  petal length (cm)  \
    Mean                    5.843333          3.054000           3.758667
    Median                  5.800000          3.000000           4.350000
    Max                     7.900000          4.400000           6.900000
    Min                     4.300000          2.000000           1.000000
    Variance                0.685694          0.188004           3.113179
    Skewness                0.308641          0.327401          -0.268999
    Least Freq.                  NaN               NaN                NaN
    Most Freq.                   NaN               NaN                NaN
    Class Balance                NaN               NaN                NaN
    Num Levels                   NaN               NaN                NaN
    Arity                        NaN               NaN                NaN
    Missing                 0.000000          0.000000           0.000000

                   petal width (cm)  species
    Mean                   1.198667      NaN
    Median                 1.300000      NaN
    Max                    2.500000      NaN
    Min                    0.100000      NaN
    Variance               0.582414      NaN
    Skewness              -0.102906      NaN
    Least Freq.                 NaN     0.00
    Most Freq.                  NaN     2.00
    Class Balance               NaN     1.00
    Num Levels                  NaN     3.00
    Arity                       NaN     0.02
    Missing                0.000000     0.00
    """
    X, cols = check_dataframe(X, cols=None, assert_all_finite=False)
    n_samples = X.shape[0]

    # There are some operations we'll compute on all variable types:
    #   * n_missing
    n_missing = X.isnull().sum().values

    # compute stats on each continuous col
    continuous = get_continuous_columns(X)
    scont_cols = set(continuous.columns.tolist())
    other_cols = [c for c in cols
                  if c not in scont_cols]

    continuous_stats_cols = ["Mean", "Median", "Max", "Min",
                             "Variance", "Skewness"]
    categorical_stats_cols = ["Least Freq.", "Most Freq.",
                              "Class Balance", "Num Levels",
                              "Arity"]
    if scont_cols:
        # For each continuous feature, compute the following:
        #   * mean
        #   * median
        #   * max
        #   * min
        #   * variance
        #   * Fisher-Pearson skewness
        #
        # We can largely vectorize each of these over the axis...
        means = continuous.mean().values
        medians = continuous.median().values
        maxes = continuous.max().values
        mins = continuous.min().values
        var = continuous.var().values
        s = np.sqrt(var)  # used just internally

        # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm
        skew_numer = (((continuous - means) ** 3.) / continuous.shape[0]).sum()
        fp_skew = (skew_numer / (s ** 3.)).values
        stats = pd.DataFrame.from_records(
            data=[means, medians, maxes, mins, var, fp_skew],
            columns=continuous.columns.tolist(),
            index=continuous_stats_cols).T

        # for each column in the categorical statistics (that we won't be
        # computing over these features), set as NaN
        for stat in categorical_stats_cols:
            stats[stat] = np.nan
        stats = stats.T

    # otherwise we need "stats" in the namespace
    else:
        stats = None

    # Now for each categorical feature, compute the following:
    #   * Least populated class
    #   * Most populated class
    #   * Ratio of most-populous class: least-populous
    #   * Arity (num unique factor levels/num samples)
    def categ_summary(feature):
        vc = feature.value_counts()
        idcs, values = vc.index.values, vc.values
        n_levels = idcs.shape[0]

        # if there is only one value we have to return it as both
        # the most populous and the least-populous...
        if n_levels == 1:
            least_pop = most_pop = idcs[0]
            ratio = 1.
        else:
            least_pop = idcs[-1]
            most_pop = idcs[0]
            ratio = values[-1] / float(values[0])

        arity = n_levels / float(n_samples)
        return least_pop, most_pop, ratio, n_levels, arity

    # apply the categorical function
    if other_cols:
        # Compute and then transpose so we can tack this onto
        # the continuous statistics
        categ_results = pd.DataFrame.from_records(
            data=np.array(
                X[other_cols].apply(categ_summary)
                             .values
                             .tolist()).T,
            columns=other_cols,
            index=categorical_stats_cols).T

        # for each stat in the continuous stats (that we won't compute for
        # these features), set to np.nan
        for stat in continuous_stats_cols:
            categ_results[stat] = np.nan

        # select in this order to make sure our index will be in the right
        # order after we transpose back
        categ_results = categ_results[continuous_stats_cols +
                                      categorical_stats_cols].T

        # cbind to stats if it's defined, otherwise just rename it
        if stats is not None:
            stats = pd.concat([stats, categ_results], axis=1)
        else:
            stats = categ_results

    # Make sure we're in order
    stats = stats[cols]
    stats.loc["Missing"] = n_missing
    return stats
