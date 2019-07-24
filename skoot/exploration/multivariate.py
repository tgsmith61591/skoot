# -*- coding: utf-8 -*-

from ..utils.validation import check_dataframe
from ..utils.dataframe import get_continuous_columns
from .univariate import fisher_pearson_skewness, kurtosis

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
        * Kurtosis

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

    Notes
    -----
    The skewness of a normal distribution is zero, and symmetric data should
    exhibit a skewness near zero. Positive values for skewness indicate the
    data is skewed right, and negative indicate they're skewed left. If the
    data are multi-modal, this may impact the sign of the skewness.

    Examples
    --------
    >>> import skoot
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=20,
    ...                            n_informative=12, random_state=1)
    >>> X = pd.DataFrame.from_records(X[:, :5],
    ...                               columns=['a', 'b', 'c', 'd', 'e'])
    >>> # Make one into a binary column
    >>> X['d'] = (np.random.RandomState(1).rand(X.shape[0]) > 0.9).astype(int)
    >>> skoot.summarize(X)
                          a         b         c         d         e
    Mean          -1.036419 -0.382853 -0.007993       NaN  0.394417
    Median        -0.968732 -0.382114 -0.047757       NaN  0.283779
    Max            4.559433  9.863773  2.991107       NaN  7.344063
    Min           -6.147430 -8.301872 -2.679137       NaN -5.866428
    Variance       3.324646  5.832246  0.985764       NaN  3.938836
    Skewness      -0.059496  0.148757  0.121908       NaN  0.021251
    Kurtosis       0.069795 -0.040619 -0.098477       NaN -0.187570
    Least Freq.         NaN       NaN       NaN      (1,)       NaN
    Most Freq.          NaN       NaN       NaN      (0,)       NaN
    Class Balance       NaN       NaN       NaN  0.113586       NaN
    Num Levels          NaN       NaN       NaN         2       NaN
    Arity               NaN       NaN       NaN     0.002       NaN
    Missing        0.000000  0.000000  0.000000         0  0.000000

    References
    ----------
    .. [1] Measures of Skewness and Kurtosis
           https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm
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
                             "Variance", "Skewness", "Kurtosis"]
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
        #   * Kurtosis
        #
        # We can largely vectorize each of these over the axis...
        means = continuous.mean().values
        medians = continuous.median().values
        maxes = continuous.max().values
        mins = continuous.min().values
        var = continuous.var().values

        # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm
        fp_skew = continuous.apply(fisher_pearson_skewness).values
        kurt = continuous.apply(kurtosis).values
        stats = pd.DataFrame.from_records(
            data=[means, medians, maxes, mins, var, fp_skew, kurt],
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
            # there might be ties, so use masks to determine all classes
            # that are least/most populated & return those as tuples
            least_pop = tuple(idcs[values == values[-1]])
            most_pop = tuple(idcs[values == values[0]])

            # only care about the ratio of the LEAST populous class to the most
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
