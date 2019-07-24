# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# The SMOTE balancer

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_random_state
from sklearn.utils import safe_indexing

from .base import _validate_X_y_ratio_classes
from ..utils import safe_vstack, safe_mask_samples

__all__ = [
    'smote_balance'
]


def _perturb(consider_vector, nearest, random_state):
    return (consider_vector - nearest) * \
        random_state.rand(nearest.shape[0], 1) + consider_vector


def _interpolate(consider_vector, nearest, _):
    # the danger here is that if there are not enough samples, we'll
    # interpolate with the same value. Hrmm. Maybe add some entropy? # todo
    return np.average((consider_vector, nearest), axis=0)


# Define after the _perturb and _interpolate functions are defined
STRATEGIES = {
    'perturb': _perturb,
    'interpolate': _interpolate
}


def _nearest_neighbors_for_class(X, label, label_encoder, y_transform,
                                 target_count, random_state, strategy,
                                 n_neighbors, algorithm, leaf_size, p,
                                 metric, metric_params, n_jobs):
    func = STRATEGIES[strategy]

    # transform the label, get the subset
    transformed_label = label_encoder.transform([label])[0]
    X_sub = safe_mask_samples(X, y_transform == transformed_label)

    # make sure it's a numpy array for all of this...
    if isinstance(X_sub, pd.DataFrame):
        X_sub = X_sub.values

    # if the count >= the ratio, skip this label
    count = X_sub.shape[0]
    if count >= target_count:
        return X, y_transform, None

    # define the nearest neighbors model
    model = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm,
                             leaf_size=leaf_size, p=p, metric=metric,
                             metric_params=metric_params, n_jobs=n_jobs)

    # get the observations that map to the transformed label
    amt_required = target_count - count

    # fit the model once, query the tree once. n_neighbors MUST
    # be ONE PLUS n_neighbors, since the zero'th index will always
    # be the index of the observation itself (i.e., obs 0 is its own
    # nearest neighbor).
    model.fit(X_sub)

    # draw the nearest neighbors ONCE. There is an interesting corner
    # case here... sklearn's nearest neighbors will draw the actual
    # observation as its own nearest neighbor so we need to query for k + 1,
    # and remove the first index (handled in the while loop)
    k_neighbors = min(X_sub.shape[0], n_neighbors + 1)
    nearest = model.kneighbors(X_sub, n_neighbors=k_neighbors,
                               return_distance=False)  # type: np.ndarray
    indices = np.arange(count)

    # append the labels to y_transform - do this once to avoid the
    # overhead of repeated concatenations
    y_transform = np.concatenate([
        y_transform,
        np.ones(amt_required, dtype=np.int16) * transformed_label])

    # sample while under the requisite ratio
    while amt_required > 0:
        # randomly select some observations. since each selection will produce
        # n_neighbors synthetic points, take the first
        # amt_required // n_neighbors
        draw_count = max(1, int(round(amt_required / n_neighbors)))
        random_indices = random_state.permutation(indices)[:draw_count]

        # select the random sample, get nearest neighbors for the
        # sample indices. SHUFFLE, because of the next draw step.
        synthetic = random_state.permutation(np.vstack([
            func(X_sub[consideration, :],

                 # take 0th out (the observation vector under consideration):
                 X_sub[nearest[consideration][1:], :],
                 random_state)

            # using the language of smote... ("vector under consideration")
            for consideration in random_indices
        ]))

        # append to X. Since the round up earlier might cause a slight
        # error in count, make sure to truncate the synthetically-drawn
        # examples to amt_required
        if isinstance(X, pd.DataFrame):
            synthetic = pd.DataFrame.from_records(
                synthetic, columns=X.columns).iloc[:amt_required]
        else:
            synthetic = synthetic[:amt_required]
        X = safe_vstack(X, synthetic)

        # determine whether we need to recurse for this class (if
        # there were too few samples)
        amt_required -= synthetic.shape[0]

    return X, y_transform, model


def smote_balance(X, y, return_estimators=False, balance_ratio=0.2,
                  strategy='perturb', n_neighbors=5, algorithm='kd_tree',
                  leaf_size=30, p=2, metric='minkowski', metric_params=None,
                  n_jobs=1, random_state=None, shuffle=True):
    """Balance a dataset using SMOTE.

    Synthetic Minority Oversampling TEchnique (SMOTE) is a class balancing
    strategy that samples the k-Nearest Neighbors from each minority class,
    perturbs them with a random value between 0 and 1, and adds the difference
    between the original observation and the perturbation to the original
    observation to generate a synthetic observation. This is repeated until
    the minority classes are represented at a prescribed ratio to the
    majority class.

    Alternative methods involve interpolating the distance between the
    original observation and the nearest neighbors using either the median
    or the mean. This strategy can be set using the ``strategy`` arg (one of
    'perturb' or 'interpolate').

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The training array. Samples from the minority class(es) in this array
        will be interpolated until they are represented at ``balance_ratio``.

    y : array-like, shape (n_samples,)
        Training labels corresponding to the samples in ``X``.

    return_estimators : bool, optional (default=False)
        Whether or not to return the dictionary of fit
        :class:``sklearn.neighbors.NearestNeighbors`` instances. If True,
        the return value will be a tuple, with the first index being the
        balanced ``X`` matrix, the second index being the ``y`` values, and
        the third index being a dictionary of the fit estimators. If False,
        the return value is simply a tuple of the balanced ``X`` matrix and
        the corresponding labels.

    balance_ratio : float, optional (default=0.2)
        The minimum acceptable ratio of ``$MINORITY_CLASS : $MAJORITY_CLASS``
        representation, where 0 < ``ratio`` <= 1

    strategy : str, optional (default='perturb')
        The strategy used to construct synthetic examples from existing
        examples. The original SMOTE paper suggests a strategy by which a
        random value between 0 and 1 scales the difference between the
        nearest neighbors, and the difference is then added to the original
        vector. This is the default strategy, 'perturb'. Valid strategies
        include:

          * 'perturb' - a random value between 0 and 1 scales the difference
             between the nearest neighbors, and the difference is then added
             to the original vector.

          * 'interpolate' - the ``interpolation_method`` ('mean' or 'median')
            of the nearest neighbors constitutes the synthetic example.

    n_neighbors : int, optional (default=5)
        Number of neighbors to use by default for ``kneighbors`` queries.
        This parameter is passed to each respective ``NearestNeighbors call.``

    algorithm : str or unicode, optional (default='kd_tree')
        Algorithm used to compute the nearest neighbors. One of
        {'auto', 'ball_tree', 'kd_tree', 'brute'}:

        - 'ball_tree' will use ``sklearn.neighbors.BallTree``
        - 'kd_tree' will use ``sklearn.neighbors.KDtree``
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to ``fit`` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force. This parameter
        is passed to each respective ``NearestNeighbors`` call.

    leaf_size : int, optional (default=30)
        Leaf size passed to ``BallTree`` or ``KDTree``.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    p : integer, optional (default=2)
        Parameter for the Minkowski metric from
        ``sklearn.metrics.pairwise.pairwise_distances``. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric : string or callable, optional (default='minkowski')
        Metric to use for distance computation. Any metric from scikit-learn
        or ``scipy.spatial.distance`` can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.
        Distance matrices are not supported.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean',
          'l1', 'l2', 'manhattan']
        - from ``scipy.spatial.distance``: ['braycurtis',
          'canberra', 'chebyshev', 'correlation', 'dice', 'hamming',
          'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
          'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
          'sokalsneath', 'sqeuclidean', 'yule']

        See the documentation for ``scipy.spatial.distance`` for details
        on these metrics.

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
        Affects only ``kneighbors`` and ``kneighbors_graph`` methods.

    shuffle : bool, optional (default=True)
        Whether to shuffle the output.

    random_state : int or None, optional (default=None)
        The seed to construct the random state to generate random selections.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, random_state=42,
    ...                            n_classes=2, weights=[0.99, 0.01])
    >>> X_bal, y_bal = smote_balance(X, y, balance_ratio=0.2, random_state=42)
    >>> ratio = round((y_bal == 1).sum() / float((y_bal == 0).sum()), 1)
    >>> assert ratio == 0.2, ratio

    Note that the count of samples is now greater than it initially was:

    >>> assert X_bal.shape[0] > 1000

    References
    ----------
    .. [1] N. Chawla, K. Bowyer, L. Hall, W. Kegelmeyer,
           "SMOTE: Synthetic Minority Over-sampling Technique"
           https://www.jair.org/media/953/live-953-2037-jair.pdf
    """
    # validate the cheap stuff before copying arrays around...
    X, y, n_classes, present_classes, \
        counts, majority_label, target_count = \
        _validate_X_y_ratio_classes(X, y, balance_ratio)

    # validate n_neighbors is at least one
    if n_neighbors < 1:
        raise ValueError('n_neighbors must be at least 1')
    if strategy not in STRATEGIES:
        raise ValueError('strategy must be one of %r' % STRATEGIES)

    # get the random state
    random_state = check_random_state(random_state)

    # encode y, in case they are not numeric (we need them to be for np.ones)
    le = LabelEncoder()
    le.fit(present_classes)
    y_transform = le.transform(y)  # make numeric

    # get the nearest neighbor models
    models = dict()
    for label in present_classes:

        # skip the pass to the method and continue if label is majority
        if label == majority_label:
            models[label] = None
            continue

        # X and y_transform will progressively be updated throughout the runs
        X, y_transform, models[label] = \
            _nearest_neighbors_for_class(
                X=X, label=label, label_encoder=le,
                y_transform=y_transform, target_count=target_count,
                random_state=random_state, strategy=strategy,
                n_neighbors=n_neighbors, algorithm=algorithm,
                leaf_size=leaf_size, p=p, metric=metric,
                metric_params=metric_params, n_jobs=n_jobs)

    # now that X, y_transform have been assembled, inverse_transform
    # the y_t back to its original state:
    y = le.inverse_transform(y_transform)

    # finally, shuffle both and return
    output_order = np.arange(X.shape[0])
    if shuffle:
        output_order = random_state.permutation(output_order)

    # select the output order, then return
    X, y = safe_indexing(X, output_order), y[output_order]
    if return_estimators:
        return X, y, models
    return X, y
