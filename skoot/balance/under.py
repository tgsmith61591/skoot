# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# The under-sampling balancer

from sklearn.utils.validation import check_random_state
import numpy as np

from .base import _validate_X_y_ratio_classes, _reorder
from ..utils.dataframe import safe_drop_samples

__all__ = [
    'under_sample_balance'
]


def under_sample_balance(X, y, balance_ratio=0.2, random_state=None,
                         shuffle=True):
    """Under sample the majority class to a specified ratio.

    One strategy for balancing data is to under-sample the majority
    class until it is represented at the prescribed ``balance_ratio``.
    This can be effective in cases where the training set is already
    quite large, and diminishing its size may not prove detrimental.

    The under-sampling procedure behaves differently than the over-sampling
    technique: its objective is only to under-sample the *majority* class,
    and will down-sample it until the *second-most* represented class is
    present at the prescribed ratio.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The training array. Samples from this array that correspond to the
        majority class will be omitted until the minority class is represented
        at the ``balance_ratio``.

    y : array-like, shape (n_samples,)
        Training labels corresponding to the samples in ``X``.

    balance_ratio : float, optional (default=0.2)
        The minimum acceptable ratio of ``$MINORITY_CLASS : $MAJORITY_CLASS``
        representation, where 0 < ``ratio`` <= 1

    random_state : int, None or numpy RandomState, optional (default=None)
        The seed to construct the random state to generate random selections.

    shuffle : bool, optional (default=True)
        Whether to shuffle the output.

    Notes
    -----
    You should only use the under sampling method when you have lots of data
    and can afford to lose some training samples. Moreover, using this in
    conjunction with a high variance modeling method can pose a higher risk of
    over-fitting, since they typically require more data.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, random_state=42,
    ...                            n_classes=2, weights=[0.99, 0.01])
    >>> X_bal, y_bal = under_sample_balance(X, y, balance_ratio=0.2,
    ...                                     random_state=42)
    >>> ratio = round((y_bal == 1).sum() / float((y_bal == 0).sum()), 1)
    >>> assert ratio == 0.2, ratio

    Note that the number of samples in the data is now lower than it
    initially was:

    >>> assert X_bal.shape[0] < 1000
    """
    random_state = check_random_state(random_state)

    # validate before copying arrays around...
    X, y, n_classes, present_classes, \
        counts, majority_label, _ = \
        _validate_X_y_ratio_classes(X, y, balance_ratio)

    # get the second-most populous count, compute target
    sorted_counts = np.sort(counts)
    if sorted_counts[-1] == sorted_counts[-2]:  # corner case
        return _reorder(X, y, random_state, shuffle)

    target_count = max(int(sorted_counts[-2] / balance_ratio), 1)

    # select which rows gotta go...
    idcs = np.arange(X.shape[0])
    mask = (y == majority_label)  # type: np.ndarray
    remove = random_state.permutation(
        idcs[mask])[:mask.sum() - target_count]  # sum is > target count

    # remove them
    X = safe_drop_samples(X, remove)
    y = np.delete(y, remove)

    # reorder if needed
    return _reorder(X, y, random_state, shuffle)
