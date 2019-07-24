# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# The over-sampling balancer

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_random_state
from sklearn.utils import safe_indexing

from .base import _validate_X_y_ratio_classes, _reorder
from ..utils import safe_vstack
import numpy as np

__all__ = [
    'over_sample_balance'
]


def over_sample_balance(X, y, balance_ratio=0.2, random_state=None,
                        shuffle=True):
    """Over sample a minority class to a specified ratio.

    One strategy for balancing data is to over-sample the minority class
    until it is represented at the prescribed ``balance_ratio``. While there
    is significant literature to show that this is not the best technique,
    and can sometimes lead to over-fitting, there are instances where it can
    work well.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The training array. Samples from this array will be resampled with
        replacement for the minority class.

    y : array-like, shape (n_samples,)
        Training labels corresponding to the samples in ``X``.

    balance_ratio : float, optional (default=0.2)
        The minimum acceptable ratio of ``$MINORITY_CLASS : $MAJORITY_CLASS``
        representation, where 0 < ``ratio`` <= 1

    random_state : int, None or numpy RandomState, optional (default=None)
        The seed to construct the random state to generate random selections.

    shuffle : bool, optional (default=True)
        Whether to shuffle the output.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, random_state=42,
    ...                            n_classes=2, weights=[0.99, 0.01])
    >>> X_bal, y_bal = over_sample_balance(X, y, balance_ratio=0.2,
    ...                                    random_state=42)
    >>> ratio = round((y_bal == 1).sum() / float((y_bal == 0).sum()), 1)
    >>> assert ratio == 0.2, ratio

    Note that the count of samples is now greater than it initially was:

    >>> assert X_bal.shape[0] > 1000
    """
    random_state = check_random_state(random_state)

    # validate before copying arrays around...
    X, y, n_classes, present_classes, \
        counts, majority_label, target_count = \
        _validate_X_y_ratio_classes(X, y, balance_ratio)

    # encode y, in case they are not numeric (we need them to be for np.ones)
    le = LabelEncoder()
    le.fit(present_classes)
    y_transform = le.transform(y)  # make numeric

    # we'll vstack/concatenate to these
    out_X, out_y = X.copy(), y_transform.copy()

    # iterate the present classes
    for label in present_classes:
        if label == majority_label:
            continue

        # get the transformed label
        label_transform = le.transform([label])[0]

        while True:
            # use the out_X, out_y copies. Since we're oversamping,
            # it doesn't matter if we're drawing from the out_X matrix.
            # also, this way we can better keep track of how many we've drawn.
            mask = out_y == label_transform
            n_req = target_count - mask.sum()

            # terminal case
            if n_req == 0:
                break

            # draw a sample, take first n_req:
            idcs = np.arange(out_X.shape[0])[mask]  # get the idcs, mask them
            sample = safe_indexing(out_X,
                                   random_state.permutation(idcs)[:n_req])

            # vstack
            out_X = safe_vstack(out_X, sample)

            # concatenate. Use sample length, since it might be < n_req
            out_y = np.concatenate([
                out_y, np.ones(sample.shape[0],
                               dtype=np.int16) * label_transform])

    return _reorder(out_X, le.inverse_transform(out_y), random_state, shuffle)
