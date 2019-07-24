# -*- coding: utf-8 -*-
#
# Profiling utilities
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from sklearn.pipeline import Pipeline

__all__ = [
    'profile_estimator'
]


def _profile_times(est):
    # Get the profile times from an estimator, if they exist
    return tuple((nm, getattr(est, nm))
                 for nm in dir(est) if nm.endswith("_time_")
                 )


def profile_estimator(estimator):
    """Profile the timed functions of an estimator.

    Get a list of runtimes for estimators that have used the
    ``timed_instance_method`` decorator. This is useful for diagnosing
    bottlenecks in a pipeline.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to profile.

    Returns
    -------
    method_times : list or tuple
        The list of method times.

    Examples
    --------
    >>> from skoot.datasets import load_iris_df
    >>> from skoot.preprocessing import SelectiveStandardScaler
    >>> iris = load_iris_df(include_tgt=False)
    >>> scl = SelectiveStandardScaler().fit(iris)
    >>> profile_estimator(scl)  # doctest: +SKIP
    (('fit_time_', 0.001055002212524414),)

    Profiling also works on pipelines:
    >>> from sklearn.pipeline import Pipeline
    >>> from skoot.preprocessing.skewness import YeoJohnsonTransformer
    >>> from skoot.feature_selection import MultiCorrFilter
    >>> pipe = Pipeline([('scl', SelectiveStandardScaler()),
    ...                  ('yj', YeoJohnsonTransformer()),
    ...                  ('mcf', MultiCorrFilter())
    ... ]).fit(iris)
    >>> profile_estimator(pipe)  # doctest: +SKIP
    [('scl', (('fit_time_', 0.0016548633575439453),)),
     ('yj', (('fit_time_', 0.0282437801361084),))]

    Notes
    -----
    This will only work if the attribute names provided for profiling
    end with "_time_". All skoot estimators, for instance, save the attribute
    for the ``fit`` method as "fit_time_", but a custom transformer that uses
    a different suffix may not be captured in the profiling process.
    """
    if isinstance(estimator, Pipeline):
        method_times = []
        for step_name, step_cls in estimator.steps:
            times = profile_estimator(step_cls)
            # Only append if it was profiled
            if times:
                method_times.append((step_name, times))
        return method_times

    # otherwise if it's not a pipeline, just return the profile times
    return _profile_times(estimator)
