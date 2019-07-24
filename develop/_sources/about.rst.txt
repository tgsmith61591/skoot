.. _about:

=================
About the project
=================

Skoot is designed to provide as much flexibility as possible while offering
implementations to common challenges, such as categorical & model-based
imputation transformers, transformers to rectify skewness (i.e., box-cox &
Yeo-Johnson transformations), as well as many wrappers to scikit-learn
transformers that enable applications to selected columns only. Every
transformer in skoot is designed for maximum flexibility and to minimize
impact on existing pipelines. Each transformer has been tested to function in
the scope of scikit-learn pipelines and grid searches, and offers the same
persistence model.

.. raw:: html

   <br/>

In addition, skoot provides transformer profiling utilities to help you
identify bottlenecks, determine whether a model will perform sufficiently
in production, and help you make any other performance-based considerations.

|

.. code-block:: python

    from skoot.preprocessing import BoxCoxTransformer
    from skoot.preprocessing import SelectiveStandardScaler
    from skoot.utils.profiling import profile_estimator
    from sklearn.datasets import load_iris
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    # load data
    X, y = load_iris(return_X_y=True)

    # fit a pipeline
    pipe = Pipeline([
        ('transform', BoxCoxTransformer(n_jobs=2)),
        ('scale', SelectiveStandardScaler(cols=[0, 1])),
        ('clf', LogisticRegression())
    ]).fit(X, y)

    # profile
    print(profile_estimator(pipe))

Which yields::

    [('transform', (('fit_time_', 0.1410069465637207),)), ('scale', (('fit_time_', 0.0011677742004394531),))]
