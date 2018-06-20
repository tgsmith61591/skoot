.. skoot documentation master file, created by
   sphinx-quickstart on Tue Apr 17 19:10:31 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

============================================
Skoot: Accelerate your data science workflow
============================================

Skoot's aim is to expedite and automate away many of the common pain points
data scientists experience as they work through their exploratory data analysis
& data cleansing/preparation stages. It does so by wrapping and augmenting
useful functions and transformers in scikit-learn, adapting them for use with
Pandas, as well as by providing its own custom transformer classes to solve
common problems that typically demand bespoke solutions.

.. raw:: html

   <br/>

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

If you have a common data preparation or transformation task you feel could be
written into a transformer, please consider :ref:`contrib`!

Two minutes to model-readiness
------------------------------

Real world data is nasty. Most data scientists spend the majority of their time
tackling data cleansing tasks. With skoot, we can automate away so much of the
bespoke hacking solutions that consume data scientists' time.

In this example, we'll examine a common dataset (the
`adult dataset <https://archive.ics.uci.edu/ml/datasets/Adult>`_ from the UCI
machine learning repo) that requires significant pre-processing.

.. code-block:: python

   from skoot.datasets import load_adult_df
   from skoot.feature_selection import FeatureFilter
   from skoot.decomposition import SelectivePCA
   from skoot.preprocessing import DummyEncoder
   from skoot.utils.dataframe import get_numeric_columns
   from skoot.utils.dataframe import get_categorical_columns
   from sklearn.model_selection import train_test_split
   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score

   # load the dataset with the skoot-native loader & split it
   adult = load_adult_df(tgt_name="target")
   y = adult.pop("target")
   X_train, X_test, y_train, y_test = train_test_split(
       adult, y, random_state=42, test_size=0.2)

   # get numeric and categorical feature names
   num_cols = get_numeric_columns(X_train).columns
   obj_cols = get_categorical_columns(X_train).columns

   # remove the education-num from the num_cols since we're going to remove it
   num_cols = num_cols[~(num_cols == "education-num")]

   # build a pipeline
   pipe = Pipeline([
       # drop out the ordinal level that's otherwise equal to "education"
       ("dropper", FeatureFilter(cols=["education-num"])),

       # decompose the numeric features with PCA
       ("pca", SelectivePCA(cols=num_cols)),

       # dummy encode the categorical features
       ("dummy", DummyEncoder(cols=obj_cols, handle_unknown="ignore")),

       # and a simple classifier class
       ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
   ])

   pipe.fit(X_train, y_train)

   # produce predictions
   preds = pipe.predict(X_test)
   print("Test accuracy: %.3f" % accuracy_score(y_test, preds))



.. toctree::
   :maxdepth: 2
   :hidden:

   API Reference <./modules/classes.rst>
   Examples <./auto_examples/index.rst>
   User Guide <./user_guide.rst>


.. raw:: html

   <br/>

Indices and tables
==================

To search for specific sections or class documentation, visit the index.

* :ref:`genindex`
