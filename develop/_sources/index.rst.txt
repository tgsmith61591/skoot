.. skoot documentation master file, created by
   sphinx-quickstart on Tue Apr 17 19:10:31 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

============================================
Skoot: Accelerate your data science workflow
============================================

.. raw:: html

   <!-- Block section -->
   <script src="_static/js/jquery.min.js"></script>

   <a href="https://circleci.com/gh/tgsmith61591/skoot"><img alt="Build status" src="https://circleci.com/gh/tgsmith61591/skoot.svg?style=svg" /></a>
   <a href="https://codecov.io/gh/tgsmith61591/skoot"><img alt="Coverage" src="https://codecov.io/gh/tgsmith61591/skoot/branch/master/graph/badge.svg" /></a>
   <a href="https://github.com/tgsmith61591/skoot"><img id="nutrition" alt="gluten free" src="https://img.shields.io/badge/gluten_free-100%25-brightgreen.svg" /></a>

Skoot's aim is to expedite and automate away many of the common pain points
data scientists experience as they work through their exploratory data analysis
& data cleansing/preparation stages. It does so by wrapping and augmenting
useful functions and transformers in scikit-learn, adapting them for use with
Pandas, as well as by providing its own custom transformer classes to solve
common problems that typically demand bespoke solutions.

|

Example: Two minutes to model-readiness
---------------------------------------

Real world data is nasty. Most data scientists spend the majority of their time
tackling data cleansing tasks. With skoot, we can automate away so much of the
bespoke hacking solutions that consume data scientists' time.

|

In this example, we'll examine a common dataset (the
`adult dataset <https://archive.ics.uci.edu/ml/datasets/Adult>`_ from the UCI
machine learning repo) that requires significant pre-processing, and show how
Skoot enables us to quickly clean up data and prepare it for modeling.

|

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


Voíla! The entire pre-processing and modeling process is achievable in a single pipeline.

.. raw:: html

   <br/>


If you have a common data preparation or transformation task you feel could be
written into a transformer, please consider :ref:`contrib`!

.. toctree::
   :maxdepth: 2
   :hidden:

   API Reference <./modules/classes.rst>
   Examples <./auto_examples/index.rst>
   User Guide <./user_guide.rst>


.. raw:: html

   <br/>

Quick refs, indices and tables
==============================

Helpful quickstart sections:

* :ref:`about`
* :ref:`setup`
* :ref:`building_on_unix`
* :ref:`building_on_windows`
* :ref:`testing`
* :ref:`contrib`
* :ref:`api_ref`

To search for specific sections or class documentation, visit the index.

* :ref:`genindex`
