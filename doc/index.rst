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
