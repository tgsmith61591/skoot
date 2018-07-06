.. _datasets:

======================
The datasets submodule
======================

.. currentmodule:: skoot.datasets

The datasets module, much like ``sklearn.datasets``, contains loadable
toy datasets for use in prototyping algorithms and transformers. Unlike sklearn,
the output of skoot's datasets are Pandas DataFrames.

|

The general interface for loading an example dataframe is shared between loaders:

.. code-block:: python

    # There are several dataset loaders: adult, iris, boston, breast_cancer
    from skoot.datasets import load_adult_df
    df = load_adult_df(include_tgt=True, tgt_name="target")


See the following references for examples on how to use the datasets module, or
specific documentation around each loader function:

|

* :ref:`datasets_ref`
* :ref:`datasets_examples`

.. raw:: html

   <br/>
