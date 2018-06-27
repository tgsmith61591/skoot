.. _balance:

=====================
The balance submodule
=====================

The balance submodule provides methods for rectifying class imbalance by
either augmenting or down-sampling the dataset. As with most skoot methods,
each of these functions works on either Numpy array objects or Pandas
DataFrames.

As of version 0.19, there are three methods of balancing data:

* Under-sampling (the majority class)
* Over-sampling (the minority class)
* SMOTE

These balancing functions are *not* transformers, since balancing should never
be applied to test data. They are simply functions that should be applied to
training data prior to fitting a model.

.. raw:: html

   <br/>

See :ref:`balance_examples`

.. raw:: html

   <br/>
