.. _api_ref:

=============
API Reference
=============

This is the class and function reference for skoot. Please refer to
the :ref:`full user guide <user_guide>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.


.. _base_ref:

:mod:`skoot.base`: Base metaclasses and utility functions
=========================================================

.. automodule:: skoot.base
    :no-members:
    :no-inherited-members:

Base classes
------------
.. currentmodule:: skoot

.. autosummary::
    :toctree: generated/
    :template: class.rst

    base.BasePDTransformer


.. _decorator_ref:

:mod:`skoot.decorators`: Decorator utilities
============================================

.. automodule:: skoot
    :no-members:
    :no-inherited-members:

Decorator methods
-----------------
.. currentmodule:: skoot

.. autosummary::
    :toctree: generated/
    :template: function.rst

    decorators.overrides
    decorators.suppress_warnings


.. _balance_ref:

:mod:`skoot.balance`: Class imbalance remedies
==============================================

.. automodule:: skoot.balance
    :no-members:
    :no-inherited-members:

**User guide:** See the :ref:`balance` section for further details.

Balancing functions
-------------------
.. currentmodule:: skoot

.. autosummary::
    :toctree: generated/
    :template: function.rst

    balance.over_sample_balance
    balance.smote_balance
    balance.under_sample_balance


.. _datasets_ref:

:mod:`skoot.datasets`: Dataset loaders
======================================

.. automodule:: skoot.datasets
    :no-members:
    :no-inherited-members:

**User guide:** See the :ref:`datasets` section for further details.

Dataset loading functions
-------------------------
.. currentmodule:: skoot

.. autosummary::
    :toctree: generated/
    :template: function.rst

    datasets.load_boston_df
    datasets.load_breast_cancer_df
    datasets.load_iris_df


.. _decomposition_ref:

:mod:`skoot.decomposition`: Various matrix decompositions
=========================================================

.. automodule:: skoot.decomposition
    :no-members:
    :no-inherited-members:

**User guide:** See the :ref:`decomposition` section for further details.

Decomposition classes
---------------------
.. currentmodule:: skoot

.. autosummary::
    :toctree: generated/
    :template: class.rst

    decomposition.SelectiveIncrementalPCA
    decomposition.SelectiveKernelPCA
    decomposition.SelectiveNMF
    decomposition.SelectivePCA
    decomposition.SelectiveTruncatedSVD
    decomposition.QRDecomposition


.. _exploration_ref:

:mod:`skoot.exploration`: Exploratory data analysis
===================================================

.. automodule:: skoot.exploration
    :no-members:
    :no-inherited-members:

**User guide:** See the :ref:`exploration` section for further details.

Exploratory analysis functions
------------------------------
.. currentmodule:: skoot

.. autosummary::
    :toctree: generated/
    :template: function.rst

    exploration.summarize


.. _feature_extraction_ref:

:mod:`skoot.feature_extraction`: Feature extraction methods
===========================================================

.. automodule:: skoot.feature_extraction
    :no-members:
    :no-inherited-members:

**User guide:** See the :ref:`feature_extraction` section for further details.

Feature extraction estimators
-----------------------------
.. currentmodule:: skoot

.. autosummary::
    :toctree: generated/
    :template: class.rst

    feature_extraction.InteractionTermTransformer


.. _feature_selection_ref:

:mod:`skoot.feature_selection`: Feature selection methods
=========================================================

.. automodule:: skoot.feature_selection
    :no-members:
    :no-inherited-members:

**User guide:** See the :ref:`feature_selection` section for further details.

Feature selection estimators
----------------------------
.. currentmodule:: skoot

.. autosummary::
    :toctree: generated/
    :template: class.rst

    feature_selection.BaseFeatureSelector
    feature_selection.FeatureFilter
    feature_selection.LinearCombinationFilter
    feature_selection.MultiCorrFilter
    feature_selection.NearZeroVarianceFilter
    feature_selection.SparseFeatureFilter


.. _model_validation_ref:

:mod:`skoot.model_validation`: Model validation & monitoring
============================================================

.. automodule:: skoot.model_validation
    :no-members:
    :no-inherited-members:

**User guide:** See the :ref:`model_validation` section for further details.

Model validators & monitoring classes
-------------------------------------
.. currentmodule:: skoot

.. autosummary::
    :toctree: generated/
    :template: class.rst

    model_validation.CustomValidator
    model_validation.DistHypothesisValidator

Pipelines with built-in monitoring at each stage
------------------------------------------------
.. currentmodule:: skoot

.. autosummary::
    :toctree: generated/
    :template: function.rst

    model_validation.make_validated_pipeline


.. _preprocessing_ref:

:mod:`skoot.preprocessing`: Pre-processing transformers
=======================================================

.. automodule:: skoot.preprocessing
    :no-members:
    :no-inherited-members:

**User guide:** See the :ref:`preprocessing` section for further details.

Continuous feature binning
--------------------------
.. currentmodule:: skoot

.. autosummary::
    :toctree: generated/
    :template: class.rst

    preprocessing.BinningTransformer

Dataframe schema transformers
-----------------------------
.. currentmodule:: skoot

.. autosummary::
    :toctree: generated/
    :template: class.rst

    preprocessing.SchemaNormalizer

Encoding transformers
---------------------
.. currentmodule:: skoot

.. autosummary::
    :toctree: generated/
    :template: class.rst

    preprocessing.DummyEncoder

Scalers/normalizers
-------------------
.. currentmodule:: skoot

.. autosummary::
    :toctree: generated/
    :template: class.rst

    preprocessing.SelectiveMaxAbsScaler
    preprocessing.SelectiveMinMaxScaler
    preprocessing.SelectiveRobustScaler
    preprocessing.SelectiveStandardScaler

Skewness transformers
---------------------
.. currentmodule:: skoot

.. autosummary::
    :toctree: generated/
    :template: class.rst

    preprocessing.BoxCoxTransformer
    preprocessing.YeoJohnsonTransformer


.. _utils_ref:

:mod:`skoot.utils`: Common utility functions
============================================

.. automodule:: skoot.utils
    :no-members:
    :no-inherited-members:

**User guide:** See the :ref:`utils` section for further details.

Iterable utilities
------------------
.. currentmodule:: skoot

.. autosummary::
    :toctree: generated/
    :template: function.rst

    utils.get_numeric_columns
    utils.is_iterable

Validation utilities
--------------------
.. currentmodule:: skoot

.. autosummary::
    :toctree: generated/
    :template: function.rst

    utils.check_dataframe
    utils.type_or_iterable_to_col_mapping
    utils.validate_multiple_cols
    utils.validate_multiple_rows
    utils.validate_test_set_columns

.. raw:: html

   <br/>
