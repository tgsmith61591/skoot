.. _setup:

=====
Setup
=====

Skoot depends on several prominent Python packages:

* `Numpy <https://github.com/numpy/numpy>`_
* `SciPy <https://github.com/scipy/scipy>`_
* `Scikit-learn <https://github.com/scikit-learn/scikit-learn>`_ (>= 0.18)
* `Pandas <https://github.com/pandas-dev/pandas>`_

To install, simply use ``pip install skoot``.

|

.. _building:

Building from source
--------------------

If you hope to contribute to the package, you'll need to build from source.
Since there are C & Fortran dependencies, the compilation procedure adds some
new dependencies:

* `Cython <https://github.com/cython/cython>`_
* A Fortran compiler (see following sections)

|

.. _building_on_unix:

Building on Linux/Unix machines (Mac OS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Building on a unix machine is easier than building on :ref:`building_on_windows`.
All you need is ``gcc``, ``g++`` and ``gfortran``. These can be downloaded via
several methods:

**Mac OS**:

.. code-block:: bash

    $ brew install gcc

|

**Linux**:

.. code-block:: bash

    $ apt-get install gcc
    $ apt-get install g++
    $ apt-get install gfortran

|

After you have a Fortran compiler, you can build the package. To build in
development mode (to be able to run tests):

.. code-block:: bash

    $ python setup.py develop

To install the egg in your site-packages:

.. code-block:: bash

    $ python setup.py install

|

.. _building_on_windows:

Building on Windows
~~~~~~~~~~~~~~~~~~~

tl;dr: **don't do it**. It's painful... but it can be done. Like with
:ref:`building_on_unix`, you'll need a Fortran compiler. You can either use
``gfortran``, ``mingw`` or ``MSVC``. After you have a Fortran compiler, you
can build the package. To build in development mode (to be able to run tests),
you can use the same command as with Linux:

.. code-block:: bash

    $ python setup.py develop

You can also install from a wheel, e.g.:

.. code-block:: bash

    $ python setup.py bdist_wheel bdist_wininst
    $ pip wheel --wheel-dir=dist .
    $ pip install dist/skoot-...-win32.whl

|

.. _testing:

Testing
-------

The following are some guidelines for creating and running unit test cases.

|

Creating test cases
~~~~~~~~~~~~~~~~~~~

Skoot uses ``nose`` or ``pytest`` to unit test. The pattern for these frameworks
is that each submodules should contain a "tests" directory and individual scripts prefixed with
"test" for each script in the submodule::


    some_submodule/
        |
        |_ tests/
            |_ test_script_a.py
            |_ ...
        |_ __init__.py
        |_ script_a.py
        |_ ...


Each unit test function within the test script should be prefixed with "test":

.. code-block:: python

    # test_script_a.py
    def test_some_function_in_script_a():
        assert something()

**Note** that no ``__main__`` section is required for pytest or nose. The frameworks
themselves will find testing functions and evaluate them. This means some care has to
be taken when naming your functions. Any function that contains the word "test" is
liable to be evaluated by the testing framework. To avoid this, use this trick:

.. code-block:: python

    def some_benign_function_that_contains_word_test(**kwargs):
        return do_something(**kwargs)

    # Avoid conflict with nose/pytest:
    some_benign_function_that_contains_word_test.__test__ = False

Running unit tests
~~~~~~~~~~~~~~~~~~

Running the unit tests is exceedingly simple.
After you've built the package in ``develop`` mode, you can run the unit tests via pytest:

.. code-block:: bash

    $ pytest

And with coverage, if you have the ``coverage`` package:

.. code-block:: bash

    $ pytest --cov
