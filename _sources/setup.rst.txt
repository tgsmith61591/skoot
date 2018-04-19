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

Linux/Unix machines (Mac OS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Windows
~~~~~~~

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