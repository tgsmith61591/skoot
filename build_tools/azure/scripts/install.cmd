@rem https://github.com/numba/numba/blob/master/buildscripts/incremental/setup_conda_environment.cmd
@rem The cmd /C hack circumvents a regression where conda installs a conda.bat
@rem script in non-root environments.
set CONDA_INSTALL=cmd /C conda install -q -y
set PIP_INSTALL=pip install -q

@echo on

@rem Deactivate any environment
call deactivate
@rem Clean up any left-over from a previous build
conda remove --all -q -y -n testvenv
conda create -n testvenv -q -y python=%PYTHON_VERSION% numpy scipy cython scikit-learn pytest wheel pillow joblib pandas blas
conda install -c anaconda mkl-rt

call activate testvenv

python --version
pip --version

@rem Install the build and runtime dependencies of the project.
python setup.py bdist_wheel bdist_wininst

@rem Install the generated wheel package to test it
pip install --pre --no-index --find-links dist\ skoot

if %errorlevel% neq 0 exit /b %errorlevel%
