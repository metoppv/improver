Dependencies
================

.. contents:: Contents
    :depth: 2


IMPROVER builds on the functionality provided by a range of open source
libraries.

Some of these libraries are widely used throughout the IMPROVER code, so are
considered as required for IMPROVER as a whole.

Other libraries are only used in specific parts of IMPROVER.
These libraries are optional for IMPROVER as a whole, but are required to use
related parts of IMPROVER.
Optional installation of these libraries allows for smaller sized installations
of IMPROVER and reduces conda environment dependency solving difficulties.

Required
-----------------

cartopy
~~~~~~~~~~~~~~~~~
Cartopy is used for grid projections and coordinate transformations.

https://scitools.org.uk/cartopy/docs/latest/


cftime
~~~~~~~~~~~~~~~~~
cftime provides functions for handling time in NetCDF files according to the
Climate and Forecast (CF) conventions.

https://unidata.github.io/cftime/


cf-units
~~~~~~~~~~~~~~~~~
cf-units provides units conversion following the Climate and Forecast (CF)
conventions.

https://cf-units.readthedocs.io/en/stable/


Clize
~~~~~~~~~~~~~~~~~
Clize automatically generates command line interfaces (CLI) from Python function
signatures.

https://clize.readthedocs.io/en/stable/


Dask
~~~~~~~~~~~~~~~~~
Dask lazy-loaded arrays are used (often via Iris) to reduce the amount of data
loaded into memory at once.

https://dask.org/


Iris
~~~~~~~~~~~~~~~~~
Iris cubes are used as a primary data structure throughout the IMPROVER code.

https://scitools-iris.readthedocs.io/en/stable/


NetCDF4
~~~~~~~~~~~~~~~~~
Python library for reading and writing NetCDF data files via Iris.

https://unidata.github.io/netcdf4-python/


Numpy
~~~~~~~~~~~~~~~~~
Multidimensional numerical array library, used as the basis for Iris cubes and
dask arrays.

https://numpy.org/doc/stable/


Scipy
~~~~~~~~~~~~~~~~~
Scientific python library, used for a variety of statistical, image processing,
interpolation and spatial functions.

https://docs.scipy.org/doc/scipy/reference/


Sigtools
~~~~~~~~~~~~~~~~~
Sigtools provides introspection tools for function signatures.
Sigtools is required by clize, so this dependency is needed anyway
despite minor usage in IMPROVER.

https://sigtools.readthedocs.io/en/stable/


Sphinx
~~~~~~~~~~~~~~~~~
Sphinx is a documentation library for Python. IMPROVER requires it at runtime
to generate help strings from CLI function docstrings via clize.

https://www.sphinx-doc.org/en/master/


Optional dependencies
---------------------

python-stratify
~~~~~~~~~~~~~~~~~~
Vectorised (cython) interpolation, particularly for vertical levels of the
atmosphere.

https://github.com/SciTools/python-stratify

Required for CLIs: ``interpolate-using-difference``, ``phase-change-level``

statsmodels
~~~~~~~~~~~~~~~~~~
Estimation of statistical models, used for
:doc:`EMOS <improver.calibration.emos_calibration>`.

https://www.statsmodels.org/stable/

Required for CLIs: ``estimate-emos-coefficients``

numba
~~~~~~~~~~~~~~~~~~
JIT compiler for numerical Python code, used for better computational performance.

https://numba.readthedocs.io/en/stable/

Optionally used by CLIs: ``generate-realizations``, ``generate-percentiles``, ``spot-extract``, ``apply-emos-coefficients``

PySTEPS
~~~~~~~~~~~~~~~~~~
Probabilistic nowcasting of radar precipitation fields, used for nowcasting.

https://pysteps.github.io/

Required for CLIs: ``nowcast-accumulate``, ``nowcast-extrapolate``,
``nowcast-optical-flow-from-winds``

pytz
~~~~~~~~~~~~~~~~~
Timezone database for Python.

https://pythonhosted.org/pytz/

LightGBM
~~~~~~~~~~~~~~~~~~
Gradient boosted decision tree ensemble framework, used for RainForests
calibration.

https://lightgbm.readthedocs.io/en/latest/

Required for CLIs: ``apply-rainforests-calibration``

Treelite
~~~~~~~~~~~~~~~~~~
Lightweight binary format for specifying decision tree models, used for
RainForests calibration.

https://treelite.readthedocs.io/en/latest/index.html

Required for CLIs: ``apply-rainforests-calibration``

TL2cgen (TreeLite 2 C GENerator)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model compiler for decision tree models, used for more efficient computation
of GBDT models required for RainForests calibration.

https://tl2cgen.readthedocs.io/en/latest/index.html

Required for CLIs: ``apply-rainforests-calibration``
