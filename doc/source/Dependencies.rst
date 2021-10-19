Dependencies
================

.. contents:: Contents
    :depth: 2


IMPROVER uses a range of open source libraries.
Some of these are widely used throughout the IMPROVER code and are required
for IMPROVER as a whole.

Other libraries only used in specific parts of IMPROVER are optional.
This allows for smaller sized installations of IMPROVER.
Generally, the library must be installed for the specific part of IMPROVER
to function.


Required
----------

cartopy
~~~~~~~~~~
Cartopy is used for grid projections and coordinate transformations.
https://scitools.org.uk/cartopy/docs/stable/


cftime
~~~~~~~~~~
cftime provides functions for handling time in NetCDF files according to the
Climate and Forecast (CF) conventions.
https://unidata.github.io/cftime/


cf-units
~~~~~~~~~~
cf-units provides units conversion following the Climate and Forecast (CF)
conventions.
https://cf-units.readthedocs.io/en/stable/


Clize
~~~~~~~~~~
Clize automatically generates command line interfaces (CLI) from Python function
signatures.
https://clize.readthedocs.io/en/stable/


Dask
~~~~~~~~~~
Dask lazy-loaded arrays are used (often via Iris) to reduce the amount of data
loaded into memory at once.
https://dask.org/


Iris
~~~~~~~~~~
Iris cubes are used as a primary data structure throughout the IMPROVER code.
https://scitools-iris.readthedocs.io/en/stable/


NetCDF4
~~~~~~~~~~
Python library for reading and writing NetCDF data files via Iris.
https://unidata.github.io/netcdf4-python/


Numpy
~~~~~~~~~~
Multidimensional numerical array library, used as the basis for Iris cubes and
dask arrays.
https://numpy.org/doc/stable/


Scipy
~~~~~~~~~~
Scientific python library, used for a variety of statistical, image processing,
interpolation and spatial functions.
https://docs.scipy.org/doc/scipy/reference/


Sigtools
~~~~~~~~~~
Sigtools provides introspection tools for function signatures.
Sigtools is required by clize, so this dependency is needed anyway
despite minor usage in IMPROVER.
https://sigtools.readthedocs.io/en/stable/


Sphinx
~~~~~~~~~~
Sphinx is a documentation library for Python. IMPROVER requires it at runtime
to generate help strings from CLI function docstrings via clize.
https://www.sphinx-doc.org/en/master/


Optional dependencies
----------------

python-stratify
~~~~~~~~~~~
Vectorised (cython) interpolation, particularly for vertical levels of the
atmosphere.
https://github.com/SciTools/python-stratify

Required for CLIs: ``interpolate-using-difference``, ``phase-change-level``

statsmodels
~~~~~~~~~~~
Estimation of statistical models, used in EMOS.
https://www.statsmodels.org/stable/

Required for CLIs: ``estimate-emos-coefficients``

numba
~~~~~~~~~~~
JIT compiler for numerical Python code, used for performance enhancement.
https://numba.readthedocs.io/en/stable/

Required for CLIs: ``generate-timezone-mask-ancillary``

pysteps
~~~~~~~~~~~
https://pysteps.github.io/

Required for CLIs: ``nowcast-accumulate``, ``nowcast-extrapolate``,
``nowcast-optical-flow-from-winds``

python-dateutil
~~~~~~~~~~
https://dateutil.readthedocs.io/en/stable/

Required for CLIs: FIXME

pytz
~~~~~~~~~~
Timezone database for Python.
https://pythonhosted.org/pytz/

Required for CLIs: ``generate-timezone-mask-ancillary``

timezonefinder
~~~~~~~~~~~
https://timezonefinder.readthedocs.io/en/stable/


Required for CLIs: ``generate-timezone-mask-ancillary``
