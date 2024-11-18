Running at your site
====================

Installation
------------

The latest release of IMPROVER can be installed using
`conda <https://docs.conda.io/en/latest/>`_ from the
`conda-forge <https://anaconda.org/conda-forge/improver>`_ channel
using:

.. code:: bash

   conda install -c conda-forge improver

If you would like to install the development version (or install in
development mode) you can do this via
`setuptools <https://setuptools.readthedocs.io/en/latest/>`_ by
downloading the code from this repository, installing the required
dependencies and then using either ``pip install`` (``pip install -e``)
or ``python setup.py install`` (``python setup.py develop``). Note that
``pip install`` will not work in an empty environment due to problems
with installation of the dependency ``iris`` via pip.

Example environments are included in the repository ``envs`` directory.
These environment files are used to run the test suite on Github actions,
so they should stay up to date with any dependency changes. See also
documentation about :doc:`use of dependencies in IMPROVER <Dependencies>`.

Alternatively, you can manually 'install' by downloading the code and
putting the IMPROVER ``bin/`` directory in your PATH.

If you have particular setup requirements such as `environment
modules <https://modules.readthedocs.io/en/latest/>`_, you can specify
them in an ``etc/site-init`` script. This is a bash script that is
sourced before IMPROVER sub-commands are run via the ``bin/improver``
top level command.

Example ``etc/site-init`` script:

.. code:: bash

   #!/bin/bash
   set -eu
   # Load a particular version of Python.
   # You can do this however you want - modules, modifying PATH, etc.
   module load pythonlatest
   # Set a default location for IMPROVER_ACC_TEST_DIR to pick up input and output test files.
   export IMPROVER_ACC_TEST_DIR=${IMPROVER_ACC_TEST_DIR:-$HOME/improver_acc_tests/}

Basic step-by-step usage example
--------------------------------

The steps below will install dependencies on a typical Linux system such
as Ubuntu, Fedora, Debian or Red Hat.

IMPROVER has been succesfully run on Apple macOS, however this is not
the team’s main focus for development. Setup on macOS is very similar
but you will need to download `Conda for
macOS <https://docs.conda.io/en/latest/miniconda.html>`_ rather than
for Linux.

IMPROVER has not been tested on Windows. However, IMPROVER is likely to
work as-is via `Windows Subsystem for
Linux <https://docs.microsoft.com/en-us/windows/wsl/>`_ - set up WSL,
then once you have a Linux environment, follow these instructions.

.. code:: bash

   # Install conda
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   # Follow the conda installer steps - accept the licence agreement,
   # choose an installation path and do the initialisation
   # Close and re-open your shell to pick up the conda initialisation
   # Install IMPROVER from conda-forge (note you may want to do this in a conda environment)
   conda install -c conda-forge improver
   # Run IMPROVER via command line
   improver threshold --help
   improver threshold --threshold-values=273.15,280,290 /path/to/your/data/air_temperature.nc
   # Install iris sample data files (used in the Python example below)
   conda install -c conda-forge iris-sample-data
   # Start the python interpreter interactively and use IMPROVER as a library
   python3

At the Python interpreter prompt:

.. code:: python

   # Load an Iris cube with sample data
   import iris
   cube = iris.load(iris.sample_data_path("E1_north_america.nc"))[0]
   print(cube)
   # Load the IMPROVER library
   import improver.cli as imprcli
   # Call IMPROVER from within Python
   output = imprcli.threshold.process(cube, threshold_values=[273.15, 280.0, 290.0])
   # Print the output cube and save as NetCDF
   print(output)
   iris.save(output, "output.nc")

Test suite
----------

Tests can be run from the top-level directory using bin/improver-tests
or directly using `pytest <https://docs.pytest.org/en/latest/>`_.

The unit tests use data which is included in the test code and these
tests are quick to run. Unit tests are run as part of the test suite on
`Github actions <https://github.com/metoppv/improver/actions>`_.

.. code:: bash

   # Run unit tests via improver-tests wrapper
   bin/improver-tests unit
   bin/improver-tests --help # Prints out the help information
   # Use pytest directly with marker to run only the unit tests
   pytest -m 'not acc'
   # To run a particular function within a unit test, you can use the :: notation
   pytest -m improver_tests/test_unit_test.py::Test_function

The CLI (command line interface) acceptance tests use known good output
(KGO) files for validating that the behaviour is as expected. This data
can be found in the `improver_test_data` open source repository on GitHub.

The path to the acceptance test data is set using the
``IMPROVER_ACC_TEST_DIR`` environment variable. Acceptance tests will be
skipped if this environment variable is not defined.
To run the acceptance tests you can use the following:

.. code:: bash

   export IMPROVER_ACC_TEST_DIR=/path/to/acceptance/data/repo
   # Use pytest marker to run only the acceptance tests
   pytest -m acc
   # Acceptance tests can be run significantly faster in parallel using the pytest-xdist plugin
   pytest -n 8
   # An example of running just one particular acceptance test
   pytest -v -s -m acc -k test_cli_name.py

To run all tests together at once, the following command can be input

.. code:: bash

   bin/improver-tests # runs all tests
