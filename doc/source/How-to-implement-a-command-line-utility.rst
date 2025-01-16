How to implement a command line utility
=======================================

What is an IMPROVER CLI?
------------------------

An IMPROVER command line interface (CLI) is a Python module located in
``improver/cli`` that uses `clize <https://github.com/epsy/clize>`_ to
provide both a command line interface and a Python function interface.
Clize automatically handles command line argument parsing, errors and
help, including generated ``--help`` text based on the Python docstring.

When called as a Python function, input and output are via
`Iris <https://github.com/SciTools/iris>`_ cubes. When run from the
command line, input and output are via netCDF files which are loaded and
saved by Iris. These netCDF files contain a single cube with an
ensemble-CDF related dimension as the first dimension - one of
realization, percentiles, or probability over thresholds.

IMPROVER CLI utilities invoke one or more plugin classes and produce a
result. For example, ``improver/cli/nbhood.py`` applies a neighbourhood
processing plugin using code from ``improver/nbhood.py``.

IMPROVER plugins
----------------

Plugins are Python classes with unit tests. See the :doc:`Code-Style-Guide`
section on plugins for more information.

Command line testing
--------------------

IMPROVER CLIs should have corresponding acceptance tests located in
``improver_tests/acceptance``. These tests are written using the
`pytest <https://docs.pytest.org/en/latest/>`_ test framework.

The point of acceptance tests is to provide reference test data to
explain and test normal use of the CLI. These acceptance tests consist
of reference NetCDF input files and a 'known good output' (KGO) NetCDF
output file.

See the :doc:`Running-at-your-site`
page for information on how to run the acceptance tests.

Recreating acceptance test data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a new test has been written and the input data has been put in
place it is possible to generate the expected output:

.. code:: bash

   export RECREATE_KGO=/path/to/new/kgos
   pytest -m acc -k combine
   unset RECREATE_KGO

The ``RECREATE_KGO`` path must be distinct from the input path
(``IMPROVER_ACC_TEST_DIR``, see :doc:`Running-at-your-site`).
It is also possible to use this method to recreate all KGOs following
changes that may affect them, for example modifications to metadata.

.. code:: bash

   export RECREATE_KGO=/path/to/new/kgos
   bin/improver-tests cli

In the singular or bulk case, it is important that the resulting KGO are
sanity checked to ensure the test data remains sensible.

Acceptance test checksums
-------------------------

The input and output files for acceptance tests are identified by
checksums listed in
`improver_tests/acceptance/SHA256SUMS
<https://github.com/metoppv/improver/blob/master/improver_tests/acceptance/SHA256SUMS>`_.
Changes to the checksum file in the code repository identify when
changes are made to the code requiring corresponding changes to the data
files. The acceptance test data files are maintained in a public repository:
`https://github.com/metoppv/improver_test_data`, with the directory
structure being based on the CLI plugin names.

Use of checksums
~~~~~~~~~~~~~~~~

As stated above, the checksums are used to ensure the acceptance test
data being used to test the code are the correct versions for the code
revision being tested. To run all the acceptance tests use:
``bin/improver-tests cli``

The first test to be run is
``improver_tests/acceptance/test_checksums.py`` which will fail if the
test data and checksums do not correspond and stop any further tests.
Running an individual test, e.g. ``pytest -m acc -k threshold`` will
also compare the input data against the expected checksums and report if
there is a mismatch.

It is possible to bypass the comparison of data against checksums using:
``export IMPROVER_IGNORE_CHECKSUMS=true``

This may be useful when in the midst of making changes.

Updating acceptance test data checksums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Updating acceptance test data checksums is achieved using the command:
``bin/improver-tests recreate_checksums``

The updated
`improver_tests/acceptance/SHA256SUMS
<https://github.com/metoppv/improver/blob/master/improver_tests/acceptance/SHA256SUMS>`_
file will need to be committed to the IMPROVER repository along with the
code that has required / led to the modified test inputs / outputs.
