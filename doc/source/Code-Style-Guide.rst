Code style guide
================

.. contents:: Contents
    :depth: 3

Developer advice
----------------

General comments
~~~~~~~~~~~~~~~~

* IMPROVER uses a number of commonly used Python modules
  e.g. `numpy <https://numpy.org/>`_,
  `scipy <https://www.scipy.org/>`_. Additionally IMPROVER makes use
  of `Iris <https://scitools.org.uk/iris/docs/latest/index.html>`_ for
  the in-memory data representation.
* Aim for smaller methods with a single purpose.
* Aim for fewer logical branches in each method to make it easier to
  test each route through the code and make it easier to maintain.
* Aim for simpler process methods (see the plugin example below),
  preferably that just call other methods in the plugin.
* Centralise functions that check input (i.e. checking input is a cube)
  so we only have to test them once.
* Aim for simple plugins with better defined purposes.
* Think about fixing metadata separately after main array manipulation
  (although there are cases where it may make sense to do it together,
  like when adding or removing dimensions).
* Some things that could be functions, should be functions, rather than
  classes (if appropriate).
* Use inline comments where functionality is not immediately obvious,
  eg where we do complex array or cube manipulation. Consider the top
  level goal: that code should be readable to a non-expert developer.
* Document in plugin docstrings and CLI help when we add or remove a
  dimension from a cube.
* Ensure that functions and classes are unit tested and CLIs (command
  line interfaces) have acceptance tests.
* Use type annotations on function and method interfaces, except for
  CLIs.
* Avoid adding functionality to CLIs beyond calling the relevant plugin,
  all other functionality should be added to the plugin itself.

Pull requests
~~~~~~~~~~~~~

Smaller pull requests (low hundreds of lines) to the IMPROVER code base
are preferred to larger ones (thousand lines or higher), if reasonably
possible. Very large pull requests are difficult to review and can lead
to mistakes - but are sometimes necessary for more efficient or
ambitious changes.

In general, developers should consider whether a change that is becoming
large (> 800 lines) can be submitted as a series of smaller,
self-contained PRs for more manageable review. Reviewers should use
their discretion in requesting PRs to be broken down if larger than
this.

Guidance (eg
https://smartbear.com/learn/code-review/best-practices-for-peer-code-review/)
suggests that for reviews to be effective reviewers should:

* Take time over reviews: don’t review faster than 500 lines of code per hour
* Not review more than 400 lines of code at a time, or for more than one hour
  continuously

Code formatting
---------------

In general, follow the very sensible `Google style
guide <https://google.github.io/styleguide/pyguide.html>`_.

Code format should match that produced by the
`black <https://github.com/psf/black>`_ code formatter - this is
checked as part of the test suite on Github Actions. Codacy will give
you a mark based around ``pylint`` (with some caveats). A pylint score
of >8 is OK, >9 is good - skip pylint-misled errors for e.g. numpy
(e.g. with ``pylint --extension-pkg-whitelist=numpy``).

Modular code
~~~~~~~~~~~~

Encapsulate non-trivial code in classes with unit tests in modules less
than 1000 lines (ideally). If it is a lot of related code (for example
>2-3 modules at >2000 lines apiece), consider making a package of
modules. This is especially true if you think that the code will grow
later.

It is perfectly OK to write small pieces of functionality as
self-contained functions rather than self-contained classes. In general,
if you find that you need more than one function to do what you want,
put them in their own class.

Datatypes
~~~~~~~~~

IMPROVER diagnostic data and coordinates are in general expected to
conform to the following datatypes:

::

       +-------------------------------+-------------+
       | Name                          | Datatype    |
       +===============================+=============+
       | diagnostic / probability data | np.float32  |
       +-------------------------------+-------------+
       | projection_x_coordinate       | np.float32  |
       +-------------------------------+-------------+
       | projection_y_coordinate       | np.float32  |
       +-------------------------------+-------------+
       | time                          | np.int64    |
       +-------------------------------+-------------+
       | forecast_reference_time       | np.int64    |
       +-------------------------------+-------------+
       | forecast_period               | np.int32    |
       +-------------------------------+-------------+

In general, numerical data should be 32-bit (either float32 or int32 is
acceptable), with the exception of absolute times, which cannot be
handled with sufficient precision as 32-bit. Absolute times are
therefore handled as 64-bit integers, in units of 'seconds since
1970-01-01 00:00:00'.

Avoiding 64-bit floats
^^^^^^^^^^^^^^^^^^^^^^

The Python float and numpy default is 64 bit floating point, which
equates to 15 or 16 significant digits. This is excessive for most of
our problems, where e.g. temperature to the hundredth of a Kelvin or
probabilities to 0.01 are good enough. Plugin code should avoid 64 bit
quantities and arithmetic wherever possible and appropriate. 64 bit
floating point is OK for example for Unix time values with non-integer
seconds, but not for most physical quantities or probabilities.

This means passing in ``dtype=np.float32`` to most numpy array
constructor functions (e.g. ``array``, ``full``, ``zeros``, ``ones``,
``arange``) and avoiding Python floating point numbers. You could use
'astype' to cast if your array is already 64 bit, but it is best for
performance to try to track down 64 bit computation at the places it
comes in.

.. code:: python

   # Bad
   foo = (bar + baz) / 2.0  # 2.0 is float64, so even if bar and baz are float32 foo will come out float64.
   qux = np.ones((1, 2, 3))  # Default np array is float64.
   wibble = np.array([wobble, wubble])  # Even if wobble and wubble are float32, no dtype is given, so float64
   fred = float(garply)  # 'float' is 64 bit floating point

   # Good
   foo = (bar + baz) / np.float32(2.0)  # Or alternatively np.float32((bar + baz) / 2.0).
   qux = np.ones((1, 2, 3), dtype=np.float32)
   wibble = np.array([wobble, wubble], dtype=np.float32)
   fred = np.float32(garply)

Plugins
-------

Docstrings
~~~~~~~~~~

These should follow PEP8 and PEP257 - examples are shown in the example
plugin code below.

Note that private methods do not always required complete doc-strings,
particularly if their behaviour is very obvious and the arguments to the
method have been defined in other doc-strings. However, if the private
method is complex, or arguments are being passed in that have not been
defined elsewhere, a doc-string is advisable. If any argument needs
defining for a private method, all arguments to that method should be
defined in its docstring to avoid partial information.

Due to the use of `Sphinx <http://www.sphinx-doc.org/en/stable/>`_ for
documentation building, a number of `docstring section
headers <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html#docstring-sections>`_
are supported. Favoured docstring section headers are:

* Args: Compulsory arguments.
* Keyword Args : Keyword arguments.
* Raises: Exceptions raised.
* Returns: Variables returned by the function or method.
* References: Link to available documentation.
* Warns: Warnings raised.

'Napolean Google style' Returns: only displays properly in Sphinx if
there is only one variable being returned. If more than one variable is
being returned then the return value should be a list of each item
returned in the tuple. See the example below.

Type annotations
~~~~~~~~~~~~~~~~

All function and method interfaces, except for those in CLIs (see
below), should have `type
annotations <https://docs.python.org/3/library/typing.html>`_. Type
annotations have been part of Python since version 3.5. Here is a simple
example showing type annotations for a function that takes a string and
returns a string.

.. code:: python

   def greeting(name: str) -> str:
       return 'Hello ' + name

Types are available from the typing module. For example

.. code:: python

   from typing import List

   def first_in_list_of_str(list_of_str: List[str]) -> str:
       return list_of_str[0]

If an argument or return value can have multiple types, use Union.

.. code:: python

   from typing import List, Union

   def length_of_str_or_list(arg: Union[str, List[str]]) -> int:
       return len(arg)

Do not put types in the docstring (except for CLIs, see next section).

.. code:: python

   def greeting(name: str) -> str:
       """My greeting.

       Args:
           name:
               The name to greet.

       Returns:
           The greeting with appropriate name.
       """
       return 'Hello ' + name

If a function or method has multiple returns then the return type is a
Tuple. The return in the docstring should be a list (in rst/markdown
style).

.. code:: python

   from typing import Tuple

   def first_and_last(list_of_str: List[str]) -> Tuple[str, str]:
       """First and last items.

       Args:
           list_of_str:
               A list of strings.

       Returns:
           - First item.
           - Second item.
       """
       return list_of_str[0], list_of_str[-1]

See the plugin example below, and throughout the existing codebase for
more examples.

Variable types in CLIs
~~~~~~~~~~~~~~~~~~~~~~

Note that we use clize in the CLIs which uses type annotations at
runtime, hence the need to define these clearly within the docstring.

Within docstrings, when specifying a variable type, Python built-in data
types can be used directly e.g.

* int
* float
* str
* bool

In order for variable types to link correctly within
`readthedocs <http://improver.readthedocs.io/en/latest/?badge=latest>`_,
the ``intersphinx_mapping`` needs to be updated to link to the
documentation of the module where the variable type originates. For
example:

* numpy.ndarray
* datatime.datetime

The full name of the module is required, rather than an alias e.g. numpy rather
than np.

When defining variables with non-trivial shapes (e.g. arrays, cubes), if
useful information about the shape of these variables can be included,
it should be.

When the method returns multiple variables (example 1 below), or a
variable whose internal structure is non-trivial (example 2 below), this
should be documented using the mypy syntax. This is also the case for
complex structured passed as input arguments (example 3 below). This
structure is not currently used throughout IMPROVER, but will be adopted
over time.

Example 1:

::

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]:
            Tuple containing the modified data arrays for A and B.

Example 2:

::

    Returns:
        Dict[pathlib.Path, str]:
            Dict with keys being relative paths and values being hexadecimal checksums

Example 3:

::

    Args:
        structured_input (Dict[pathlib.Path, str]):
            Dict with keys being relative paths and values being hexadecimal checksums

Further examples of this syntax can be `found
here <https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html>`_.

Constants
~~~~~~~~~

If you are adding a constant and it is very specific to a particular
piece of code, include it in that code (example: Von Karman’s constant
for wind downscaling). If it could apply to more than one piece of code
(e.g. G - gravitational constant) then put it in improver/constants.py.

Variable Names
~~~~~~~~~~~~~~

2-letter variable names are OK if they are obvious quantities
(e.g. ``dx`` or ``dt``).

Plugin Example
~~~~~~~~~~~~~~

Plugins (classes) should be an example of a non-trivial algorithm or set
of algorithms for a particular purpose. They should be set up via the
``__init__`` method and then invoked on a particular iris Cube ``cube``
using a ``process`` method - e.g. using ``process(cube)``. See
e.g. `Threshold <https://github.com/metoppv/improver/blob/master/lib/improver/threshold.py>`_
class. In some limited cases an iris ``CubeList`` may be preferable.
Avoid writing code that can do both. Class names use
`PascalCase <https://en.wikipedia.org/wiki/PascalCase>`_ whilst
variable names and method names use
`snake_case <https://en.wikipedia.org/wiki/Snake_case>`_. ``__repr__``
methods are not required, though they may be found in existing code.

.. code:: python

   """ module for MyPlugin. """

   import warnings

   from iris.cube import Cube


   class MyPlugin(object):
       """Title sentence to describe purpose of MyPlugin.

       Further description to help create a meaningful docstring.
       """
       # Simple variables can be passes into the __init__ method.
       def __init__(self, simple_variable: float) -> None:
           """Description of what's done in __init__.

           e.g. set up processing for MyPlugin.

           Args:
               simple_variable:
                   A simple variable to demonstrate how a variable is passed to
                   the __init__ method.
           """
           self.simple_variable = simple_variable

       @staticmethod
       def my_static_method(cube: Cube, multiplier: int = 2) -> Cube:
           """Description of what my_static_method is trying to do.

           Args:
               cube:
                   An example cube for processing by my_static_method.
               multiplier:
                   An argument with a default value.

           Returns:
               Output cube after manipulation.
           """
           new_cube = cube * multiplier
           return new_cube

       @staticmethod
       def my_static_method_multiple_returns(cube: Cube) -> Tuple[float, float]:
           """Description of what my_static_method_multiple_returns
           is trying to do.

           Args:
               cube:
                   An example cube for processing.

           Returns:
               - The max value of the cube data.
               - The min value of the cube data.
           """
           max_value = cube.data.max()
           min_value = cube.data.min()
           return max_value, min_value

       def _my_private_method(cube: Cube) -> str:
           """Description of what _my_private_method is trying to do.

           As this is a private method, taking only an argument already
           defined in the doc-string of the calling method, we do not
           need to define the input argument again.

           Returns:
               The name of the diagnostic within the cube.
           """
           return cube.name()

       def my_method(self, multiplier: float) -> float:
           """Description of what my_method is trying to do.

           This method uses the instance of the class, and therefore
           shouldn't be a static method.

           Args:
               multiplier:
                   A multiplier.

           Returns:
               The multiplied value.

           Raises:
               ValueError: If the value exceeds the allowed
                   upper limit of 100.
               ValueError: If the value is below or equal to
                   zero.

           Warns:
               Warning: If the value is outside of the
                   expected range (> 0 and <= 50).

           References:
               Bauer, P., Thorpe, A., Brunet, G. (2015) The quiet
                revolution of numerical weather prediction.
               Nature, Vol 525, pp 47-55
           """
           updated_simple_variable = (self.simple_variable * 2) / 3
           if updated_simple_variable > 100:
               msg = (
                   "An updated simple variable of {} exceeds "
                   "the allowable upper limit of 100.".format(updated_simple_variable)
               )
               raise ValueError(msg)
           elif updated_simple_variable > 50:
               msg = (
                   "The updated simple variable of {} "
                   "is higher than expected. "
                   "Expected range is > 0 and <= 50.".format(updated_simple_variable)
               )
               raise warnings.warn(msg)
           elif updated_simple_variable <= 0:
               msg = (
                   "An updated simple variable of {} is "
                   "below the allowable lower limit of 0.".format(updated_simple_variable)
               )
               raise ValueError(msg)
           return updated_simple_variable

       def process(self, cube: Cube) -> Cube:
           """Description for what's done in the process method.

           Args:
               cube:
                   An example cube for processing.

           Returns:
               Output cube after multiplying the input cube by the
               simple variable.
           """
           # Inline comments can be added, if required.
           cube = self.my_static_method(cube)
           (max_val, min_val) = self.my_static_method_multiple_returns(cube)
           new_cube = cube
           if min_val > 0.0 and max_val < 100.0:
               new_simple_variable = self.my_method()
               new_cube = cube * new_simple_variable
           return new_cube

Helper Functions/Methods
~~~~~~~~~~~~~~~~~~~~~~~~

Helper functions or methods may live in one of several places. Where
they should live depends on how they will be used.

**Case 1:** A function used in more than one module.

* In this case the function should be located in a shared location e.g.
  utilities.py

**Case 2:** A function used by several classes within one module.

* The function should be kept outside of any one class, but within the module.

**Case 3:** A function used only within a single class.

* Should be kept within the class; as a static method if it makes no use of
  self.

Command Line Interface (CLI)
----------------------------

Add a command line interface (improver/cli/<cli_name>.py) to invoke plugins
that can be used as a standalone utility or executable within a suite context
(e.g. wind downscaling, neighbourhood processing, spot data extraction).
These CLIs are invoked using ``bin/improver <cli-name>`` (note that the
CLI filename uses underscores, but the call to use the CLI uses hyphens).

IMPROVER CLIs should only have ``from improver import cli`` as the top
level imports. Other imports are placed inside the function that uses
them. This gives the benefit of a more rapid response to the command
``bin/improver <cli-name> -h`` when those other (often slow) imports are
not needed.

Each CLI should have a process function. This will require a
``@cli.clizefy`` decorator to gain the functionality of
`clize <https://clize.readthedocs.io/en/stable/>`_. If you want the CLI
to save a cube to disk, it will need the decorator ``@cli.with_output``,
this will mean on the command line, the ``--output`` flag can be used to
specify an output path.

As mentioned above, it is important to ensure that no functionality
other than calling the plugin exists within the CLI layer.
Any checks on the data or input requirements should be done in the plugin itself.

To load the cubes, each cube argument will need a type. For a basic cube
this will be ``cube: cli.inputcube``. If there is a default argument to
go with the typed variable, spaces are required around the ``=`` for
example ``weights: cli.inputcube = None``. There are other types which
can be used such as:

* the python standards

  * ``float``
  * ``int``
  * ``bool``

* specific additions

  * ``cli.inputcube``

    * Where a string is given which is a path to a cube to load

  * ``cli.inputjson``

    * Where a string is given which is a path to a json file to load

  * ``cli.comma_separated_list``

    * This will convert the argument into that format and deal with error
      handling if no conversion is possible.

A complete list of local added variable types can be found by
identifying all the ``@value_converter`` decorated functions in
`cli/__init__.py <https://github.com/metoppv/improver/blob/master/improver/cli/__init__.py>`_.

Arguments into the process function should start with the cubes. After
all the cubes, there should be an argument of ``*``, this separates the
positional arguments from the keyword arguments. If you are loading a
cube list of unspecified number of cubes ``*cubelist`` will take all the
cubes, load them and return them as a tuple.

All arguments after the ``*`` will need to be given with keywords on the
command line.

Due to the use of ``*``, required arguments can be used before and after
the star. for example

.. code:: python

   from improver import cli
   @cli.clizefy
   def process(cube: cli.inputcube,
               weights: cli.inputcube = None,
               *,
               coord_for_masking,
               radius: float = None)

The required arguments in this example are:

* cube
* coord_for_masking

Testing
-------

Unit tests test individual functions and classes by comparing the output
from a function or class to the expected in-memory result. CLI (command
line interface) acceptance tests use known good output files on disk for
validating that the behaviour is as expected. In IMPROVER, GitHub
Actions are used to run a series of tests on each pull request to ensure
that the pull request meets the expected standards. Tests can be run
from the top-level directory using bin/improver-tests or using
`pytest <https://docs.pytest.org/en/latest/>`_.

Unit Testing
~~~~~~~~~~~~

Add unit tests for functions and methods of classes where reasonable.

You should add a unit test if:

* the logic in the function or method is not reasonably obvious from just
  looking at it, or
* the code is not otherwise covered by unit tests

You don’t have to add unit tests for every function or method, including
private ones, if the above is true.

Writing unit tests usually saves time in the long run and helps document
the effect of your code.

Unit tests should:

1. Usually pass in representative inputs with the expected metadata and
   dimensions (consistent with what is enforced by the 'load' module).
2. Use centralised test `cube set-up
   utilities <https://github.com/metoppv/improver/blob/master/improver/synthetic_data/set_up_test_cubes.py>`_
   where possible.
3. Consider the most likely uses of the plugin and ensure these are represented
   in the unit tests.
4. Consider possible edge cases e.g. cubes with different input dimensions.
5. Ensure the correct output is generated in good cases.
6. Ensure exceptions are raised as required for bad cases.

Unit test structure
^^^^^^^^^^^^^^^^^^^

Unit tests for classes should be in separate files, named as:

``test_<class name in camel case>.py``

These files for each Class should be in a sub-directory with the name of
source file:

``e.g. DayNightMask in utilities/solar.py --> improver_tests/utilities/solar/test_DayNightMask.py``

Each sub-directory must have a ``__init__.py``

Tests for files containing functions external to classes should be in
test files named for the source file:

``e.g. utilities/spatial.py --> improver_tests/utilities/test_spatial.py``

New unit test in IMPROVER should be written adhering to pytest style.
These include:

* no test classes
* tests are written as functions
* test data is provided by fixtures

Many existing tests use a different approach, but these will slowly be
migrated towards this format over time.

All unit tests should have a first line title in the docstring like
this:

.. code:: python

   """Test the thing."""

rather than:

.. code:: python

   """
   Test the thing.
   """

Numpy arrays within unit tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Within unit tests, numpy arrays are often added to check that a plugin
is generating the expected results. Within unit tests, the examples
below indicate how to include numpy arrays, so that they’re compatible
with pep8 and pylint.

Example 1. In this case, spaces have been removed compared to printing a
numpy array with the default printing options.

.. code:: python

   expected = np.array(
       [[1., 1., 1., 1., 1.],
        [1., 0.88888889, 0.88888889, 0.88888889, 1.],
        [1., 0.88888889, 0.88888889, 0.88888889, 1.],
        [1., 0.88888889, 0.88888889, 0.88888889, 1.],
        [1., 1., 1., 1., 1.]])

Example 2. It is also acceptable to pad floating point values with
zeros, so that the numpy array will appear as a grid, which is often
convenient for our usage.

.. code:: python

   expected = np.array(
       [[1.000000, 1.000000, 1.000000, 1.000000, 1.000000],
        [1.000000, 0.888889, 0.888889, 0.888889, 1.000000],
        [1.000000, 0.888889, 0.888889, 0.888889, 1.000000],
        [1.000000, 0.888889, 0.888889, 0.888889, 1.000000],
        [1.000000, 1.000000, 1.000000, 1.000000, 1.000000]])

This padding can be achieved using the following lines to print a
compatible numpy array:

.. code:: python

   np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
   print repr(expected)

Multiple returns
^^^^^^^^^^^^^^^^

Having more than one return statement in a method or function - fine if
they make the code easier to understand, e.g. by decreasing the
necessary nesting.

CLI acceptance tests
~~~~~~~~~~~~~~~~~~~~

See the :doc:`How-to-implement-a-command-line-utility` page.

Licence information
-------------------

The following licence information should be added to each new file:

::

   # (C) Crown copyright, Met Office. All rights reserved.
   #
   # This file is part of IMPROVER and is released under a BSD 3-Clause license.
   # See LICENSE in the root of the repository for full licensing details.

Making a new release
--------------------

New release steps:

1. Inform the core developers across institutions and wait for approval.
2. On the command line, check out the commit on master that you want
   to tag. For a given version such as 1.1.0, run:
   `git tag -a 1.1.0 -m "IMPROVER release 1.1.0"`. Then run:
   `git push upstream 1.1.0`.
3. Go to `Draft a new
   release <https://github.com/metoppv/improver/releases/new>`_ page.
   Select your new tag under 'tag version'.
   The **release title** should be the version number (e.g., ``1.1.0``).
   Publish the release after adding any description text.
4. Update the version number and sha256 checksum in the ``meta.yaml``
   file of the conda-forge recipe by opening a pull request in the
   `improver-feedstock <https://github.com/conda-forge/improver-feedstock>`_
   repository. A pull request may be opened automatically for you, in which
   case just check it. The checksum of the compressed ``.tar.gz`` IMPROVER
   source code can be obtained via ``openssl sha256 <file name>``.
   Currently the people with write access to the improver-feedstock
   repository are @benfitzpatrick, @PaulAbernethy, @tjtg, @cpelley and
   @dementipl.
   You can ping one of these people to merge your pull request.
