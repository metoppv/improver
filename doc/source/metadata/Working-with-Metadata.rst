Developer guide to working with metadata
========================================

.. contents:: Contents
    :depth: 3

Background
----------

Object-oriented programming requires an understanding of the metadata
(name, coordinates, attributes, cell methods) associated with a data object.
In IMPROVER, an iris cube is used within the code as a proxy for
a NetCDF file object,
and the conversions from cube to NetCDF-specific metadata are handled
in the load and save wrappers.

This page aims to assist developers in making the correct decisions
about metadata when implementing code, specifically:

* How to deal with metadata when writing new functions / plugins
* When and how metadata needs to be considered when implementing
  a step in the suite
* When and how metadata treatment should be changed when code is modified

Principles of objects and metadata
----------------------------------

In object-oriented programming, metadata is intrinsically linked to the data.
Any publicly callable routine (function or public class method)
that updates the data inside a cube must also update any appropriate metadata,
so that the object returned is correct and self-consistent.

For example, the threshold plugin converts the data in a cube into probabilities.
Metadata wise, it must also do the following:

* Produce a complete and correct threshold-type coordinate describing
  the threshold values and the nature of the data relative to the threshold
  (``greater than``, ``less than``, etc)
* Update the name and units of the cube to reflect that it now contains
  probabilities (e.g. ``probability_of_X_above_threshold``)
* Update any cell methods to be consistent with thresholded data

Encapsulation / responsibility
******************************

Public interfaces **must** take full responsibility for the objects they act upon.
Any function or method that does not update **all** of the required metadata
should either:

* Operate only on the data matrix
  (i.e. take ``cube.data`` rather than ``cube`` as an argument), or
* If within a plugin, be a private method (not callable from outside the plugin)

Extending the example above: any method within the threshold plugin that returns
a cube with thresholded data, but without a new probability name, should be
a private method.

The corollary of this is that functions or classes **must not** modify
data or metadata that is outside of their scope.
A plugin or function's purpose therefore defines precisely which metadata
it must update: no less, and no more.

Writing new functions or plugins
--------------------------------

The code in IMPROVER generally falls into one of four categories:

Post-processing code
    This modifies an existing parameter, for example through
    spatial post-processing
    (neighbourhood processing or recursive filtering),
    or converts between different
    representations of the probability distribution
    (realizations, probabilities and percentiles).
    The name of the underlying parameter (e.g. air_temperature) remains the same.

New parameter creation
    This takes various inputs to create a new parameter,
    for example lapse rate or weather symbols.
    A new parameter name is required.

Calculation of post-processing parameters
    For example EMOS coefficients or reliability tables.
    The output is generally not a data cube, but a cube of parameters
    that will be used to post-process the data at a later stage.
    Plugins to apply these parameters would be post-processing code.

General utility code
    Code that does not fall into any of the three categories above.

There are two abstract base classes in IMPROVER:
``BasePlugin`` and ``PostProcessingPlugin``.
These classes apply some metadata updates automatically,
so it is important to choose the correct type. 
Post-processing code (as defined above) should use 
the ``PostProcessingPlugin`` class.
At the moment, any other plugins that process data should use
the ``BasePlugin`` class.
Plugins that produce ancillaries or have no “process” method 
should not use either of these base classes.

General post-processing
***********************

Post-processing, such as neighbourhood processing,
does not change the underlying nature of the parameter.
Such plugins can therefore copy a cube and modify specific metadata
(coordinates, attributes), but can safely inherit all other existing metadata
as it will remain correct.  Most ``PostProcessingPlugin`` instances
will copy cubes in this way.

New parameters
**************

When creating new parameters, developers **should not** 
copy an input cube directly,
but should make use of the utility to create new parameter cubes.
This makes use of ONLY the coordinates from a template cube,
and adds specific attributes and cell methods as required.
Developers should take care to:

* Provide a correct template cube, i.e. by removing any scalar coordinates
  that are not relevant to the new parameter
* Provide suitable mandatory attributes
  (``source``, ``title`` and ``institution``).
  These should usually be derived using the “generate mandatory attributes”
  function from **all** input parameters.
  (E.g. in weather symbols, all the different fields - 
  precipitation, cloud, lightning, etc - should be read into this function.)
* **NOT** simply pass in all attributes from the template cube,
  as these may be inappropriate to the new parameter
* **NEVER** copy cell methods from the template cube,
  as these will be inappropriate to the new parameter
* Consider whether this parameter may be needed as a level 3 blended field,
  or as input to weather symbols.
  If so, it will need an option to inherit a model ID attribute.

Minimal metadata
****************

The IMPROVER metadata principles include that the metadata should be
the **minimum** required to fully describe the parameter,
and that the metadata should be **correct**.
The main setting where developers need to understand this is in
creating new parameters.
Practical implications include:

Positive selection
    Choosing a specific set of attributes to **include**,
    rather than a specific set to **exclude**.
    This means a new parameter plugin does not inherit anything
    unexpected by default, which may not be “correct” for the new parameter.

Clear internal responsibility
    Defining within the plugin **all** new attributes and / or cell methods 
    which are required to describe this new dataset.

The **only** case for a plugin not taking full responsibility for metadata
is if organisation-specific details - such as the name of the model ID attribute
- need to be passed in via the command line.
Even in these cases, the plugin should take as much responsibility as possible,
requiring minimal information from the user to inform metadata updates.
For example, in the model ID attribute case,
the user is required to provide the name of the attribute from which to read
model information, rather than a ``name: value`` pairing
to be directly applied.
This maximises code flexibility and minimises the chances of
bugs or inconsistencies by clearly recording the expected metadata
within the code, where it can be covered by automated tests.

Implementing a step in the suite
--------------------------------

Metadata is almost exclusively dealt with at the code level,
with plugins taking responsibility for updating the
appropriate metadata internally.
However, there are a few limited cases where the code needs information
to be provided via the command line in order to make the correct updates:

Standardisation
    In the Met Office implementation, the “standardise” step at the start of
    each suite chain has been configured to remove unnecessary attributes
    from incoming data.

New parameters
    If a new parameter is to be blended,
    the name of the model ID attribute needs to be provided via the suite app
    so that this attribute can be included on the parameter file.
    If this argument is omitted,
    the file will not contain source model information and will not be able
    to be blended.

Spot extracted data
    This requires a ``title``, which must currently be provided
    via a command line argument.
    If not provided, the title will default to ``unknown``.

Modifying existing functions or plugins
---------------------------------------

When modifying an existing function or plugin it will not usually be necessary
to change how metadata are treated.
However, it is worth developers considering the following specific questions:

* Have I significantly changed the amount of post-processing
  this plugin is doing?
  If so, does it need to change from a ``BasePlugin``
  to a ``PostProcessingPlugin`` or vice versa?
* Have I changed what this plugin is doing,
  i.e. from producing coefficients or generating a correction to applying them?
  Does it now need to be a ``PostProcessingPlugin``
  where previously it was a general object?
* Is this plugin as a whole taking the right level of responsibility
  for the changes it is making?
  Are there any public methods that take only partial responsibility,
  and so should be private?
* Should this function be a plugin (e.g. feels_like_temperature)?

Some of these are 'nice-to-have' questions, which should be considered
if refactoring a piece of code more widely
(as opposed to one-line changes or small bug fixes),
to help guide the new design.

Using the metadata interpreter
------------------------------

A tool has been developed to help developers identify whether code outputs
are compliant with the IMPROVER standard.

.. note::

    It is probable that the metadata interpreter itself will need to be
    updated or modified in future to accommodate new metadata that is required.

This tool provides the following outputs:

Returns
    A human-readable description of the cube or file contents

Raises
    A list of collated errors if the file is not compliant with the standard

Collates
    A list of warnings if the file has metadata which may not be compliant
    with the “minimal” metadata principle

When using this tool, the developer should consider:

* Whether or not the human-readable output corresponds to 
  their understanding of what the file should contain
* Whether any warnings raised are valid (e.g. regarding unwanted attributes),
  and what to do about them

If errors are raised, the developer is advised to re-run the interpreter
after fixing all the errors, to ensure no further issues are present.

The syntax for using the tool in a Python programme or notebook is:

.. code-block:: python

    from improver.developer_tools.metadata_interpreter import MOMetadataInterpreter, display_interpretation
    interpreter = MOMetadataInterpreter()
    interpreter.run(cube)
    print(display_interpretation(interpreter))

If the supplied cube is not compliant,
a useful error message will be raised by line 3 which can be trapped
and demoted to print a list of the errors
if you want to test multiple cubes at once like this:

.. code-block:: python

    try:
        interpreter.run(cube)
    except:
        print(interpreter.errors)
    else:
        print(display_interpretation(interpreter))

The syntax for the command-line tool is:

.. code-block:: python

    Usage: improver interpret-metadata [OPTIONS] [file-paths...]

    Intepret the metadata of an IMPROVER output into human readable format
    according to the IMPROVER standard. An optional verbosity flag,
    if set to True, will specify the source of each interpreted element.

    This tool is intended as an aid to developers in adding and modifying
    metadata within the code base.

    Arguments:
        file-paths...     File paths to netCDF files for which the metadata 
                          should be interpreted. (type: INPUTPATH)

    Options:
        --verbose         Boolean flag to output information about sources of
                          metadata interpretation.
        --failures-only   Boolean flag that, if set, means only information
                          about non-compliant files is printed.