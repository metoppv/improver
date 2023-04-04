Introduction
============

.. contents:: Contents
    :depth: 3

Overview
--------

Metadata is data that describes other data.

In IMPROVER this takes the form of attributes either within a netCDF file
or within an in-memory Iris cube. 

The principles applied to the metadata within IMPROVER are: 

 * Conformance to the CF Metadata Conventions, building on this where necessary
 * Clear purpose, with just enough metadata to describe the data sufficiently
 * Nothing misleading or unnecessarily restrictive for wide usage
 * (Ideally) support for referencing more detailed external documentation

Looking at an example
---------------------

The easiest way to explain the IMPROVER metadata from a user perspective
is to dive straight in and look at an actual example.
For this purpose, we will consider
gridded probabilities for a 12-hour maximum temperature 
exceeding a range of thresholds
on an extended UK domain generated from Met Office model data. 

There are two common views of the metadata:

* How the ncdump utility would display the netCDF file metadata 
* How iris would display the cube

We will mainly focus on the file view here as the default output
provides a fuller view
(although iris can be used to fully explore the metadata).

Full ncdump of netCDF file metadata
***********************************

Using the command ``ncdump -v threshold filename`` 
will yield the following output for our sample file.

.. literalinclude:: temp12max_prob_ncdump.txt
    :linenos:
    :tab-width: 4

Full iris listing of cube metadata
**********************************

Using the python command ``print(cube)`` will yield the following output
for our sample file.

.. literalinclude:: temp12max_prob_irisprint.txt
    :linenos:
    :tab-width: 4

Global attributes
-----------------

These provide the general information about the file contents
(although they actually appear at the end of the ncdump output).

.. literalinclude:: temp12max_prob_ncdump.txt
    :tab-width: 4
    :lines: 66-72
    :emphasize-lines: 2, 5-7

The four highlighted attributes are part of the `CF Metadata Conventions`_:

Conventions
    Indicates conformance to version of the `CF Metadata Conventions`_
    (determined by the version of iris, used to write the data). 

institution
    Where the original data was produced.

source
    Method of production of the original data.
    For the model data feeding into this
    in this example, this will be “Met Office Unified Model”, 
    but as IMPROVER applies significant processing to multiple inputs,
    the output of IMPROVER can be considered as original data. 

title
    Succinct description of what is in the file.
    A specific model is specified where data is from a single model 
    and no significant post-processing has been applied. 

The other two attributes are specific to IMPROVER,
which is why they are prefixed by ``mosg__``. 
This is intended to indicate a MOSG (Met Office standard grid)
namespace to show that they are separate from the 
`CF Metadata Conventions`_ attributes.

mosg__model_configuration
   This provides a space separated list of model identifiers
   denoting which sources have contributed to the blend.
   The naming is fairly arbitary, but at the Met Office
   we have chosen to indicate the models in a coded form:

   * ``gl`` = global model
   * ``uk`` = high-resolution UK domain model
   * ``nc`` = (extrapolation-based) nowcast

   with a secondary component indicating whether the 
   source is deterministic (``det``) or an ensemble (``ens``).
   
   For example, ``uk_ens`` indicates our UK ensemble model,
   MOGREPS-UK.

mosg__model_run
   This attribute extends the information provided by
   ``mosg__model_configuration``, to detail the contribution 
   of specific model runs (also known as cycles) to the blend. 
   This is represented as a list of new line (``\n``) separated
   composite entries of the form:

   ``model identifier:cycle time in format yyyymmddTHHMMZ:weight``

Although Met Office examples are provided above, these are configurable.
For example, the ``mosg__model_configuration`` attribute is named
in the CLI calls by specifying the ``model_id_attr`` argument.
Likewise the ``mosg__model_run attribute`` is set
using the ``record_run_attr`` argument.

Dimensions
----------

These do what the name suggests and provide the name and extent
of the dimensions for the variable arrays. 
In this example, three of these are the dimensions of coordinate variables
and the last is a more general dimension. 

.. literalinclude:: temp12max_prob_ncdump.txt
    :tab-width: 4
    :lines: 2-6

projection_y_coordinate
    Number of points in the horizontal y-direction

projection_x_coordinate
    Number of points in the horizontal x-direction. 

threshold
    Number of probability thresholds for the probabilities. 

bnds
    Used to dimension variables that require an upper and lower bound
    (e.g. gridded data grid square boundaries,
    and time step boundaries used to indicate statistical processing
    preiods for maximum and minimum temperatures and 
    precipitation accumulations). 

Conventionally, the coordinate variables are given the same name as their dimensions,
so, in the file metadata you will see the declaration for the threshold variable
is ``threshold(threshold)``.
However, slightly confusingly, when the dimension appears in the iris cube metadata
(see file snippet below),
the actual dimension name
(which is stored in the iris cube as ``var_name=”threshold”``) 
in the first and third lines is replaced by the ``standard_name`` 
(``air_temperature``) of the coordinate variable associated
with this dimension (also ``threshold``). 

.. literalinclude:: temp12max_prob_irisprint.txt
    :tab-width: 4
    :lines: 1-5
    :emphasize-lines: 1, 3

Another dimension that will also be seen is:

percentile
    Number of percentiles in files holding percentile values
    rather then probabilities


Variables
---------

Main probability variable
*************************

In this example, the main variable is
``probability_of_air_temperature_above_threshold``,
which represents the probability of the 12-hour maximum temperature
exceeding a set of thresholds.
It has 3 dimensions and 5 attributes that describe the meteorological quantity
and its relationship to other variables in the metadata.

.. literalinclude:: temp12max_prob_ncdump.txt
    :tab-width: 4
    :lines: 8-14

The variable attributes are:

least_significant_digit
    Specifies the precision that is maintained when 'bit-shaving'
    is applied to provide improved file compression.
    The example value of ``3LL`` indicated that a precision of 3 decimal places
    is preserved, i.e. values precise to the nearest 0.001
    (actually implemented as the power of 2 nearest 0.001). 

long_name
    A descriptive name that is not governed by CF.
    If a `CF Standard Name`_ exists for the quantity, 
    it will be present and the ``long_name`` will usually be omitted
    (one of these two should always be present). 

units
    the units of measurement for the quantity.
    These will always be SI units. 
    In this example the unit is 1 as the variable is 
    a probability rather than a temperature.

cell_methods
    Used to describe the statistical processing applied to the quantity
    that usually changes the interpretation of the data.
    ``time: maximum`` indicates the maximum over the period of the time bounds.
    The ``comment: of air_temperature`` in brackets
    is to clarify that the maximum
    is not of the probability, but of the underlying quantity,
    which is the temperature in this example.
    Cell methods are covered in more detail in the :ref:`stat-section`

grid_mapping
    Although in this case, the name of the projection used,
    this is actually only a label pointing to a separate grid mapping variable,
    which more fully describes the map projection. 

coordinates
    This lists the scalar coordinates,
    i.e. those that do not appear as dimensions of the main variable. 


Coordinate variables
********************

In this example, there 8 coordinate variables.
Three that appear as dimensions on the variable 
and 5 scalar coordinates listed in the coordinates attribute
(both highlighted in the code snippet below)

.. literalinclude:: temp12max_prob_ncdump.txt
    :tab-width: 4
    :lines: 8-14
    :emphasize-lines: 1, 7

In summary these are:

* threshold 
* projection_y_coordinate 
* projection_x_coordinate 
* blend_time
* forecast_period 
* forecast_reference_time 
* height 
* time 


.. warning::

    ``forecast_period`` and ``forecast_reference_time`` 
    are deprecated and will be dropped in the future.
    There more discussion on this in the sections below.

Probability threshold coordinate variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As the main variable in this example is a probability of exceeding a threshold, 
a further dimensional coordinate variable is required to allow the data 
to be fully interpreted.
This holds the set of thresholds for the probabilities,
which in this case are a set of 12-hour maximum temperature values.
Both, these are shown in the code snippets below

.. literalinclude:: temp12max_prob_ncdump.txt
    :tab-width: 4
    :lines: 24-27, 73-83

The variable attributes are:

units
    The units of measure for the quantity, these will always be SI units. 

standard_name
    A descriptive name, in this case it is a `CF Standard Name`_ 
    from the governed list of names, but may instead be a ``long_name``
    if there is no suitable ``standard_name``.
    This represents the quantity for which the probabilities are specified,
    in this example, ``air_temperature``.

ssp__relative_to_threshold
    This attributes is specific to IMPROVER,
    which is why they are prefixed by ``spp__``. 
    This is intended to indicates a SPP (statistical post-processing)
    namespace to show that it is seperate from the 
    `CF Metadata Conventions`_ attributes.
    It is used to indicate the nature of the threshold inequality,
    and takes one of the four values:

    * ``greater_than`` 
    * ``greater_than_or_equal_to``
    * ``less_than`` 
    * ``less_than_or_equal_to`` 

Time coordinate variables
^^^^^^^^^^^^^^^^^^^^^^^^^

At present, most parameters have two time coordinate variables, 
``time`` and ``blend_time``, 
with a further variable providing the bounds of the time step for parameters 
where this information is required.

.. literalinclude:: temp12max_prob_ncdump.txt
    :tab-width: 4
    :lines: 40-43, 59-64

time
    The time at which the parameter value is valid.

blend_time
    Has been added to indicate when the data was processed (blended)
    to generate this forecast, and can be used to indicate how 'fresh'
    the data is. 
    (For the Met Office continually updating dataset, 
    the ``blend_time`` will not be the same for all validity times,
    as forecasts in the near future are updated more frequently).

.. previously, this included text the text below,
    but is this still true?
    There is one special case where blend_time is not set at present:
    wind direction – more information on this is given later. 

time_bnds
    Describes the start and end points of the time step
    (for the maximum temperature example here,
    it would represent a 12-hour period). 
    In IMPROVER (as is standard practice for meteorological parameters),
    the time is at the end of the time step defined by these bounds. 


There are two further time coordinate variables which a have been **deprecated**
and will be removed in the future:

forecast_reference_time
    This is also a `CF Standard Name`, and used to represent
    the nominal data time or start time of a model forecast run.
    However, as IMPROVER generates a blend from multiple sources
    with different start times, there is no unique data time,
    so the use of ``blend_time`` is more appropriate.

forecast_period
    This usually represent the interval between the ``forecast_reference_time``
    and the validity time (``time``), but as stated above,
    there is no unique ``forecast_reference_time``, and forecasts valid at
    different times may have a different ``blend_time``, so at best
    ``forecast_period`` is unhelpful, at worst it is confusing.

The time coordinate variables share a common set of attributes:

standard_name / long_name
    A descriptive name, either from the `CF Standard Name`_ list 
    (e.g. ``time``) or  
    a non-standard ``long_name`` (e.g. ``blend_time``)

units
    Units of measure for the quantity.
    As these are usually in seconds relative to midnight on 1st January 1970,
    so usually some formatting is required to be human-readable. 

calendar
    Indicates that a Gregorian (standard) calendar is used. 

bounds
    Pointer to the variable defining the start and end of the time period,
    if present. 


Horizontal coordinate variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two horizontal coordinate variables, 
here projection_y_coordinate and projection_x_coordinate
(but would be latitude and longitude for an equirectangular 
projection grid, such as the Met Office global domain),
provide the coordinates of the grid points at the centre of the grid cell,
with two further variables defining the cell bounds.

.. literalinclude:: temp12max_prob_ncdump.txt
    :tab-width: 4
    :lines: 28-39

The horizontal coordinates variables share a common set of attributes:

standard_name
    A descriptive name from the `CF Standard Name`_ list.

units
    The units of measure for the quantity, these will always be SI units,
    with the exception that degrees are used in preference to radians

axis
    Indicates whether the coordinate should be regarded as an “X” or “Y” 
    Cartesian coordinate.  

bounds
    Pointer to the variable defining the edges of the grid cells 

Vertical coordinate variable 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some parameters also have a vertical scalar coordinate, height
(in the example here, it is ``1.5 m`` representing screen level): 

.. literalinclude:: temp12max_prob_ncdump.txt
    :tab-width: 4
    :lines: 55-58

It's attributes are:

standard_name
    A descriptive name, either from the `CF Standard Name`_ list.

units
    The units of measure for the quantity, these will always be SI units.  

positive
    Indicates the direction in which values of the vertical coordinate increase,
    i.e. where the vertical coordinate is pressure,
    the ``positive`` attribute is ``down``


Grid mapping variable
*********************

This describes the grid map projection.
The example here is for the Lambert azimuthal equal area (LAEA) grid 
used by the Met Office for the UK domain,
but a Latitude-Longitude (strictly, equirectangular) projection
is usually used for the global domain.
The set of attributes will vary depending on map projection,
so to get the exact meanings, it is best to look at 
“Appendix F: Grid Mappings” in the `CF Metadata Conventions`_

.. literalinclude:: temp12max_prob_ncdump.txt
    :tab-width: 4
    :lines: 15-23

Other forms of IMPROVER data
----------------------------

Percentiles
***********
The example explored so far in the document describes the metadata
for probabilities of exceeding a threshold, 
but IMPROVER can also generate forecasts as a set of percentiles, 
for which the metadata will differ slightly. 
These difference are described below, but the full ncdump listing
is shown in an appendix.

Dimensions
^^^^^^^^^^

The ``threshold`` dimension is replaced by ``percentile`` dimension

.. literalinclude:: temp12max_perc_ncdump.txt
    :tab-width: 4
    :lines: 2-6
    :emphasize-lines: 2
 
percentile
    The dimension for the coordinate describing the percentile values. 

Main percentile variable 
^^^^^^^^^^^^^^^^^^^^^^^^

In the example here, the main variable is a 12 hour maximum temperature,
but representing the Nth percentile of the temperature forecast probability distribution
at each point. 

.. literalinclude:: temp12max_perc_ncdump.txt
    :tab-width: 4
    :lines: 8-14

The variable attributes are the same as the probability example
previously described in detail, except: 

standard_name
    Can be used rather than a ``long_name``, as ``air_temperature``
    exists as a descriptive name in the `CF Standard Name`_ list. 

cell_methods
    Used to describe statistical post-processing applied to the quantity.
    Cell methods no longer require the additional non-standardized part, 
    ``(comment: of air_temperature)``, as they now refer to the main variable
    that is a maximum temperature over 12 hours. 

Percentile coordinate variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As the main variable provides the air temperature at a set of percentiles,
a further dimensional coordinate variable is required to allow the data to be 
fully interpreted. This holds the set of percentile values. 

.. literalinclude:: temp12max_perc_ncdump.txt
    :tab-width: 4
    :lines: 24-26, 72-74

The variable attributes are:

long_name
   ``percentile`` which is used for all IMPROVER percentile variables. 
   It is not in the `CF Standard Name`_ list.

units
    The units here are a percentage. 


Spot data 
*********

Broadly, the spot data parallels the form of the gridded data, 
both probabilities and percentiles are produced,
but there are some notable differences because of the nature of the data.
These difference are described below, but the full ncdump listing
is shown in an appendix.

Global attributes 
^^^^^^^^^^^^^^^^^

Only the title is different for the spot data. 

.. literalinclude:: temp12max_spotperc_ncdump.txt
    :tab-width: 4
    :lines: 60-66
    :emphasize-lines: 6

Dimensions
^^^^^^^^^^

The dimensions for the horizontal coordinate variables have been replaced by
a dimension for an index coordinate for the sites, 
and a dimension for the WMO identifier string length has been added. 

.. literalinclude:: temp12max_spotperc_ncdump.txt
    :tab-width: 4
    :lines: 2-7

spot_index
    The dimension for the index for the set of sites. 

string5 / string8
    Constants used to dimension the character length of the string variable
    holding zero padded WMO identifier and Met Office identifiers, respectively.

Variables
^^^^^^^^^

The majority of variables are the same for both gridded data and spot data.
Only the differences will be discussed here.
The variable attributes are common to the gridded data, so these will not be discussed. 

.. literalinclude:: temp12max_spotperc_ncdump.txt
    :tab-width: 4
    :lines: 18-23, 43-50, 57-58

spot_index
    An arbitrary integer index for the sites. 

altitude
    Height of site above sea level. 

latitude, longitude
    Site positions are only provided in latitude and longitude,
    regardless of the projection used for the corresponding grid.
    These can be considered as relative the WGS84
    or the World Geodetic System 1984 datum,
    although this is not explicit in the metadata. 

met_office_site_id
    This is an 8-character string, zero-padded ID number
    used by the Met Office to label all sites.
    The name is user configurable, such that it can be changed
    for different institutions / indices.

wmo_id
    For WMO sites, this is a 5-character string, zero-padded ID number. 
    The non-WMO sites are set to the string "None" as iris cannot save NaN.


Special parameters
------------------

Weather code
************

At present, a single weather code is offered,
which is derived from the full set of probabilistic data. 
For this reason, some of the metadata are slightly different. 
These difference are described below, but the full ncdump listing
is shown in an appendix.

As this is deterministic, there is no dimension and coordinate variable for 
``threshold`` or ``percentile``. 

There are two new coordinate variables.

.. literalinclude:: wx_code_ncdump.txt
    :tab-width: 4
    :lines: 11-12

weather_code
    Is the integer code used to represent the weather type in the main variable data. 

weather_code_meaning
    Provides corresponding short descriptions of each weather type in the weather_code list. 

Wind direction
**************

At present, probabilities and percentiles are not offered for wind direction, 
just a single set of deterministic values, which are a mean across realizations
(ensemble members) from a single model. 
For this reason, some of the metadata are slightly different 
from the nearest equivalent percentiles. 
These difference are described below, but the full ncdump listing
is shown in an appendix.

.. note::

    In future, we expect to at least be able to generate 
    percentiles of wind direction.


The global attribute ``mosg__model_configuration`` is always a single model 
(for the Met Office ``uk_ens`` or ``gl_ens``)

For gridded data only, the ``title`` only identifies the single model used
(for the Met Office, this would be either
``Post-processed MOGREPS-UK Model Forecast on UK 2 km Standard Grid`` or 
``Post-processed MOGREPS-G Model Forecast on UK 2 km Standard Grid``),
rather than referring to the IMPROVER Blend
(although the spot data is still labelled as IMPROVER,
as a site extraction has taken place). 

As this is a mean, there is no dimension and coordinate variable for 
``threshold`` or ``percentile``. 

The main variable has a ``cell_methods`` attribute, set to
``realization: mean``.

At present there is no ``blend_time`` coordinate variable.
For now an idea of the time at which the data was generated 
can be obtained from the ``forecast_reference_time``


References
----------

`CF Metadata Conventions`_

`CF Standard Name`_


Appendices
----------

Full ncdump of netCDF file metadata for gridded percentiles
***********************************************************

Using the command ``ncdump -v percentile filename`` 
will yield the following output for our sample file.

.. literalinclude:: temp12max_perc_ncdump.txt
    :linenos:
    :tab-width: 4

Full ncdump of netCDF file metadata for spot percentiles
********************************************************

Using the command ``ncdump -v percentile filename`` 
will yield the following output for our sample file.

.. literalinclude:: temp12max_spotperc_ncdump.txt
    :linenos:
    :tab-width: 4

Full ncdump of netCDF file metadata for spot weather codes
**********************************************************

Using the command ``ncdump -h filename`` 
will yield the following output for our sample file.

.. literalinclude:: wx_code_ncdump.txt
    :linenos:
    :tab-width: 4

Full ncdump of netCDF file metadata for wind direction
******************************************************

Using the command ``ncdump -h filename`` 
will yield the following output for our sample file.

.. literalinclude:: windir_mean_ncdump.txt
    :linenos:
    :tab-width: 4

.. -----------------------------------------------------------------------------------
.. Links
.. _`CF Metadata Conventions`:
    http://cfconventions.org/

.. _`CF Standard Name`:
    http://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html