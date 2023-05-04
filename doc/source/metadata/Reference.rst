Reference
=========

.. contents:: Contents
    :depth: 3

Usage
-----

This section is intended as a quick look-up on the use of metadata items
within IMPROVER; they are listed alphabetically. 
It also indicates whether the metadata item is part of 
the basic `NetCDF`_  metadata,
the Climate and Forecast or `CF Metadata Conventions`_
or specific to IMPROVER
and whether it **must**, **should** or **could** be present.

Metadata items
--------------

altitude
********

This CF coordinate variable holds the height above sea level
for site data
and has the ``standard_name`` attribute set to ``altitude``.

This **should** be present for the site data to allow
the data to be fully exploited.

axis
****

This CF attribute uses a single capitalised character to indicate
how a coordinate variable should be intepreted.
It can take the values ``X``, ``Y``, ``Z`` and ``T``
representing the three spatial directions and time.

Axis values **should** be set for non-scalar coordinate variables,
which for IMPROVER gridded data are ``X`` and ``Y``.
IMPROVER also has ``threshold`` and ``percentile`` 
coordinate variables, but there are no standards for labelling these.

blend_time
**********

This IMPROVER-specific variable 
has been added to indicate when the data was processed (blended)
to generate this forecast, and can be used to indicate how 'fresh'
the data is.
This has the ``long_name`` attribute set to ``blend_time``,
but otherwise takes the same form as the ``time`` variable.

Ideally, this **should** be present.

bounds
******

This CF attribute provides a label pointing to a separate
variable defining the bounds of a point on an axis,
most commonly the start and end of a time period.

This **must** be present if an associated 'bounds variable' exists.

calendar
********

For IMPROVER, this CF attribute is set to ``gregorian`` to indicate
that a Gregorian (standard) calendar is used.

cell_methods
************

This CF attribute can be used to describe the application of simple
statistical processing to a variable 
(e.g. a maximum of the temperature over a period of time).
It is a string comprised of a list of blank-separated words of the form
``name: method``.
The ``name`` can be a dimension of the variable, a scalar coordinate variable,
a valid standard name, or the word ``area``.
The ``method`` should be selected from a standard list 
described in the `CF Metadata Conventions`_ 
It is also possible to included additional information
in the form of a ``comment``

If any method other than ``point`` is specified for a given axis,
then bounds **must** also be provided for that axis.

For example, ``time: maximum`` would indicate the maximum 
over the period of time described by the time bounds.

Cell methods are covered in more detail in the :ref:`stat-section`.

Conventions
***********

This netCDF attribute specifies a space-separated
(or comma-separated if conventions have spaces in their titles) 
list of metadata conventions that the file conforms to.
Up until CF version 1.6, 
strictly only the `CF Metadata Conventions`_ were allowed to be declared here,
but a change at 1.7 allowed multiple conventions. 

This **must** be set to include the appropriate version of the CF Convention
which **should** include any other conventions that are used
(although, at present, there is no entry set automatically to indicate the
extensions used to support enhancements used by IMPROVER).

coordinates
***********

This CF attribute lists any coordinates that do not appear as
dimensioned coordinate variables, 
i.e. those that do not appear as dimensions of the main variable. 
This covers both scalar coordinate variables
(single-valued coordinates, with no dimension)
and auxillary coordinate variables
(variables that contain coordinate data but are not coordinate variables,
usually because they depend on more than one dimension).

This **should** be included where coordinates are present that
do not appear as dimensioned coordinate variables.
For IMPROVER gridded data this would typically be the
scalar coordinate variables:
``blend_time``, ``height`` and ``time`` 
and for spot data the scalar coordinate variables:
``altitude``, ``blend_time``, ``latitude``, ``longitude``, 
``met_office_site_id`` and ``time wmo_id``.

forecast_reference_time
***********************
    
This CF variable represents the nominal data time or start time of a
model forecast run,
and has the ``standard_name`` attribute set to ``forecast_reference_time``.

Ideally, this **should no longer** be used for IMPROVER data.

.. warning::

    Use of ``forecast_reference_time`` in IMPROVER is deprecated
    as it is at best unhelpful and at worst it is confusing,
    as IMPROVER generates a blend from multiple sources
    with different start times so there is no unique data time.

forecast_period
***************

This CF variable represents the interval between
the ``forecast_reference_time`` and the validity time (``time``)
and has the ``standard_name`` attribute set to ``forecast_period``.

Ideally, this **should no longer** be used for IMPROVER data.

.. warning::

    Use of ``forecast_period`` in IMPROVER is deprecated
    as it is at best unhelpful and at worst it is confusing,
    as IMPROVER generates a blend from multiple sources
    with different start times so there is no unique data time.

grid_mapping
************

This CF attribute provides a label pointing to a separate 
grid mapping variable, which more fully describes the map projection. 

This **must** be present for gridded data,
as **must** the associated grid mapping variable.

height
******

This CF vertical coordinate variable is included in some
cases to fully describe the quantity of interest,
for single-level variables appearing as a scalar coordinate variable.

This **should** be included if there is any ambiguity in the interpretation
of quantity of interest if it is omitted.
(e.g. an inclusion of ``height`` with a value of ``1.5 m``
for the representatiion of screen level.) 

history
*******

Ideally, this netCDF attribute should provide a list of the applications
that have modified the original data (i.e. an audit trail),
with recommended practice being to add a date/time stamp
(in the form ``YYYY-MM-DDThh:mm:ssZ``) and identify the software package.
However, in practice, this is far from straightforward for IMPROVER
as it processes a range of model runs,
so there is no single, sequential processing chain
from which to generate such an audit trail,
making it impossible to accurately maintain previous history information.

This is **not currently set** in IMPROVER.

institution
***********

This CF attribute specifies where the original data was produced.

This **must** be present and **should** take the name of the institution from
where the data originated if only data from a single model has been processed.
However, it **should** be set to the institution running the post-processing
for multi-model blended data.

latitude
********

This coordinate variable represents one half of the positional
information for gridded data held on a
Latitude-Longitude (strictly, equirectangular) projection.
This is also used for site positions, which are only provided
in latitude and longitude.
It has the ``standard_name`` attribute set to ``latitude``
and ``units`` set to ``degrees``.
Unless explicitly stated in the metadata,
the latitude and longitude can be considered as relative the WGS84
or the World Geodetic System 1984 datum.

All data **must** contain either this or ``projection_y_coordinate`` variable.

For gridded data, if any statistical processing over the coordinate 
has been applied,
there **must** also be an associated ``latitude_bnds`` variable
providing the bounds over which ``cell_methods`` are applied,
although this is often included anyway to define the cell boundaries.
The ``latitude_bnds`` variable has no attributes as it is tied to the 
main coordinate variable.

least_significant_digit
***********************

This is a variable attribute used by netCDF-writing software to
specify the precision that is maintained when 'bit-shaving'
is applied to provide improved file compression.
The example value of ``3LL`` indicates that a precision of 3 decimal places
is preserved, i.e. values precise to the nearest 0.001.
As 'bit-shaving' is zeroing bits
(that are providing an unrequired level precision),
this would actually be implemented as the power of 2 nearest 0.001.

This is usually included automatically where the precision is limited.

The driver for the use of 'bit-shaving' is that although it requires
no extension to the software to read the data (the number formats
in the file are not changed), it facilitates more effective 
reduction in file size, when lossless compression is applied.

long_name
*********

This netCDF-specific variable attribute provides
a descriptive name that is not governed by CF.
If a `CF Standard Name`_ exists for the quantity, 
this should be used and the ``long_name`` is usually omitted.s

A ``standard_name`` or ``long_name`` **must** be present. 

longitude
*********

This coordinate variable represents one half of the positional
information for gridded data held on a
Latitude-Longitude (strictly, equirectangular) projection.
This is also used for site positions, which are only provided
in latitude and longitude.
It has the ``standard_name`` attribute set to ``longitude``
and ``units`` set to ``degrees``.
Unless explicitly stated in the metadata,
the latitude and longitude can be considered as relative the WGS84
or the World Geodetic System 1984 datum.

All data **must** contain either this or ``projection_x_coordinate`` variable.

For gridded data, if any statistical processing over the coordinate 
has been applied,
there **must** also be an associated ``longitude_bnds`` variable
providing the bounds over which ``cell_methods`` are applied,
although this is ofsten included anyway to define the cell boundaries.
The ``longitude_bnds`` variable has no attributes as it is tied to the 
main coordinate variable.

met_office_site_id
******************

This IMPROVER-specific coordinate variable
is an 8-character string, zero-padded ID number
used by the Met Office to label all sites.
Within the IMPROVER code, the name is user configurable,
such that it can be changed for different institutions / indices.

Although this precise variable is not appropriate for most users
other than the Met Office, it is **advisable** to implement
some form of site identification that has unique elements
and is complete. 

mosg\__
*******

This is intended to indicate a MOSG (Met Office standard grid)
namespace.
It prefixes attributes to show that they are separate from the 
`CF Metadata Conventions`_ attributes.

mosg__model_configuration
*************************

This is an IMPROVER-specific global attribute and
provides a space-separated list of model identifiers
denoting which sources have contributed to the blend.
The naming is fairly arbitary, but at the Met Office
we have chosen to indicate the models in a coded form:

   * ``gl`` = global model
   * ``uk`` = high-resolution UK domain model
   * ``nc`` = (extrapolation-based) nowcast

with a secondary component indicating whether the 
source is deterministic (``det``) or an ensemble (``ens``).
   
For example, ``uk_ens`` indicates our UK ensemble model, MOGREPS-UK.

mosg__model_run
***************

This is an IMPROVER-specific global attribute
which extends the information provided by
``mosg__model_configuration``, to detail the contribution 
of specific model runs (also known as cycles) to the blend. 
This is represented as a list of new line (``\n``) separated
composite entries of the form:

   ``model identifier:cycle time in format yyyymmddTHHMMZ:weight``

percentile
**********

This is an IMPROVER-specific coordinate variable that holds
the set of percentile levels for which values of the variable of
interest are generated.
It has a ``long_name`` attribute set to ``percentile``
and a ``units`` attribute set to ``%``

This **must** be present for percentile variables.

positive
********

Indicates the direction in which values of the vertical coordinate increase,
i.e. where the vertical coordinate is pressure,
the ``positive`` attribute is ``down``.

This **should** be present for vertical coordinates.

projection_x_coordinate
***********************

This coordinate variable represents one half of the positional
information for gridded data held on non-Latitude-Longitude projections.
For example, the Met Office uses a Lambert azimuthal equal area (LAEA) 
projection for the IMPROVER UK domain.
It has a ``standard_name`` attribute set to ``projection_x_coordinate``,
and in the case of the LAEA projection,
the ``units`` attribute is set to ``m``. 

This **must** be provided for gridded data
on a non-Latitude-Longitude projection.
For gridded data, if any statistical processing over the coordinate 
has been applied,
there **must** also be an associated ``projection_x_coordinate_bnds`` variable
providing the bounds over which ``cell_methods`` are applied,
although this is often included anyway to define the cell boundaries.
The ``projection_x_coordinate_bnds`` variable has no attributes
as it is tied to the main coordinate variable.

.. note::
    
    For Met Office data using Lambert azimuthal equal area (LAEA) projection,
    the coordinate can be considered as relative to ETRS89
    or the European Terrestrial Reference System 1989 
    although this is not explicit in the metadata.
    The European Terrestrial Reference System 1989 is a a datum
    based on WGS84, but fixed on 1-Jan-1989
    to be anchored to the Eurasian continental plate. 
    This is realised through a TRF
    (the European Terrestrial Reference Frame or ETRF).
    ETRS89 is ideal for a Europe-wide consistent mapping and datasets,
    and is an EU INSPIRE directive standard.
    In practice, it is close enough to WGS84 to make no difference
    for most applications of post-processed meteorological data.

projection_y_coordinate
***********************

This coordinate variable represents one half of the positional
information for gridded data held on non-Latitude-Longitude projections.
For example, the Met Office uses a Lambert azimuthal equal area (LAEA) grid 
for the IMPROVER UK domain.
It has a ``standard_name`` attribute set to ``projection_y_coordinate``,
and in the case of the LAEA projection,
the ``units`` attribute is set to ``m``. 

This **must** be provided for gridded data
on a non-Latitude-Longitude projection.
For gridded data, if any statistical processing over the coordinate 
has been applied,
there **must** also be an associated ``projection_y_coordinate_bnds`` variable
providing the bounds over which ``cell_methods`` are applied,
although this is often included anyway to define the cell boundaries.
The ``projection_y_coordinate_bnds`` variable has no attributes
as it is tied to the main coordinate variable.

realization
***********

This CF coordinate variable is used for indexing ensemble members
and has the ``standard_name`` attribute set to ``realization``.
This is not usually seen in the metadata of IMPROVER output files,
IMPROVER usually generates probabilities of exceedance or percentiles.
However, it will be seen in the input file metadata
and may be seen in the output data ``cell_methods``
where processing has been applied over realizations
(e.g. ``realization: mean`` for mean wind direction).
By convention, realization zero is the unperturbed or control member.

source
******

This CF attribute specifies the method of production of the original data.

This **must** be present and **should** take the value of the original source
of the data (typically an NWP model)
when no significant post-processing has been applied.
However, where significant adjustment of the data has occurred
or a number input sources have been blended,
it **should** be set to ``IMPROVER``.
Often, careful consideration of when it is appropriate to set this
to reference ``IMPROVER`` is required to avoid the metadata being misleading.
It is probably not worth including a version of the IMPROVER software,
unless this can be reliably supplied.

spot_index
**********

This IMPROVER-specific dimension is used as an increasing integer value
index for sites.

ssp\__
******

This is intended to indicate a SPP (statistical post-processing)
namespace.
It prefixes atributes to show that they are separate from the 
`CF Metadata Conventions`_ attributes.

ssp__relative_to_threshold
**************************

This is an IMPROVER-specific varaible attribute
indicating the nature of the threshold inequality for a probability
and takes one of the following four values:

* ``greater_than`` 
* ``greater_than_or_equal_to``
* ``less_than`` 
* ``less_than_or_equal_to``

standard_name
*************

This CF attribute provides a descriptive name
from the governed `CF Standard Name`_ list.
If no ``standard_name`` exists for the quantity, 
a ``long_name`` must be used.

A ``standard_name`` or ``long_name`` **must** be present. 

string5 / string8
*****************

These IMPROVER-specific arbitary constants are used to dimension
the character length of the string variable holding
zero padded WMO identifiers and Met Office identifiers, respectively.

threshold
*********

This is an IMPROVER-specific coordinate variable that holds
the set of values of the variable of interest for which the
probability values are generated.
The IMPROVER code uses ``var_name="threshold"`` to detect a threshold variable
as a different ``standard_name`` or ``long_name`` attributes will be set for 
different quantities to represent the variable of interest.
The appropriate ``units`` for this will also be set.

This **must** be present for probability variables.

time
****

This CF Variable provides the time at which the parameter value is valid,
and has a ``standard_name`` attribute set to ``time``.
This is an 64-bit integer in ``units`` of ``seconds since 1970-01-01 00:00:00``

This **must** be present.
If any statistical processing over time has been applied
(e.g. accumulation, maxiumum, etc),
there **must** also be ``time_bnds`` variable
providing the time bounds over which ``cell_methods`` are applied.
``time_bnds`` has no attributes as it is tied to the main time variable.

title
*****

This netCDF global attribute provides a succinct description
of what is in the file and should be something that could be used on a plot
to help describe the data. 

This **must** be present, but there is no generally prescribed form
that is must take.

units
*****

This netCDF variable attribute provides the units of measurement for the quantity
in a string form recognised by the Unidata's `UDUNITS package`_

This **must** be present and for IMPROVER this **must** be SI units,
with the exception that ``degrees`` (rather than ``radians``)
are used for wind direction. 
Non-dimensional quantities, such as IMPROVER probabilities,
have units set to ``1``.

weather_code
************

This IMPROVER variable provides a weather code in the form of an integer value.
It has a ``long_name`` attribute set to ``weather_code``
and a ``units`` attribute set to ``1``.
It also has ``weather_code`` and ``weather_code_meaning`` attributes
which can used to map code values to a short description;
the values use for the Met Office IMPROVER implementation are
shown in the table below.

.. csv-table:: Met Office weather codes
   :header: "Code", "Description"
   :widths: 5, 15
   :file: weather_codes.csv


wmo_id
******

This IMPROVER-specific coordinate variable
is a 5-character string, zero-padded ID number for WMO sites.
For non-WMO sites it is set to the string ``None``.
It has a ``long_name`` attribute set to ``wmo_id``.

This is **optional** and only relevant for WMO sites.


References
----------

`CF Metadata Conventions`_

`CF Standard Name`_

`NetCDF`_

`UDUNITS Package`_


.. -----------------------------------------------------------------------------------
.. Links
.. _`CF Metadata Conventions`:
    http://cfconventions.org/

.. _`CF Standard Name`:
    http://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html

.. _`NetCDF`:
    https://docs.unidata.ucar.edu/netcdf-c/current/index.html

.. _`UDUNITS Package`:
    https://www.unidata.ucar.edu/software/udunits/