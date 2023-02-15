Reference
=========

.. contents:: Contents
    :depth: 3

Usage
-----

This section is intended as a quick look-up on the use of metadata items,
listed alphabetically, within IMPROVER. 
This indicates whether the item is part of the netCDF metadata standard,
the Climate and Forecast (CF) Metadata Convention 
or specific to IMPROVER.
See also the `CF Metadata Conventions`_ and `NetCDF`_ 
for more further information.

Metadata items
--------------

altitude
********

This coordinate variable holds the height above sea level
for site data
and has the ``standard_name`` attribute set to ``atlitude``.

This **should** be present for the site data to allow
the data to be fully exploited.

axis
****

This CF attribute uses a single capitalised character to indicate
how a coordinate variable should be intpreted.
It can take the values ``X``, ``Y``, ``Z`` and ``T``
representing the three spatial directions and time.

Axis values **should** be set for non-scalar coordinate variables,
which for IMPROVER gridded data is ``X`` and ``Y``.
IMPROVER also has ``threshold`` and ``percentile`` 
coordinate varaibles, but there are no standards for labelling these.

bounds
******

This CF attribute provides a label pointing to a separate
variable defining the start and end of the time period.

This **must** be present if an asociated 'bounds variable' exists.


calendar
********

This CF attribute indicates that a Gregorian (standard) calendar is used,
taking the value ``gregorian``.

cell_methods
************

This CF attribute describes any statistical processing applied to the quantity
that usually changes the interpretation of the data.
It is a string comprising a list of blank-separated words of the form
``name: method``, where
the ``name`` can be a dimension of the variable, a scalar coordinate variable,
a valid standard name, or the word ``area`` and
the ``method`` should be selected from a standard list 
described in the `CF Metadata Conventions`_ 
It is also possible to included additional information
in the form of a ``comment``

If any method other than ``point`` is specified for a given axis,
then bounds **should** also be provided for that axis.

For example, ``time: maximum`` would indicate the maximum 
over the period of the time described by the time bounds.

Cell methods are covered in more detail in the :ref:`stat-section`.

Conventions
***********

This netCDF attribute specifies a space-separated
(or, if necessary, comma-separated) 
list of metadata conventions that the file conforms to.
Up until CF version 1.6, 
strictly only the CF convention was allowed to be declared,
but a change at 1.7 allows multiple conventions. 

This **must** be set to include the appropriate version of CF
and should include any other conventions that are used.

blend_time
**********

This is an IMPROVER-specific variable 
has been added to indicate when the data was processed (blended)
to generate this forecast, and can be used to indicate how 'fresh'
the data is.
This has the ``long_name`` attribute ``time``.
but otherwise takes the same form as the ``time`` variable.

Ideally, this **should** be present.

coordinates
***********

This CF attribute lists the scalar coordinates,
i.e. those that do not appear as dimensions of the main variable. 

This **should** be included where scalar varaiables are present.
For IMPROVER gridded data this should typically include
``blend_time height time`` 
and for spot data this would typically include
``altitude blend_time latitude longitude met_office_site_id time wmo_id``.

forecast_reference_time
***********************
    
This represents the nominal data time or start time of a model forecast run,
and has the ``standard_name`` attribute set to ``time``.

Ideally, this **should not** be used for IMPROVER data.

.. warning::

    Use of ``forecast_reference_time`` in IMPROVER is deprecated
    as it is at best unhelpful and at worst it is confusing,
    as IMPROVER generates a blend from multiple sources
    with different start times so there is no unique data time.

forecast_period
***************

This represents the interval between the ``forecast_reference_time``
and the validity time (``time``)
and has the ``standard_name`` attribute set to ``time``.

Ideally, this **should not** be used for IMPROVER data.

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

This CF scalar coordianate variable is included in some
cases to fully describe the quantity of interest.

This **should** be included if there is any ambiguity in
quantity of interest if it is excluded.
(e.g. an inclusion of ``height`` with a value of ``1.5 m``
for the representatiion of screen level.) 

history
*******

Ideally, it should should provide a list of the applications
that have modified the original data (i.e. an audit trail),
with recommended practice being to add a date/time stamp
(in the form YYYY-MM-DDThh:mm:ssZ) and identify the software package.
However, in practice, this is far from straightforward,
as IMPROVER processes a range of model runs,
so there is no single, sequential processing chain
from which to generate such an audit trail,
making it impossible to accurately maintain previous history information.

This netCDF attribute is **not currently set** in IMPROVER.

institution
***********

This CF attribute specifies where the original data was produced.

This **must** be present and **should** take the name of the institute
from where the data originated from for data from a single model,
but **should** be set to the institution running the post-processing
for multi-model blended data.

latitude
********

This coordinate variable represents one half of the positional
information for gridded data held on a
Latitude-Longitude (strictly, equirectangular) projection.
This is also used for site positions, which are are only provided
in latitude and longitude.
It has the ``standard_name`` attribute set to ``latitude``
and ``units`` set to ``degrees``.
These can be considered as relative the WGS84
or the World Geodetic System 1984 datum,
although this is not explicit in the metadata.

This **must** be provided for site data and for gridded data
on a Latitude-Longitude projection.
For gridded data, if any statistical processing over coordinate 
has been applied (e.g. mean, etc),
there **must** also be ``latitude_bnds`` variable
providing the bounds over which ``cell_methods`` are applied.
This has no attributes as it is tied to the main coordinate variable


least_significant_digit
***********************

This is a variable attribute used by netCDF-writing software to
specify the precision that is maintained when 'bit-shaving'
is applied to provide improved file compression.
The example value of ``3LL`` indicated that a precision of 3 decimal places
is preserved, i.e. values precise to the nearest 0.001.
As 'bit-having' is zeroing bit providing unrequired precision,
this woulf actually be implemented as the power of 2 nearest 0.001. 

long_name
*********

This netCDF-specific variable attribute provides
a descriptive name that is not governed by CF.
If a `CF Standard Name`_ exists for the quantity, 
this should be used and the ``long_name`` is usually be omitted.

A ``standard_name`` or ``long_name`` **must** be present. 

longitude
*********

This coordinate variable represents one half of the positional
information for gridded data held on a
Latitude-Longitude (strictly, equirectangular) projection.
This is also used for site positions, which are are only provided
in latitude and longitude.

These can be considered as relative the WGS84
or the World Geodetic System 1984 datum,
although this is not explicit in the metadata.

This **must** be provided for site data and for gridded data
on a Latitude-Longitude projection.
For gridded data, if any statistical processing over coordinate 
has been applied (e.g. mean, etc),
there **must** also be ``longitude_bnds`` variable
providing the bounds over which ``cell_methods`` are applied.
This has no attributes as it is tied to the main coordinate variable

met_office_site_id
******************

This IMPROVER-specific coordinate variable
is an 8-character string, zero-padded ID number
used by the Met Office to label all sites.
Within the IMPROVER code, the name is user configurable,
such that it can be changed for different institutions / indices.

Although this precise variable is not appropriate for most users
other than the Met Office, it is **advisable** to implemented
some form of site identification that has unique elements
and is complete. 

mosg\__
*******

This is intended to indicate a MOSG (Met Office standard grid)
namespace.
It prefixes atributes to show that they are separate from the 
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
the set of percentile level for which values of the variable of
interest are  generated.
It has a ``long_name`` attribute set to ``percentile``
and a ``units`` attribute set to ``%``

This **must** be present for percentile variables.

positive
********

Indicates the direction in which values of the vertical coordinate increase,
i.e. where the vertical coordinate is pressure,
the ``positive`` attribute is ``down``.

This **should** be present for height coordinates.

projection_x_coordinate
***********************

This coordinate variable represents one half of the positional
information for gridded data held on non-Latitude-Longitude projections.
For example, the Met Office uses a Lambert azimuthal equal area (LAEA) grid 
for the IMPROVER UK domain.
It has a ``standard_name`` attribute set to ``projection_x_coordinate``,
and in the case of the LAEA projection,
the ``units`` attribute is set to ``m``. 
This can be considered as relative to ETRS89
or the European Terrestrial Reference System 1989 
although this is not explicit in the metadata.

This **must** be provided for gridded data
on a non-Latitude-Longitude projection.
If any statistical processing over coordinate 
has been applied (e.g. mean, etc),
there **must** also be ``projection_x_coordinate_bnds`` variable
providing the bounds over which ``cell_methods`` are applied.
This has no attributes as it is tied to the main coordinate variable

projection_y_coordinate
***********************

This coordinate variable represents one half of the positional
information for gridded data held on non-Latitude-Longitude projections.
For example, the Met Office uses a Lambert azimuthal equal area (LAEA) grid 
for the IMPROVER UK domain.
It has a ``standard_name`` attribute set to ``projection_y_coordinate``,
and in the case of the LAEA projection,
the ``units`` attribute is set to ``m``. 
This can be considered as relative to ETRS89
or the European Terrestrial Reference System 1989 
although this is not explicit in the metadata.

This **must** be provided for gridded data
on a non-Latitude-Longitude projection.
If any statistical processing over coordinate 
has been applied (e.g. mean, etc),
there **must** also be ``projection_y_coordinate_bnds`` variable
providing the bounds over which ``cell_methods`` are applied.
This has no attributes as it is tied to the main coordinate variable

.. note::
    
    European Terrestrial Reference System 1989 is a a datum,
    based, on WGS84, but fixed on 1-Jan-1989,
    to be anchored to the Eurasian continental plate. 
    This is realised through the a TRF
    (the European Terrestrial Reference Frame or ETRF).
    ETRS89 is ideal for a Europe-wide consistent mapping and data sets and
    an EU INSPIRE directive standard.
    In practice, it is close enough WGS84 to make no difference
    for most applications of post-processed meteorological data.

realization
***********

This CF coordinate variable is used for indexing ensemble members
and has the ``standard_name`` attribute set to ``realization``.
This is not usually seen in IMPROVER output fiel metadata
as these usually contain  probabilities of exceedance or percentiles,
but it will be seen in the input file metadata
and may be seen in the output data ``cell_methods``
where processing has been applied over realizations
(e.g. ``realization: mean`` for mean wind direction)
.
source
******

This CF attribute specifies the method of production of the original data.

This **must** be present and **should** take the value of the original source data
where no significant post-processing has been applied 
but should be set “IMPROVER” where significant correction has occurred;
consideration of where it is appropriate to set this to reference IMPROVER
so that the metadata does not become misleading
needs carefully consideration.
It is probably not worth including a version of the IMPROVER software,
unless this can be reliably supplied.

spot_index
**********

This is a dimension for sites, and is simply an increasing integer value.

ssp\__
******

This is intended to indicate a SPP (statistical post-processing)
namespace.
It prefixes atributes to show that they are separate from the 
`CF Metadata Conventions`_ attributes.

ssp__relative_to_threshold
**************************

This is an IMPROVER-specific varaible attribute
indicates the nature of the threshold inequality for a probability
and takes one of the four values:

* ``greater_than`` 
* ``greater_than_or_equal_to``
* ``less_than`` 
* ``less_than_or_equal_to``

standard_name
*************

This CF attribute provides a descriptive name,
from the governed `CF Standard Name`_ list.
If no `standard_name`` exists for the quantity, 
a ``long_name`` must be used.

A ``standard_name`` or ``long_name`` **must** be present. 

string5 / string8
*****************

Arbitary constants used to dimension the character length of the string variable
holding zero padded WMO identifier and Met Office identifiers, respectively.

threshold
*********

This is an IMPROVER-specific coordinate variable that holds
the set of values of the variable of interest for which the
probability values are generated.
It has a ``long_name`` attribute set to ``threshold``.

This **must** be present for probability variables.

time
****

Variable providing the time at which the parameter value is valid,
which has a ``standard_name`` attribute set to ``time``.
This is an 64-bit integer in ``units`` of
``seconds since 1970-01-01 00:00:00``

This **must** be present.
If any statistical proceessing over time 
has been applied (e.g. accumulation, maxiumum, etc),
there **must** also be ``time_bnds`` variable
providing the time bounds over which ``cell_methods`` are applied.
This has no attributes as it is tied to the main time variable.

title
*****

This netCDF global attribute provides a succinct description of what is in the file 
and should be something that could be used on a plot to help describe the data. 

This **must** be present, but there is no generally prescribed form that is must take.

units
*****

This netCDF variable attribute provides the units of measurement for the quantity.
in a string form recognised by the Unidata's `UDUNITS package`_

This **must** be present,and for IMPROVER this **must** be SI units,
with the exception that degrees are used rather than radians. 
Non-dimensional quantities, such as IMPROVER probabilities
have units set to "1".

weather_code
************

This IMPROVER variable provides a weather code in the forms
of an integer value.
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
For non-WMO sites it it set to the string "None".
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