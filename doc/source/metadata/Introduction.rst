Introduction
============

.. contents:: Contents
    :depth: 3

Overview
--------

Metadata is data that describes other data.

In IMPROVER this takes the form of attributes either within a netCDF file
or within an in-memory Iris cube. 

The principles applied to the metadata witin IMPROVER are: 

 * Conformance to the CF Metadata Conventions, building on this where necessary
 * Clear purpose, with just enough metadata to describe the data sufficiently
 * Nothing misleading or unnecessarily restrictive for wide usage
 * (Ideally) support for referencing more detailed external documentation

Looking at an example
---------------------

The easiest way to explain the IMPROVER metadata from a user perspective
is to dive straight in and look at an actual example of the metadata.
For this purpose, we will consider
gridded probabilities for a 1-hour precipitation_accumulation 
exceeding a range of thresholds
on an extended UK domain generated from Met Office model data. 

There are two common views of the metadata:

* How the ncdump utility would display the netCDF file metadata 
* How iris would display the cube

We will mainly focus on the file view as here as the default output
provides a fuller view
(although iris can be used to fully explore the metadata).

Note that to aid readability, in both cases
``probability_of_lwe_thickness_of_precipitation_amount_above_threshold`` 
has been shortened to ``prob_precip``
and for the ncdump output trailing semicolons have also been removed
and the text for the global attribute
``mosg__model_run`` has line breaks inserted.


Full ncdump of netCDF file metadata
***********************************

Using the command ``ncdump -h filename`` will yield the following output
for our sample file.

.. literalinclude:: rainrate_probs_ncdump.txt
    :lines: -73, 79

Full iris listing cube metadata
*******************************

Using the python command ``print(cube)`` will yield the following output
for our sample file.

.. literalinclude:: rainrate_probs_irisprint.txt
    :linenos:

Global attributes
-----------------

These provide the general information about the file contents
(although they actually appear at the end of the ncdump output).

.. literalinclude:: rainrate_probs_ncdump.txt
    :lines: 62-73
    :emphasize-lines: 2, 10-12

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
    but as IMPROVER applies significant processing of multiple inputs,
    the output of IMPROVER can be considered as original data. 

title
    Succinct description of what is in the file.
    A specific model is specified where data is from a single model 
    and no significant post-processing has been applied. 

The other two attributes are specific to IMPROVER,
originally used in separate Met Office process to 'standardise'
the model output, which is why they are prefixed by ``mosg__``. 
This is intended to indicates a MOSG (Met Office standard grid)
namespace to show that they are seperate from the 
`CF Metadata Conventions`_ attributes.

mosg__model_configuration
   This provides a space separated list of model identifiers
   denoting which sources have contributed to the blend.
   The naming is fairly arbitary, but at the Met Office
   we have chosen to indicate the models in a coded form:
   * ``nc`` = (extrapolation-based) nowcast
   * ``uk`` = high-resolution UK domain model
   * ``gl`` = global model
   with a secondary component indicating whether the 
   source is deterministic (``det``) or an ensemble (``ens``).
   For example, ``uk_ens`` indicates our UK ensemble model,
   MOGREPS-UK.

mosg__model_run
   This attribute extends the information provided by
   ``mosg__model_configuration``, to detail the contribution 
   blend of specific model runs(or cycles). 
   More recently, this has been extended to include the weight
   given to this contribution in the blend.
   This is represented as a list of space-separated composite entries
   of the form:

   ``model identifier:cycle time in format yyyymmddTHHMMZ:weight\n``

Dimensions
----------

These do what the name suggests and provide the name and extent
of the dimensions for the variable arrays. 
In this example, three of these are the dimensions of coordinate variables
and the last is a more general dimension. 

.. literalinclude:: rainrate_probs_ncdump.txt
    :lines: 2-6

projection_y_coordinate
    Number of points in the horizontal x-direction

projection_x_coordinate
    Number of points in the horizontal y-direction. 

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
in the first and third lineslines is replaced by the standard name 
(``lwe_thickness_of_precipitation_amount```) of the coordinate variable associated
with this dimension (also ``threshold``). 

.. literalinclude:: rainrate_probs_irisprint.txt
    :lines: 1-5
    :emphasize-lines: 1, 3

Another dimension that will also be seen is:

percentile
    Number of percentile levels in the files holding precentile values
    rather then probabilities


Variables
---------

Main probability variable
*************************

In this example, the main variable is ``prob_precip`` 
(in the real file it would be,
``probability_of_lwe_thickness_of_precipitation_amount_above_threshold``).
This represents the probability of the 1-hour precipitation
accumulation exceeding a set of thresholds.
It has 3 dimensions and 5 attributes that describe the meteorological quantity
and its relationship to other variables in the metadata.

.. literalinclude:: rainrate_probs_ncdump.txt
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
    it will be present and the long_name will usually be omitted
    (one of these two should always be present). 

units
    the units of measurement for the quantity.
    These will always be SI units. 

cell_methods
    Used to describe the statistical processing applied to the quantity
    that usually changes the interpretation of the data.
    The example here is a slightly strange one, as although
    ``time: sum`` indicates a summing (or accumulation), this in
    actually already captured in the ``standard_name`` or ``long_name``. 
    The ``comment: of lwe_thickness_of_precipitation_amount`` in brackets
    is to clarity that it the summing (or accumulation)
    is not of the probability, but of the underlying quantity,
    the preciputation accumulation, in this exmaple.
    Cell methods are covered in more detail in the User Guide

.. add link to User Guide

grid_mapping
    Although in this case, the name of the projection used,
    this is actually only a label pointing to separate grid mapping variable,
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

.. literalinclude:: rainrate_probs_ncdump.txt
    :lines: 8-14
    :emphasize-lines: 1, 7

In summary these are:

* threshold 
* projection_y_coordinate 
* projection_x_coordinate 
* height 
* time 
* blend_time 
* forecast_period 
* forecast_reference_time 

.. warning::

    ``forecast_period`` and ``forecast_reference_time`` 
    are deprecated and will be dropped in the future.
    There more discussion on this in the sections below.

Probability threshold coordinate variable
*****************************************

As the main variable in this example is a probability of exceeding a threshold, 
a further dimensional coordinate variable is required to allow the data 
to be fully interpreted.
This holds the set of thresholds for the probabilities,
which in this case are a set of 1-hour precipition accumulation values.
Both, these are shown in the code snippets below
(to display the thresholds use ``ncdump -v threshold filename``).


.. literalinclude:: rainrate_probs_ncdump.txt
    :lines: 24-27

.. literalinclude:: rainrate_probs_ncdump.txt
    :lines: 75-78

The variable attributes are:

units
    The units of measure for the quantity, these will always be SI units. 

standard_name
    A descriptive name, in this case it is a `CF Standard Name`_ 
    from the governed list of names, but may instead be a ``long_name``
    if there is no suitable ``standard_name``.
    This represents the quantity for which the probabilities are specified,
    in this example, ``lwe_thickness_of_precipitation_amount`` or 
    precipitation accumulation
    ("lwe" = liquid water equivalent, as this caters for both accumulated
    liquid and ice phase water).

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


References
----------

`CF Metadata Conventions`_


.. -----------------------------------------------------------------------------------
.. Links
.. _`CF Metadata Conventions`:
    http://cfconventions.org/

.. _`CF Standard Name`:
    http://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html