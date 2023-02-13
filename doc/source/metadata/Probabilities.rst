.. _prob-section:

Probabilistic Information
=========================

.. contents:: Contents
    :depth: 3

Context
-------

As IMPROVER is inherently probabalistic,
it seemed appropriate to have a section specifically to focussed this area.
This also looks at the use of `CF Metadata Conventions`_ cell methods
which are also heavily used within IMPROVER.


Representation of probabilistic information
-------------------------------------------

Background
**********

Probabilistic information can be represented in three different ways: 

* Ensemble members (realizations or scenarios) 
* Probabilities of being above, below or between a set of thresholds 
* Percentile values, representing the value of the diagnostics 
  at a set of probability levels 

These all have different strengths and weaknesses in different situations,
which will not be discussed here, but they also have different metadata, 
which will be described here. 

Ensemble members
****************

This is the simplest form, as it is a natural extension to deterministic data,
incorporating a number of realizations (or versions of the forecast). 
As such its representation can be accommodated through the inclusion
of an additional coordinate variable. 

For ensemble member data, the following must be present:

* Dimension ``realization``
* Coordinate variable ``realization``, with:

  * Units ``1``
  * Standard name ``realization``

An example would be the Met Office MOGREPS-UK model,
which runs every hour to generate a 3-member ensemble:

.. code-block:: python

    dimensions:
        realization = 3 ;
    variables: 
        float air_temperature(realization) ;
            air_temperature:standard_name = "air_temperature" ;
            air_temperature:units = "K" ;
        int realization(realization) ;
            realization:units = "1" ;
            realization:standard_name = "realization"  ;

    data: 
     realization = 0, 1, 2 ;


Percentiles
***********

This is probably the second simplest form,
as again it still represents actual sets of values of the diagnostic,
but instead of a set of realizations (consistent over time),
the set of percentile values represent the values at a set of probability levels.
This can again be incorporated by adding a coordinate variable.  

For percentile data, the following must be present:

* Dimension ``percentile``
* Coordinate variable ``percentile``, with:

  * Units ``%``
  * Long name ``percentile``


An example would be a set of percentile values for temperature:

.. code-block:: python

    dimensions:
	    percentile = 13 ;
    variables:
	    float air_temperature(percentile) ;
		    air_temperature:standard_name = "air_temperature" ;
		    air_temperature:units = "K" ;
	    float percentile(percentile) ;
		    percentile:units = "%" ;
		    percentile:long_name = "percentile" ;

    data:
     percentile = 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95 ;0 ;


Probabilities
*************

This is a more interesting form,
as the diagnostics values are transformed into a set of probabilities,
so the metadata is more substantially changed.
This can be catered for with a new coordinate variable to represent
the set of probability thresholds.

For probability data, the following must be present:

* Dimension “threshold”
* Coordinate variable “threshold”, with:

  * Units appropriate to the original diagnostic (``V``, see below) 
  * Standard_name or long_name (as appropriate) set to that of the original diagnostic
    (``V`` in the section below) 

* Main variable, with:

  * Units ``1``
  * Long name set to one of the following (as appropriate): 

    * ``probability_of_V_above_threshold``
    * ``probability_of_V_below_threshold``

    where ``V”`` was the standard or long name of the original variable

* A new non-CF attribute ``spp__relative_to_threshold`` 
    which is used to indicate the nature of the threshold inequality,
    and takes one of the four values:

    * ``greater_than`` 
    * ``greater_than_or_equal_to``
    * ``less_than`` 
    * ``less_than_or_equal_to``

An example would be a set of set of probabilities of temperature
exceeding a set of 79 thresholds:

.. code-block:: python

    dimensions:
        threshold = 79 ; 
    variables:
        float probability_of_air_temperature_above_threshold(threshold) ;
            probability_of_air_temperature_above_threshold:long_name = "probability_of_air_temperature_above_threshold" ;
            probability_of_air_temperature_above_threshold:units = "1" ;
        float threshold(threshold) ;
            threshold:units = "K" ;
            threshold:standard_name = "air_temperature" ;
            threshold:spp__relative_to_threshold = "greater_than_or_equal_to" ;

    data:
     threshold = 213.15, 218.15, 223.15, 228.15, 233.15, 238.15, 243.15, ....


Cell methods
------------

Background
**********

`CF Metadata Conventions`_ provides the attribute cell_methods,
which can be used to describe the characteristic of a field that is represented
by cell values, where a simple statistical method has been apply to a variable.
This is represented as a string attribute comprising
a list of blank-separated words of the form ``name: method``. 
Each ``name: method`` pair indicates that for an axis identified by name,
the cell values representing the field have been determined
or derived by the specified method.

name
    Can be a dimension of the variable, a scalar coordinate variable,
    a valid `CF Standard Name`_, or the word "area". 

method
    Should be selected from a list:
    point,
    maximum,
    maximum_absolute_value,
    median,
    mid_range,
    minimum,
    minimum_absolute_value,
    mean, mean_absolute_value,
    mean_of_upper_decile,
    mode, 
    range,
    root_mean_square,
    standard_derivation,
    sum, sum_of_squares,
    variance.  

If any method other than ``point`` is specified for a given axis,
then bounds should also be provided for that axis.
For example, a one-dimensional array of maximum air temperatures,
could be represented as:

.. code-block:: python

    variables: 
        float air_temperature(time) ;
            air_temperature:cell_methods = "time: maximum" ;
        int64 time ;
            time:bounds = "time_bnds" ;
            time:units = "seconds since 1970-01-01 00:00:00" ;
            time:standard_name = "time" ;
            time:calendar = "gregorian" ;
	int64 time_bnds(bnds) ;

The ``time_bnds`` values define the periods over which 
the statistical processing has been calculated. 
If more than one cell method is to be indicated,
they should be arranged in the order they were applied.
The left-most operation is assumed to have been applied first. 

It is possible to indicate more precisely how the cell method was applied
by including extra information in parentheses after the method.
This information includes standardized and non-standardized parts.
The only standardized option is ``interval``,
used to provide the typical interval between the original data values
to which the method was applied,
in the situation where the present data values are statistically representative 
of original data values which had a finer spacing.

.. warning::

    It is important to understand that the following example, 
    does not (necessarily) represent a 1-hour maximum temperature,
    as the period over which the maximum is derived is included in the bounds
    (not shown), but rather that the temperature data used to calculate
    the maximum was provided in 1-hour steps.

        .. code-block:: python

            air_temperature: cell_methods = "time: maximum (interval: 1 hour)" ;

The non-standardised part is included in the same way,
unless there is also a standardised part,
in which case it is preceded by a ``comment:`` statement.
For example:

.. code-block:: python
        
    air_temperature: cell_methods = "time: mean (time-weighted)" ;
 
Cell methods can be used to describe the characteristic of a field
that is represented by cell values,
where a simple statistical method has been apply to a variable.
However, this leads to an issue,
in that there is a decision to be taken over whether a statistical process
that has been applied is significant for, or of relevance to, the end user,
which is likely to depend on who that user is. 
Two extreme examples of how the cell methods might be presented for the same diagnostic,
in this case a maximum temperature in a period:

1. Simple version, just describing the diagnostic;
   note that without the cell_methods,
   this is a different diagnostic, an instantaneous temperature:

.. code-block:: python

    air_temperature:cell_methods = "time: mean (time-weighted)"

2. Complex version, including a whole chain of processes that have been applied
   to the diagnostic:

.. code-block:: python

    air_temperature:cell_methods = "time: maximum realization: mean area: mean (neighbourhood: square topographic) forecast_reference_time: mean (time-weighted) area: mean (recursive-filter) model: mean (model-weighted)" ;

The issue with the complex version is that the ``“time: maximum”`` 
is required by any user to correctly interpret and use the diagnostic,
whereas the other processing steps tell you more about how it was generated
than how should be interpreted, and, to some extent,
are acting as a substitute for provenance metadata,
and these can obscure the essential statistical information,
making it harder to understand what the diagnostic actually represents.

.. add cross-reference to the Principles section to cover
   different types of metadata

Use of cell methods in IMPROVER
*******************************

IMPROVER should only use cell methods to represent the 'what' metadata of the variable,
i.e. information that is required to correctly interpret the variable.

The use of the ``interval`` within the extra information in cell methods
is not helpful within IMPROVER, as it can be confusing, and so should be omitted

Examples of valid uses of cell methods would be:

* Maximum, minimum and mean value over time, 
  using a cell methods statement of the form (note that there is no ``interval``):

.. code-block:: python

    air_temperature:cell_methods = "time: maximum" ;

* Value within a vicinity, with cell methods using a maximum or minimum, 
  and taking the form:

.. code-block:: python

    air_temperature:cell_methods = "area: maximum(vicinity: radius=50km)" ;

.. need to check the example above is correct - better, replace with actual code
   and add further examples of IMPROVER-specific usage


References
----------

`CF Metadata Conventions`_

`CF Standard Name`_


.. -----------------------------------------------------------------------------------
.. Links
.. _`CF Metadata Conventions`:
    http://cfconventions.org/

.. _`CF Standard Name`:
    http://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html