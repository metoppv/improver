.. _prob-section:

Probability Distribution
========================

.. contents:: Contents
    :depth: 3

Context
-------

As IMPROVER is inherently probabilistic,
it seems appropriate to have a section specifically focused
on the representation of probability distributions in the metadata.
This includes some extensions to the `CF Metadata Conventions`_ 
which provide limited support in this area.

Probability distributions can be represented in one of three different ways: 

* Ensemble members - a set of realizations (or scenarios) each holding
  a possible value of the diagnostic of interest;
* Probabilities of the value of the diagnostic being above, below or between
  a set of thresholds;
* Percentile values representing thresholds of the distribution of the
  diagnostic below which the value will occur with fixed relative frequency.

These all have different strengths and weaknesses in different situations,
which will not be discussed here, but they also have different metadata, 
which will be described here. 

Ensemble members
----------------

This is the simplest form of representing forecast uncertainty,
as it is a natural extension to deterministic forecasts,
incorporating a number of realizations (or versions of the forecast). 
As such, its representation can be accommodated through the inclusion
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
-----------

This is probably the second most straightforward form,
as again it still represents actual sets of values of the diagnostic.
Instead of realizations
(separate scenarios, each self-consistent over time),
the set of percentiles represent the values of a set of thresholds
below which the value of the diagnostic will occur with 
fixed relative frequency.
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
     percentile = 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95 ;


Probabilities
-------------

This is a more interesting form,
as the diagnostics values are transformed into a set of probabilities
of being above, below, or between a set of thresholds,
so the metadata is more substantially changed.
This can be catered for with a new coordinate variable to represent
the set of thresholds.

For probability data, the following must be present:

* Dimension ``threshold``
* Coordinate variable ``threshold``, with:

  * Units appropriate to the original diagnostic
    (indicated by ``V`` in the following text) 
  * Standard_name or long_name (as appropriate) set to that of 
    the original diagnostic (``V``) 

* Main variable, with:

  * Units ``1``
  * Long name set to one of the following (as appropriate): 

    * ``probability_of_V_above_threshold``
    * ``probability_of_V_below_threshold``

    where ``V`` is the standard or long name of the original variable

* A new non-CF attribute ``spp__relative_to_threshold`` 
    which is used to indicate the nature of the threshold inequality,
    and takes one of the four values:

    * ``greater_than`` 
    * ``greater_than_or_equal_to``
    * ``less_than`` 
    * ``less_than_or_equal_to``

An example would be a set of probabilities of temperature
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