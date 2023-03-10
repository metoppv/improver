.. _stat-section:

Statistical Processsing
=======================

.. contents:: Contents
    :depth: 3

Context
-------

As IMPROVER both processes forecast data that has been
statistically process end supports a range of statistical processing steps.
As a result, it makes significant use of the `CF Metadata Conventions`_ 
cell methods.

Overview of cell methods
------------------------

`CF Metadata Conventions`_ provide the attribute ``cell_methods``
which can be used to describe the application of simple statistical
processing to a variable
(e.g. a maximum of the temperature over a period of time).
This is represented as a string attribute comprising
a list of blank-separated words of the form ``name: method``. 
Each ``name: method`` pair indicates that for an axis identified by ``name``,
the values representing the variable have been determined
or derived by the specified ``method``.

name
    Can be a dimension of the variable, a scalar coordinate variable,
    a valid `CF Standard Name`_, or the word ``area``. 

method
    Should be selected from a list:
    ``point``,
    ``maximum``,
    ``maximum_absolute_value``,
    ``median``,
    ``mid_range``,
    ``minimum``,
    ``minimum_absolute_value``,
    ``mean, mean_absolute_value``,
    ``mean_of_upper_decile``,
    ``mode``,
    ``range``,
    ``root_mean_square``,
    ``standard_derivation``,
    ``sum, sum_of_squares``,
    ``variance``.

If any method other than ``point`` is specified for a given axis,
then bounds should also be provided for that axis. The ``point`` method 
is unique in that it essentially means that no statistical processing 
has been applied on that axis and is usually omitted,
though could be included if it improved clarity.
As an example of the use of cell methods, consider a one-dimensional array
of maximum air temperatures, which could be represented as:

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

The non-standardised part is included in the same way
(unless there is also a standardised part), for example:

.. code-block:: python
        
    air_temperature: cell_methods = "time: mean (time-weighted)" ;
 
If both standardised and non-standardised parts are present,
the latter must be proceeded by ``comment:`` keyword;
for an example of this, see the `CF Metadata Conventions`_.

Cell methods can be used to describe any simple statistical method
that has been apply to a variable.
However, this leads to the question of whether a statistical process
that has been applied is significant for, or of relevance to, the end user?
This is likely to depend on exactly who that user is
and what they need to know about the data.
Two extreme examples of how the cell methods might be presented
for the same variable (in this case a maximum temperature in a period)
are shown below:

1. Simple version, just describing the information required to correctly
   interpret the variable; note that without the ``cell_methods``,
   this would be a different variable, an instantaneous temperature:

.. code-block:: python

    air_temperature:cell_methods = "time: maximum"

2. Complex version, including a whole chain of processes that have been applied
   to the variable:

.. code-block:: python

    air_temperature:cell_methods = "time: maximum realization: mean area: mean (neighbourhood: square topographic) forecast_reference_time: mean (time-weighted) area: mean (recursive-filter) model: mean (model-weighted)" ;

The issue with the complex version is that only the ``time: maximum`` 
is required by any user to correctly interpret and use the variable.
The other processing steps tell you more about how it was generated
and are really acting as a substitute for provenance metadata.
This can obscure the essential statistical information,
making it harder to understand what the variable actually represents.

Use of cell methods in IMPROVER
*******************************

IMPROVER should only use cell methods to represent the **what** metadata
of the variable,
i.e. information that is required to correctly interpret the variable.
See the section on :ref:`principles-CF-conformance-label` 
in :ref:`principles-label`.

The use of the ``interval`` within the extra information in cell methods
is unhelpful and potentially confusing within IMPROVER
and should be omitted.s

There are two main ways in which cell methods are used
within IMPROVER at present:

1. Maximum, minimum and sum methods applied to time for percentile values,
   using a cell methods statement of the form below:

.. code-block:: python

	float air_temperature(percentile) ;
		air_temperature:standard_name = "air_temperature" ;
		air_temperature:units = "K" ;
        air_temperature:cell_methods = "time: maximum" ;

2. Maximum, minimum and sum methods applied to time for probability values,
   using a cell methods statement of the form below;
   note that there is a non-standard comment to indicate that the
   statistical processing is over the base variable 
   rather than the probability.

.. code-block:: python

    float probability_of_air_temperature_above_threshold(threshold) ;
        probability_of_air_temperature_above_threshold:long_name = "probability_of_air_temperature_above_threshold" ;
        probability_of_air_temperature_above_threshold:units = "1" ;
        probability_of_air_temperature_above_threshold:cell_methods = "time: maximum (comment: of air_temperature)" ;




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