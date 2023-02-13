Reference
=========

.. contents:: Contents
    :depth: 3

Intent
------

This section is intended as a quick look-up on the use of metadata items,
listed alphabetically, within IMPROVER. 
This indicates whether the item is par tof the netCDF metadata standard,
the Climate and Forecast (CF) Metadata Convention 
or specific to IMPROVER.
See also the `CF Metadata Conventions`_ and `NetCDF`_ 
for more further information.

Metadata items
--------------

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


title
*****

This netCDF attribute provides a succinct description of what is in the file 
and should be something that could be used on a plot to help describe the data. 

This **must** be present, but there is no generally prescribed form that is must take.




References
----------

`CF Metadata Conventions`_

`CF Standard Name`_

`NetCDF`_


.. -----------------------------------------------------------------------------------
.. Links
.. _`CF Metadata Conventions`:
    http://cfconventions.org/

.. _`CF Standard Name`:
    http://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html

.. _`NetCDF`:
    https://docs.unidata.ucar.edu/netcdf-c/current/index.html