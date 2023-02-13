Principles
==========

.. contents:: Contents
    :depth: 3

Purpose
-------

The principles applied to the metadata within IMPROVER have been listed
in the Introduction, but are discussed in more detail in this section.


Conformance to the CF Metadata Conventions
------------------------------------------

As far as possible the implementation should conform to 
`CF Metadata Conventions`_ (CF), building on this where necessary.
However, whilst CF does a good job with 'where' and 'what'
metadata (see note below), 
it does not really cover 'who' metadata.
Therefore, it is helpful to keep CF
(and direct extensions to this) focussed on these aspects
and not try to stretch it to cover 'who'.

An example is CF cell methods, which are intended to cover 'what'
(i.e. to ensure that data can be correctly interpreted) 
but, if care is not taken, can unwittingly be extended into 'who'
(e.g. to try to describe elements of the data processing), 
which can complicate its main purpose. 

.. note::

    **Metadata Categorization** 

    To inform decisions about the implementation of metadata, 
    it can be helpful to consider three types of metadata in terms
    of what they are trying to achieve:

    * **Where** - location in space and time, and additional 'data space' information

    * **What** - what is this value? i.e. phenomenon, units, extra characteristics

    * **Who** - data identity, i.e. where it came from, provenance, data processing, etc

Clear purpose
-------------

Metadata should not be present, if it does not have a purpose
(i.e. just because it might be useful in future), as,
if no one is using it, it is unlikely to be properly maintained across all files.

There should be just enough metadata to describe the data sufficiently

Nothing misleading or unnecessarily restrictive
-----------------------------------------------

Metadata that misleading, or simply wrong, should not be present.

Also, if at all possible, metadata that restricts wider use
should be avoided. 

Referencing more detailed external documentation
------------------------------------------------

Ideally, the metadata should support the ability to refer out
to more detailed descriptions of the data to avoid 'overloading'
the files with metadata. This allows more detail of the 
provence of the data and the processing applied (the 'where' metadata
described in an earlier note) to be provided.

One approach to help support thia that has been partially adopted within IMPROVER
is the use of such as name-spacing for new (non-CF) attributes.
This has two purposes:

 * Clearly separates the attribute from CF standard attributes,
   identifying it as belonging to a separate statistical metadata convention.
 * Allowing it to be resolved to a definition held in an external registry in the future

Namespacing is implemented using a double underscore to maintain CF conformance
whilst clearly identiying the namespace part of the name.
For example, the "ssp" is used as statistical post-processing namespace,
with the attributes in this namespace prefixed by ``spp__``.
At present this is only used for a single attribute ``ssp__relative_to_threshold``,
which takes one of the four values:

    * ``greater_than`` 
    * ``greater_than_or_equal_to``
    * ``less_than`` 
    * ``less_than_or_equal_to`` 

The other namespace used is "mosg", which is used to indicate
a Met Office standard grid attribute.
At present this is used for two attributes:
``mosg__model_configuration``, which identifies the 
sources that have contributed to the blend, and
``mosg__model_run``, which extends the information provided by
the contribution of specific model runs and 
the level of their contribution.
 
At present, IMPROVER doesss not provide explicit References
to external metadata. 
Earlier in the development process,
the use of NetCDF Linked Data (see note below) was explored,
but the method of implementation a the time created issues
that outweighed the benefits.
The intent is to revisit ways of referencing external
metadata in the future.

.. note::

    **NetCDF Linked Data (NetCDF-LD)**

    `NetCDF Linked Data`_ was a proposed standard being developed by the
    Open Geospatial Consortium (OGC) NetCDF Standards Group,
    building on Linked Data, the widely used standard for managing relations between data items. 
    The NetCDF-LD standard allows proxies to URIs to act as resolvable identifiers,
    which can reference to external metadata held in registries.
    This approach can be used to link the files to a much richer set of metadata
    (such as the details of the model which generated the data). 


Practical considerations
------------------------

IMPROVER code is ususally implemented in a series of processing chains,
so it useful to consider the metadata in this context. 
From a processing perspctive, a useful way of dividing the metadata 
into three different types is:

1. Low-level, specific to a particular plug-in (Open Source)
2. Centralised, appropriate to all users (Open Source)
3. Organisation-specific metadata (Bespoke)

Metadata is set, updated or removed at three stages in the process:

1. Start - standardise CLI / amend_metadata plug-in
2. Plug-ins - reflecting changes made to the data and enforce wider standards
3. End - set metadata in the files that will be shared

.. note::

    amend_metadata makes use of JSON dictionaries to flexibly update metadata
    (delete, set)

The general approach proposed is to be conservative with metadata;
get rid of everything that is not needed.
In particular, processing stage 1, will remove or transform most organisation-specific metadata,
to ensure that the metadata does not become out of date.
for example, only 6 global attributes are expected
(being either retained or set at the start).

* Conventions
* institution
* source
* title
* mosg__model_configuration
* mosg__model_run

.. add note to Reference section

Organisation-specific metadata may be added in at the end of the processing chain.

Low-level metadata is will usually only be transitory,
required for certain processing steps, but out exposed in the final output.

Centralised metadata is key to the use of the final output,
providing the required information to understand and exploit the data.
This will be continually updated and, at times, added to,
as the data are tranformed in the processing steps.
Some of the most significant of these changes are:

* Thresholding to generate probabilities:

  * variable name - prefixed with ``probability_of _``
  * standard_name or long_name 'top and tail' with
    ``probability_of_`` and 
    ``_above_threshold`` or ``_below_threshold``, respectively
  * units - set ``1``

* Blend grid (multi-model):

  * source - change to be ``IMPROVER``
  * title - change to describe the blend appropriately
  * mosg__model_configuration - set to list of model identifiers
  * mosg__model_run - set to list of model runs and weights


References
----------

`CF Metadata Conventions`_

`NetCDF Linked Data`_


.. -----------------------------------------------------------------------------------
.. Links
.. _`CF Metadata Conventions`:
    http://cfconventions.org/

.. _`NetCDF Linked Data`:
    https://github.com/opengeospatial/netCDF-Classic-LD
