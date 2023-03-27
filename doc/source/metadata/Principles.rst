.. _principles-label:

Principles
==========

.. contents:: Contents
    :depth: 3

Overview
--------

The principles applied to the metadata within IMPROVER have been listed
in the Introduction, but are discussed in more detail in this section.

.. _principles-CF-conformance-label:

Conformance to the CF Metadata Conventions
------------------------------------------

As far as possible the use of metadata within IMPROVER should conform to the
`CF Metadata Conventions`_ (CF),
whilst recognising the need to extend this in some areas.

However, whilst CF does a good job with 'where' and 'what' metadata
(see note below), it does not really cover 'who' metadata.
Therefore, it is helpful to keep CF (and direct extensions to this)
focused on these aspects and not try to stretch it to cover 'who'.

An example is CF cell methods, which are intended to cover 'what'
(i.e. to ensure that data can be correctly interpreted) 
but can unwittingly be extended into 'who'
(e.g. to try to describe the post-processing that has been applied), 
which can complicate its main purpose. 

.. note::

    **Metadata Categorization** 

    To inform decisions about the implementation of metadata, 
    it can be helpful to consider three types of metadata in terms
    of what they are trying to achieve:

    * **Where** - location in space and time,
      and additional 'data space' information

    * **What** - what is this value?
      i.e. phenomenon, units, extra characteristics

    * **Who** - data identity,
      i.e. where it came from, provenance, data processing, etc

Clear purpose
-------------

Metadata should only be present, if it has a purpose. It should not
be included just because it might be useful in future, as if no one is
using it, it is unlikely to be properly maintained.

There should be just enough metadata to describe the data sufficiently.

Nothing misleading or unnecessarily restrictive
-----------------------------------------------

Metadata that is misleading, or simply wrong, should not be present.

Also, metadata that restricts wider use should be avoided, if possible.

Referencing more detailed external documentation
------------------------------------------------

Ideally, the metadata should support the ability to 'refer out'
to further information about the data to avoid 'overloading'
the files with metadata.
This makes it easier to provide more detail of the provenance of the data
and the post-processing applied
(the 'where' metadata described in the earlier note).

One approach to help support this that has been partially adopted within
IMPROVER is the use of name-spacing for new (non-CF) attributes.
This has two purposes:

 * It clearly separates the attribute from CF standard attributes,
   identifying it as belonging to a separate metadata convention.
 * It provides the potential for this attribute to be 'resolved' to a
   definition held in an external registry in the future

Namespacing is implemented using a double underscore to maintain CF conformance
whilst clearly identifying the namespace part of the attribute name.
For example, "ssp" is used to denote the statistical post-processing namespace,
with the attributes in this namespace prefixed by ``spp__``.
At present this is only used for a single attribute ``ssp__relative_to_threshold``,
which takes one of the four values:

    * ``greater_than`` 
    * ``greater_than_or_equal_to``
    * ``less_than`` 
    * ``less_than_or_equal_to`` 

The other namespace used is "mosg", which is used to indicate
a Met Office standard grid attribute.
Despite the name this can, and generally should, be used.
At present this is used for two attributes:
``mosg__model_configuration``, which identifies the 
sources that have contributed to the blend, and
``mosg__model_run``, which extends the information provided by
the contribution of specific model runs and 
the level of their contribution.
 
At present, IMPROVER does not provide explicit references to external metadata.
Earlier, in the development process,the use of NetCDF Linked Data was explored
(see note below), but the method of implementation at the time created issues
that outweighed the benefits.
The intent is to revisit ways of referencing external metadata in the future.

.. note::

    **NetCDF Linked Data (NetCDF-LD)**

    `NetCDF Linked Data`_ was a proposed standard being developed by the
    Open Geospatial Consortium (OGC) NetCDF Standards Group,
    building on Linked Data, the widely used standard for managing relations
    between data items. 
    The NetCDF-LD standard allows proxies to URIs to act as resolvable
    identifiers, which can reference to external metadata held in registries.
    This approach can be used to link the files to a much richer set of
    metadata (such as the details of the model which generated the data). 


Practical considerations
------------------------

IMPROVER code is usually implemented in a series of processing chains,
so it makes sense to consider the metadata in this context. 
From a post-processing perspective, it is helpful to divide the metadata 
into three different types:

1. Low-level, specific to a particular plug-in (Open Source)
2. Centralised, appropriate to all users (Open Source)
3. Organisation-specific metadata (Bespoke)

Metadata is set, updated or removed at three stages in a post-processing chain:

1. Start - standardise CLI / amend_metadata plug-in
2. Plug-ins - reflecting changes made to the data and enforce wider standards
3. End - set metadata in the files that will be shared

.. note::

    amend_metadata makes use of JSON dictionaries to flexibly update metadata
    (delete, set)

The general approach proposed is to be conservative with metadata;
remove attributes on the source data that do not serve a purpose for the
post-processed data.
In particular, processing stage 1, will remove or transform most
organisation-specific metadata, to ensure that the metadata does
not become out of date.
For this reason, only 6 global attributes are expected
(being either retained or set at the start).

* Conventions
* institution
* source
* title
* mosg__model_configuration
* mosg__model_run

.. add note to Reference section

Organisation-specific metadata may be added in at the end of the processing chain.

Low-level metadata will usually only be transitory,
required for certain processing steps, but not exposed in the final output.

Centralised metadata is key to the use of the final output,
providing the required information to understand and exploit the data.
This will be continually updated and added to as the data are tranformed
in the post-processing steps.
A couple of the most significant of these changes are:

* Thresholding to generate probabilities:

  * variable name - prefixed with ``probability_of _``
  * standard_name or long_name 'top and tailed' with
    ``probability_of_`` and 
    ``_above_threshold`` or ``_below_threshold``, respectively
  * units - set to ``1``

* Blend grid (multi-model):

  * source - changed to be ``IMPROVER``
  * title - updated to describe the blend appropriately
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
