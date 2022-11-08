Introduction
============

.. contents:: Contents
    :depth: 3

Overview
--------

Metadata is data that describes other data, for example, 
within a file or an in-memory object. 
For IMPROVER, this generally means the attributes within a netCDF1 file or
an Iris2 cube. 
The principles applied are: 

 - Conformance to the CF Metadata Conventions, building on this where necessary. 

- Clear purpose, with just enough metadata to describe the data sufficiently. 

- Nothing misleading or unnecessarily restrictive for non-Met Office usage. 

- (Ideally) support for referencing more detailed external documentation.  

IMPROVER provides data as both gridded and spot values for the UK and global domains.
It generates different types of output, primarily probabilities and percentiles, 
but is some special cases deterministic.