What is IMPROVER?
-----------------

IMPROVER: Integrated Model post-PROcessing and VERification

The Met Office in the UK is leading an open source project to create a probabilistic post processing system for use with ensemble weather prediction models. The system is designed to extract the wealth of information provided by ensemble forecasts and present it in a format that is suitable for the production of automated weather forecasts, and for use by forecasters and the public.

Using IMPROVER with weather forecasts from a range of forecasting centres, it is possible to generate a consolidated blend of information that better captures the range of potential weather outcomes. Additional techniques, such as spatial and temporal neighbourhooding, are available within IMPROVER to further increase the spread of forecasts, capturing more of this range. Functionality also exists to include older forecasts into the final blend, weighting them appropriately to capture the fall off in forecast skill at longer lead times. The end result is the consolidation of tens or hundreds of representations of a weather situation into an interrogable probabilistic output.

An example IMPROVER forecast product is shown below, here for wind speed. Four fields are plotted at different wind speed thresholds that may be of interest to the user, each field shows the probability of exceeding the relevant threshold.


.. figure:: ../files/wind_probabilities.jpg
   :align: center

   Probability of exceeding wind speed thresholds


Structure of IMPROVER
---------------------

IMPROVER is designed as a modular post-processing system. The final product is created through the application of a sequence of processing steps, where the sequence can be readily modified to achieve different outcomes. The output of each step in the chain can be written out, allowing for verification against observations at each stage (using a suitable verification package, which is not part of IMPROVER). This enables the user to determine whether a given step in the chain is improving or harming the forecast quality.

Any given step in the processing can be applied using the included command line interfaces (CLIs). A complex system is built be calling the CLIs in sequence. The simple schematic below gives an example processing chain.

.. figure:: ../files/processing_chain.jpg
   :align: center

   An example processing chain with IMPROVER


Using IMPROVER
--------------

IMPROVER does not currently include installation functionality (e.g. setup.py). The code can be used by cloning the `GitHub repository`_ and calling the command line interfaces (CLIs) from a linux/unix terminal or by importing the modules directly into Python.

.. _GitHub repository: https://github.com/metoppv/improver

.. code:: console

    git clone https://github.com/metoppv/improver.git <local directory>

The list of dependencies can be found in the `environment.yml`_ file.

.. _environment.yml: https://github.com/metoppv/improver/blob/master/environment.yml

Example use of a CLI
====================

Here we give a simple example of using an IMPROVER CLI to threshold data, moving into probability space. This CLI invocation is from the root directory of a local copy of IMPROVER.

.. code:: console

    bin/improver threshold input_file.nc output_file.nc 5 10 15 20 --threshold_units m/s


* input_file.nc is a netCDF file containing a forecast diagnostic data cube, e.g. wind speeds across an x-y grid at a given time.
* for each threshold specified (5, 10, 15, 20 m/s) a new x-y grid of data will be created. Each point in the grid will contain a 0 if the input wind speed at that point was below the threshold, or 1 if it was above the threshold.
* output_file.nc will be a new netCDF file containing the resulting data cube with an additional leading dimension that corresponds to the given thresholds (5, 10, 15, 20 m/s).

This simple example covers one step in a processing chain. Additional information about using any CLI can be found on the command line using `-h`, e.g.:

.. code:: console

    bin/improver nbhood -h


Publications & Presentations
----------------------------

Below are links to publicly accessible publications & presentations that relate to IMPROVER.

1. `Generating probabilistic forecasts from convection permitting ensembles`_

   - Nigel Roberts

2. `Creating a probabilistic, multi-model post-processing system (IMPROVER) at the Met Office`_

   - Gavin Evans

3. `Topographic neighbourhood processing`_

   - Fiona Rust

.. _Generating probabilistic forecasts from convection permitting ensembles: https://presentations.copernicus.org/EMS2017-277_presentation.pdf
.. _Creating a probabilistic, multi-model post-processing system (IMPROVER) at the Met Office: https://presentations.copernicus.org/EMS2018-20_presentation.pdf
.. _Topographic neighbourhood processing: https://presentations.copernicus.org/EMS2018-70_presentation.pdf



Contributing
------------

IMPROVER is freely available to use and we welcome contributions to code development, but please note that we are unable to provide support for use of the software at this time.

For details about contributing to IMPROVER, please refer to the `How to Contribute`_ page on GitHub.

.. _How to Contribute: https://github.com/metoppv/improver/blob/master/CONTRIBUTING.md
