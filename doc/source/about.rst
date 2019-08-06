What is IMPROVER?
-----------------

IMPROVER: Integrated Model post-PROcessing and VERification

The Met Office in the UK is leading an open source project to create a probabilistic post processing system for ensemble weather prediction systems. The system is designed to extract the wealth of information provided by ensemble forecasts and present it in a format that is suitable for the production of automated weather forecasts, and for the use of forecasters and the public.

Using IMPROVER with weather forecasts from a range of forecasting centres, it is possible to generate a consolidated blend of information that better captures the range of potential weather outcomes. Additional techniques, such as spatial and temporal neighbourhooding, are available within IMPROVER to further increase the spread of forecasts, capturing more of this range. Functionality also exists to include older forecasts into the final blend, weighting them appropriately to capture the fall off in forecast skill at longer lead times. The end result is the consolidation of tens or hundreds of representations of a weather situation into an interrogable probabilistic output.

An example IMPROVER forecast product is shown below, here for wind speed. Four fields are plotted at different wind speed thresholds that may be of interest to the user, each field shows the probability of exceeding the relevant threshold.


.. figure:: ../files/wind_probabilities.jpg
   :align: center

   Probability of exceeding wind speed thresholds


Structure of IMPROVER
---------------------

IMPROVER is designed as a modular post-processing system. The final product is created through the application of a sequence of processing steps, where the sequence can be readily modified to achieve different outcomes. The output of each step in the chain can be written out, allowing for verification against observations at each stage (using a suitable verification package, which is not part of IMPROVER). This enables the user to determine whether a given step in the chain is improving or harming the forecast quality.

Any given step in the processing can be applied using the included command line interfaces (CLIs). A complex system is built be calling the CLIs in sequence. The simple schematic below gives an example processing chain:

.. figure:: ../files/processing_chain.jpg
   :align: center

   An example processing chain with IMPROVER
