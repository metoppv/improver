#######################################
RainForests calibration
#######################################

RainForests calibration is a situation dependent non-parametric method for the calibration
of rainfall. It is loosely based on the ECPoint method of Hewson and Pillosu 
`Hewson & Pillosu, 2021`_.

.. _Hewson & Pillosu, 2021: https://www.nature.com/articles/s43247-021-00185-9

****************************************************
Sub-grid variability as a means of calibration
****************************************************

Like ECPoint, RainForests aims to calibrate grid-scale rainfall forecasts by accounting 
for sub-grid variability. Sub-grid variability in this context describes the relationship 
between the point observations one would expect to measure within a grid box and the
areal average grid-scale NWP forecast value. This relationship is represented by a mapping that 
takes each NWP forecast value and maps it to a distribution of expected observed values.

Naturally the relationship between this distribution and the forecast value is contingent
on the underlying rainfall processes at play. For instance, the distribution associated
with a rainband will be characteristically different to that associated with post-frontal
showers and different still to that associated with deep-tropical convection. To this end,
a suitable set of meteorological variables can be used to distinguish different rainfall
regimes and identify the associated distribution that describes the underlying sub-grid
variability.

Rainforests (like ECPoint) processes each ensemble member separately to produce a per-realization output that can be
considered a pseudo-ensemble representing the
conditional probability distribution that describes the likelihood of observing a given outcome when the
realised atmospheric state is consistent with that represented in the input ensemble member forecast.
The predicted distributions for different ensemble members are then blended in probability space to produce 
the final calibrated probability output.

One advantage of the ECPoint approach is that the calibration is inherently non-local. As the calibration is done by
identifying distinct weather types, the model bias and scale difference should be independent of any given location and time as 
the underlying physical process should be identical. Thus a grid-point can be calibrated using data from any location, provided the 
underlying weather type is the consistent. This enables effective calibration to be applied to areas that are typically lacking 
sufficient cases to calibrate against.


****************************
The RainForests method
****************************

RainForests is an adaptation of the ECPoint method that uses machine learning based
tree models to replace the manually constructed tree model of ECPoint. Here we use gradient-boosted
decision tree (GBDT) models.

Our aim is to produce a probability distribution of the expected rainfall, given the NWP 
forecasts of relevant variables. We define a set of rainfall thresholds, suitably spaced so as 
to accurately model the distribution. Then we train a separate GBDT model for each lead time and 
threshold (note that each GBDT model itself consists of several hundred trees).

The set of variables that feed into the GBDT models are what allows the calibration to distinguish between
different weather situations and build the appropriate probability distribution accordingly. These input
variables are typically diagnostic variables for the ensemble realization (including the variable of interest),
but can include static and dynamic ancillary variables, such as local solar time, and whole-of-ensemble
values for diagnostic variables, such as mean or standard deviation.

Here we use LightGBM for training the models, and compile the models with Treelite for efficient prediction.

================================
GBDT vs manually constructed DT
================================

The choice of using GBDT models in place of the manually constructed decision tree of ECPoint comes with
some advantages, but at the expense of some trade-offs:

**Advantages:**

* GBDT is a sum of many trees rather than a single tree. This means outputs are
  near-continuous relative to the inputs.
* Trees are built algorithmically, not manually, with each branch split automatically
  chosen to be optimal relative to the loss function. In principle this gives better
  accuracy, and makes it easier to retrain on new data.

**trade-offs:**

* By using an ensemble of trees, the intuitive connection between weather type and feature
  variables becomes obscured.
* Using a series of decision tree ensembles in place of a single decision tree increases the
  computational demand significantly.
* Some initial effort is required to select a good set of model hyper-parameters that neither
  under- or over-fit. This process is more challenging and less transparent than ECPoint,
  however is required only once rather than each time the decision tree(s) are constructed.

****************************
Implementation details
****************************

===========================
Model training
===========================

..
    TODO: Add more specific details when model training Plugin is incorporated into IMPROVER.

The model training process is relatively simple and involves collating a series of
forecast-observation pairs with the associated feature variables into a single pandas
dataframe. In general, each ensemble member yields a separate row of the dataframe, although for 
reasons of computational efficiency it may be desirable to only use a subset of members, for example 
by using only the control realization in each ensemble. 
As each model predicts the probability that rainfall will exceed a particular threshold, 
the output is expected to be a number between 0 and 1. However, on occasion the GBDT models may
predict values slightly outside of this range and so the model predictions are capped to back onto this
interval; such values are possible due to the current choice of loss function used in training, namely the
mean squared loss function which is akin to Brier score when working in probability space.

Currently model training is done offline, using a minimum 12-month period to capture the
full seasonal cycle.

===========================
Forecast calibration
===========================

Forecast calibration uses the trained GBDT models, along with the forecast cube and associated
feature cubes. The tree-models are passed in via a model-config json which identifies
the appropriate tree-model file for each threshold.

The decision-tree models are used to construct representative probability distributions for
each input ensemble member which are then blended to give the calibrated
distribution for the full ensemble.

The distributions for the individual ensemble members are formed in a two-step process:

1. Evaluate the CDF defined over the specified model thresholds for each ensemble member.
Each threshold exceedance probability is evaluated using the corresponding
decision-tree model.
2. Interpolate each ensemble member distribution to the output thresholds.

Deterministic forecasts can also be calibrated using the same approach to produce a calibrated
CDF; in this case inputs are treated as an ensemble of size 1.
