#######################################
RainForests calibration
#######################################

RainForests calibration is a situation dependent non-parametric method for the calibration
of rainfall. It is loosely based on the ECPoint method of Hewson and Pillosu 
`Hewson & Pillosu, 2021`_, but with some significant changes.

.. _Hewson & Pillosu, 2021: https://www.nature.com/articles/s43247-021-00185-9

****************************************************
Sub-grid variability as a means of calibration
****************************************************

Like ECPoint, RainForests aims to calibrate grid-scale rainfall forecasts by accounting 
for sub-grid variability. Sub-grid variability in this context is the reliationship
between the point observations one would measure within a grid box, and the 
areal average grid-scale NWP forecast value. This relationship is modelled by a mapping that 
maps each NWP ensemble member forecast to a distribution of expected observed values.

Naturally the relationship between this distribution and the forecast value is contingent
on the underlying rainfall processes at play. For instance, the distribution associated
with a rainband will be characteristically different to that associated with post-frontal
showers and different still to that associated with deep-tropical convection. To this end,
a suitable set of meteorological variables can be used to distinguish different rainfall
regimes and identify the associated distribution that describes the underlying sub-grid
variability.


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

Our model uses the five variables found in ECPoint, plus 2 additional variables describing the 
ensemble characteristics. The variables in common with ECPoint are precipitation, convective precipitation,
wind speed, and convective and potential energy (CAPE), and solar irradiance (specifically, the 
daily accumulated clearsky solar irradiance per unit area). The first four of these are directly 
from the NWP forecast, and solar irradiance is a modelled value. In addition, we included as features 
the mean and standard deviation of the ensemble predictions of precipitation. These ensemble features 
allow the model to distinguish situations where all ensemble members are similar (and therefore the calibration 
should not adjust each member of the NWP forecast too much), versus situations where there is wide variance, 
and therefore each member should be adjusted to be close to the ensemble mean. 

We use LightGBM for training the models, and compile the models with Treelite for efficient prediction.

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
dataframe. Each ensemble member corresponds to a single row of the dataframe. 
Each model predicts the probability that rainfall will exceed a particular threshold, 
so the target variable is a 0/1 variable given by thresholding the observations. We use 
the regression objective in LightGBM, with the mean squared error loss. Although this is 
not a traditional choice for classification tasks (where the aim is to predict a probability), 
we found it gave better results.

Currently model training is done offline, using a minimum 12-month period to capture the
full seasonal cycle.

===========================
Forecast calibration
===========================

Forecast calibration uses the trained GBDT models, along with the forecast cube and associated
feature cubes. The tree-models are passed in via a model-config json which identifies
the appropriate tree-model file for each error-threshold.

Forecast calibration proceeds via a 2-step process:

1. Evaluate the thresholded CDF for each ensemble member.

2. Average the predicted probabilities at each threshold over all ensemble members.

Deterministic forecasts can also be calibrated using the same approach to produce a calibrated
CDF; in this case inputs are treated as an ensemble of size 1.
