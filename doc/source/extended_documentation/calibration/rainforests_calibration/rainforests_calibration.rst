#######################################
RainForests calibration
#######################################

RainForests calibration is a situation dependent non-parametric method for the calibration
of rainfall based on the ECPoint method of Hewson and Pillosu `Hewson & Pillosu, 2021`_.

.. _Hewson & Pillosu, 2021: https://www.nature.com/articles/s43247-021-00185-9

****************************************************
Sub-grid variability as a means of calibration
****************************************************

ECPoint is based on the principle of using sub-grid variability as a means to calibrate
grid-scale rainfall forecasts. Sub-grid variability in this context is the connection
between the point observations one would measure within a grid-box when presented with
a given areal average grid-scale forecast value. This relationship is encapsulated in the
mapping that takes a grid-scale forecast value and maps it onto a distribution of typical
point observation values, or equivalently a set of forecast error values, and is central
to this calibration method.

Naturally the relationship between this distribution and the forecast value is contingent
on the underlying rainfall processes at play. For instance, the distribution associated
with a rainband will be characteristically different to that associated with post-frontal
showers and different still to that associated with deep-tropical convection. To this end,
a suitable set of meteorological variables can be used to distinguish different rainfall
regimes and identify the associated distribution that describes the underlying sub-grid
variability.

Equipped with the relevant distribution that describes the sub-grid variability for a given
forecast, the grid-scale input forecast value can be mapped onto a series of realisable
point-scale values, yielding a form of conditional bias-correction. This process is done
on a per realization basis, using the mapping function most consistent with the input 
ensemble member forecast. In this way we calibrate to point scale and correct model bias
within each ensemble member independently, with each grid-scale ensemble member producing
a series of realisable point-scale forecast values.

The output produced from each ensemble member can be considered a pseudo-ensemble which
represents a conditional probability distribution that describes the likelihood of observing
a given outcome when the realised atmospheric state is consistent with that represented in the
input ensemble member forecast.

Combining these ensemble member pseudo-ensembles, we can produce an intra-model "super-ensemble".
Usage of the term super-ensemble here is distinct from the more common usage which refers to the
inter-model super-ensemble formed by combining multiple NWP ensemble models. Herein the usage of
super-ensemble refers to the intra-model super-ensemble. Rather than containing a range of atmospheric
states to which a single outcome is associated, the super-ensemble describes a range of possible
outcomes given sourced from each atmospheric state. 

This super-ensemble is represented by a series of realization values, and the associated probability
distribution can be sourced using the same approach one would apply to the input forecast ensemble.

Hewson & Pillosu propose a framework (ECPoint) to determine and apply these distributions
across a variety of weather types through the use of a manually constructed decision tree model
taking an appropriate set of feature parameters (meteorological variables) as inputs.
Details of this method can be found in `Hewson & Pillosu, 2021`_. 

One advantage of the ECPoint approach is that the calibration is inherently non-local. As
the calibration is done by identifying distinct weather types, the model bias and scale
difference should be independent of any given location and time as the underlying physical
process should be identical. Thus a grid-point can be calibrated using data from any location,
provided the underlying weather type is the consistent. This enables effective calibration to
be applied to areas that are typically lacking sufficient cases to calibrate against.

****************************
The RainForests method
****************************

RainForests is an adaptation of the ECPoint method that seeks to use machine learning based
tree models to replace the manually constructed tree model of ECPoint. Here we use gradient-boosted
decision tree (GBDT) ensemble models.

The underlying principle of the calibration methodology is essentially the same, namely using
a set of feature variables to map onto error distributions which can be applied to the
forecast values to produce calibrated forecast values. However, the way in which these
distributions are constructed differs somewhat within the RainForests framework.

Our approach is to use a series of GBDT models, taking the feature variables as inputs, to
produce exceedance probability values for representative error thresholds. Collectively
these represent the error distributions as a cumulative distribution function.

Each CDF is then mapped onto a series of equispaced percentile values to yield a series of
representative error values which can be applied to the grid-scale forecast value to produce
the series of possible point-scale (calibrated) values.

Calibration of the ensemble forecast proceeds by determining the underlying distribution on
a per realization basis and applying this to each forecast value to produce a series of
distinct calibrated forecast values for each realization. Collectively these values form the
so-called super-ensemble, which we subsequently sample to produce the calibrated forecast
ensemble output.

Another point of difference between RainForests and ECPoint is in the choice of error value
underlying the error distribution. For RainForests we have chosen to use additive error in
place of a multiplicative error (forecast error ratio in ECPoint) to allow us to calibrate
input forecast values of zero-rainfall.

================================
GBDT vs manually constructed DT
================================

The choice of using GBDT models in place of the manually constructed DT of ECPoint comes with
some advantages, but at the expense of some trade-offs:

**Advantages:**

* GBDT is an ensemble of many trees rather than a single tree. This means outputs are
  near-continuous relative to the inputs.
* Trees are built algorithmically, not manually, with each branch split automatically
  chosen to be optimal relative to some loss function. In principle this gives better
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
dataframe.

Using this dataframe, a series of lightGBM binary-classification models are trained against
the truth values for exceedances of the forecast error relative to a series of error-threshold
values, resulting in one model per error threshold.

As the calibration method works on a per realization basis, the errors used in training
must be representative of the underlying systematic model biases for a single ensemble
member rather than those of the ensemble as a whole. This requires the use of a single
representative ensemble forecast member in training. It also requires the underlying
weather type associated with the forecast and observation in the forecast-observation
pair be as consistent as possible.

For the forecast values used in training, we use the control at the shortest available
lead-times that distinctly represent of the underlying weather types. Specifically, we
take the earliest 24-hour period for daily accumulations, and all lead-times over the
earliest 24-hour period for sub-daily accumulations to capture the full cycle of diurnal
variations.

Currently model training is done offline, using a minimum 12-month period to capture the
full seasonal cycle.

===========================
Forecast calibration
===========================

Forecast calibration uses the trained GBDT models, along with the forecast cube and associated
feature cubes. The tree-models are passed in via a model-config json which identifies
the appropriate tree-model file for each error-threshold.

Forecast calibration proceeds via a 2-step process:

1. Evaluate the error CDF defined over the series of error-thresholds used in model training.
   Each exceedance probability is evaluated using the corresponding tree-model, and the feature
   variables as inputs.

2. Interpolate the CDF to extract a series of percentile values for the error distributions.
   The error percentiles are then added to each associated ensemble realization from the
   forecast variable to produce a series of realisable forecast values.

Collectively these series form the calibrated super-ensemble which is obtained by collapsing
the two realization dimensions into one. This is then sampled to provide the calibrated
ensemble forecast.

Deterministic forecasts can also be calibrated using the same approach to produce a calibrated
pseudo-ensemble; in this case inputs are treated as an ensemble of size 1.

**A typical usage example:** we typically use around 25 error threshold values to construct
the CDF for the distribution of forecast errors. For each error threshold we have an associated
GBDT model which is used to evaluate the exceedance probabilities that describe the CDF.
So starting with an input ensemble forecast consisting of 50 realizations, we evaluate 25
threshold probability values for each realization to construct a forecast error CDF for each
realization (50 distributions in total, each containing 25 threshold values).

For each distribution, we then interpolate between the threshold probabilities to extract
20 evenly-spaced percentiles. These are then applied to each forecast realization to produce
20 calibrated forecast realizations, resulting in 50 * 20 (1000) forecast values which
collective for the calibrated "super-ensemble". Finally, we sample the super-ensemble by
taking 100 equispaced percentile values to be representative realizations for the calibrated
forecast ensemble.

This final step is not required, but ensures efficient processing in downstream CLIs.

The number of error-percentiles used in the interim step, and the number of output ensemble
realizations are taken as input parameters.
