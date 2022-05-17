#######################################
RainForests calibration
#######################################

RainForests calibration is a situation dependent non-parametric method for the
calibration of rainfall based on the ECPoint method of Hewson and Pillosu 
`Hewson & Pillosu, 2021`_.

.. _Hewson & Pillosu, 2021: https://www.nature.com/articles/s43247-021-00185-9

****************************************************
Sub-grid variability as a means of calibration
****************************************************

ECPoint is based on the principle of using sub-grid variability as a means to
calibrate grid-scale rainfall forecasts. Sub-grid variability in this context is
the relationship between the distribution of point observations one would within
a grid-box for given areal average grid-scale forecast value.

Naturally the relationship between the resultant observation distribution is
contingent on the underlying rainfall processes at play. For instances, the
distribution associated with a rain-band will be characteristically different
than that associated with post-frontal showers and different still to that
associated with deep-tropical convection.

In knowing the distribution describing the sub-grid variability for a given
forecast, one can map the grid-scale forecast value onto a series of point-scale
values, yielding a form of conditional bias-correction.

`Hewson & Pillosu, 2021`_ propose a method to determine and apply these distributions
across a variety of weather types through the use of a manually tuned decision tree model coupled
with an appropriate set of feature parameters (meterological variables) to describe
the distinct weather types.

****************************
The RainForests method
****************************

RainForests is an adaptation of the ECPoint method that seeks to use machine learning
based tree model methods to replace the manually trained tree model. Here we use
gradient-boosted decision tree (GBDT) ensemble models.

The underlying principle of the calibration methodology is essentially the same, namely
using a set of feature variables to map onto an error distribution which is applied to
the forecast values to produce calibrated forecast values. However, the way in which the distributions are constructed differs somewhat within the RainForests
framework.

Our approach is to use a series of GBDT models, taking the feature variables as inputs,
to produce exceedence probability values for representative error thresholds. Collectively
these describe a cumulative distribution function for the error distribution.

The error CDF is then mapped onto a series of equispaced percentile values which
provide representative error values which can be applied to the forecast value to
produce the series of point-scale (calibrated) values.

Calibration of the ensemble forecast proceeds by determining the underlying
distribution on a per realization basis to produce a distinct series of calibrated forecast
values for each realization. Collectively these values form a so called super-ensemble,
which we subsequently sample to produce the calibrated forecast ensemble.

Another point of difference between RainForests and ECPoint is in the choice of
error value underlying the error distribution. For RainForests we have chosen to
use additive error in place of a multiplicative error (forecast error ratio in ECPoint)
to allow us to calibrate input forecast values of zero-rainfall.

===========================
GBDT vs manually trained DT
===========================

The choice of using GBDT models in place of the manually trained DT of ECPoint comes
with some advantages, but at the expense of some tradeoffs:

**Advantages:**

* GBDT is an ensemble of many trees rather than a single tree. This means outputs are 
  near-continuous relative to the inputs.
* Trees are built algorithmically, not manually, with each branch split automatically chosen
  to be optimal relative to some loss function. In principle this gives better accuracy, and
  makes it easier to retrain on new data.

**tradeoffs:**

* By using an ensemble of trees, the intuitive connection between weather type and feature
  variables is lost.
* Using a series of decision tree ensembles in place of a single decision tree increases
  the computational demand significantly.

****************************
Implementation details
****************************

===========================
Model training
===========================

..
    TODO: Add more specific details when model training Plugin is incorporated into
    IMPROVER.

The model training process is relatively simple and involves collating a series of
forecast-observation pairs with the associated feature variables into a single
pandas dataframe.

Using this dataframe, a series of lightGBM binary-classification models are trained
on the forecast error relative to a series of error-threshold values (one model per
error threshold).

As the method works on a per realization basis, the errors used in training must be
representative of the underlying systematic model biases. For this we require the
underlying weather type corresponding to the forecast-observation pair to be as
consistent as possible.

The forecast values used in training are the control forecast at the shortest available lead-times
that representative of the underlying weather situation. That is to say, we take the
earliest 24-hour period for daily accumulations, and all lead-times over the earliest
24-hour period for sub-daily accumulations to capture the full cycle of diurnal variations.

Currently model training is done offline, using a minimum 12-month period to capture
the full seasonal cycle.

===========================
Forecast calibration
===========================

The forecast calibration process involves a 2-step process:

1. Evaluate the error CDF defined over the series of error-thresholds used
   in model training. Each exceedence probability is evaluated using the
   corresponding tree-model, and the feature variables as inputs.

   The error CDF is evaluated concurrently for each forecast realization,
   so the associated error-probability cube will contain two ensemble dimensions,
   a threshold dimension (associated with the forecast error distributions) and a
   realization dimension (associated with the input ensemble).
2. Interpolate the CDF to extract a series of percentiles over for the error
   distributions. The error percentiles are then added to each associated ensemble
   realization from the forecast variable to produce a series of realisable forecast
   values.

   Collectively these series form a calibrated super-ensemble which is obtained by
   collapsing the two realization dimensions into one. This is then sub-sampled to
   provide the calibrated forecast.
