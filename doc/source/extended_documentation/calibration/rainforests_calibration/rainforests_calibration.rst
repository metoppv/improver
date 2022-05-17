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

