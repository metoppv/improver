**Use of ensemble mean and ensemble variance as location and scale parameters**

The EMOS coefficients represent adjustments to the ensemble mean and ensemble
variance, in order to generate the location and scale parameters that, for the
chosen distribution, minimise the CRPS. The coefficients can therefore be used
to construct the location parameter, :math:`\mu`, and scale parameter,
:math:`\sigma^{2}`, for the calibrated forecast from today's ensemble mean, or
ensemble realizations, and the ensemble variance.

.. math::

    \mu = a + b\bar{X}

    \sigma^{2} = c + dS^{2}

Note here that this procedure holds whether the distribution is normal, i.e.
where the application of the EMOS coefficients to the raw ensemble mean results
in a calibrated location parameter that is equivalent to a calibrated ensemble
mean (e.g. for screen temperature), and where the distribution is e.g.
truncated normal (e.g. for wind speed). For a truncated normal distribution,
the result of applying the EMOS coefficients to an uncalibrated forecast is a
location parameter and scale parameter describing the calibrated distribution.
