# Ensemble Model Output Statistics (EMOS)

Ensemble Model Output Statistics (EMOS), otherwise known as
Nonhomogeneous Gaussian Regression (NGR), is a technique for calibrating
an ensemble forecast.

## Estimating EMOS coefficients

Following [Gneiting et al., 2005](https://doi.org/10.1175/MWR2904.1),
Ensemble Model Output Statistics for a normal distribution can be
represented by the equation:

$$\mathcal{N}(a + b_1X_1 + ... + b_mX_m, c + dS^{2})$$

where the location parameter is a bias-corrected weighted average of the
ensemble member forecasts, or alternatively, where the location
parameter is a bias-corrected ensemble mean:

$$\mathcal{N}(a + b\bar{X}, c + dS^{2})$$

If a different distribution is required, for example, using a truncated
normal distribution for wind speed, then the equations remain the same,
apart from updating the distribution chosen. The preferred distribution
will depend upon the variable being calibrated.

The a, b, c and d coefficients within the equations above are computed
by minimising the Continuous Ranked Probability Score (CRPS), in terms
of $\alpha, \beta, \gamma$ and $\delta$ (explained below). The
distribution is accounted for through the formulation of the CRPS that
is minimised e.g. please see [Gneiting et al.,
2005](https://doi.org/10.1175/MWR2904.1) for an example using a normal
distribution and [Thorarinsdottir and Gneiting,
2010](https://doi.org/10.1111/j.1467-985X.2009.00616.x) for an example
using a truncated normal distribution.

### Distribution description

A normal (Gaussian) distribution is often represented using the syntax:

$$\mathcal{N}(\mu,\,\sigma^{2})$$

where $\mu$ is mean and $\sigma^{2}$ is the variance. The normal
distribution is a special case, where $\mu$ can be interpreted as both
the mean and the location parameter and $\sigma^{2}$ can be interpreted
as both the variance and the square of the scale parameter. For an
alternative distribution, such as a truncated normal distribution that
has been truncated to lie within 0 and infinity, the distribution can be
represented as:

$$\mathcal{N^0}(\mu,\,\sigma^{2})$$

In this case, the $\mu$ is strictly interpreted as the location
parameter and $\sigma^{2}$ is strictly interpreted as the square of the
scale parameter.

### What is the location parameter?

The location parameter indicates the shift in the distribution from the
\"centre\" of the standard normal.

### What is the scale parameter?

The scale parameter indicates the width in the distribution. If the
scale parameter is large, then the distribution will be broader. If the
scale is smaller, then the distribution will be narrower.

## Implementation details

In this implementation, we will choose to define the distributions using
the scale parameter (as this matches scipy\'s expectation), rather than
the square of the scale parameter:

$$\mathcal{N}(\mu,\,\sigma)$$

The full equation when estimating the EMOS coefficients using the
ensemble mean is therefore:

$$\mathcal{N}(a + b\bar{X}, \sqrt{c + dS^{2}})$$

This matches the equations noted in [Allen et al.,
2021](https://doi.org/10.1002/qj.3983).

## Estimating EMOS coefficients using the ensemble mean

If the predictor is the ensemble mean, coefficients are estimated as
$\alpha, \beta, \gamma$ and $\delta$ based on the equation:

$$\mathcal{N}(a + b\bar{X}, \sqrt{c + dS^{2}})$$

where N is a chosen distribution and values of a, b, c and d are solved
in the format of $\alpha, \beta, \gamma$ and $\delta$, see the equations
below.

$$a = \alpha$$

$$b = \beta$$

$$c = \gamma^2$$

$$d = \delta^2$$

The $\gamma$ and $\delta$ values are squared to ensure c and d are
positive and therefore more interpretable.

## Estimating EMOS coefficients using the ensemble realizations

If the predictor is the ensemble realizations, coefficients are
estimated for $\alpha, \beta, \gamma$ and $\delta$ based on the
equation:

$$\mathcal{N}(a + b_1X_1 + ... + b_mX_m, \sqrt{c + dS^{2}})$$

where N is a chosen distribution, the values of a, b, c and d relate to
alpha, beta, gamma and delta through the equations above with the
exception that $b=\beta^2$, and the number of beta terms depends on the
number of realizations provided. The beta, gamma, and delta values are
squared to ensure that b, c and d are positive values and therefore are
more easily interpretable. Specifically for the b term, the squaring
ensures that the the b values can be interpreted as a weighting for each
realization.

## Applying EMOS coefficients

The EMOS coefficients represent adjustments to the ensemble mean and
ensemble variance, in order to generate the location and scale
parameters that, for the chosen distribution, minimise the CRPS. The
coefficients can therefore be used to construct the location parameter,
$\mu$, and scale parameter, $\sigma$, for the calibrated forecast from
today\'s ensemble mean, or ensemble realizations, and the ensemble
variance.

$$\mu = a + b\bar{X}$$$$\sigma = \sqrt{c + dS^{2}}$$

Note here that this procedure holds whether the distribution is normal,
i.e. where the application of the EMOS coefficients to the raw ensemble
mean results in a calibrated location parameter that is equivalent to a
calibrated ensemble mean (e.g. for screen temperature), and where the
distribution is e.g. truncated normal (e.g. for wind speed). For a
truncated normal distribution, the result of applying the EMOS
coefficients to an uncalibrated forecast is a location parameter and
scale parameter describing the calibrated distribution.
