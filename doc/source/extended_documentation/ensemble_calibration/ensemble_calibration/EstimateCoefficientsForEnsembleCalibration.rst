**Distribution description**

A normal (Gaussian) distribution is often represented using the syntax:

.. math::

    \mathcal{N}(\mu,\,\sigma^{2})

where :math:`\mu` is mean and :math:`\sigma^{2}` is the variance. The normal
distribution is a special case, where :math:`\mu` can be interpreted as both
the mean and the location parameter and :math:`\sigma^{2}` can be interpreted
as both the variance and the scale parameter. For an alternative distribution,
such as a truncated normal distribution that has been truncated to lie within
0 and infinity, the distribution can be represented as:

.. math::

    \mathcal{N^0}(\mu,\,\sigma^{2})

In this case, the :math:`\mu` is strictly interpreted as the location parameter
and :math:`\sigma^{2}` is strictly interpreted as the scale parameter.

**What is the location parameter?**

The location parameter indicates the shift in the distribution from the
"centre" of the standard normal. Examples of the location parameter include
the mean (as for a normal distribution), median or mode.

**What is the scale parameter?**

The scale parameter indicates the width in the distribution. If the scale
parameter is large, then the distribution will be broader. If the scale is
smaller, then the distribution will be narrower.

**Ensemble Model Output Statistics**

Following `Gneiting et al., 2005`_, Ensemble Model Output Statistics for a
normal distribution can be represented by the equation:

.. _Gneiting et al., 2005: https://doi.org/10.1175/MWR2904.1

.. math::

    \mathcal{N}(a + b_1X_1 + ... + b_mX_m, c + dS^{2})

where the location parameter is a bias-corrected weighted average of the
ensemble member forecasts, or alternatively, where the location parameter is a
bias-corrected ensemble mean:

.. math::

    \mathcal{N}(a + \bar{X}, c + dS^{2})

If a different distribution is required, for example, using a truncated
normal distribution for wind speed, then the equations remain the same, apart
from updating the distribution chosen. The preferred distribution will depend
upon the variable being calibrated.

The a, b, c and d coefficients within the equations above are computed by
minimising the Continuous Ranked Probability Score (CRPS). The distribution is
accounted for through the formulation of the CRPS that is minimised e.g.
please see `Gneiting et al., 2005`_ for an example using a normal distribution
and or `Thorarinsdottir and Gneiting, 2010`_ for an example using a truncated
normal distribution.

.. _Gneiting et al., 2005: https://doi.org/10.1175/MWR2904.1
.. _Thorarinsdottir and Gneiting, 2010: https://doi.org/10.1111/j.1467-985X.2009.00616.x

In order to constrain the scale parameters (c and d) to be non-negative and the
regression coefficients (b) to be non-negative, the coefficients that are
solved take the form:

.. math::

    a = \alpha

    b = \beta^{2}

    c = \gamma^{2}

    d = \delta^{2}

The coefficients resulting from the CRPS minimisation are therefore actually
:math:`\alpha, \beta, \gamma` and :math:`\delta`, which are squared when the
coefficients are applied.
