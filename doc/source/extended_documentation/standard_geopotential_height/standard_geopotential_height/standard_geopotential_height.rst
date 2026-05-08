
**Standard Geopotential Height Calculation**
If :math:`\beta = 0`:

.. math::
   Z_{\mathrm{std}}(p)
   =
   Z_b
   -
   \frac{R T_b}{g}
   \ln\!\left(\frac{p}{p_b}\right)

If :math:`\beta \ne 0`:

.. math::
   Z_{\mathrm{std}}(p)
   =
   Z_b
   +
   \frac{T_b}{\beta}
   \left[
     \left(\frac{p}{p_b}\right)^{-\beta R / g}
     -
     1
   \right]
``

where

:math:`p_b` is the pressure at base (hPa)
:math:`Z_b` is the (geopotential height at base, m)
local vapour pressure, :math:`m` is
the engine mixing ratio, :math:`T` is the ambient air
