**Standard Geopotential Height Calculation**

The standard geopotential height calculation is based on the International Civil Aviation Organisation (ICAO) Standard Atmosphere definitions,
which vary depending on the layer of the atmosphere.

The standard atmosphere definition can be taken from a look up table, whereby the standard
constants for a layer are dependent on the pressure level of interest (e.g. 500 hPa).

+---------------+--------------------------------------+-----------------------------+---------------------------+---------------------------+
| Layer         | Vertical temperature gradient, β     | Geopotential height at base | Temperature at base Tb    | Pressure at base, Pb      |
|               | [K m⁻¹]                              | Zb [m]                      | [K]                       | [hPa]                     |
+===============+======================================+=============================+===========================+===========================+
| Troposphere   | -0.0065                              | 0                           | 288.15                    | 1013.25                   |
+---------------+--------------------------------------+-----------------------------+---------------------------+---------------------------+
| Stratosphere  | 0.0000                               | 11,000                      | 216.65                    | 226.32                    |
+---------------+--------------------------------------+-----------------------------+---------------------------+---------------------------+
| Mesosphere    | 0.0010                               | 20,000                      | 216.65                    | 54.75                     |
+---------------+--------------------------------------+-----------------------------+---------------------------+---------------------------+
| Thermosphere  | 0.0028                               | 32,000                      | 228.65                    | 8.68                      |
+---------------+--------------------------------------+-----------------------------+---------------------------+---------------------------+

The formulae based on the table above are:

If :math:`\beta = 0`:

.. math::
   Z_{\mathrm{std}}(p)
   =
   Z_b
   -
   \frac{R T_b}{g}
   \ln\left(\frac{p}{p_b}\right)

If :math:`\beta \ne 0`:

.. math::
   Z_{\mathrm{std}}(p)
   =
   Z_b
   +
   \frac{T_b}{\beta}
   \left[
     \left(\frac{p}{p_b}\right)^{-\frac{\beta R}{g}}
     -
     1
   \right]


where:

:math:`\beta` is the vertical temperature gradient (Km⁻¹)

:math:`Z_b` is the geopotential height at base (m)

:math:`T_b` is the temperature at base (T)

:math:`p_b` is the pressure at base (hPa)

:math:`p` is the input pressure (hPa)

:math:`R` is the universal gas constant (287.0 Jkg⁻¹K⁻¹)

:math:`g` is the standard gravitational acceleration (9.81 ms⁻²)
