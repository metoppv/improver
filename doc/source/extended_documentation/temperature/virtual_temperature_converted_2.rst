Virtual Temperature Calculations
================================

Two virtual temperature calculations are currently provided.

1. VirtualTemperature Class Calculation
---------------------------------------

Calculation input diagnostics:

#. temperature (T)
#. humidity_mixing_ratio (q)

This IMPROVER plugin uses a first order approximation for virtual temperature (:math:`T_v`).

You can show:

.. math::

   T_v = T \left[ rac{q}{\epsilon} + (1-q) ight]
       = T \left[ 1 + q\left(rac{1}{\epsilon}-1ight) ight]

Since:

.. math::

   rac{1}{\epsilon} - 1 pprox 0.62

Gives:

.. math::

   T_v pprox T(1 + 0.62q)

where :math:`\epsilon` is the ratio of the gas constants (or molecular weights)
of dry air and water vapour:

.. math::

   \epsilon = rac{R_d}{R_v}

where:

* :math:`R_d pprox 287.05\,\mathrm{J\,kg^{-1}\,K^{-1}}` is the gas constant for dry air.
* :math:`R_v pprox 461.5\,\mathrm{J\,kg^{-1}\,K^{-1}}` is the gas constant for water vapour.

So:

.. math::

   \epsilon pprox rac{287.05}{461.5} pprox 0.622

An equivalent definition using molecular weights is:

.. math::

   \epsilon = rac{M_v}{M_d}

where:

* :math:`M_v pprox 18.016\,\mathrm{g\,mol^{-1}}` (water vapour)
* :math:`M_d pprox 28.964\,\mathrm{g\,mol^{-1}}` (dry air)

Giving the same value:

.. math::

   \epsilon pprox 0.622

2. VirtualTemperatureFromSpecificHumidityClass Calculation
----------------------------------------------------------

Calculation input diagnostics:

#. temperature (T)
#. humidity_mixing_ratio (q)
#. cloud_water_mixing_ratio
#. cloud_ice_mixing_ratio

This calculates virtual temperature using the specific humidity, which is desirable in the calculation of air density.

This virtual temperature (:math:`T_v`) calculation also uses condensates, if provided.
Condensed water (rain, snow and liquid cloud droplets) add weight to a volume of air without contributing to the pressure.

When the data on the mixing ratios of liquid water (:math:`q_{cl}`) and ice (:math:`q_{cf}`) are available, a slightly improved estimate is:

.. math::

   T_v = T \left[ \left(rac{q}{\epsilon}ight) + (1 - q - q_{cl} - q_{cf}) ight]

This is only significant lower down in the atmosphere, and where there is cloud.

Given the densest clouds have condensate specific humidities typically no higher than 5 g kg-1, the adjustment is unlikely to ever be more than 0.5%.

where:

* :math:`T_v` = virtual temperature (K)
* :math:`T` = temperature (K)
* :math:`q` = specific humidity (kg kg-1)
* :math:`\epsilon = 0.62198` ratio of gas constants of dry to moist air
* :math:`q_{cl}` = mixing ratio of liquid water
* :math:`q_{cf}` = mixing ratio of ice

Excluding condensates from the virtual temperature calculation will reduce the amount of moisture and therefore reduce density. In most cases the adjustment is very small and unlikely to exceed 0.5%.
