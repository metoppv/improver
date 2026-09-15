Air Density Calculation
-----------------------

Density is calculated using the Ideal Gas Law equation of state with
moisture effects included via the virtual temperature calculation.

.. math::


   \rho=\frac{p}{R_d T_v}

where:

:math:`\rho` = air density (kgm⁻³)

:math:`p` = pressure (Pa)

:math:`R_d` = gas constant for dry air = 287.05 Jkg⁻¹K⁻¹ (See R_DRY_AIR constant in IMPROVER.constants)

:math:`T_v` = virtual temperature (K), which needs calculating

Key interpretation
------------------

If temperature goes up, density goes down.

If moisture goes up, density goes down [*]_

If pressure goes up, density goes up.

.. [*] Derivable from Avogadro’s Law (i.e. equal volumes of different gases
  under the same conditions hold the same number of molecules) and
  molecular weights. Water molecules have a lower atomic weight (16u)
  than oxygen molecules (32u) and nitrogen molecules (18u).
