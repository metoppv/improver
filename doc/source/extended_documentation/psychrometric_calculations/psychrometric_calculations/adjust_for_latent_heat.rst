The latent heat energy :math:`Q` released through condensation of water is
equal to the mass of water condensed (:math:`m`) multiplied by the constant
latent heat of condensation of water (:math:`L`) defined in the
:doc:`improver.constants`.

.. math::

    Q = L m

The temperature change of the air parcel is the latent heat energy divided
by the specific heat of dry air (:math:`C_p`) defined in the
:doc:`improver.constants`.

.. math::

    \Delta T = \frac{Q}{C_p}

The mass of water condensed is dependent on the difference between
the initial parcel temperature and the final parcel temperature, because
the saturated mass mixing ratio :math:`q_{sat}` of water in air is
dependent on the temperature and pressure of the air. The derivation of
:math:`q_{sat}` is in :py:meth:`saturated_humidity`.

Therefore, when ascending a saturated parcel of air from pressure level :math:`P_1`
to :math:`P_2`, the resulting temperature is

.. math::

    T_2 = T_1 + \frac{L}{C_p}(q_{sat(T_1)} - q_{sat(T_2)})

This requires an iterative solver.

The solver takes in the initial values of pressure (:math:`P_1`) and
humidity mixing ratio (:math:`q_1`), the dry-adiabatically adjusted
temperature (:math:`T_2`) (see :py:meth:`dry_adiabatic_temperature`),
and a guess of the final saturated humidity mixing ratio
(:math:`q_{sat(T_{2}^{'})}`).
From these, it calculates the mass of water unaccounted for:

.. math::

    \Delta q = q_{sat(T_2)} - q_{sat(T_{2}^{'})}

Where

.. math::

    T_{2}^{'} = T_2 + \frac{L}{C_p}(q_{sat(T_2)} - q_{sat(T_{2}^{'})})

The solver iterates until :math:`\Delta q < 1 \times 10^{-6}`. The final
temperature (:math:`T_{2}^{'}`) is calculated from the :py:meth:`saturated_humidity`
method and returned along with the final humidity mixing ratio.
