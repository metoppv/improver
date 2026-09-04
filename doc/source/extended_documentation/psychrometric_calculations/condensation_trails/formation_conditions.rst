**Condition 1**

The local vapour pressure sits above the tangent to the saturated 
vapour pressure curve,

.. math::
    C_1 = e_{local} - mT > I_{critical} \ \ \text{,}

where :math:`e_{local}` is the local vapour pressure, :math:`m` is 
the engine mixing ratio, :math:`T` is the ambient air temperature, 
and :math:`I_{critical}` is the critical intercept.

**Condition 2**

The air temperature is below the critical temperature,

.. math::
    C_2 = T < T_{critical} \ \ \text{.}

**Condition 3**

The local vapour pressure is higher than the atmospheric saturated 
vapour pressure (with respect to ice),

.. math::
    C_3 = e_{local} > e_{s,ice}(T, P) \ \ \text{.}

**Condition 4**

The air temperature is below the freezing point of water,

.. math::
    C_4 = T < 273.15 \ \text{K} \ \ \text{.}

**Will contrails form?**

.. math::
    \begin{aligned}
    \text{Any} &= C_1 \land C_2 \\
    \text{Persistent} &= \text{Any} \land C_3 \land C_4 \\
    \text{Non-persistent} &= \text{Any} \land \lnot \ \text{Persistent} \ \ \text{.}
    \end{aligned}