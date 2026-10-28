Scientific Background
=====================

Numerical weather prediction (NWP) models represent terrain using a smoothed
orography field. Some smoothing arises from the finite model resolution, and
additional smoothing is typically applied to maintain numerical stability and
avoid errors associated with steep terrain gradients. Consequently, terrain
features such as hills, ridges and escarpments are not fully resolved by the
model.

These unresolved terrain features can modify near-surface winds through local
speed-up and sheltering effects. This plugin applies an unresolved-orography
correction based on the linear theory of neutral flow over hills described by
Howard and Clark (2007).

The method uses information describing the unresolved terrain together with the
forecast wind profile to estimate the terrain-induced wind-speed perturbation.
At a high level, the workflow is:

    describe the unresolved terrain
        -> describe the wind profile
        -> calculate the expected terrain effect
        -> apply the correction

The unresolved terrain is approximated as a sinusoidal ridge-trough system
derived from model orography parameters. The resulting terrain-induced
perturbation decays with height and is added to the model wind field through a
multiplicative speed-up factor.


Method Overview
===============

1. Describe the unresolved terrain
----------------------------------

The unresolved-terrain height is calculated at each grid point as:

.. math::

   \Delta h = h_{\mathrm{unsmoothed}} - h_{\mathrm{model}}.

This difference describes how much higher or lower the true (e.g. unsmoothed)
terrain is than the terrain represented by the NWP model.

The horizontal scale of the unresolved terrain must also be estimated. The
same terrain-height difference spread across a broad hill will affect the flow
differently from one occurring across a narrow ridge. Following Howard and
Clark (2007), the model-orography standard deviation and silhouette roughness
are combined to estimate a characteristic terrain wavenumber. Large
wavenumbers represent short-scale terrain; small wavenumbers represent broader
terrain features.

The half peak-to-trough amplitude is estimated from the orography standard
deviation by assuming sinusoidal terrain. The inverse length scale is then
constrained to the configured bounds before conversion to a wavenumber:

.. math::

   A = \sqrt{2}\,\sigma,
   \qquad
   k = \pi\,\operatorname{clip}\!\left(
       \frac{r}{\max(A, A_{\min})},
       \frac{1}{\ell_{\max}},
       \frac{1}{\ell_{\min}}
   \right),

where \(\sigma\) is the sub-grid orography standard deviation, \(r\) is the
dimensionless silhouette roughness, and \(A_{\min}\) is a minimum amplitude
that avoids division by very small values. The defaults are \(A_{\min}=1\) m,
\(\ell_{\min}=500\) m, and \(\ell_{\max}=4000\) m. The calculation is masked
where standard deviation is below 2 m or silhouette roughness is negative.

The unresolved-orography height and characteristic wavenumber together describe
the amplitude and horizontal scale of the terrain not represented by the
model.

2. Describe the model wind profile
----------------------------------

The model provides wind speeds at a series of heights above ground level. At
each grid point, a Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) is
fitted through the model wind profile. PCHIP provides a smooth,
shape-preserving representation of the profile, avoiding the overshoots that
can occur with conventional cubic interpolation.

The spline provides a continuous representation of the model wind profile and
is used to evaluate wind speeds at arbitrary heights within the forecast
profile. If target output heights are requested, it interpolates wind speeds
from the model levels onto those heights. It does not extrapolate beyond the
input height range.

3. Calculate a reference wind speed
-----------------------------------

The terrain effect depends on the strength of the wind flowing over it. A
representative background wind is therefore required to scale the terrain
response.

For each grid point, the reference height is calculated from the characteristic
terrain wavenumber:

.. math::

   z_s = \frac{1}{k}.

The PCHIP spline is evaluated at this height to obtain the reference wind
speed. Broad terrain features have smaller wavenumbers and therefore larger
reference heights; short-scale terrain features have larger wavenumbers and
lower reference heights. The reference wind represents the model wind at the
height scale associated with the unresolved terrain.

4. Estimate an effective roughness length
-----------------------------------------

The flow-over-terrain calculation requires an estimate of aerodynamic
roughness length, which describes the influence of surface friction on the
near-surface flow.

At each grid point, an effective roughness length is estimated by fitting a
neutral logarithmic wind profile to positive model wind speeds between the
surface and 300 m above ground level:

.. math::

   U(z) = \frac{u_*}{\kappa}\,
          \ln\!\left(\frac{z + z_0}{z_0}\right),

where \(U(z)\) is wind speed at height \(z\), \(u_*\) is friction velocity,
\(\kappa=0.4\) is the von Karman constant, and \(z_0\) is effective roughness
length.

This is different to the legacy system, which needed an explicitly provided
roughness-length ancillary. 

5. Calculate the terrain effect
--------------------------------

The flow-over-terrain calculation combines unresolved-orography height,
characteristic wavenumber, reference wind speed, effective roughness length,
and target height. Surface friction is represented by an inner-layer response
calculated using modified Bessel functions. The terrain response also decays
with height, with shorter-scale terrain decaying more rapidly.

In the implementation, the terrain-induced perturbation at target height \(z\)
is:

.. math::

   \Delta U(z) = U(z_s)\,F(z)\,e^{-kz}\,\tanh(k\Delta h),

where \(F(z)\) is the inner-layer response. The perturbation describes the
expected wind-speed change due to terrain unresolved by the NWP model.

6. Apply the correction
-----------------------

The perturbation is expressed relative to the background wind to produce a
multiplicative correction factor:

.. math::

   f(z) = \operatorname{clip}\!\left(
       \frac{\Delta U(z)}{U_{\mathrm{background}}(z)}, 0, 1
   \right),
   \qquad
   U_{\mathrm{corrected}}(z) =
       U_{\mathrm{background}}(z)\,[1 + f(z)].

The correction factor is therefore between 1 and 2. Negative perturbations are
clipped to zero, so this implementation applies speed-ups only; it does not
apply terrain-induced slow-downs. Invalid ancillary values receive a neutral
correction factor of 1, while masked target winds remain masked.

Corrected winds are returned on either the original forecast height levels or
user-specified target heights if these are supplied.

Input Requirements
==================

Wind forecast:

* Height coordinates must be in metres above ground level.
* At least two height levels must be available.
* The input profile must extend to at least 300 m above ground level.
* Ensemble forecasts are supported; each realization is processed independently.

Orography ancillaries:

* Unsmoothed (e.g. true) orography, in metres.
* Model orography, in metres.
* Sub-grid orography standard deviation, in metres.
* Sub-grid silhouette roughness, dimensionless.

Ancillary inputs must be on the same horizontal grid as the wind forecast. If
the wind forecast has been regridded from its native model grid (e.g. by StaGE),
the ancillary fields should be regridded consistently.
