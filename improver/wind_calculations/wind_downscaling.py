# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing wind downscaling plugins."""

import itertools
from typing import Optional, Tuple, Union

import iris
import numpy as np
from cf_units import Unit
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from numpy import ndarray

from improver import BasePlugin, PostProcessingPlugin
from improver.constants import RMDI

# Fractional tolerance used when deciding whether an absolute correction to
# the computed reference height is significant. Corrections smaller than this
# value are treated as noise and ignored.
ABSOLUTE_CORRECTION_TOL = 0.04

# Multiplier used to convert terrain‑related variability (e.g. orographic
# standard deviation or silhouette roughness) into an effective reference
# height for the logarithmic wind‑profile calculation.
HREF_SCALE = 2.0

# Von Karman's constant, used in the logarithmic wind‑profile equation.
VONKARMAN = 0.4

# Default roughness length (in metres) assigned to sea grid cells when no
# surface roughness information is available.
Z0M_SEA = 0.0001


class FrictionVelocity(BasePlugin):
    """Compute friction velocity u_star using the logarithmic wind profile:
        u_star = K * u(h_ref) / ln(h_ref / z0)
    where:
      - u(h_ref) is wind speed evaluated at the reference height h_ref,
      - z0 is the aerodynamic roughness length,
      - K is the von Karman constant.

    Notes:
      - h_ref and model_z0 must share the same units.
      - The returned u_star has the same velocity units as wspeed_at_h_ref.
    """

    def __init__(
        self,
        wspeed_at_h_ref: ndarray,
        h_ref: ndarray,
        model_z0: ndarray,
        ustar_mask: ndarray,
    ) -> None:
        """Initialize the friction-velocity calculator.

        Args:
            wspeed_at_h_ref: Wind speed evaluated at the reference height h_ref.
            h_ref: Effective reference height h_ref.
            model_z0: Vegetative roughness length z0.
            ustar_mask: 2D boolean array, True where u_star should be computed.

        Raises:
            ValueError: If any input array has a different size to the others.
        """
        self.wspeed_at_h_ref = wspeed_at_h_ref
        self.h_ref = h_ref
        self.model_z0 = model_z0
        self.ustar_mask = ustar_mask

        # Check array sizes all the same
        sizes = [
            np.size(wspeed_at_h_ref),
            np.size(h_ref),
            np.size(model_z0),
            np.size(ustar_mask),
        ]
        if not all(s == sizes[0] for s in sizes):
            raise ValueError(
                "Input arrays must have identical sizes, but sizes are: "
                f"wspeed_at_h_ref={sizes[0]}, "
                f"h_ref={sizes[1]}, "
                f"model_z0={sizes[2]}, "
                f"ustar_mask={sizes[3]}"
            )

    def process(self) -> ndarray:
        """Compute friction velocity (u_star) using the logarithmic wind profile.

        The friction velocity is computed according to the neutral-stability
        logarithmic wind law:
            u_star = K * u(h_ref) / ln(h_ref / z0)
        where:
          - u(h_ref) is wind speed evaluated at the reference height h_ref,
          - z0 is the aerodynamic roughness length,
          - K is the von Karman constant.

        This method applies the calculation only at locations where
        ustar_mask is True. All other points are filled with
        missing-data indicators.

        Returns:
            2D float32 array of friction velocity u_star.
        """
        ustar = np.full(self.wspeed_at_h_ref.shape, RMDI, dtype=np.float32)

        # Values at locations where u_star is to be computed
        wind_vals = self.wspeed_at_h_ref[self.ustar_mask]

        # Compute log(h_ref / z0)
        with np.errstate(invalid="ignore"):
            log_term = np.log(
                self.h_ref[self.ustar_mask] / self.model_z0[self.ustar_mask]
            )

        # Compute u_star following the log-profile relation
        ustar[self.ustar_mask] = VONKARMAN * (wind_vals / log_term)

        return ustar


class WindTerrainAdjustmentUtilities:
    """Compute wind-speed corrections related to surface roughness and
    height differences.

    Provides methods to apply roughness and height adjustments to forecast data
    using ancillary inputs:

    - model_silhouette_roughness (ndarray):
        Dimensionless measure of sub-grid terrain steepness and associated drag.
    - model_orog_stddev (ndarray):
        Standard deviation of sub-grid orography height (m), describing terrain variability.
    - model_z0 (ndarray):
        Vegetative roughness length (m), representing surface drag from land cover.
    - target_orog (ndarray):
        High-resolution (target) orography (m) to which winds are downscaled.
    - model_orog (ndarray):
        Model orography (m), representing smoothed terrain.
    - height_levels (ndarray):
        Heights (m) corresponding to the wind field vertical coordinate.
    - wind_field (ndarray):
        3D wind speed field defined on the height levels.

    Note that all input fields must be defined on the same grid as the wind field
    (i.e. the target/post-processed grid). In particular, ancillary inputs derived on
    the model grid (e.g. model_orog) should be regridded to this target grid before use.
    """

    def __init__(
        self,
        model_silhouette_roughness: ndarray,
        model_orog_stddev: ndarray,
        model_z0: ndarray,
        target_orog: ndarray,
        model_orog: ndarray,
        output_res: float,
        model_res: float,
    ) -> None:
        """Initialise roughness and height-correction parameters.

        Args:
            model_silhouette_roughness: 2D array of dimensionless silhouette roughness, describing
                sub-grid terrain steepness and associated drag.

            model_orog_stddev: 2D array of orographic standard deviation (m), representing
                sub-grid terrain height variability.

            model_z0: 2D array of vegetative roughness length (m), controlling
                near-surface wind drag from land cover.

            target_orog: 2D array of high-resolution (target) orography (m) to which
                winds are downscaled.

            model_orog: 2D array of model orography (m), representing smoothed terrain.

            output_res: Horizontal resolution of the target grid (m).

            model_res: Horizontal resolution of the raw NWP model grid (m).
        """
        self.model_silhouette_roughness = model_silhouette_roughness
        self.model_z0 = model_z0
        self.target_orog = target_orog
        self.model_orog = model_orog

        # Half peak‑to‑trough orographic height
        self.h_half = self.model_orog_stddev_to_h_half(model_orog_stddev)

        # Height‑correction and roughness‑correction masks
        self.hc_mask, self.rc_mask = self._setmask()

        # Replace non‑positive roughness values with the default sea roughness
        if self.model_z0 is not None:
            self.model_z0[self.model_z0 <= 0] = Z0M_SEA

        # Minimum resolvable scale on the post-processing grid
        self.dx_min = output_res / 2.0

        # Maximum unresolved scale on the model grid
        self.dx_max = 3.0 * model_res

        # Wavenumber of terrain variability: k = 2π / L
        self.wavenumber = self._calc_wavenumber()

        # Reference height used for roughness correction.
        self.h_ref = self._calc_h_ref()

        # Update height correction mask for missing orography
        self._refinemask()

        # Height difference between post-processing and model orography
        self.h_at0 = self._delta_height()

    def _refinemask(self) -> None:
        """Refine the height-correction mask based on invalid orography values.

        The height-correction mask (hc_mask) must be set to False wherever either
        the post-processing or model orography contains invalid values (e.g.
        missing data indicators or NaNs).

        This cannot be done earlier because hc_mask is used when calculating
        the wavenumber, and the wavenumber should be computed for all points
        where both h_half and model_silhouette_roughness are valid (even if the
        corresponding orography values are not).
        """
        self.hc_mask[
            np.equal(self.target_orog, RMDI)
            | np.equal(self.model_orog, RMDI)
            | np.isnan(self.target_orog)
            | np.isnan(self.model_orog)
        ] = False

    def _setmask(self) -> Tuple[ndarray, ndarray]:
        """Create the height-correction (hc_mask) and roughness-correction
        (rc_mask) masks.

        The height-correction mask acts like a land-sea mask: both h_half and
        model_silhouette_roughness are zero over the sea, and a standard deviation
        of 0 results in a missing data indicator for h_half.

        The roughness-correction mask begins as hc_mask but also excludes
        points where the vegetative roughness length (model_z0)
        is missing or non-positive.
        """
        # Height‑correction mask
        hc_mask = np.full(self.h_half.shape, True, dtype=bool)
        hc_mask[self.h_half <= 0] = False
        hc_mask[self.model_silhouette_roughness <= 0] = False
        hc_mask[np.isnan(self.h_half)] = False
        hc_mask[np.isnan(self.model_silhouette_roughness)] = False

        # Roughness‑correction mask
        rc_mask = np.copy(hc_mask)
        if self.model_z0 is not None:
            rc_mask[self.model_z0 <= 0] = False
            rc_mask[np.isnan(self.model_z0)] = False

        return hc_mask, rc_mask

    @staticmethod
    def model_orog_stddev_to_h_half(model_orog_stddev: ndarray) -> ndarray:
        """Convert orography standard deviation into half the peak-to-trough
        height.

        The ancillary data used to estimate the peak to trough height
        contains the standard deviation of height in a cell. For
        sine-waves, this relates to the amplitude of the wave as:
            amplitude = model_orog_stddev * sqrt(2)

        This amplitude corresponds to half the peak-to-trough height (h_half).

        Args:
            model_orog_stddev: 2D float32 array containing the standard deviation of terrain
                height within each grid cell (metres).

        Returns:
            2D float32 array of half the peak-to-trough height (h_half).
            Points with zero or missing input are assigned the missing-data
            indicator.
        """
        h_half = np.full(model_orog_stddev.shape, RMDI, dtype=np.float32)
        valid = model_orog_stddev > 0
        h_half[valid] = model_orog_stddev[valid] * np.sqrt(2.0)
        return h_half

    def _calc_wavenumber(self) -> ndarray:
        """Calculate the wavenumber k associated with the orographic length
        scale.

        The orographic length scale L is estimated from the half
        peak-to-trough height (h_half) and the silhouette-roughness field
        (average of up-slopes per unit length over several cross-sections
        through a grid cell) using the relationship
            L = 2 * h_half / model_silhouette_roughness

        The corresponding wavenumber k is
            k = 2π / L = (model_silhouette_roughness * π) / h_half

        h_half is derived from the standard deviation of sub-grid terrain
        height (model_orog_stddev) as in model_orog_stddev_to_h_half:
            h_half = model_orog_stddev * sqrt(2).

        Wavenumbers are then limited to the smallest and largest scales that
        can be represented by the post-processing grid and the model grid.

        Grid points where h_half is zero or missing are given the missing-data
        indicator for the wavenumber.

        Returns:
            2D float32 array of wavenumbers, in units of
            inverse units of supplied h_half.
        """
        wavenumber = np.full(
            self.model_silhouette_roughness.shape, RMDI, dtype=np.float32
        )

        # Compute wavenumber k for valid height‑correction points
        valid = self.hc_mask
        wavenumber[valid] = (
            self.model_silhouette_roughness[valid] * np.pi
        ) / self.h_half[valid]

        # Apply upper/lower bounds determined by smallest+largest resolvable scales
        wavenumber[wavenumber > (np.pi / self.dx_min)] = np.pi / self.dx_min
        wavenumber[self.h_half == 0] = RMDI
        wavenumber[np.abs(wavenumber) < (np.pi / self.dx_max)] = np.pi / self.dx_max

        return wavenumber

    def _calc_h_ref(self) -> ndarray:
        """Calculate the reference height h_ref for roughness correction.

        The reference height marks the height below which the flow is
        considered to be in equilibrium with the vegetative roughness.
        This height is proportional to 1 / wavenumber (Howard & Clark, 2007).

        Vosper (2009) and Clark (2009) argue that at the reference
        height, the perturbation should have decayed to a fraction
        epsilon (ABSOLUTE_CORRECTION_TOL).

        The factor alpha implements eq. 1.3 in Clark (2009): UK Climatology
        - Wind Screening Tool. See also Vosper (2009) for a motivation. It is
        defined as alpha = -log(ABSOLUTE_CORRECTION_TOL)

        alpha is the log of scale parameter to determine reference
        height which is currently set to 0.04 (this corresponds to
        epsilon in both Vosper and Clark)

        Returns:
            ndarray: 2D array of reference height h_ref for roughness
                correction.

        References:
            Howard T., Clark P. 2007. Correction and downscaling of NWP wind
            speed forecasts. Meteorological Applications 14(2), 105-116.
        """
        alpha = -np.log(ABSOLUTE_CORRECTION_TOL)

        tunable = np.full(self.wavenumber.shape, RMDI, dtype=np.float32)
        h_ref = np.full(self.wavenumber.shape, RMDI, dtype=np.float32)

        # Compute tunable parameter for valid points
        valid = self.hc_mask
        tunable[valid] = alpha + np.log(self.wavenumber[valid] * self.h_half[valid])
        tunable = np.clip(tunable, 0.0, 1.0)

        # Compute reference height
        h_ref[valid] = tunable[valid] / self.wavenumber[valid]

        # Enforce lower and upper bounds on h_ref
        h_ref = np.maximum(h_ref, 1.0)
        h_ref = np.minimum(h_ref, HREF_SCALE * self.h_half)
        h_ref = np.maximum(h_ref, 1.0)

        # For points outside hc_mask, no roughness correction is applied
        h_ref[~self.hc_mask] = 0.0

        return h_ref

    def calc_roughness_correction(
        self,
        height_above_orog: ndarray,
        wspeed_original: ndarray,
        rc_mask: ndarray,
    ) -> ndarray:
        """Apply the roughness correction.

        Args:
            height_above_orog: 3D or 1D float32 array giving height above orography.
            wspeed_original: 3D float32 array containing the original wind speeds.
            rc_mask: 2D boolean array. True where roughness correction is valid
                (e.g. land points with valid vegetative roughness length),
                False elsewhere.

        Returns:
            3D float32 array of roughness-corrected windspeeds.
            Above the reference height h_ref, values remain unchanged from
            wspeed_original.
        """
        wspeed_new = np.copy(wspeed_original)

        # Windspeed at the reference height
        u_at_href = self._interpolate_wspeed_to_height(
            wspeed_original, height_above_orog, self.h_ref, rc_mask
        )

        # Friction velocity u_star
        ustar = FrictionVelocity(u_at_href, self.h_ref, self.model_z0, rc_mask)()

        # h_ref = 0 where roughness correction does not apply
        h_ref = np.copy(self.h_ref)
        h_ref[~rc_mask] = 0.0

        # Ensure broadcast correctly (expand 1D to 3D)
        if height_above_orog.ndim == 1:
            height_above_orog = height_above_orog[np.newaxis, np.newaxis, :]
        ustar_3d = ustar[:, :, np.newaxis] * np.ones_like(height_above_orog)
        z0_3d = self.model_z0[:, :, np.newaxis] * np.ones_like(height_above_orog)

        # Apply the roughness correction below the reference height
        below_href = height_above_orog < h_ref[:, :, np.newaxis]
        log_term = np.log(height_above_orog / z0_3d)[below_href]
        ustar_term = ustar_3d[below_href]
        wspeed_new[below_href] = (ustar_term * log_term) / VONKARMAN

        return wspeed_new

    def _interpolate_wspeed_to_height(
        self,
        wspeed_in: ndarray,
        height_levels_in: ndarray,
        height_target: ndarray,
        valid_mask: ndarray,
        use_log_interpolation: bool = False,
    ) -> ndarray:
        """Interpolate wind speed from input height levels to a target height.

        Args:
            wspeed_in: 3D float32 array of wind speed defined on height_levels_in.
                Last dimension is height.
            height_levels_in: 3D or 1D float32 array of heights corresponding to wspeed_in.
            height_target: 2D float32 array giving the target height at which to interpolate.
            valid_mask: 2D boolean array. True where interpolation is permitted.
            use_log_interpolation: If True, perform logarithmic interpolation. Otherwise linear.

        Returns:
            2D float32 array of interpolated wind speed.
        """

        # Mask invalid (negative) heights and speeds
        wspeed_in = np.ma.masked_less(wspeed_in, 0.0)
        height_levels_in = np.ma.masked_less(height_levels_in, 0.0)
        height_target = np.ma.masked_less(height_target, 0.0)

        # Find indices of the first height above and below the target height
        above_idx = np.argmax(
            height_levels_in > height_target[:, :, np.newaxis], axis=2
        )

        # Index of the height just below target
        below_idx = np.argmin(
            np.ma.masked_less(height_target[:, :, np.newaxis] - height_levels_in, 0.0),
            axis=2,
        )

        # Extract bounding heights (upper and lower)
        if height_levels_in.ndim == 3:
            flat_stride = height_levels_in.shape[2]
            h_upper = height_levels_in.take(
                above_idx.flatten()
                + np.arange(0, above_idx.size * flat_stride, flat_stride)
            )
            h_lower = height_levels_in.take(
                below_idx.flatten()
                + np.arange(0, below_idx.size * flat_stride, flat_stride)
            )
        else:
            h_upper = height_levels_in[above_idx].flatten()
            h_lower = height_levels_in[below_idx].flatten()

        # Extract bounding wind‑speed values (upper and lower)
        flat_stride_u = wspeed_in.shape[2]
        u_upper = wspeed_in.take(
            above_idx.flatten()
            + np.arange(0, above_idx.size * flat_stride_u, flat_stride_u)
        )
        u_lower = wspeed_in.take(
            below_idx.flatten()
            + np.arange(0, below_idx.size * flat_stride_u, flat_stride_u)
        )

        # Choose interpolation method
        valid_mask_flat = valid_mask.flatten()
        u_at_height = np.full(valid_mask_flat.shape, RMDI, dtype=np.float32)
        if use_log_interpolation:
            u_at_height[valid_mask_flat] = self._interpolate_log(
                h_upper[valid_mask_flat],
                h_lower[valid_mask_flat],
                height_target.flatten()[valid_mask_flat],
                u_upper[valid_mask_flat],
                u_lower[valid_mask_flat],
            )
        else:
            u_at_height[valid_mask_flat] = self._interpolate_1d(
                h_upper[valid_mask_flat],
                h_lower[valid_mask_flat],
                height_target.flatten()[valid_mask_flat],
                u_upper[valid_mask_flat],
                u_lower[valid_mask_flat],
            )

        # Reshape to 2D field
        return np.reshape(u_at_height, height_target.shape)

    @staticmethod
    def _interpolate_1d(
        x_upper: ndarray,
        x_lower: ndarray,
        x_target: ndarray,
        y_upper: ndarray,
        y_lower: ndarray,
    ) -> ndarray:
        """Simple 1D linear interpolation for 2D grid inputs level.

        Args:
            x_upper:Upper x-coordinates (e.g., upper heights).
            x_lower:Lower x-coordinates (e.g., lower heights).
            x_target:Target x-values to interpolate at.
            y_upper:Values at x_upper.
            y_lower:Values at x_lower.

        Returns:
            Interpolated y-values at x_target. Missing-data indicator is
            returned where interpolation cannot be performed.
        """
        interp = np.full(x_upper.shape, RMDI, dtype=np.float32)
        diff = x_upper - x_lower

        # Standard linear interpolation when x_upper != x_lower
        valid = diff != 0
        interp[valid] = y_lower[valid] + (x_target[valid] - x_lower[valid]) / diff[
            valid
        ] * (y_upper[valid] - y_lower[valid])

        # Fallback for x_upper == x_lower
        collapse = ~valid
        interp[collapse] = x_target[collapse] / x_upper[collapse] * y_upper[collapse]

        return interp

    @staticmethod
    def _interpolate_log(
        x_upper: ndarray,
        x_lower: ndarray,
        x_target: ndarray,
        y_upper: ndarray,
        y_lower: ndarray,
    ) -> ndarray:
        """Simple 1D log interpolation y(x), except if lowest layer is
        ground level.

        Args:
            x_upper: Upper x-coordinates (e.g., upper heights).
            x_lower: Lower x-coordinates (e.g., lower heights).
            x_target: Target x-values to interpolate at.
            y_upper: Values at x_upper.
            y_lower: Values at x_lower.

        Returns:
            Interpolated y-values at x_target.
        """
        out = np.full(x_upper.shape, RMDI, dtype=np.float32)
        ratio = x_upper / x_lower

        # Case 1: x_upper != x_lower and x_target != x_upper
        # Log interpolation
        normal = (ratio != 1.0) & (x_target != x_upper)
        a = np.full(x_upper.shape, RMDI, dtype=np.float32)
        a[normal] = (y_upper[normal] - y_lower[normal]) / np.log(ratio[normal])
        out[normal] = (
            a[normal] * np.log(x_target[normal] / x_upper[normal]) + y_upper[normal]
        )

        # Case 2: x_upper == x_lower
        # Collapse to linear scaling
        collapse = ratio == 1.0
        out[collapse] = (x_target[collapse] / x_upper[collapse]) * y_upper[collapse]

        # Case 3: x_target == x_upper
        same_level = x_target == x_upper
        out[same_level] = y_upper[same_level]

        return out

    def _calc_height_corr(
        self,
        wspeed_outer: ndarray,
        height_above_orog: ndarray,
        valid_mask: ndarray,
        onemfrac: Union[float, ndarray],
    ) -> ndarray:
        """Calculate the additive height correction.

        Args:
            wspeed_outer: 2D float32 array of wind speed at the reference height.
            height_above_orog: 1D or 3D float32 array of heights above orography.
            valid_mask: 3D boolean array where the correction should be applied.
            onemfrac (float or ndarray): Currently, scalar = 1. But can be a function
                of position and height, e.g. a 3D array (float32)

        Returns:
            3D float32 array of additive height correction.

        Comments:
            The height correction is a disturbance of the flow that
            decays exponentially with height. The larger the vertical
            offset (the higher the unresolved hill), the larger the
            disturbance.

            The more smooth the disturbance (the larger the horizontal
            scale of the disturbance), the smaller the height
            correction (hence, a larger wavenumber results in a larger
            disturbance).

            A final factor of 1 is assumed and omitted for the Bessel
            function term.
        """
        nx, ny = wspeed_outer.shape

        # Ensure heights are 3D for broadcasting
        if height_above_orog.ndim == 1:
            nz = height_above_orog.shape[0]
            height_above_orog = height_above_orog[np.newaxis, np.newaxis, :]
        else:
            nz = height_above_orog.shape[2]

        # Amplitude term
        amp = self.h_at0 * self.wavenumber

        # Exponential decay factor exp(-k * z)
        decay = np.ones((nx, ny, nz), dtype=np.float32)
        kz = self.wavenumber[:, :, np.newaxis] * height_above_orog
        decay[kz > 1e-4] = np.exp(-kz[kz > 1e-4])

        # Full additive height correction
        hc_add = (
            decay * wspeed_outer[:, :, np.newaxis] * amp[:, :, np.newaxis] * onemfrac
        )

        # Zero correction where mask is False
        hc_add[~valid_mask] = 0.0

        return hc_add

    def _delta_height(self) -> ndarray:
        """Calculate the difference between post-processing grid height and
        model grid height.

        Returns:
            2D float32 array of height difference, defined as target_orog - model_orog.
        """
        delt_z = np.full(self.target_orog.shape, RMDI, dtype=np.float32)
        valid = self.hc_mask
        delt_z[valid] = self.target_orog[valid] - self.model_orog[valid]
        return delt_z

    def _mask_missing_data(self, height_above_orog, wspeed_original):
        """Return a boolean mask: True where either RC or HC may be applied.

        Args:
            height_above_orog: 1D or 3D float32 array of heights above local orography.
            wspeed_original: 3D float32 array of wind speed defined on height_above_orog.

        Returns:
            3D float32 array of wind speed after applying RC and HC.
        """
        valid = np.ones_like(self.rc_mask, dtype=bool)

        # Disable RC/HC where height inputs contain missing values
        if height_above_orog.ndim == 3:
            missing_h = (height_above_orog == RMDI).any(axis=2)
            valid[missing_h] = False

        # Disable RC/HC wherever the vertical wind profile is missing
        missing_w = (wspeed_original == RMDI).any(axis=2)
        valid[missing_w] = False

        return valid

    def do_rc(self, height_above_orog, wspeed_original):
        if self.model_z0 is None:
            raise ValueError(
                "Roughness correction (RC) was requested, but no roughness-length "
                "field (z0_cube) was supplied."
            )

        # Mask where missing data in height and wind fields
        valid = self._mask_missing_data(height_above_orog, wspeed_original)
        mask_rc = np.copy(self.rc_mask)
        mask_rc[~valid] = False

        return self.calc_roughness_correction(
            height_above_orog, wspeed_original, mask_rc
        )

    def do_hc(self, height_above_orog, wspeed_original):
        # Mask where missing data in height and wind fields
        valid = self._mask_missing_data(height_above_orog, wspeed_original)
        mask_hc = np.copy(self.hc_mask)
        mask_hc[~valid] = False

        # Height correction
        # Requires wind speed at the reference height, so interpolate first
        uhref_orig = self._interpolate_wspeed_to_height(
            wspeed_original,
            height_above_orog,
            1.0 / self.wavenumber,
            mask_hc,
        )

        # HC only where u(h_ref) is positive
        mask_hc[uhref_orig <= 0.0] = False

        # Setting this value to 1, is equivalent to setting the
        # Bessel function to 1. (Friedrich, 2016)
        # Example usage if the Bessel function was not set to 1 is:
        # onemfrac = 1.0 - BfuncFrac(nx,ny,nz,heightvec,z_0,waveno, Ustar, UI)
        onemfrac = 1.0
        hc_add = self._calc_height_corr(
            uhref_orig, height_above_orog, mask_hc, onemfrac
        )

        # Apply HC additively
        wspeed_out = wspeed_original + hc_add

        # Enforce non-negative wind speeds
        # HC can be negative if target_orog < model_orog
        wspeed_out[wspeed_out < 0.0] = 0.0

        return wspeed_out.astype(np.float32)

    def do_rc_and_hc(self, height_above_orog, wspeed_original):
        """
        Apply roughness correction (RC) followed by height correction (HC)
        to the wind field.
        """
        wspeed_rc = self.do_rc(height_above_orog, wspeed_original)
        wspeed_hc = self.do_hc(height_above_orog, wspeed_rc)
        return wspeed_hc.astype(np.float32)


class WindTerrainAdjustment(PostProcessingPlugin):
    """Plugin to orographically-correct 3d wind speeds."""

    zcoordnames = ["height", "model_level_number"]
    tcoordnames = ["time", "forecast_time"]

    def __init__(
        self,
        model_silhouette_roughness_cube: Cube,
        model_orog_stddev_cube: Cube,
        target_orog_cube: Cube,
        model_orog_cube: Cube,
        model_res: float,
        model_z0_cube: Optional[Cube] = None,
        height_levels_cube: Optional[Cube] = None,
        mode: str = "hc_and_rc",
    ) -> None:
        """Initialise the WindTerrainAdjustment plugin.

        Args:
            model_silhouette_roughness_cube:
                2D model silhouette roughness (dimensionless). Describes how steep
                and rugged unresolved terrain is within a model grid box, and hence
                the amount of drag and turbulence it introduces.
                This is a static model ancillary field.

            model_orog_stddev_cube:
                2D standard deviation of model orography height (m). Represents the
                vertical variability of unresolved terrain within a grid box (i.e.
                how large the sub-grid hills and valleys are).
                This is a static model ancillary field.

            target_orog_cube:
                2D high-resolution (true) orography (m) that winds are downscaled to.

            model_orog_cube:
                2D model orography (m), representing the smoothed terrain used by
                the model.
                This is a static model ancillary field.

            model_res:
                Native horizontal resolution of the model orography (m), prior to
                interpolation onto the standard grid.

            model_z0_cube:
                2D vegetative roughness length (m), representing drag from vegetation
                and land cover. Controls the near-surface wind profile.
                Historically static, but may now be time-varying (e.g. from StaGE).

            height_levels_cube:
                1D or 3D height levels of the input wind field (m).

            mode:
                Which correction(s) to apply: "hc_and_rc" (default), "hc", or "rc".

        Notes:
            All ancillary inputs must be defined on the same grid as the wind field
            (the target / post-processed grid). Fields originating on the model grid
            must be regridded prior to use.

        References:
            Howard T., Clark P. 2007. Correction and downscaling of NWP wind
            speed forecasts. Meteorological Applications 14(2), 105-116.
        """
        valid_modes = ("hc_and_rc", "hc", "rc")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}")
        self.mode = mode

        # Roughness correction cannot be performed without providing roughness length
        if "rc" in self.mode and model_z0_cube is None:
            raise ValueError(
                f"Roughness correction (RC) requested via mode={self.mode!r}, "
                "but no model_z0_cube was supplied. Provide a roughness-length cube or use mode='hc'."
            )

        model_res = np.float32(model_res)
        x_name, y_name, _, _ = self.find_coord_names(target_orog_cube)

        # Check grid consistency
        if not self.check_ancils(
            model_silhouette_roughness_cube,
            model_orog_stddev_cube,
            model_z0_cube,
            target_orog_cube,
            model_orog_cube,
        ):
            raise ValueError("Ancillary grids are not consistent.")

        # Extract 2D [y, x] slices
        self.model_silhouette_roughness = next(
            model_silhouette_roughness_cube.slices([y_name, x_name])
        )
        self.model_orog_stddev = next(model_orog_stddev_cube.slices([y_name, x_name]))

        try:
            self.model_z0 = next(model_z0_cube.slices([y_name, x_name]))
        except AttributeError:
            self.model_z0 = model_z0_cube

        self.target_orog = next(target_orog_cube.slices([y_name, x_name]))
        self.model_orog = next(model_orog_cube.slices([y_name, x_name]))

        # Grid resolutions
        self.output_res = self.calc_av_output_res(target_orog_cube)
        self.model_res = model_res

        # Optional height levels
        self.height_levels = height_levels_cube

        # Store coordinate names
        self.x_name = x_name
        self.y_name = y_name
        self.z_name = None
        self.t_name = None

    def find_coord_names(self, cube: Cube) -> Tuple[str, str, str, str]:
        """Extract x, y, z, and time coordinate names.

        Args:
            cube: Cube from which coordinate names will be extracted.

        Returns:
            (x_name, y_name, z_name, t_name)
        """
        coord_names = {coord.name() for coord in cube.coords()}

        # x coordinate
        try:
            x_name = cube.coord(axis="x").name()
        except CoordinateNotFoundError as exc:
            print(f"'{exc}' while determining x_name. Args: {exc.args}")
            x_name = None

        # y coordinate
        try:
            y_name = cube.coord(axis="y").name()
        except CoordinateNotFoundError as exc:
            print(f"'{exc}' while determining y_name. Args: {exc.args}")
            y_name = None

        # Check spatial coordinates exist
        missing = [
            name for name, value in (("x", x_name), ("y", y_name)) if value is None
        ]
        if missing:
            raise ValueError(
                f"Cube is missing required spatial coordinate(s): {', '.join(missing)}"
            )

        # z coordinate
        z_matches = coord_names.intersection(self.zcoordnames)
        z_name = next(iter(z_matches), None)

        # time coordinate
        t_matches = coord_names.intersection(self.tcoordnames)
        t_name = next(iter(t_matches), None)

        return x_name, y_name, z_name, t_name

    def calc_av_output_res(self, input_cube: Cube) -> float:
        """Calculate the average horizontal resolution of the given cube.

        Args:
            input_cube: Cube from which to determine the grid spacing.

        Returns:
            Average horizontal grid resolution (metres).
        """
        # Identify horizontal coordinate names
        x_name, y_name, _, _ = self.find_coord_names(input_cube)

        # Expected coordinates and units
        expected_x = "projection_x_coordinate"
        expected_y = "projection_y_coordinate"
        expected_units = Unit("m")

        if x_name != expected_x or y_name != expected_y:
            raise ValueError("Cannot calculate resolution: unexpected horizontal axes.")

        x_coord = input_cube.coord(x_name)
        y_coord = input_cube.coord(y_name)

        # Use bounds if available, else use point spacing
        if x_coord.bounds is None and y_coord.bounds is None:
            xres = np.diff(x_coord.points).mean()
            yres = np.diff(y_coord.points).mean()
        else:
            xres = np.diff(x_coord.bounds).mean()
            yres = np.diff(y_coord.bounds).mean()

        # Ensure units are metres
        if x_coord.units != expected_units or y_coord.units != expected_units:
            raise ValueError("Post-processing grid axes must have units of metres.")

        # Mean absolute resolution
        return (abs(xres) + abs(yres)) / 2.0

    @staticmethod
    def check_ancils(
        model_silhouette_roughness_cube: Cube,
        model_orog_stddev_cube: Cube,
        model_z0_cube: Optional[Cube],
        target_orog_cube: Cube,
        model_orog_cube: Cube,
    ) -> bool:
        """
        Check ancillary inputs for grid consistency and expected units.

        Ensures all ancillary cubes are on the same spatial grid and have
        appropriate units for wind downscaling.

        Args:
            model_silhouette_roughness_cube:
                Dimensionless field describing sub-grid terrain ruggedness.
            model_orog_stddev_cube:
                Standard deviation of sub-grid orography height (m).
            model_z0_cube:
                Vegetative roughness length (m), representing surface drag
                from land cover.
            target_orog_cube:
                High-resolution (target) orography (m) used for downscaling.
            model_orog_cube:
                Model orography (m) representing the smoothed terrain.

        Returns:
            bool:
                True if all ancillary fields share the same x/y grid;
                False otherwise.
        """
        required = [
            model_silhouette_roughness_cube,
            model_orog_stddev_cube,
            target_orog_cube,
            model_orog_cube,
        ]
        required_units = [1, Unit("m"), Unit("m"), Unit("m")]

        # Coords to strip before comparing horizontal grids
        drop_coords = [
            "time",
            "height",
            "model_level_number",
            "forecast_time",
            "forecast_reference_time",
            "forecast_period",
        ]

        # Clean and check required cubes
        for field, expected in zip(required, required_units):
            for name in drop_coords:
                try:
                    field.remove_coord(name)
                except CoordinateNotFoundError:
                    pass
            if field.units != expected:
                raise ValueError(
                    f"{field.name()} ancillary has unexpected unit: "
                    f"expected {expected}, got {field.units}"
                )

        # Clean and check z0 if supplied
        if model_z0_cube is not None:
            for name in drop_coords:
                try:
                    model_z0_cube.remove_coord(name)
                except CoordinateNotFoundError:
                    pass
            if model_z0_cube.units != Unit("m"):
                raise ValueError(
                    f"z0 ancillary has unexpected unit: "
                    f"expected m, got {model_z0_cube.units}"
                )
            required.append(model_z0_cube)

        # Pairwise x/y grid compatibility check across all ancils
        ok_pairs: list[bool] = []
        for a, b in itertools.permutations(required, 2):
            try:
                same_y = a.coord(axis="y") == b.coord(axis="y")
                same_x = a.coord(axis="x") == b.coord(axis="x")
                ok_pairs.append(bool(same_x & same_y))
            except CoordinateNotFoundError:
                ok_pairs.append(False)

        return all(ok_pairs)

    def find_coord_order(self, mcube: Cube) -> Tuple[int, int, int, int]:
        """Return the dimension indices of the x, y, z, and time coordinates.

        Use coord_dims to assess the dimension associated with a particular
        dimension coordinate. If a coordinate is not a dimension coordinate,
        then a NaN value will be returned for that coordinate.

        Returns:
            (x_dim, y_dim, z_dim, t_dim) with NaN for coordinates not found.
        """
        coord_names = [self.x_name, self.y_name, self.z_name, self.t_name]
        positions = [np.nan, np.nan, np.nan, np.nan]

        for idx, coord_name in enumerate(coord_names):
            # Skip missing coord names
            if coord_name is None:
                continue
            # Record coordinate axis number
            try:
                if mcube.coords(coord_name, dim_coords=True):
                    (positions[idx],) = mcube.coord_dims(coord_name)
            except CoordinateNotFoundError:
                # Coordinate does not exist, so leave as NaN
                pass

        return tuple(positions)

    def find_heightgrid(self, wind: Cube) -> ndarray:
        """Find the height grid to use for interpolation.

        If no height-levels cube is supplied, use the vertical coordinate
        from the wind cube. Otherwise use the provided height-levels cube.

        Args:
            wind: 3D or 4D wind-speed cube.

        Returns:
            1D or 3D array of heights (metres).
        """
        # Case 1: No external height-levels cube provided
        # -> use wind cube's z‑axis
        if self.height_levels is None:
            return wind.coord(self.z_name).points

        # Case 2: Use the height-levels cube
        hld = iris.util.squeeze(self.height_levels)
        if np.isnan(hld.data).any() or (hld.data == RMDI).any():
            raise ValueError("Height grid contains invalid points.")
        if hld.ndim == 3:
            try:
                x_dim, y_dim, z_dim, _ = self.find_coord_order(hld)
                hld = hld.transpose([y_dim, x_dim, z_dim])
            except Exception:
                raise ValueError("Height grid does not align with wind grid.")
        elif hld.ndim == 1:
            try:
                hld = next(hld.slices([self.z_name]))
            except CoordinateNotFoundError:
                raise ValueError("Height grid z‑coordinate differs from wind grid.")
        else:
            raise ValueError(f"Height grid must be 1D or 3D, got ndim = {hld.ndim}.")

        return hld.data

    def check_wind_ancil(self, xwp: int, ywp: int) -> None:
        """Verify that the wind field and ancillary grids share the same
        horizontal orientation.

        Args:
            xwp: Dimension index of the x-axis in the wind cube.
            ywp: Dimension index of the y-axis in the wind cube.

        Raises:
            ValueError: If ancillary grids do not share the same x/y dimension
                ordering as the wind cube.
        """
        # Dim-order of ancillary post-processing-grid orography
        xap, yap, _, _ = self.find_coord_order(self.target_orog)

        # Compare relative ordering of (x,y) dimensions
        if xwp - ywp != xap - yap:
            if np.isnan(xap) or np.isnan(yap):
                raise ValueError("Ancillary grid differs from wind grid.")
            raise ValueError(
                "XY dimension ordering differs between wind and ancillary grids."
            )

    def process(self, input_cube: Cube) -> Cube:
        """Apply roughness (RC) and height (HC) corrections to a 4D wind cube.

        Args:
            input_cube: Wind-speed cube (x, y, z, time), defined on height levels for all
                desired forecast times.

        Returns:
            The wind cube with RC and HC applied.

        Raises:
            TypeError: If input_cube is not an iris Cube.
            ValueError: If any time slice contains invalid wind data.
        """
        if not isinstance(input_cube, iris.cube.Cube):
            raise TypeError(f"Wind input is not a cube, but {type(input_cube)}")

        # Determine coordinate names and dimension ordering for the wind cube
        self.x_name, self.y_name, self.z_name, self.t_name = self.find_coord_names(
            input_cube
        )
        xwp, ywp, zwp, twp = self.find_coord_order(input_cube)

        # Reorder wind cube so dimensions are consistently (y, x, z [, t])
        if np.isnan(twp):
            input_cube.transpose([ywp, xwp, zwp])
        else:
            input_cube.transpose([ywp, xwp, zwp, twp])

        z0_data = None if self.model_z0 is None else self.model_z0.data
        rc_utils = WindTerrainAdjustmentUtilities(
            self.model_silhouette_roughness.data,
            self.model_orog_stddev.data,
            z0_data,
            self.target_orog.data,
            self.model_orog.data,
            self.output_res,
            self.model_res,
        )
        self.check_wind_ancil(xwp, ywp)
        height_grid = self.find_heightgrid(input_cube)

        corrected_list = iris.cube.CubeList()
        for time_slice in input_cube.slices_over("time"):
            # Validate wind field (e.g. not contain NaNs or negative values)
            if np.isnan(time_slice.data).any() or (time_slice.data < 0.0).any():
                tcoord = time_slice.coord(self.t_name)
                raise ValueError(f"{tcoord} has invalid wind data")
            # Compute windspeed correction/s
            corrected_cube = time_slice.copy()
            if self.mode == "rc":
                corrected_cube.data = rc_utils.do_rc(height_grid, time_slice.data)
            elif self.mode == "hc":
                corrected_cube.data = rc_utils.do_hc(height_grid, time_slice.data)
            elif self.mode == "hc_and_rc":
                corrected_cube.data = rc_utils.do_rc_and_hc(
                    height_grid, time_slice.data
                )
            corrected_list.append(corrected_cube)
        output_cube = corrected_list.merge_cube()

        # Restore the original dimension ordering of both input and output
        if np.isnan(twp):
            order = np.argsort([ywp, xwp, zwp])
            input_cube.transpose(order)
            output_cube.transpose(order)
        else:
            input_cube.transpose(np.argsort([ywp, xwp, zwp, twp]))
            output_cube.transpose(np.argsort([twp, ywp, xwp, zwp]))

        return output_cube
