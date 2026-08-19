# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing a plugin for orographic wind-speed correction."""

import numpy as np
from iris.cube import Cube, CubeList
from scipy.interpolate import PchipInterpolator
from scipy.special import kv

from improver import BasePlugin

VON_KARMAN_CONSTANT = 0.4


class WindDownscaling(BasePlugin):
    """Plugin-style interface for orographic wind-speed correction."""

    def __init__(
        self,
        high_res_orog_cube: Cube,
        model_orog_cube: Cube,
        model_orog_stddev_cube: Cube,
        model_silhouette_roughness_cube: Cube,
        landmask_cube: Cube,
    ) -> None:
        """Initialise plugin with ancillary cubes required for downscaling.

        Args:
            high_res_orog_cube:
                High-resolution orography cube.

            model_orog_cube:
                Model-resolution orography cube.

            model_orog_stddev_cube:
                Sub-grid orography standard deviation cube.

            model_silhouette_roughness_cube:
                Sub-grid silhouette roughness cube.

            landmask_cube:
                Land-sea mask cube.
        """
        self.high_res_orog_cube = high_res_orog_cube
        self.model_orog_cube = model_orog_cube
        self.model_orog_stddev_cube = model_orog_stddev_cube
        self.model_silhouette_roughness_cube = model_silhouette_roughness_cube
        self.landmask_cube = landmask_cube

    def process(
        self,
        wind_profile_cube: Cube,
        target_wind_speed_cube: Cube,
        target_height_levels: list[float] | None = None,
    ) -> Cube:
        """Apply wind downscaling using stored ancillary cubes.

        Args:
            wind_profile_cube:
                Wind-speed cube used to fit vertical profiles for roughness
                and reference-wind estimation. Must contain wind speed on
                heights between ground and 300 m above ground level.

            target_wind_speed_cube:
                Wind-speed cube to be corrected.

            target_height_levels:
                Optional target heights in metres. If omitted, heights are
                taken from target_wind_speed_cube.

        Returns:
            Cube:
                Corrected wind-speed cube.

        Raises:
            ValueError:
                If supplied cubes do not share a common horizontal grid.
        """
        target_heights = get_target_height_levels(
            target_wind_speed_cube,
            target_height_levels,
        )
        wind_profile_cube = crop_wind_profile_cube(
            wind_profile_cube,
            target_heights,
        )
        cubes_to_check = get_cubes_to_check(
            wind_profile_cube,
            target_wind_speed_cube,
            self.high_res_orog_cube,
            self.model_orog_cube,
            self.model_orog_stddev_cube,
            self.model_silhouette_roughness_cube,
            self.landmask_cube,
        )
        check_same_grid(*cubes_to_check)
        for cube in (
            self.high_res_orog_cube,
            self.model_orog_cube,
            self.model_orog_stddev_cube,
        ):
            cube.convert_units("m")

        wavenumber_cube = calculate_characteristic_wavenumber(
            self.model_orog_stddev_cube,
            self.model_silhouette_roughness_cube,
            self.landmask_cube,
        )
        reference_height_cube = calculate_reference_height(wavenumber_cube)
        unresolved_orography_height_cube = calculate_unresolved_orography_height(
            self.high_res_orog_cube,
            self.model_orog_cube,
        )

        spline = fit_spline_wind_profile(wind_profile_cube)

        target_wind_speeds = get_target_wind_speeds(
            target_wind_speed_cube,
            target_heights,
        )

        z0 = fit_log_wind_profile(wind_profile_cube)
        reference_wind_speed = evaluate_spline_at_reference_heights(
            spline,
            reference_height_cube,
        )

        speed_up_factor = calculate_speed_up_factor(
            wavenumber_cube.data,
            unresolved_orography_height_cube.data,
            target_heights,
            target_wind_speeds,
            reference_wind_speed,
            z0,
            self.landmask_cube.data,
        )

        corrected_wind_speeds = target_wind_speeds * speed_up_factor

        return create_corrected_wind_speed_cube(
            target_wind_speed_cube,
            corrected_wind_speeds,
            target_heights,
        )


def get_cubes_to_check(
    wind_profile_cube: Cube,
    target_wind_speed_cube: Cube,
    high_res_orog_cube: Cube,
    model_orog_cube: Cube,
    model_orog_stddev_cube: Cube,
    model_silhouette_roughness_cube: Cube,
    landmask_cube: Cube,
) -> list[Cube]:
    """
    Build the list of cubes that must share a common horizontal grid.

    Args:
        wind_profile_cube:
            Wind-speed cube used for profile fitting.

        target_wind_speed_cube:
            Wind-speed cube that will be corrected.

        high_res_orog_cube:
            High-resolution orography cube.

        model_orog_cube:
            Model-resolution orography cube.

        model_orog_stddev_cube:
            Sub-grid orography standard deviation cube.

        model_silhouette_roughness_cube:
            Sub-grid silhouette roughness cube.

        landmask_cube:
            Land-sea mask cube.

    Returns:
        list[Cube]:
            Cubes that should be checked with check_same_grid.
    """
    return [
        wind_profile_cube,
        target_wind_speed_cube,
        high_res_orog_cube,
        model_orog_cube,
        landmask_cube,
        model_orog_stddev_cube,
        model_silhouette_roughness_cube,
    ]


def get_target_height_levels(
    target_wind_speed_cube: Cube,
    target_height_levels: list[float] | None,
) -> np.ndarray:
    """
    Determine the heights to apply wind-speed corrections to.

    Explicitly requested target heights are used when supplied. Otherwise,
    the height levels from target_wind_speed_cube are used.

    Args:
        target_wind_speed_cube:
            Cube containing winds to be corrected.

        target_height_levels:
            Optional list of heights above ground level, in metres.
    Returns:
        np.ndarray:
            Target heights above ground level, in metres.
    """
    if target_height_levels is None:
        target_height_levels = get_height_levels_from_cube(
            target_wind_speed_cube,
        )

    return np.sort(np.asarray(target_height_levels, dtype=float))


def crop_wind_profile_cube(
    wind_profile_cube: Cube,
    target_heights: np.ndarray,
    minimum_upper_height: float = 300.0,
) -> Cube:
    """
    Crop wind-profile levels to those needed for downscaling calculations.

    The upper crop bound is the greater of 300 m and the maximum requested
    target height. This keeps profile data needed for spline evaluation at
    target heights while limiting memory use from unnecessary high levels.

    Args:
        wind_profile_cube:
            Wind-speed cube on height levels used for profile fitting.

        target_heights:
            Requested target heights in metres.

        minimum_upper_height:
            Minimum upper crop bound in metres. Defaults to 300 m.

    Returns:
        Cube:
            Wind-profile cube cropped in height.

    Raises:
        ValueError:
            If no wind-profile height levels are at or below the crop bound.
    """
    if wind_profile_cube.ndim < 3:
        return wind_profile_cube

    heights = np.asarray(get_height_levels_from_cube(wind_profile_cube), dtype=float)
    finite_target_heights = np.asarray(target_heights, dtype=float)
    finite_target_heights = finite_target_heights[np.isfinite(finite_target_heights)]
    max_target_height = (
        float(np.max(finite_target_heights))
        if finite_target_heights.size
        else float(minimum_upper_height)
    )
    upper_height = max(max_target_height, float(minimum_upper_height))

    keep_indices = np.where(heights <= upper_height)[0]
    if keep_indices.size == 0:
        raise ValueError(
            "wind_profile_cube has no height levels at or below "
            f"the crop limit ({upper_height} m)."
        )

    if keep_indices.size == heights.size:
        return wind_profile_cube

    return wind_profile_cube[keep_indices]


def prepare_target_wind_speeds(
    target_wind_speeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert target-wind input to mask and float array forms.

    Args:
        target_wind_speeds:
            Target wind speeds as ndarray or masked array.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            A tuple of (mask, values), where mask is a boolean array marking
            invalid points and values is a float ndarray with masked entries
            filled by NaN.
    """
    winds = np.ma.asarray(target_wind_speeds, dtype=float)
    return np.ma.getmaskarray(winds), np.ma.filled(winds, fill_value=np.nan)


def get_target_wind_speeds(
    target_wind_speed_cube: Cube,
    target_heights: np.ndarray,
) -> np.ma.MaskedArray:
    """
    Get target wind speeds from target cube at requested heights.

    Args:
        target_wind_speed_cube:
            Wind-speed cube to be corrected.

        target_heights:
            Target heights in metres.

    Returns:
        np.ma.MaskedArray:
            Target wind speeds with shape (n_heights, y, x).

    Raises:
        ValueError:
            If target_height_levels differ from a single-level target cube.
    """
    cube_heights = np.asarray(
        get_height_levels_from_cube(target_wind_speed_cube),
        dtype=float,
    )
    cube_winds = np.ma.asarray(target_wind_speed_cube.data, dtype=float)

    if target_wind_speed_cube.ndim == 2:
        cube_winds = cube_winds[np.newaxis, ...]

    if cube_winds.shape[0] != cube_heights.size:
        raise ValueError(
            "target_wind_speed_cube data first dimension must match "
            "its height coordinate size."
        )

    if np.array_equal(cube_heights, target_heights):
        return cube_winds.copy()

    if cube_heights.size == 1:
        raise ValueError(
            "Cannot interpolate target_wind_speed_cube to target_height_levels "
            "when only one source height is available."
        )

    target_spline = fit_spline_wind_profile(target_wind_speed_cube)
    return np.ma.asarray(target_spline(target_heights), dtype=float)


def create_corrected_wind_speed_cube(
    template_cube: Cube,
    corrected_wind_speeds: np.ndarray,
    target_heights: list[float] | np.ndarray,
) -> Cube:
    """
    Create a corrected wind-speed cube on the requested height levels.

    Args:
        template_cube:
            Cube used as the metadata template for the output.

        corrected_wind_speeds:
            Corrected wind speeds with shape
            (n_target_heights, y, x).

        target_heights:
            Target heights above ground level, in metres.

    Returns:
        Cube containing corrected wind speeds on the target heights.

    Raises:
        ValueError:
            If corrected_wind_speeds first dimension does not match the
            number of target heights.
    """
    target_heights = np.asarray(
        target_heights,
        dtype=np.float32,
    )

    corrected_wind_speeds = np.ma.asarray(
        corrected_wind_speeds,
        dtype=np.float32,
    )

    if corrected_wind_speeds.shape[0] != target_heights.size:
        raise ValueError(
            "The first dimension of corrected_wind_speeds must "
            "match the number of target heights."
        )

    if template_cube.ndim == 2:
        template_2d = template_cube
    else:
        template_2d = template_cube[0]

    output_cubes = CubeList()

    for height_index, target_height in enumerate(target_heights):
        output_cube = template_2d.copy(data=corrected_wind_speeds[height_index])

        height_coord = output_cube.coord("height")

        height_coord.convert_units("m")
        height_coord.points = np.array(
            [target_height],
            dtype=np.float32,
        )
        height_coord.bounds = None

        output_cubes.append(output_cube)

    return output_cubes.merge_cube()


def calculate_speed_up_factor(
    characteristic_wavenumber: np.ndarray,
    unresolved_orography_height: np.ndarray,
    target_heights: np.ndarray,
    target_wind_speeds: np.ndarray,
    reference_wind_speed: np.ndarray,
    roughness_length: np.ndarray,
    land_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculate a multiplicative wind-speed correction for unresolved terrain.

    The correction estimates the perturbation to the background wind caused
    by terrain that is not resolved by the model orography. Its magnitude
    depends on the unresolved terrain height and horizontal length scale,
    while its influence decreases with height above the surface.

    Args:
        characteristic_wavenumber:
            Characteristic wavenumber of the unresolved terrain, in m-1.
            Larger values represent shorter-scale terrain. Shape: (y, x).

        unresolved_orography_height:
            High-resolution orography minus model orography, in metres.
            Shape: (y, x).

        target_heights:
            Heights above ground at which corrected winds are required,
            in metres. Shape: (n_heights,).

        target_wind_speeds:
            Background wind speeds at the target heights, in m s-1.
            Shape: (n_heights, y, x).

        reference_wind_speed:
            Background wind speed at the reference height 1/k, in m s-1.
            This provides the velocity scale for the terrain perturbation.
            Shape: (y, x).

        roughness_length:
            Aerodynamic roughness length, in metres, used to describe the
            near-surface response of the flow. Shape: (y, x).

        land_mask:
            Optional land-sea mask with land > 0 and sea <= 0. If provided,
            speed_up_factor is forced to 1.0 at sea.

    Returns:
        Multiplicative wind-speed correction with shape
        (n_heights, y, x). A value of 1 leaves the background wind unchanged.
    """
    von_karman_constant = VON_KARMAN_CONSTANT

    # Convert masked inputs to arrays with NaN at invalid points.
    characteristic_wavenumber = np.ma.filled(
        characteristic_wavenumber,
        np.nan,
    )
    unresolved_orography_height = np.ma.filled(
        unresolved_orography_height,
        np.nan,
    )
    reference_wind_speed = np.ma.filled(
        reference_wind_speed,
        np.nan,
    )
    roughness_length = np.ma.filled(roughness_length, np.nan)

    # Broadcast the two-dimensional fields over target height.
    characteristic_wavenumber = characteristic_wavenumber[np.newaxis, ...]
    unresolved_orography_height = unresolved_orography_height[np.newaxis, ...]
    reference_wind_speed = reference_wind_speed[np.newaxis, ...]
    roughness_length = roughness_length[np.newaxis, ...]
    target_heights = target_heights[:, np.newaxis, np.newaxis]
    target_wind_mask, target_wind_speeds = prepare_target_wind_speeds(
        target_wind_speeds
    )

    # Calculate the inner-layer response. This describes how surface
    # friction modifies the response of the flow to unresolved terrain.
    roughness_scaled_wavenumber = characteristic_wavenumber * np.log(
        target_heights / roughness_length
    )
    bessel_argument_at_height = (
        (1.0 + 1.0j)
        * np.sqrt(roughness_scaled_wavenumber * target_heights)
        / von_karman_constant
    )
    bessel_argument_at_roughness_height = (
        (1.0 + 1.0j)
        * np.sqrt(roughness_scaled_wavenumber * roughness_length)
        / von_karman_constant
    )
    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        inner_layer_response = np.real(
            1.0
            - kv(0, bessel_argument_at_height)
            / kv(0, bessel_argument_at_roughness_height)
        )

    # The effect of a terrain feature decreases with height: shorter-scale
    # terrain, represented by larger wavenumbers, decays more rapidly.
    vertical_decay = np.exp(-characteristic_wavenumber * target_heights)

    # Describe the unresolved terrain height relative to its characteristic
    # horizontal scale
    terrain_amplitude = characteristic_wavenumber * unresolved_orography_height

    # Calculate the terrain-induced change in wind speed.
    # The reference wind sets the velocity scale, while the remaining terms
    # describe the strength and vertical structure of the terrain response.
    wind_speed_perturbation = (
        reference_wind_speed * inner_layer_response * vertical_decay * terrain_amplitude
    )
    fractional_perturbation = np.divide(
        wind_speed_perturbation,
        target_wind_speeds,
        out=np.zeros_like(
            target_wind_speeds,
            dtype=float,
        ),
        where=(
            np.isfinite(wind_speed_perturbation)
            & np.isfinite(target_wind_speeds)
            & (target_wind_speeds > 0.0)
        ),
    )

    # Limit the terrain-induced change so that its magnitude cannot exceed
    # the background wind speed
    fractional_perturbation = np.clip(
        fractional_perturbation,
        -1.0,
        1.0,
    )

    # Speed-up factor = multiplicative correction to the background wind.
    speed_up_factor = 1.0 + fractional_perturbation

    if land_mask is not None:
        land = np.ma.filled(land_mask, 0) > 0
        speed_up_factor = np.where(
            land[np.newaxis, ...],
            speed_up_factor,
            1.0,
        )

    invalid = (
        target_wind_mask
        | ~np.isfinite(characteristic_wavenumber)
        | ~np.isfinite(unresolved_orography_height)
        | ~np.isfinite(reference_wind_speed)
        | ~np.isfinite(roughness_length)
        | ~np.isfinite(speed_up_factor)
    )

    return np.where(invalid, 1.0, speed_up_factor)


def evaluate_spline_at_reference_heights(
    spline: PchipInterpolator,
    reference_height_cube: Cube,
) -> np.ma.MaskedArray:
    """
    Evaluate wind-speed splines at spatially varying reference heights.

    Args:
        spline:
            PCHIP interpolator fitted to wind profiles with shape
            (height, y, x).

        reference_height_cube:
            Reference height at each horizontal grid point, in metres.

    Returns:
        Wind speed at the reference height for each grid point,
        with shape (y, x).
    """
    reference_heights = np.ma.asarray(reference_height_cube.data, dtype=float)

    input_heights = spline.x

    valid = (
        ~np.ma.getmaskarray(reference_heights)
        & (reference_heights.data >= input_heights.min())
        & (reference_heights.data <= input_heights.max())
    )

    # Find which spline interval contains each reference height.
    interval = (
        np.searchsorted(
            input_heights,
            reference_heights.data,
            side="right",
        )
        - 1
    )

    # A point exactly at the highest input level belongs to
    # the final spline interval.
    interval = np.clip(
        interval,
        0,
        len(input_heights) - 2,
    )

    y_indices, x_indices = np.indices(reference_heights.shape)

    # PCHIP stores polynomial coefficients for each interval as:
    # c[0] * dx**3
    # + c[1] * dx**2
    # + c[2] * dx
    # + c[3]
    coefficients = spline.c[
        :,
        interval,
        y_indices,
        x_indices,
    ]

    dx = reference_heights.data - input_heights[interval]

    reference_wind_speed = (
        coefficients[0] * dx**3
        + coefficients[1] * dx**2
        + coefficients[2] * dx
        + coefficients[3]
    )

    return np.ma.array(reference_wind_speed, mask=~valid)


def _approximate_roughness_length(
    fit_heights: np.ndarray,
    fit_winds: np.ndarray,
    valid: np.ndarray,
    valid_count: np.ndarray,
    min_roughness_length: float,
    max_roughness_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate z0 via linear regression in log-height space and return a
    bracketing search interval for the exact refinement step.

    Args:
        fit_heights:
            Heights used in the fit, shape (n_fit_heights,).

        fit_winds:
            Wind speeds at fit heights, shape (n_fit_heights, y, x).

        valid:
            Boolean mask marking valid fitting points, same shape as fit_winds.

        valid_count:
            Number of valid height levels per grid point, shape (y, x).

        min_roughness_length:
            Lower bound for roughness length z0 in metres.

        max_roughness_length:
            Upper bound for roughness length z0 in metres.

    Returns:
        lower_z0, upper_z0: search bounds in linear z0 space.
    """
    log_heights = np.log(fit_heights)[:, np.newaxis, np.newaxis]
    count = np.maximum(valid_count, 1)

    mean_log_height = np.sum(np.where(valid, log_heights, 0.0), axis=0) / count
    mean_wind = np.sum(np.where(valid, fit_winds, 0.0), axis=0) / count

    log_height_anomaly = log_heights - mean_log_height[np.newaxis, ...]
    wind_anomaly = fit_winds - mean_wind[np.newaxis, ...]

    numerator = np.sum(np.where(valid, log_height_anomaly * wind_anomaly, 0.0), axis=0)
    denominator = np.sum(np.where(valid, log_height_anomaly**2, 0.0), axis=0)

    A = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0.0,
    )
    B = mean_wind - A * mean_log_height

    approximate_z0 = np.exp(np.clip(-B / A, -50.0, 50.0))

    valid_initial_z0 = np.isfinite(approximate_z0) & (approximate_z0 > 0.0)

    lower_z0 = np.clip(
        np.where(valid_initial_z0, approximate_z0 / 2.0, min_roughness_length),
        min_roughness_length,
        max_roughness_length,
    )
    upper_z0 = np.clip(
        np.where(valid_initial_z0, approximate_z0 * 2.0, max_roughness_length),
        min_roughness_length,
        max_roughness_length,
    )

    bad_interval = lower_z0 >= upper_z0
    lower_z0 = np.where(bad_interval, min_roughness_length, lower_z0)
    upper_z0 = np.where(bad_interval, max_roughness_length, upper_z0)

    return lower_z0, upper_z0


def _evaluate_log_profile_fit(
    fit_heights: np.ndarray,
    fit_winds: np.ndarray,
    valid: np.ndarray,
    valid_count: np.ndarray,
    roughness_length: np.ndarray,
    von_karman_constant: float,
    min_friction_velocity: float,
    max_friction_velocity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the best-fit friction velocity and squared error for a given z0 field.

    Args:
        fit_heights:
            Heights used in the fit, shape (n_fit_heights,).

        fit_winds:
            Wind speeds at fit heights, shape (n_fit_heights, y, x).

        valid:
            Boolean mask marking valid fitting points.

        valid_count:
            Number of valid height levels per grid point.

        roughness_length:
            Candidate roughness length z0 field, shape (y, x).

        von_karman_constant:
            Von Karman constant used by the log profile.

        min_friction_velocity:
            Lower clipping bound for fitted friction velocity.

        max_friction_velocity:
            Upper clipping bound for fitted friction velocity.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            Fitted friction velocity and squared error, both shape (y, x).
    """
    heights_3d = fit_heights[:, np.newaxis, np.newaxis]
    z0 = roughness_length[np.newaxis, ...]
    log_term = np.log((heights_3d + z0) / z0)

    numerator = np.sum(np.where(valid, log_term * fit_winds, 0.0), axis=0)
    denominator = np.sum(np.where(valid, log_term**2, 0.0), axis=0)

    profile_scale = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0.0,
    )
    friction_velocity = np.clip(
        von_karman_constant * profile_scale,
        min_friction_velocity,
        max_friction_velocity,
    )

    fitted_winds = friction_velocity[np.newaxis, ...] / von_karman_constant * log_term
    squared_error = np.sum(
        np.where(valid, (fit_winds - fitted_winds) ** 2, 0.0), axis=0
    )
    squared_error = np.where(valid_count >= 2, squared_error, np.inf)

    return friction_velocity, squared_error


def _refine_roughness_length(
    fit_heights: np.ndarray,
    fit_winds: np.ndarray,
    valid: np.ndarray,
    valid_count: np.ndarray,
    lower_z0: np.ndarray,
    upper_z0: np.ndarray,
    refinement_iterations: int,
    von_karman_constant: float,
    min_friction_velocity: float,
    max_friction_velocity: float,
) -> np.ndarray:
    """
    Refine z0 estimates via a vectorised golden-section search in log z0 space.

    Args:
        fit_heights:
            Heights used in the fit, shape (n_fit_heights,).

        fit_winds:
            Wind speeds at fit heights, shape (n_fit_heights, y, x).

        valid:
            Boolean mask marking valid fitting points.

        valid_count:
            Number of valid height levels per grid point.

        lower_z0:
            Lower z0 search bound per grid point.

        upper_z0:
            Upper z0 search bound per grid point.

        refinement_iterations:
            Number of golden-section iterations.

        von_karman_constant:
            Von Karman constant used by the log profile.

        min_friction_velocity:
            Lower clipping bound for fitted friction velocity.

        max_friction_velocity:
            Upper clipping bound for fitted friction velocity.

    Returns:
        np.ndarray:
            Refined roughness length z0 field with shape (y, x).
    """
    golden_ratio = (np.sqrt(5.0) - 1.0) / 2.0
    lower = np.log(lower_z0)
    upper = np.log(upper_z0)

    def _error(log_z0: np.ndarray) -> np.ndarray:
        roughness_length = np.exp(log_z0)
        _, err = _evaluate_log_profile_fit(
            fit_heights,
            fit_winds,
            valid,
            valid_count,
            roughness_length,
            von_karman_constant,
            min_friction_velocity,
            max_friction_velocity,
        )
        return err

    for _ in range(refinement_iterations):
        left = upper - golden_ratio * (upper - lower)
        right = lower + golden_ratio * (upper - lower)
        choose_left = _error(left) <= _error(right)
        upper = np.where(choose_left, right, upper)
        lower = np.where(choose_left, lower, left)

    return np.exp(0.5 * (lower + upper))


def fit_log_wind_profile(
    wind_speed_cube: Cube,
    lower_height_limit: float = 0.0,
    upper_height_limit: float = 300.0,
    refinement_iterations: int = 20,
) -> np.ndarray:
    """
    Fit a neutral logarithmic wind profile at every grid point.

    A fast linear approximation is first used to estimate z0. This
    estimate is then refined using the exact logarithmic wind profile:

        U(z) = (u_star / kappa) * log((z + z0) / z0)

    The refinement is vectorised across all horizontal grid points.

    Args:
        wind_speed_cube:
            Wind-speed cube on height levels with shape (height, y, x).

        lower_height_limit:
            Exclusive lower bound on heights included in fitting, in metres.

        upper_height_limit:
            Inclusive upper bound on heights included in fitting, in metres.

        refinement_iterations:
            Number of refinement iterations for roughness-length search.

    Returns:
        roughness_length:
            Fitted z0 values with shape (y, x).
    """
    von_karman_constant = VON_KARMAN_CONSTANT
    min_friction_velocity = 0.001
    max_friction_velocity = 5.0
    min_roughness_length = 1e-5
    max_roughness_length = 5.0

    heights = np.asarray(get_height_levels_from_cube(wind_speed_cube), dtype=float)
    wind_speeds = np.ma.asarray(wind_speed_cube.data, dtype=float).filled(np.nan)

    valid_heights = (
        np.isfinite(heights)
        & (heights > lower_height_limit)
        & (heights <= upper_height_limit)
    )
    fit_heights = heights[valid_heights]
    fit_winds = wind_speeds[valid_heights]
    valid = np.isfinite(fit_winds) & (fit_winds > 0.0)
    valid_count = np.sum(valid, axis=0)

    lower_z0, upper_z0 = _approximate_roughness_length(
        fit_heights,
        fit_winds,
        valid,
        valid_count,
        min_roughness_length,
        max_roughness_length,
    )
    roughness_length = _refine_roughness_length(
        fit_heights,
        fit_winds,
        valid,
        valid_count,
        lower_z0,
        upper_z0,
        refinement_iterations,
        von_karman_constant,
        min_friction_velocity,
        max_friction_velocity,
    )
    friction_velocity, _ = _evaluate_log_profile_fit(
        fit_heights,
        fit_winds,
        valid,
        valid_count,
        roughness_length,
        von_karman_constant,
        min_friction_velocity,
        max_friction_velocity,
    )

    invalid = (
        (valid_count < 2)
        | ~np.isfinite(friction_velocity)
        | ~np.isfinite(roughness_length)
    )
    roughness_length[invalid] = np.nan

    return roughness_length


def fit_spline_wind_profile(
    wind_speed_cube: Cube,
) -> PchipInterpolator:
    """
    Fit shape-preserving cubic splines to wind-speed profiles.

    All horizontal grid points are fitted simultaneously.

    Args:
        wind_speed_cube:
            Iris cube with shape (height, y, x) containing wind speeds.

    Returns:
        PCHIP interpolator for all grid-point wind profiles.

    Raises:
        ValueError:
            If wind-speed vertical dimension does not match the number of
            heights, if any height is non-finite, or if heights are not
            strictly increasing.
    """
    heights = get_height_levels_from_cube(wind_speed_cube)
    heights = np.asarray(heights, dtype=float)
    wind_speeds = np.ma.filled(wind_speed_cube.data, fill_value=np.nan).astype(float)

    if wind_speeds.shape[0] != len(heights):
        raise ValueError(
            "The first dimension of wind_speeds must correspond "
            "to the supplied height levels."
        )

    if np.any(~np.isfinite(heights)):
        raise ValueError("Height levels must all be finite.")

    if np.any(np.diff(heights) <= 0):
        raise ValueError("Height levels must be strictly increasing.")

    return PchipInterpolator(
        heights,
        wind_speeds,
        axis=0,
        extrapolate=False,
    )


def calculate_unresolved_orography_height(
    high_res_orography_cube: Cube,
    model_orography_cube: Cube,
) -> Cube:
    """
    Calculate the unresolved orography height.

    The unresolved orography height is the difference between the
    high-resolution orography and the model orography. Positive values
    indicate terrain that is higher than represented by the model,
    while negative values indicate terrain that is lower.

    Args:
        high_res_orography_cube:
            High-resolution orography, in metres.

        model_orography_cube:
            Model orography, in metres.

    Returns:
        Cube containing the unresolved orography height, in metres.
    """
    high_res_orography_cube = high_res_orography_cube.copy()
    model_orography_cube = model_orography_cube.copy()

    high_res_orography_cube.convert_units("m")
    model_orography_cube.convert_units("m")

    unresolved_orography_height = (
        high_res_orography_cube.data - model_orography_cube.data
    )

    unresolved_orography_height_cube = high_res_orography_cube.copy(
        data=unresolved_orography_height
    )

    unresolved_orography_height_cube.rename("unresolved_orography_height")
    unresolved_orography_height_cube.units = "m"

    return unresolved_orography_height_cube


def calculate_reference_height(
    wavenumber_cube: Cube,
) -> Cube:
    """
    Calculate the wave-scale reference height for the unresolved orography.

    The reference height is the inverse of the characteristic horizontal
    wavenumber:

        z_s = 1 / k

    Args:
        wavenumber_cube:
            Characteristic unresolved-orography wavenumber, in m-1.

    Returns:
        Cube containing the reference height z_s in metres.
        The mask from the wavenumber cube is preserved.
    """
    wavenumber = np.ma.asarray(wavenumber_cube.data)
    reference_height = 1.0 / wavenumber

    reference_height_cube = wavenumber_cube.copy(data=reference_height)
    reference_height_cube.rename("unresolved_orography_reference_height")
    reference_height_cube.units = "m"

    return reference_height_cube


def calculate_characteristic_wavenumber(
    orog_stddev_cube: Cube,
    silhouette_roughness_cube: Cube,
    landmask_cube: Cube,
    min_valid_orog_stddev: float = 2.0,
    min_valid_silhouette_roughness: float = 0.0,
    min_length_scale: float = 500.0,
    max_length_scale: float = 4000.0,
    min_half_amplitude: float = 1.0,
) -> Cube:
    """
    Calculate the characteristic horizontal wavenumber of unresolved
    orography over land.

    The characteristic terrain scale is estimated from silhouette roughness
    divided by the half peak-to-trough terrain amplitude. The resulting
    inverse length scale is constrained to the range represented by the
    specified minimum and maximum terrain length scales, then converted to
    wavenumber by multiplying by pi.

    Args:
        orog_stddev_cube:
            Standard deviation of sub-grid orography, in metres.

        silhouette_roughness_cube:
            Silhouette roughness of the sub-grid terrain.

        landmask_cube:
            Land-sea mask.

        min_valid_orog_stddev:
            Minimum terrain standard deviation for applying the calculation.

        min_valid_silhouette_roughness:
            Minimum silhouette roughness for applying the calculation.

        min_length_scale:
            Smallest permitted terrain length scale, in metres.

        max_length_scale:
            Largest permitted terrain length scale, in metres.

        min_half_amplitude:
            Minimum half peak-to-trough terrain amplitude used in the
            wavenumber calculation, in metres.

    Returns:
        Cube containing the characteristic terrain wavenumber, in m-1.
    """
    orog_stddev_cube = orog_stddev_cube.copy()
    orog_stddev_cube.convert_units("m")

    sigma = np.ma.asarray(orog_stddev_cube.data)
    silhouette_roughness = np.ma.asarray(silhouette_roughness_cube.data)
    landmask = np.ma.asarray(landmask_cube.data)
    half_amplitude = np.sqrt(2.0) * sigma

    valid = (
        ~np.ma.getmaskarray(sigma)
        & ~np.ma.getmaskarray(silhouette_roughness)
        & ~np.ma.getmaskarray(landmask)
        & (landmask.data > 0)
        & (sigma.data >= min_valid_orog_stddev)
        & (silhouette_roughness.data >= min_valid_silhouette_roughness)
    )
    wavenumber = np.ma.masked_all(
        sigma.shape,
        dtype=np.float64,
    )

    inverse_length_scale = silhouette_roughness.data[valid] / np.maximum(
        half_amplitude.data[valid],
        min_half_amplitude,
    )
    inverse_length_scale = np.clip(
        inverse_length_scale,
        1.0 / max_length_scale,
        1.0 / min_length_scale,
    )
    wavenumber[valid] = np.pi * inverse_length_scale

    wavenumber_cube = orog_stddev_cube.copy(data=wavenumber)
    wavenumber_cube.rename("characteristic_unresolved_orography_wavenumber")
    wavenumber_cube.units = "m-1"

    return wavenumber_cube


def check_same_grid(*cubes: Cube) -> None:
    """
    Check that all cubes share the same horizontal shape.

    Args:
        *cubes:
            Cubes to compare. If fewer than two cubes are supplied, no check
            is performed.

    Raises:
        ValueError:
            If any cube has a different trailing two-dimensional shape than
            the first cube.
    """
    if len(cubes) < 2:
        return

    ref_shape = cubes[0].shape[-2:]
    for cube in cubes[1:]:
        if cube.shape[-2:] != ref_shape:
            raise ValueError(
                f"Cube '{cube.name()}' has horizontal shape {cube.shape[-2:]}, "
                f"expected {ref_shape}."
            )


def get_height_levels_from_cube(
    height_levels_cube: Cube,
) -> np.ndarray:
    """
    Return height-coordinate points in metres.

    Args:
        height_levels_cube:
            Cube containing a height coordinate.

    Returns:
        np.ndarray:
            One-dimensional array of height points in metres.
    """
    height_coord = height_levels_cube.coord("height").copy()
    height_coord.convert_units("m")
    return np.atleast_1d(height_coord.points)
