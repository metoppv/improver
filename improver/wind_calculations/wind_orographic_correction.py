# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugin for interpolating wind speeds and applying unresolved-orography correction."""

import numpy as np
from iris.coords import Coord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from scipy.interpolate import PchipInterpolator
from scipy.special import kv

from improver import BasePlugin
from improver.metadata.utilities import check_grid_match

VON_KARMAN_CONSTANT = 0.4


class OrographicWindCorrection(BasePlugin):
    """Interpolate wind speeds to target height levels (if provided) and adjust
    using a terrain-induced speed-up factor derived from unresolved-orography
    ancillaries.

    At a high level, the process is (for each grid point):
        - describe the unresolved terrain
        - describe the wind profile
        - calculate the expected terrain effect
        - apply the correction
    """

    def __init__(
        self,
        unsmoothed_orog_cube: Cube,
        model_orog_cube: Cube,
        model_orog_stddev_cube: Cube,
        model_silhouette_roughness_cube: Cube,
    ) -> None:
        """Initialise the plugin with the orography ancillaries used for correction.

        Args:
            unsmoothed_orog_cube:
                Cube containing the unsmoothed (i.e. true) orography, in metres.
                This cube should be on the same grid as ``model_orog_cube``.

            model_orog_cube:
                Cube containing the smoothed (i.e. NWP model representation of)
                orography, in metres.

            model_orog_stddev_cube:
                Cube containing the sub-grid orography standard deviation, in
                metres. This cube should be on the same grid as
                ``model_orog_cube``.

            model_silhouette_roughness_cube:
                Cube containing the dimensionless sub-grid silhouette roughness.
                This cube should be on the same grid as ``model_orog_cube``.

        Note:
            All orography inputs must be provided on grids consistent with the
            forecast data. If the forecast and model orography data have been
            re-gridded from their native NWP model grid (e.g. using StaGE), the
            ancillary inputs should also be re-gridded to the same grid.
        """
        self.unsmoothed_orog_cube = unsmoothed_orog_cube
        self.model_orog_cube = model_orog_cube
        self.model_orog_stddev_cube = model_orog_stddev_cube
        self.model_silhouette_roughness_cube = model_silhouette_roughness_cube

    def process(
        self,
        uncorrected_wind_cube: Cube,
        target_height_levels: list[float] | None = None,
    ) -> Cube:
        """Interpolate wind speeds to target height levels (if provided) and adjust
        using a terrain-induced speed-up factor.

        Describe the unresolved terrain and model wind profile, calculate the
        expected terrain effect, and apply it as a wind-speed correction.

        For ensemble input, each realization is corrected independently.

        Args:
            uncorrected_wind_cube:
                Cube containing the uncorrected wind-speed forecast on height
                levels. The levels must span 0 m to at least 300 m above ground
                level, since the lower profile is used to estimate effective
                surface roughness. Its height coordinate must be in metres.
                It may contain multiple realizations along the "realization"
                coordinate.

            target_height_levels:
                Optional target heights in metres. If omitted, the heights
                already present on ``uncorrected_wind_cube`` are used.

        Returns:
            Cube:
                Wind-speed cube after unresolved-orography correction on the
                selected target heights.
        """
        self.original_height_levels = self._get_height_levels_from_cube(
            uncorrected_wind_cube
        )
        self.target_height_levels = self._get_target_height_levels(
            target_height_levels,
        )

        for cube_name, cube in (
            ("unsmoothed_orog_cube", self.unsmoothed_orog_cube),
            ("model_orog_cube", self.model_orog_cube),
            ("model_orog_stddev_cube", self.model_orog_stddev_cube),
        ):
            self._require_metre_units(cube, cube_name)

        # If realizations are present, correct each independently.
        try:
            uncorrected_wind_iterator = uncorrected_wind_cube.slices_over("realization")
        except CoordinateNotFoundError:
            uncorrected_wind_iterator = [uncorrected_wind_cube]
        corrected_wind_slices = CubeList()
        for uncorrected_wind_slice in uncorrected_wind_iterator:
            corrected_wind_slices.append(
                self._process_single_realization(
                    uncorrected_wind_slice,
                )
            )

        return corrected_wind_slices.merge_cube()

    def _process_single_realization(
        self,
        uncorrected_wind_cube: Cube,
    ) -> Cube:
        """Interpolate wind speeds to target height levels (if provided) and adjust
        using a terrain-induced speed-up factor, for a single realization.

        Args:
            uncorrected_wind_cube:
                Cube containing the uncorrected wind-speed forecast on height
                levels, for a single realization. The levels must span 0 m to at
                least 300 m above ground level, since the lower profile is used
                to estimate effective surface roughness. Its height coordinate
                must be in metres.

        Returns:
            Cube:
                Wind-speed cube after unresolved-orography correction on the
                selected target heights.

        Raises:
            ValueError:
                If fewer than two wind heights are provided, or if the wind
                forecast does not include heights up to at least 300 m above
                ground level.
        """
        if self.original_height_levels.size < 2:
            raise ValueError(
                "uncorrected_wind_cube must include at least two height levels."
            )

        # The lower wind profile must extend to at least 300 m so the roughness
        # estimate can be constructed from the 0-300 m log fit.
        if np.max(self.original_height_levels) < 300.0:
            raise ValueError(
                "uncorrected_wind_cube must include heights up to at least 300 m "
                "above ground level, but max height provided is "
                f"{np.max(self.original_height_levels)} m."
            )

        # Validate the grids.
        cubes_to_check = [
            uncorrected_wind_cube,
            self.unsmoothed_orog_cube,
            self.model_orog_cube,
            self.model_orog_stddev_cube,
            self.model_silhouette_roughness_cube,
        ]
        check_grid_match(cubes_to_check)

        # Describe the unresolved terrain.
        wavenumber_cube = self._calculate_characteristic_wavenumber(
            self.model_orog_stddev_cube,
            self.model_silhouette_roughness_cube,
        )
        unresolved_orography_height_cube = self._calculate_unresolved_orography_height(
            self.unsmoothed_orog_cube,
            self.model_orog_cube,
        )

        # Describe the model wind profile.
        spline = self._fit_spline_wind_profile(uncorrected_wind_cube)
        target_wind_speeds = self._get_target_wind_speeds(
            uncorrected_wind_cube,
            spline=spline,
        )

        # Calculate a reference wind speed.
        reference_height_cube = self._calculate_reference_height(wavenumber_cube)
        reference_wind_speed = self._evaluate_spline_at_reference_heights(
            spline,
            reference_height_cube,
        )

        # Estimate an effective roughness length.
        z0 = self._fit_log_wind_profile(uncorrected_wind_cube)

        # Calculate the terrain effect.
        speed_up_factor = self._calculate_speed_up_factor(
            wavenumber_cube.data,
            unresolved_orography_height_cube.data,
            self.target_height_levels,
            target_wind_speeds,
            reference_wind_speed,
            z0,
        )
        # Apply the correction.
        # Corrected wind = background wind * speed-up factor.
        corrected_wind_speeds = target_wind_speeds * speed_up_factor

        return self._create_corrected_wind_speed_cube(
            uncorrected_wind_cube,
            corrected_wind_speeds,
        )

    def _get_target_height_levels(
        self,
        target_height_levels: list[float] | np.ndarray | None,
    ) -> np.ndarray:
        """Return the height levels at which to provide corrected wind speeds.

        If target heights are explicitly provided, they are used. Otherwise,
        the height levels in the original wind speed forecast are used (and so
        no interpolation is performed).

        The returned target heights are sorted in ascending order.

        Args:
            target_height_levels:
                Optional target heights in metres. If omitted, the heights
                stored in ``self.original_height_levels`` are used, e.g. the
                height levels in the original wind speed forecast.

        Returns:
            np.ndarray:
                Target heights above ground level, in metres, sorted in ascending
                order.
        """
        if target_height_levels is None:
            target_height_levels = self.original_height_levels

        return np.sort(np.asarray(target_height_levels, dtype=float))

    @staticmethod
    def _prepare_target_wind_speeds(
        target_wind_speeds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert target wind speeds into arrays suitable for calculation.

        Extracts any mask from the input wind speeds and returns the wind values
        as a floating-point array, with masked points represented by NaN.

        Args:
            target_wind_speeds:
                Uncorrected wind speeds at the target heights, supplied as an
                array or masked array.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                Boolean mask marking masked points, followed by wind speeds as
                a floating-point array with masked entries filled by NaN.
        """
        winds = np.ma.asarray(target_wind_speeds, dtype=float)
        return np.ma.getmaskarray(winds), np.ma.filled(winds, fill_value=np.nan)

    @staticmethod
    def _extract_wind_data(wind_cube: Cube) -> np.ma.MaskedArray:
        """Return wind-speed data, as a masked array.

        Moves the height dimension to the leading axis to provide a consistent
        (height, y, x) ordering, needed for subsequent processing.

        Args:
            wind_cube:
                Wind-speed cube whose height coordinate may be scalar or
                associated with any data dimension.

        Returns:
            np.ma.MaskedArray:
                Wind-speed data with height on the leading axis.
        """
        wind_data = np.ma.asarray(wind_cube.data, dtype=float)
        height_dims = wind_cube.coord_dims("height")

        if not height_dims:
            return wind_data[np.newaxis, ...]

        return np.moveaxis(wind_data, height_dims[0], 0)

    def _get_target_wind_speeds(
        self,
        uncorrected_wind_cube: Cube,
        spline: PchipInterpolator | None = None,
    ) -> np.ma.MaskedArray:
        """Return uncorrected wind speeds at the requested height levels.

        If ``self.target_height_levels`` matches the source height coordinate, no
        interpolation is performed and a copy of the forecast data is returned.

        Otherwise, wind speeds are interpolated to
        ``self.target_height_levels``. If a PCHIP interpolator is provided via
        ``spline``, it is used directly; otherwise a new PCHIP interpolator is
        fitted from ``uncorrected_wind_cube``.

        Args:
            uncorrected_wind_cube:
                Cube containing uncorrected wind speeds on height levels. The
                height coordinate must be in metres.

            spline:
                Optional, pre-computed PCHIP interpolator to use for the
                interpolation.

        Returns:
            np.ma.MaskedArray:
                Uncorrected wind speeds at the requested heights, with shape
                (n_heights, y, x).

        Raises:
            ValueError:
                If wind speeds cannot be evaluated at the requested heights.
        """
        cube_winds = self._extract_wind_data(uncorrected_wind_cube)

        # If the target heights match the source heights, no interpolation is needed,
        # so return a copy of the wind data as-is.
        if (
            self.original_height_levels.shape == self.target_height_levels.shape
            and np.allclose(
                self.original_height_levels,
                self.target_height_levels,
                rtol=0.0,
                atol=1e-6,
            )
        ):
            return cube_winds.copy()

        if self.original_height_levels.size == 1:
            raise ValueError(
                "Cannot interpolate uncorrected_wind_cube to target_height_levels "
                "when only one source height is available."
            )

        # If no spline is provided, fit one.
        if spline is None:
            spline = self._fit_spline_wind_profile(uncorrected_wind_cube)

        return np.ma.asarray(spline(self.target_height_levels), dtype=float)

    def _create_corrected_wind_speed_cube(
        self,
        template_cube: Cube,
        corrected_wind_speeds: np.ndarray,
    ) -> Cube:
        """Create the corrected wind-speed cube on the target height levels.

        Copy metadata from the uncorrected forecast and replace the height
        points with ``self.target_height_levels``. The supplied wind speeds must
        already include the terrain-induced speed-up correction.

        Args:
            template_cube:
                Cube used as the metadata template for the output.

            corrected_wind_speeds:
                Corrected wind speeds with shape
                (n_target_heights, y, x).

        Returns:
            Wind-speed cube after unresolved-orography correction on the selected
            target heights.

        Raises:
            ValueError:
                If the template height coordinate is not in metres.
        """
        corrected_wind_speeds = np.ma.asarray(
            corrected_wind_speeds,
            dtype=np.float32,
        )

        self._require_metre_units(
            template_cube.coord("height"),
            "height coordinate",
        )

        # Use a 2D slice of the template cube as the metadata source for each
        # output level before replacing the height coordinate with a target value.
        height_dims = template_cube.coord_dims("height")
        if not height_dims:
            template_2d = template_cube
        else:
            height_dim = height_dims[0]
            index = [slice(None)] * template_cube.ndim
            index[height_dim] = 0
            template_2d = template_cube[tuple(index)]

        output_cubes = CubeList()

        for height_index, target_height in enumerate(self.target_height_levels):
            output_cube = template_2d.copy(data=corrected_wind_speeds[height_index])

            height_coord = output_cube.coord("height")

            height_coord.points = np.array(
                [target_height],
                dtype=np.float32,
            )
            height_coord.bounds = None

            output_cubes.append(output_cube)

        return output_cubes.merge_cube()

    def _compute_inner_layer_response(
        self,
        roughness_scaled_wavenumber_3d: np.ndarray,
        target_heights_broadcast: np.ndarray,
        roughness_length_field: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the inner-layer response using modified Bessel functions.

        Describes how surface friction modifies the response of the flow to
        unresolved terrain.

        Args:
            roughness_scaled_wavenumber_3d:
                Wavenumber scaled by log-height ratio, shape (n_heights, y, x).

            target_heights_broadcast:
                Target heights reshaped for broadcasting, shape (n_heights, 1, 1).

            roughness_length_field:
                Roughness length broadcasted to 3D, shape (1, y, x).

        Returns:
            np.ndarray:
                Inner-layer response values with shape (n_heights, y, x).
        """
        bessel_argument_at_target_height = (
            (1.0 + 1.0j)
            * np.sqrt(roughness_scaled_wavenumber_3d * target_heights_broadcast)
            / VON_KARMAN_CONSTANT
        )
        bessel_argument_at_roughness_level = (
            (1.0 + 1.0j)
            * np.sqrt(roughness_scaled_wavenumber_3d * roughness_length_field)
            / VON_KARMAN_CONSTANT
        )
        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            inner_layer_response = np.real(
                1.0
                - kv(0, bessel_argument_at_target_height)
                / kv(0, bessel_argument_at_roughness_level)
            )
        return inner_layer_response

    def _calculate_speed_up_factor(
        self,
        characteristic_wavenumber: np.ndarray,
        unresolved_orography_height: np.ndarray,
        target_heights: np.ndarray,
        target_wind_speeds: np.ndarray,
        reference_wind_speed: np.ndarray,
        roughness_length: np.ndarray,
    ) -> np.ndarray:
        """Calculate the expected terrain effect and express it as a speed-up factor.

        Combine terrain height and horizontal scale, reference wind, roughness
        length, and target heights using the flow-over-terrain calculation based
        on Howard and Clark (2007). The response depends on terrain height
        relative to its horizontal scale and includes the effect of surface
        friction.

        Express the resulting wind-speed perturbation relative to the background
        wind to produce a speed-up factor:

            corrected wind = background wind * speed-up factor

        Factors above 1 increase wind speed. Negative perturbations are clipped,
        so this implementation does not apply slow-downs.

        Args:
            characteristic_wavenumber:
                Characteristic wavenumber of the unresolved terrain, in m-1.
                Larger values represent shorter-scale terrain. Shape: (y, x).

            unresolved_orography_height:
                True, unsmoothed orography minus the orography used by the NWP
                model, e.g. smoothed model orography, in metres.
                Shape: (y, x).

            target_heights:
                Heights above ground at which corrected winds are required,
                in metres. Shape: (n_heights,).

            target_wind_speeds:
                Background wind speeds at the target heights, in m s-1.
                Shape: (n_heights, y, x).

            reference_wind_speed:
                Background wind speed at the unresolved-orography reference height
                z_s = 1 / k, where k is the characteristic terrain wavenumber in
                m-1. This height scale represents the dominant horizontal scale of
                the unresolved terrain and provides the velocity scale for the
                terrain perturbation. Shape: (y, x).

            roughness_length:
                Effective roughness length, in metres, used to describe the
                near-surface response of the flow. Shape: (y, x).

        Returns:
            Multiplicative wind-speed correction with shape
            (n_heights, y, x). A value of 1 leaves the background wind unchanged.

        Raises:
            ValueError:
                If ``target_heights`` is empty or contains non-finite values, or if
                ``target_wind_speeds`` does not have shape
                (n_heights, y, x) matching ``target_heights``.
        """
        # Validate input: check for non-finite target heights early.
        if np.any(~np.isfinite(target_heights)):
            raise ValueError("target_heights contains non-finite values (NaN or inf).")
        if len(target_heights) == 0:
            raise ValueError("target_heights must not be empty.")

        perturbation_limits = (0.0, 1.0)

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

        target_wind_mask, target_wind_speeds = self._prepare_target_wind_speeds(
            target_wind_speeds
        )

        if target_wind_speeds.ndim != 3:
            raise ValueError("target_wind_speeds must have shape (n_heights, y, x).")
        if target_wind_speeds.shape[0] != len(target_heights):
            raise ValueError(
                "target_wind_speeds first dimension must match target_heights."
            )

        target_spatial_shape = target_wind_speeds.shape[1:]

        characteristic_wavenumber = np.broadcast_to(
            np.asarray(characteristic_wavenumber, dtype=float),
            target_spatial_shape,
        )
        unresolved_orography_height = np.broadcast_to(
            np.asarray(unresolved_orography_height, dtype=float),
            target_spatial_shape,
        )
        reference_wind_speed = np.broadcast_to(
            np.asarray(reference_wind_speed, dtype=float),
            target_spatial_shape,
        )
        roughness_length = np.broadcast_to(
            np.asarray(roughness_length, dtype=float),
            target_spatial_shape,
        )

        # Compute terrain amplitude. This describes the unresolved terrain height
        # relative to its characteristic horizontal scale.
        terrain_amplitude = 1.0 * np.tanh(
            characteristic_wavenumber * unresolved_orography_height
        )

        # Identify grid points with invalid ancillary data (NaN, inf, or
        # negative values). These will be masked out later and assigned a
        # neutral correction factor of 1.0.
        invalid_characteristic_wavenumber = ~np.isfinite(characteristic_wavenumber)
        invalid_unresolved_orography = ~np.isfinite(unresolved_orography_height)
        invalid_reference_height = ~np.isfinite(1.0 / characteristic_wavenumber) | (
            (1.0 / characteristic_wavenumber) <= 0.0
        )
        invalid_reference_wind = (
            ~np.isfinite(reference_wind_speed) | invalid_reference_height
        )
        invalid_roughness_length = ~np.isfinite(roughness_length)
        invalid_ancillary = (
            invalid_characteristic_wavenumber
            | invalid_unresolved_orography
            | invalid_reference_wind
            | invalid_roughness_length
        )

        # Broadcast 2D ancillary fields to all non-height target-wind dimensions.
        target_heights_broadcast = np.reshape(
            np.asarray(target_heights, dtype=float),
            (len(target_heights),) + (1,) * (target_wind_speeds.ndim - 1),
        )
        wavenumber_field = characteristic_wavenumber[np.newaxis, ...]
        reference_wind_field = reference_wind_speed[np.newaxis, ...]
        roughness_length_field = roughness_length[np.newaxis, ...]
        terrain_amplitude_field = terrain_amplitude[np.newaxis, ...]

        # Calculate the inner-layer response. This describes how surface
        # friction modifies the response of the flow to unresolved terrain.
        roughness_scaled_wavenumber_3d = wavenumber_field * np.log(
            target_heights_broadcast / roughness_length_field
        )
        inner_layer_response_per_height = self._compute_inner_layer_response(
            roughness_scaled_wavenumber_3d,
            target_heights_broadcast,
            roughness_length_field,
        )

        # The effect of a terrain feature decreases with height: shorter-scale
        # terrain, represented by larger wavenumbers, decays more rapidly.
        vertical_decay_per_height = np.exp(-wavenumber_field * target_heights_broadcast)

        # Calculate the terrain-induced change in wind speed.
        # The reference wind sets the velocity scale, while the remaining terms
        # describe the strength and vertical structure of the terrain response.
        wind_speed_perturbation_per_height = (
            reference_wind_field
            * inner_layer_response_per_height
            * vertical_decay_per_height
            * terrain_amplitude_field
        )

        # Scale the terrain perturbation by the local background wind so the
        # resulting correction is expressed as a fractional speed-up.
        fractional_perturbation_per_height = np.divide(
            wind_speed_perturbation_per_height,
            target_wind_speeds,
            out=np.zeros_like(target_wind_speeds, dtype=float),
            where=(
                np.isfinite(wind_speed_perturbation_per_height)
                & np.isfinite(target_wind_speeds)
                & (target_wind_speeds > 0.0)
            ),
        )

        # Limit the terrain-induced change so that its magnitude cannot exceed
        # the background wind speed while retaining the positive-only speed-up.
        fractional_perturbation_per_height = np.clip(
            fractional_perturbation_per_height,
            perturbation_limits[0],
            perturbation_limits[1],
        )

        # The multiplicative correction factor applied to the background wind.
        speed_up_multiplier_per_height = 1.0 + fractional_perturbation_per_height

        # The target-wind mask already has the same dimensions as target_wind_speeds.
        target_wind_invalid_per_height = target_wind_mask
        invalid_ancillary_per_height = invalid_ancillary[np.newaxis, ...]

        invalid = (
            target_wind_invalid_per_height
            | invalid_ancillary_per_height
            | ~np.isfinite(speed_up_multiplier_per_height)
        )
        speed_up_factor = np.where(invalid, 1.0, speed_up_multiplier_per_height)

        return speed_up_factor

    def _evaluate_spline_at_reference_heights(
        self,
        spline: PchipInterpolator,
        reference_height_cube: Cube,
    ) -> np.ma.MaskedArray:
        """Calculate a reference wind speed from the model wind profile.

        The terrain effect depends partly on the strength of the wind flowing
        over it. Evaluate the model wind spline at each reference height to
        obtain the background wind used to scale the terrain response.

        Args:
            spline:
                PCHIP interpolator fitted to wind profiles with shape
                (height, y, x).

            reference_height_cube:
                Two-dimensional reference-height cube with shape (y, x), in
                metres. Its horizontal shape must match the wind-profile grid.

        Returns:
            Wind speed at the reference height for each grid point,
            with shape (y, x).

        Raises:
            ValueError:
                If the spline was fitted to data with dimensions other than
                (height, y, x), or if the reference-height cube does not have a
                matching two-dimensional horizontal shape.
        """
        input_heights = spline.x
        reference_heights = np.ma.asarray(reference_height_cube.data, dtype=float)
        coefficients_all = spline.c

        if coefficients_all.ndim != 4:
            raise ValueError(
                "spline must be fitted to wind profiles with shape (height, y, x)."
            )
        if reference_heights.ndim != 2:
            raise ValueError("reference_height_cube must have shape (y, x).")
        if coefficients_all.shape[-2:] != reference_heights.shape:
            raise ValueError(
                "reference_height_cube horizontal shape must match the "
                "wind-profile grid."
            )

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

        # PCHIP stores polynomial coefficients for each interval as:
        # c[0] * dx**3
        # + c[1] * dx**2
        # + c[2] * dx
        # + c[3]
        interval_index = np.reshape(
            interval,
            (1, 1) + interval.shape,
        )
        interval_index = np.broadcast_to(
            interval_index,
            (4, 1) + interval.shape,
        )
        coefficients = np.take_along_axis(
            coefficients_all,
            interval_index,
            axis=1,
        )[:, 0, ...]

        dx = reference_heights.data - input_heights[interval]

        reference_wind_speed = (
            coefficients[0] * dx**3
            + coefficients[1] * dx**2
            + coefficients[2] * dx
            + coefficients[3]
        )

        return np.ma.array(reference_wind_speed, mask=~valid)

    def _fit_log_wind_profile(
        self,
        wind_speed_cube: Cube,
        lower_height_limit: float = 0.0,
        upper_height_limit: float = 300.0,
        refinement_iterations: int = 20,
        min_friction_velocity: float = 0.001,
        max_friction_velocity: float = 5.0,
        min_roughness_length: float = 1e-5,
        max_roughness_length: float = 5.0,
    ) -> np.ndarray:
        """
        Estimate an effective roughness length at each grid point.

        An effective roughness length is calculated independently at each grid
        point by fitting a neutral logarithmic wind profile to wind speeds on
        the lowest model levels (between the surface and 300 m).

        Directly fitting the logarithmic profile at every grid point is
        computationally expensive because it requires a non-linear optimisation.
        Therefore, instead, a fast linear approximation is first used to
        estimate the effectiveroughness length. This estimate is then refined
        using the exact neutral logarithmic wind-profile relationship:

            U(z) = (u_star / kappa) * log((z + z0) / z0)

        where ``U(z)`` is the wind speed at height ``z``, ``u_star`` is the friction
        velocity, ``kappa`` is the von Kármán constant, and ``z0`` is the effective
        roughness length.

        Args:
            wind_speed_cube:
                Wind-speed cube with shape (height, y, x). Its height coordinate
                must be in metres.

            lower_height_limit:
                Exclusive lower bound on heights included in fitting, in metres.
                Default is 0.0 m.

            upper_height_limit:
                Inclusive upper bound on heights included in fitting, in metres.
                Default is 300.0 m.

            refinement_iterations:
                Number of refinement iterations for roughness-length search.
                Default is 20.

            min_friction_velocity:
                Lower clipping bound for the fitted friction velocity, in m s-1.
                Default is 0.001 m s-1.

            max_friction_velocity:
                Upper clipping bound for the fitted friction velocity, in m s-1.
                Default is 5.0 m s-1.

            min_roughness_length:
                Lower bound for the fitted effective roughness length, in metres.
                Default is 1e-5 m.

            max_roughness_length:
                Upper bound for the fitted effective roughness length, in metres.
                Default is 5.0 m.

        Returns:
            roughness_length:
                Effective roughness length at each horizontal grid point,
                shape (y, x).
        """
        wind_speeds = OrographicWindCorrection._extract_wind_data(
            wind_speed_cube
        ).filled(np.nan)

        valid_heights = (
            np.isfinite(self.original_height_levels)
            & (self.original_height_levels > lower_height_limit)
            & (self.original_height_levels <= upper_height_limit)
        )
        fit_heights = self.original_height_levels[valid_heights]
        fit_winds = wind_speeds[valid_heights]
        valid = np.isfinite(fit_winds) & (fit_winds > 0.0)
        valid_count = np.sum(valid, axis=0)

        lower_z0, upper_z0 = OrographicWindCorrection._approximate_roughness_length(
            fit_heights,
            fit_winds,
            valid,
            valid_count,
            min_roughness_length,
            max_roughness_length,
        )
        roughness_length = OrographicWindCorrection._refine_roughness_length(
            fit_heights,
            fit_winds,
            valid,
            valid_count,
            lower_z0,
            upper_z0,
            refinement_iterations,
            min_friction_velocity,
            max_friction_velocity,
        )
        friction_velocity, _ = OrographicWindCorrection._evaluate_log_profile_fit(
            fit_heights,
            fit_winds,
            valid,
            valid_count,
            roughness_length,
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

    @staticmethod
    def _approximate_roughness_length(
        fit_heights: np.ndarray,
        fit_winds: np.ndarray,
        valid: np.ndarray,
        valid_count: np.ndarray,
        min_roughness_length: float,
        max_roughness_length: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate an initial roughness length and search interval.

        A linearised approximation to the neutral logarithmic wind profile is
        used to estimate the effective roughness length at each grid point. The
        resulting estimate is used to define lower and upper bounds for the
        subsequent exact refinement step.

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
            lower_z0:
                Lower bound of the roughness-length search interval, shape (y, x).

            upper_z0:
                Upper bound of the roughness-length search interval, shape (y, x).
        """
        log_heights = np.log(fit_heights)
        count = np.maximum(valid_count, 1)

        mean_log_height = np.zeros(valid_count.shape, dtype=float)
        mean_wind = np.zeros(valid_count.shape, dtype=float)
        for i, lh in enumerate(log_heights):
            mean_log_height += np.where(valid[i], lh, 0.0)
            mean_wind += np.where(valid[i], fit_winds[i], 0.0)
        mean_log_height /= count
        mean_wind /= count

        numerator = np.zeros(valid_count.shape, dtype=float)
        denominator = np.zeros(valid_count.shape, dtype=float)
        for i, lh in enumerate(log_heights):
            lha = lh - mean_log_height
            wa = fit_winds[i] - mean_wind
            numerator += np.where(valid[i], lha * wa, 0.0)
            denominator += np.where(valid[i], lha**2, 0.0)

        A = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=float),
            where=denominator > 0.0,
        )
        B = mean_wind - A * mean_log_height

        ratio = np.divide(
            -B,
            A,
            out=np.full_like(B, np.nan, dtype=float),
            where=np.isfinite(A) & np.isfinite(B) & (A != 0.0),
        )
        approximate_z0 = np.exp(np.clip(ratio, -50.0, 50.0))

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

    @staticmethod
    def _evaluate_log_profile_fit(
        fit_heights: np.ndarray,
        fit_winds: np.ndarray,
        valid: np.ndarray,
        valid_count: np.ndarray,
        roughness_length: np.ndarray,
        min_friction_velocity: float,
        max_friction_velocity: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the logarithmic wind-profile fit for a candidate roughness length.

        For a supplied roughness-length field, compute the least-squares friction
        velocity and the corresponding sum of squared fitting errors at each grid
        point.

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

            min_friction_velocity:
                Lower clipping bound for fitted friction velocity.

            max_friction_velocity:
                Upper clipping bound for fitted friction velocity.

        Returns:
            friction_velocity:
                Best-fit friction velocity, shape (y, x).

            squared_error:
                Sum of squared profile-fitting errors, shape (y, x).
        """
        numerator = np.zeros(roughness_length.shape, dtype=float)
        denominator = np.zeros(roughness_length.shape, dtype=float)
        for i, h in enumerate(fit_heights):
            log_term = np.log((h + roughness_length) / roughness_length)
            numerator += np.where(valid[i], log_term * fit_winds[i], 0.0)
            denominator += np.where(valid[i], log_term**2, 0.0)

        profile_scale = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=float),
            where=denominator > 0.0,
        )
        friction_velocity = np.clip(
            VON_KARMAN_CONSTANT * profile_scale,
            min_friction_velocity,
            max_friction_velocity,
        )

        squared_error = np.zeros(roughness_length.shape, dtype=float)
        for i, h in enumerate(fit_heights):
            log_term = np.log((h + roughness_length) / roughness_length)
            fitted_wind = friction_velocity / VON_KARMAN_CONSTANT * log_term
            squared_error += np.where(valid[i], (fit_winds[i] - fitted_wind) ** 2, 0.0)
        squared_error = np.where(valid_count >= 2, squared_error, np.inf)

        return friction_velocity, squared_error

    @staticmethod
    def _refine_roughness_length(
        fit_heights: np.ndarray,
        fit_winds: np.ndarray,
        valid: np.ndarray,
        valid_count: np.ndarray,
        lower_z0: np.ndarray,
        upper_z0: np.ndarray,
        refinement_iterations: int,
        min_friction_velocity: float,
        max_friction_velocity: float,
    ) -> np.ndarray:
        """
        Refine roughness-length estimates using a golden-section search.

        A golden-section search is performed independently at each grid point
        in logarithmic roughness-length space. The search minimises the squared
        error between "true" wind speeds and the fitted neutral logarithmic
        wind profile.

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
                Number of golden-section refinement iterations.

            min_friction_velocity:
                Lower clipping bound for fitted friction velocity.

            max_friction_velocity:
                Upper clipping bound for fitted friction velocity.

        Returns:
            np.ndarray:
                Midpoints of the final log-space search intervals, converted to
                roughness length, with the same shape as ``valid_count``.
        """
        golden_ratio = (np.sqrt(5.0) - 1.0) / 2.0
        spatial_shape = valid_count.shape
        lower = np.log(lower_z0)
        upper = np.log(upper_z0)

        def _error(log_z0: np.ndarray) -> np.ndarray:
            """Return profile-fitting error for a candidate log roughness length."""
            roughness_length = np.exp(log_z0)
            _, err = OrographicWindCorrection._evaluate_log_profile_fit(
                fit_heights,
                fit_winds,
                valid,
                valid_count,
                roughness_length,
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

        return np.exp(0.5 * (lower + upper)).reshape(spatial_shape)

    def _fit_spline_wind_profile(
        self,
        wind_speed_cube: Cube,
    ) -> PchipInterpolator:
        """Describe the model wind profile using a PCHIP spline.

        Fit Piecewise Cubic Hermite Interpolation (PCHIP) spline through the wind
        profile at each grid point. This gives a smooth spline through the model
        levels, allowing calculation of the background wind at any arbitary
        height within the profile.

        Uses ``self.original_height_levels``, set by ``process``.

        Args:
            wind_speed_cube:
                Iris cube containing wind speeds with one height dimension and
                two horizontal dimensions. The height coordinate must be in
                metres.

        Returns:
            PCHIP interpolator that can be used to evaluate wind speeds at
            arbitrary heights within the model profile.

        Raises:
            ValueError:
                If any height in `wind_speed_cube` is non-finite, or if the
                heights are not strictly increasing.
        """
        if np.any(~np.isfinite(self.original_height_levels)):
            raise ValueError("Height levels must all be finite.")
        if np.any(np.diff(self.original_height_levels) <= 0):
            raise ValueError("Height levels must be strictly increasing.")

        wind_speeds = np.ma.filled(
            OrographicWindCorrection._extract_wind_data(wind_speed_cube),
            fill_value=np.nan,
        ).astype(float)

        return PchipInterpolator(
            self.original_height_levels,
            wind_speeds,
            axis=0,
            extrapolate=False,
        )

    def _calculate_unresolved_orography_height(
        self,
        unsmoothed_orography_cube: Cube,
        model_orography_cube: Cube,
    ) -> Cube:
        """Describe the height of the unresolved terrain.

            unresolved orography = unsmoothed orography - model orography

        This tells us how much higher or lower the "real" terrain is than the
        smoothed representation of terrain used by the NWP model.

        Args:
            unsmoothed_orography_cube:
                Cube containing the unsmoothed (i.e. true) orography, in metres.
                It must be on the same grid as ``model_orography_cube``.

            model_orography_cube:
                Cube containing the smoothed (i.e. NWP model representation of)
                orography, in metres.

        Returns:
            Cube containing the unresolved orography height, in metres.
        """
        unresolved_orography_height = (
            unsmoothed_orography_cube.data - model_orography_cube.data
        )

        unresolved_orography_height_cube = unsmoothed_orography_cube.copy(
            data=unresolved_orography_height
        )

        unresolved_orography_height_cube.rename("unresolved_orography_height")
        return unresolved_orography_height_cube

    def _calculate_reference_height(
        self,
        wavenumber_cube: Cube,
    ) -> Cube:
        """Calculate a reference height from the horizontal scale of the terrain.

            reference height = 1 / wavenumber

        Broad terrain features have a higher reference height; shorter-scale
        features have a lower one. The model wind at this height supplies the
        reference wind speed.

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

    def _calculate_characteristic_wavenumber(
        self,
        orog_stddev_cube: Cube,
        silhouette_roughness_cube: Cube,
        min_valid_orog_stddev: float = 2.0,
        min_valid_silhouette_roughness: float = 0.0,
        min_length_scale: float = 500.0,
        max_length_scale: float = 4000.0,
        min_half_peak_to_trough_amplitude: float = 1.0,
    ) -> Cube:
        """
        Calculate the characteristic horizontal wavenumber of unresolved
        orography over land.

        The same height difference over a broad hill affects the wind
        differently from that difference over a short distance: this is
        described by the `characteristic wavenumber`. Large values represent
        short-scale terrain; small values represent broader terrain.

        The characteristic wavenumber is calculated by: first estimating the
        characteristic terrain scale from the silhouette roughness and half
        half peak-to-trough terrain amplitude, then constraining it to the range
        represented given by the specified minimum and maximum terrain length
        scales, and finally converting it to a wavenumber (by multiplying by pi).

        Args:
            orog_stddev_cube:
                Cube containing the standard deviation of sub-grid orography, in
                metres.

            silhouette_roughness_cube:
                Cube containing the silhouette roughness of the sub-grid terrain.
                It must be on the same grid as ``orog_stddev_cube``.

            min_valid_orog_stddev:
                Minimum terrain standard deviation for applying the calculation.
                Default is 2.0 m.

            min_valid_silhouette_roughness:
                Minimum silhouette roughness for applying the calculation.
                Default is 0.0.

            min_length_scale:
                Smallest permitted terrain length scale, in metres.
                Default is 500.0 m.

            max_length_scale:
                Largest permitted terrain length scale, in metres.
                Default is 4000.0 m.

            min_half_peak_to_trough_amplitude:
                Minimum half peak-to-trough terrain amplitude used in the
                wavenumber calculation, in metres. Default is 1.0 m.

        Returns:
            Cube containing the characteristic terrain wavenumber, in m-1.
        """
        orog_stddev = np.ma.asarray(orog_stddev_cube.data)
        silhouette_roughness = np.ma.asarray(silhouette_roughness_cube.data)

        # Half peak-to-trough amplitude = √2*orog_stddev (assumes sinusoidal terrain)
        half_peak_to_trough_amplitude = np.sqrt(2.0) * orog_stddev

        valid = (
            ~np.ma.getmaskarray(orog_stddev)
            & ~np.ma.getmaskarray(silhouette_roughness)
            & (orog_stddev.data >= min_valid_orog_stddev)
            & (silhouette_roughness.data >= min_valid_silhouette_roughness)
        )
        wavenumber = np.ma.masked_all(
            orog_stddev.shape,
            dtype=np.float64,
        )

        inverse_length_scale = silhouette_roughness.data[valid] / np.maximum(
            half_peak_to_trough_amplitude.data[valid],
            min_half_peak_to_trough_amplitude,
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

    @staticmethod
    def _require_metre_units(cube: Cube | Coord, cube_name: str) -> None:
        """Require the given cube or coordinate to have units of metres."""
        if str(cube.units) != "m":
            raise ValueError(f"{cube_name} must be in metres; got {cube.units}.")

    @classmethod
    def _get_height_levels_from_cube(
        cls,
        height_levels_cube: Cube,
    ) -> np.ndarray:
        """
        Return height levels from given cube, in metres.

        Args:
            height_levels_cube:
                Cube containing a height coordinate in metres.

        Returns:
            np.ndarray:
                One-dimensional array of height points in metres.

        Raises:
            ValueError:
                If the height coordinate is not in metres.
        """
        height_coord = height_levels_cube.coord("height").copy()
        cls._require_metre_units(
            height_coord,
            "The height coordinate",
        )
        return np.asarray(np.atleast_1d(height_coord.points), dtype=float)
