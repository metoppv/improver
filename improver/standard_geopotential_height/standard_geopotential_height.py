# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing plugin to calculate standard geopotential height on pressure levels.

The standard geopotential height is calculated using the ICAO standard atmosphere
lookup table and barometric formulae described in the D-Factors workflow design.

Pressure handling:
    Only pressure levels within the configured range (default 10–1000 hPa) are
    processed and included in the output cube. Any levels outside this range are
    excluded from the returned cube (i.e. the pressure dimension is reduced).

Equations:
    If β = 0:
        Zstd(p) = Zb - (R Tb / g) ln(p / Pb)

    If β ≠ 0:
        Zstd(p) = Zb + (Tb / β) [ (p / Pb)^(-β R / g) - 1 ]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from cf_units import Unit
from iris.coords import Coord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError

from improver import PostProcessingPlugin
from improver.constants import EARTH_SURFACE_GRAVITY_ACCELERATION, R_DRY_AIR
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.cube_checker import check_cube_coordinates


@dataclass(frozen=True)
class _LayerParams:
    """ICAO Standard Atmosphere layer parameters.

    Pressures in hPa; temperatures in K; heights in m; beta in K m-1.
    """

    pb_hpa: float
    zb_m: float
    tb_k: float
    beta_k_m: float


# Static lookup table from D-Factors workflow design
_TROPOSPHERE = _LayerParams(pb_hpa=1013.25, zb_m=0.0, tb_k=288.15, beta_k_m=-0.0065)
_STRATOSPHERE = _LayerParams(pb_hpa=226.32, zb_m=11000.0, tb_k=216.65, beta_k_m=0.0)
_MESOSPHERE = _LayerParams(pb_hpa=54.75, zb_m=20000.0, tb_k=216.65, beta_k_m=0.0010)
_THERMOSPHERE = _LayerParams(pb_hpa=8.68, zb_m=32000.0, tb_k=228.65, beta_k_m=0.0028)


class StandardGeopotentialHeight(PostProcessingPlugin):
    """Calculate standard geopotential height on the pressure levels of an input cube.

    The input cube is used as a template. The output values depend only on the
    pressure coordinate points and are broadcast across the remaining dimensions.

    Any pressure levels outside the configured range are excluded from the output.

    Args:
        model_id_attr:
            Name of the attribute used to identify the source model. If provided,
            mandatory attributes will be generated consistently.
        pressure_min_hpa:
            Minimum pressure processed (hPa). Defaults to 10 hPa.
        pressure_max_hpa:
            Maximum pressure processed (hPa). Defaults to 1000 hPa.
    """

    def __init__(
        self,
        model_id_attr: Optional[str] = None,
        pressure_min_hpa: float = 10.0,
        pressure_max_hpa: float = 1000.0,
    ) -> None:
        self.model_id_attr = model_id_attr
        self.pressure_min_hpa = float(pressure_min_hpa)
        self.pressure_max_hpa = float(pressure_max_hpa)

    @staticmethod
    def _get_pressure_coord(cube: Cube) -> Coord:
        """Return the pressure coordinate from the cube.

        Raises:
            ValueError: If a suitable pressure coordinate is not found.
        """
        for name in ("pressure", "air_pressure"):
            try:
                return cube.coord(name)
            except CoordinateNotFoundError:
                continue
        raise ValueError(
            "Input cube must contain a pressure coordinate named 'pressure' "
            "or with standard name 'air_pressure'."
        )

    @staticmethod
    def _to_hpa(pressure_coord: Coord) -> np.ndarray:
        """Convert pressure coordinate points to hPa without mutating the cube."""
        points = np.array(pressure_coord.points, dtype=np.float64)
        in_units = pressure_coord.units
        out_units = Unit("hPa")
        try:
            return in_units.convert(points, out_units)
        except Exception as exc:
            raise ValueError(
                f"Pressure coordinate units {in_units!s} are not convertible to hPa."
            ) from exc

    @staticmethod
    def _layer_params_for_pressure(
        p_hpa: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return arrays of Pb, Zb, Tb, beta corresponding to each pressure level.

        Layer selection follows D-Factors design conditional logic:
            p <= 8.68   -> thermosphere
            p <= 54.75  -> mesosphere
            p <= 226.32 -> stratosphere
            else        -> troposphere
        """
        p_hpa = np.asarray(p_hpa, dtype=np.float64)

        therm = p_hpa <= _THERMOSPHERE.pb_hpa
        meso = (p_hpa > _THERMOSPHERE.pb_hpa) & (p_hpa <= _MESOSPHERE.pb_hpa)
        strat = (p_hpa > _MESOSPHERE.pb_hpa) & (p_hpa <= _STRATOSPHERE.pb_hpa)
        trop = p_hpa > _STRATOSPHERE.pb_hpa

        pb = np.empty_like(p_hpa)
        zb = np.empty_like(p_hpa)
        tb = np.empty_like(p_hpa)
        beta = np.empty_like(p_hpa)

        pb[therm], zb[therm], tb[therm], beta[therm] = (
            _THERMOSPHERE.pb_hpa,
            _THERMOSPHERE.zb_m,
            _THERMOSPHERE.tb_k,
            _THERMOSPHERE.beta_k_m,
        )
        pb[meso], zb[meso], tb[meso], beta[meso] = (
            _MESOSPHERE.pb_hpa,
            _MESOSPHERE.zb_m,
            _MESOSPHERE.tb_k,
            _MESOSPHERE.beta_k_m,
        )
        pb[strat], zb[strat], tb[strat], beta[strat] = (
            _STRATOSPHERE.pb_hpa,
            _STRATOSPHERE.zb_m,
            _STRATOSPHERE.tb_k,
            _STRATOSPHERE.beta_k_m,
        )
        pb[trop], zb[trop], tb[trop], beta[trop] = (
            _TROPOSPHERE.pb_hpa,
            _TROPOSPHERE.zb_m,
            _TROPOSPHERE.tb_k,
            _TROPOSPHERE.beta_k_m,
        )

        return pb, zb, tb, beta

    @staticmethod
    def _standard_geopotential_height_1d(p_hpa: np.ndarray) -> np.ndarray:
        """Compute 1D standard geopotential height (m) for pressure levels (hPa)."""
        pb, zb, tb, beta = StandardGeopotentialHeight._layer_params_for_pressure(p_hpa)

        r = float(R_DRY_AIR)
        g = float(EARTH_SURFACE_GRAVITY_ACCELERATION)

        z = np.empty_like(p_hpa, dtype=np.float64)

        beta_is_zero = np.isclose(beta, 0.0)
        if np.any(beta_is_zero):
            z[beta_is_zero] = zb[beta_is_zero] - (r * tb[beta_is_zero] / g) * np.log(
                p_hpa[beta_is_zero] / pb[beta_is_zero]
            )

        if np.any(~beta_is_zero):
            exponent = -beta[~beta_is_zero] * r / g
            z[~beta_is_zero] = zb[~beta_is_zero] + (
                tb[~beta_is_zero] / beta[~beta_is_zero]
            ) * ((p_hpa[~beta_is_zero] / pb[~beta_is_zero]) ** (exponent) - 1.0)

        return z.astype(np.float32)

    @staticmethod
    def _broadcast_to_template(
        z_1d: np.ndarray, template: Cube, pressure_coord: Coord
    ) -> np.ndarray:
        """Broadcast 1D values along the pressure dimension to match template shape."""
        pressure_dims = template.coord_dims(pressure_coord)

        if len(pressure_dims) == 0:
            # Scalar pressure coord: broadcast a single value over all points.
            return np.full(template.shape, float(z_1d.item()), dtype=np.float32)

        if len(pressure_dims) != 1:
            raise ValueError(
                "Pressure coordinate must span exactly one dimension. "
                f"Found {len(pressure_dims)} dimensions."
            )

        p_dim = pressure_dims[0]
        shape = [1] * template.ndim
        shape[p_dim] = z_1d.size
        z_reshaped = z_1d.reshape(shape)
        return np.broadcast_to(z_reshaped, template.shape).astype(np.float32)

    @staticmethod
    def _add_masking_by_pressure(
        geopotential_height_cube: Cube, p_hpa_min: float, p_hpa_max: float
    ) -> Cube:
        """
        Add masking by pressure levels, any existing masking is preserved

        Args:
            geopotential_height_cube:  cube to be masked
            p_hpa_min:  miniumum valid pressure for D-factors computation
            p_hpa_max:  maximum valid pressure for D-factors computation

        Returns:

            masked cube (same as input cube)

        """

        target_units = Unit("hPa")
        pressure_coord = StandardGeopotentialHeight._get_pressure_coord(
            geopotential_height_cube
        )
        in_units = pressure_coord.units
        points_in_hpa = [
            in_units.convert(point, target_units) for point in pressure_coord.points
        ]

        over_max = max(points_in_hpa) > p_hpa_max
        under_min = min(points_in_hpa) < p_hpa_min

        by_element_masking = over_max or under_min

        # ensure a masking araay if any pressure under the minimum or over the maximum
        if by_element_masking:
            if np.isscalar(geopotential_height_cube.data.mask):
                geopotential_height_cube.data.mask = np.zeros(
                    geopotential_height_cube.shape, dtype=bool
                )

            for i, p_hpa in enumerate(points_in_hpa):
                if p_hpa < p_hpa_min or p_hpa > p_hpa_max:
                    geopotential_height_cube.data.mask[i, :, :] = True

        return geopotential_height_cube

    def process(self, geopotential_height_cube: Cube) -> Cube:
        """Create standard geopotential height cube using pressure-filtered template.

        Pressure levels within the configured expected range are included in
        the output cube. Out-of-range pressure levels are masked out.
        """

        # Take a deep copy as the masking of the cube may be altered.
        # Input arguments should not (in general) be altered.
        template_cube = geopotential_height_cube.copy()

        # find pressure coord
        pressure_coord = self._get_pressure_coord(template_cube)

        # Compute standard geopotential height values for the various pressure levels
        p_hpa = self._to_hpa(pressure_coord)
        z_1d = self._standard_geopotential_height_1d(p_hpa)
        data = self._broadcast_to_template(z_1d, template_cube, pressure_coord)

        template_cube = self._add_masking_by_pressure(
            template_cube, self.pressure_min_hpa, self.pressure_max_hpa
        )

        if np.ma.isMaskedArray(template_cube.core_data()):
            # copy input masked array
            masked_data = np.ma.MaskedArray(data, mask=template_cube.data.mask)
        else:
            # create new mask
            masked_data = np.ma.MaskedArray(data, mask=False)

        # Create output cube with correct metadata based on the subset template
        mandatory = generate_mandatory_attributes(
            [template_cube], model_id_attr=self.model_id_attr
        )
        optional = template_cube.attributes.copy()
        optional["source"] = "IMPROVER"

        result = create_new_diagnostic_cube(
            name="standard_geopotential_height_on_pressure_levels",
            units="m",
            template_cube=template_cube,
            mandatory_attributes=mandatory,
            optional_attributes=optional,
            data=masked_data,
            dtype=np.float32,
        )

        # Ensure coordinate ordering matches the (subset) template cube
        # result is not masked
        result = check_cube_coordinates(template_cube, result)

        return result
