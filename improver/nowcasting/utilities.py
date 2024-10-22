#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module with utilities required for nowcasting."""

from typing import List, Union

import iris
import numpy as np
from cf_units import Unit
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.utilities import neighbourhood_tools
from improver.utilities.temporal import (
    extract_nearest_time_point,
    iris_time_to_datetime,
)


class ExtendRadarMask(BasePlugin):
    """Extend the mask on radar rainrate data based on the radar coverage
    composite"""

    def __init__(self) -> None:
        """
        Initialise with known values of the coverage composite for which radar
        data is valid.  All other areas will be masked.
        """
        self.coverage_valid = [1, 2]

    def process(self, radar_data: Cube, coverage: Cube) -> Cube:
        """
        Update the mask on the input rainrate cube to reflect where coverage
        is valid

        Args:
            radar_data:
                Radar data with mask corresponding to radar domains
            coverage:
                Radar coverage data containing values:
                    0: outside composite
                    1: precip detected
                    2: precip not detected & 1/32 mm/h detectable at this range
                    3: precip not detected & 1/32 mm/h NOT detectable

        Returns:
            Radar data with mask extended to mask out regions where
            1/32 mm/h are not detectable
        """
        # check cube coordinates match
        for crd in radar_data.coords():
            if coverage.coord(crd.name()) != crd:
                raise ValueError(
                    "Rain rate and coverage composites unmatched " "- coord {}".format(
                        crd.name()
                    )
                )

        # accommodate data from multiple times
        radar_data_slices = radar_data.slices(
            [radar_data.coord(axis="y"), radar_data.coord(axis="x")]
        )
        coverage_slices = coverage.slices(
            [coverage.coord(axis="y"), coverage.coord(axis="x")]
        )

        cube_list = iris.cube.CubeList()
        for rad, cov in zip(radar_data_slices, coverage_slices):
            # create a new mask that is False wherever coverage is valid
            new_mask = ~np.isin(cov.data, self.coverage_valid)

            # remask rainrate data
            remasked_data = np.ma.MaskedArray(rad.data.data, mask=new_mask)
            cube_list.append(rad.copy(remasked_data))

        return cube_list.merge_cube()


class FillRadarHoles(BasePlugin):
    """Fill in small "no data" regions in the radar composite by interpolating
    in log rainrate space.

    The log-linear transformation does not preserve non-zero rainrates of less
    than 0.001 mm/h. Since the radar composite encodes trace rain rates with a
    value of 0.03 mm/h, this should not have any effect on "real" data from the
    Met Office.
    """

    MIN_RR_MMH = 0.001

    def __init__(self) -> None:
        """Initialise parameters of interpolation

        The constants defining neighbourhood size and proportion of neighbouring
        masked pixels for speckle identification have been empirically tuned for
        UK radar data. As configured, this method will flag "holes" of up to 24
        pixels in size (30% of a 9 x 9 neighbourhood).

        The radius used to interpolate data into these holes has been chosen to
        match these constants, by defining the smallest radius that ensures there
        will always be valid data in the neighbourhood (25 pixels) over which
        averaging is performed.
        """
        # shape of neighbourhood over which to search for masked neighbours
        self.r_speckle = 4
        self.window_shape = ((self.r_speckle * 2) + 1, (self.r_speckle * 2) + 1)

        # proportion of masked neighbours below which a pixel is considered to
        # be isolated "speckle", which can be filled in by interpolation
        p_masked = 0.3
        # number of masked neighbours in neighbourhood
        self.max_masked_values = self.window_shape[0] * self.window_shape[1] * p_masked

        # radius of neighbourhood from which to calculate interpolated values
        self.r_interp = 2

    def _find_and_interpolate_speckle(self, cube: Cube) -> None:
        """Identify and interpolate "speckle" points, where "speckle" is defined
        as areas of "no data" that are small enough to fill by interpolation
        without affecting data integrity.  We would not wish to interpolate large
        areas as this may give false confidence in "no precipitation", where in
        fact precipitation exists in a "no data" region.

        Masked pixels near the borders of the input data array are not considered
        for interpolation.

        Args:
            cube:
                Cube containing rainrates (mm/h).  Data modified in place.
        """
        mask_windows = neighbourhood_tools.pad_and_roll(
            cube.data.mask, self.window_shape, mode="constant", constant_values=1
        )
        data_windows = neighbourhood_tools.pad_and_roll(
            cube.data, self.window_shape, mode="constant", constant_values=np.nan
        )

        # find indices of "speckle" pixels
        indices = np.where(
            (mask_windows[..., self.r_speckle, self.r_speckle] == 1)
            & (np.sum(mask_windows, axis=(-2, -1)) < self.max_masked_values)
        )

        # average data from the 5x5 nbhood around each "speckle" point
        bounds = slice(
            self.r_speckle - self.r_interp, self.r_speckle + self.r_interp + 1
        )
        data = data_windows[indices][..., bounds, bounds]
        mask = mask_windows[indices][..., bounds, bounds]

        for row_ind, col_ind, data_win, mask_win in zip(*indices, data, mask):
            valid_points = data_win[mask_win == 0]
            mean = np.mean(
                np.where(valid_points > self.MIN_RR_MMH, np.log10(valid_points), np.nan)
            )
            # when data value is set, mask is removed at that point
            if np.isnan(mean):
                cube.data[row_ind, col_ind] = 0
            else:
                cube.data[row_ind, col_ind] = np.power(10, mean)

    def process(self, masked_radar: Cube) -> Cube:
        """
        Fills in and unmasks small "no data" regions within the radar composite,
        to minimise gaps in the extrapolation nowcast.

        Args:
            masked_radar:
                A masked cube of radar precipitation rates

        Returns:
            A masked cube with continuous coverage over the radar composite
            domain, where missing data has been interpolated
        """
        # extract precipitation rate data in mm h-1
        masked_radar_mmh = masked_radar.copy()
        masked_radar_mmh.convert_units("mm h-1")

        # fill "holes" in data by interpolation
        self._find_and_interpolate_speckle(masked_radar_mmh)

        # return new cube in original units
        masked_radar_mmh.convert_units(masked_radar.units)
        return masked_radar_mmh


class ApplyOrographicEnhancement(BasePlugin):
    """Apply orographic enhancement to precipitation rate input, either to
    add or subtract an orographic enhancement component."""

    def __init__(self, operation: str) -> None:
        """Initialise class.

        Args:
            operation:
                Operation ("add" or "subtract") to apply to the incoming cubes.

        Raises:
            ValueError: Operation not supported.
        """
        # A minimum precipitation rate in mm/h that will be used as a lower
        # precipitation rate threshold.
        self.min_precip_rate_mmh = 1 / 32.0
        self.operation = operation

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        result = "<ApplyOrographicEnhancement: operation: {}>"
        return result.format(self.operation)

    @staticmethod
    def _select_orographic_enhancement_cube(
        precip_cube: Cube, oe_cube: Cube, allowed_time_diff: int = 1800
    ) -> Cube:
        """Select the orographic enhancement cube with the required time
        coordinate.

        Args:
            precip_cube:
                Cube containing the input precipitation fields.
            oe_cube:
                Cube containing orographic enhancement fields at one or
                more times.
            allowed_time_diff:
                The maximum permitted difference, in integer seconds,
                between the datetime of the precipitation cube and the time
                points available within the orographic enhancement cube.
                If this limit is exceeded, then an error is raised.


        Returns:
            Cube containing the orographic enhancement field at the
            required time.

        Raises:
            ValueError: If required time step is not available within tolerance
                (in theory.  In practise, the tolerance is left as the default
                None, which matches ANY available field regardless of time
                offset.  So this error will never be thrown.)
        """
        (time_point,) = iris_time_to_datetime(precip_cube.coord("time").copy())
        oe_cube_slice = extract_nearest_time_point(
            oe_cube, time_point, allowed_dt_difference=allowed_time_diff
        )
        return oe_cube_slice

    def _apply_orographic_enhancement(self, precip_cube: Cube, oe_cube: Cube) -> Cube:
        """Combine the precipitation rate cube and the orographic enhancement
        cube.

        Args:
            precip_cube:
                Cube containing the input precipitation field.
            oe_cube:
                Cube containing the orographic enhancement field matching
                the validity time of the precipitation cube.

        Returns:
            Cube containing the precipitation rate field modified by the
            orographic enhancement cube.
        """
        # Convert orographic enhancement into the units of the precipitation
        # rate cube.
        oe_cube.convert_units(precip_cube.units)

        # Set orographic enhancement to be zero for points with a
        # precipitation rate of < 1/32 mm/hr.
        original_units = Unit("mm/hr")
        threshold_in_cube_units = original_units.convert(
            self.min_precip_rate_mmh, precip_cube.units
        )

        # Ignore invalid warnings generated if e.g. a NaN is encountered
        # within the less than (<) comparison.
        with np.errstate(invalid="ignore"):
            oe_cube.data[precip_cube.data < threshold_in_cube_units] = 0.0

        # Add / subtract orographic enhancement where data is not masked
        cube = precip_cube.copy()
        if self.operation == "add":
            cube.data = cube.data + oe_cube.data
        elif self.operation == "subtract":
            cube.data = cube.data - oe_cube.data
        else:
            msg = (
                "Operation '{}' not supported for combining "
                "precipitation rate and "
                "orographic enhancement.".format(self.operation)
            )
            raise ValueError(msg)

        return cube

    def _apply_minimum_precip_rate(self, precip_cube: Cube, cube: Cube) -> Cube:
        """Ensure that negative precipitation rates are capped at the defined
        minimum precipitation rate.

        Args:
            precip_cube:
                Cube containing a precipitation rate input field.
            cube:
                Cube containing the precipitation rate field after combining
                with orographic enhancement.

        Returns:
            Cube containing the precipitation rate field where any
            negative precipitation rates have been capped at the defined
            minimum precipitation rate.
        """
        if self.operation == "subtract":
            original_units = Unit("mm/hr")
            threshold_in_cube_units = original_units.convert(
                self.min_precip_rate_mmh, cube.units
            )
            threshold_in_precip_cube_units = original_units.convert(
                self.min_precip_rate_mmh, precip_cube.units
            )

            # Ignore invalid warnings generated if e.g. a NaN is encountered
            # within the less than (<) comparison.
            with np.errstate(invalid="ignore"):
                # Create a mask computed from where the input precipitation
                # cube is greater or equal to the threshold and the result
                # of combining the precipitation rate input cube with the
                # orographic enhancement has generated a cube with
                # precipitation rates less than the threshold.
                mask = (precip_cube.data >= threshold_in_precip_cube_units) & (
                    cube.data <= threshold_in_cube_units
                )

                # Set any values lower than the threshold to be equal to
                # the minimum precipitation rate.
                cube.data[mask] = threshold_in_cube_units
        return cube

    def process(
        self, precip_cubes: Union[Cube, List[Cube]], orographic_enhancement_cube: Cube
    ) -> CubeList:
        """Apply orographic enhancement by modifying the input fields. This can
        include either adding or deleting the orographic enhancement component
        from the input precipitation fields.

        Args:
            precip_cubes:
                Cube or iterable (list, CubeList or tuple) of cubes containing
                the input precipitation fields.
            orographic_enhancement_cube:
                Cube containing the orographic enhancement fields.

        Returns:
            CubeList of precipitation rate cubes that have been updated
            using orographic enhancement.
        """
        if isinstance(precip_cubes, iris.cube.Cube):
            precip_cubes = iris.cube.CubeList([precip_cubes])

        updated_cubes = iris.cube.CubeList([])
        for precip_cube in precip_cubes:
            oe_cube = self._select_orographic_enhancement_cube(
                precip_cube, orographic_enhancement_cube.copy()
            )
            cube = self._apply_orographic_enhancement(precip_cube, oe_cube)
            cube = self._apply_minimum_precip_rate(precip_cube, cube)
            updated_cubes.append(cube)
        return updated_cubes
