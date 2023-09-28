# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Estimate and apply a rescaling of the input forecast based on the difference
in altitude between the grid point and the site."""

from typing import Union

import iris
import numpy as np
import pandas as pd
from iris.coords import AuxCoord
from iris.cube import Cube
from numpy.polynomial import Polynomial as poly1d
from numpy.polynomial.polynomial import polyfit

from improver import PostProcessingPlugin
from improver.calibration.utilities import filter_non_matching_cubes
from improver.constants import SECONDS_IN_HOUR
from improver.metadata.constants.time_types import TIME_COORDS
from improver.spotdata.neighbour_finding import NeighbourSelection


class EstimateDzRescaling(PostProcessingPlugin):

    """Estimate a rescaling of the input forecasts based on the difference in
    altitude between the grid point and the site."""

    def __init__(
        self,
        forecast_period: float,
        dz_lower_bound: Union[str, float] = None,
        dz_upper_bound: Union[str, float] = None,
        land_constraint: bool = False,
        similar_altitude: bool = False,
        site_id_coord: str = "wmo_id",
    ):
        """Initialise class.

        Args:
            forecast_period: The forecast period in hours that is considered
                representative of the input forecasts. This is required as the input
                forecasts could contain multiple forecast periods.
            dz_lower_bound: The lowest acceptable value for the difference in
                altitude between the grid point and the site. Sites with a lower
                (or more negative) difference in altitude will be excluded.
                Defaults to None.
            dz_upper_bound: The highest acceptable value for the difference in
                altitude between the grid point and the site. Sites with a larger
                positive difference in altitude will be excluded. Defaults to None.
            land_constraint:
                If True, this will return a cube containing the nearest grid point
                neighbours to spot sites that are also land points. May be used
                with the similar_altitude option.
            similar_altitude:
                If True, this will return a cube containing the nearest grid point
                neighbour to each spot site that is found, within a given search
                radius, to minimise the height difference between the two. May be
                used with the land_constraint option.
            site_id_coord:
                The name of the site ID coordinate. This defaults to 'wmo_id'.
        """
        self.forecast_period = forecast_period
        if dz_lower_bound is None:
            self.dz_lower_bound = -np.inf
        else:
            self.dz_lower_bound = np.float32(dz_lower_bound)
        if dz_upper_bound is None:
            self.dz_upper_bound = np.inf
        else:
            self.dz_upper_bound = np.float32(dz_upper_bound)
        # The degree chosen for the fitting polynomial. This is set to 1.
        # Please see numpy.polynomial.polynomial.Polynomial.fit for further information.
        self.polyfit_deg = 1

        self.neighbour_selection_method = NeighbourSelection(
            land_constraint=land_constraint, minimum_dz=similar_altitude
        ).neighbour_finding_method_name()

        self.site_id_coord = site_id_coord

    def _fit_polynomial(self, forecasts: Cube, truths: Cube, dz: Cube) -> float:
        """Create a polynomial fit between the log of the ratio of forecasts and truths,
        and the difference in altitude between the grid point and the site.

        Args:
            forecasts: Forecast cube.
            truths: Truth cube.
            dz: Difference in altitude between the grid point and the site location.

        Returns:
            A scale factor deduced from a polynomial fit. This is a single value
            deduced from the fit between the forecasts and the truths.
        """
        truths_data = np.reshape(truths.data, forecasts.shape)

        data_filter = (
            (forecasts.data != 0)
            & (truths_data != 0)
            & (dz.data >= self.dz_lower_bound)
            & (dz.data <= self.dz_upper_bound)
        )

        forecasts_data = forecasts.data.flatten()
        truths_data = truths_data.flatten()
        dz_data = np.broadcast_to(dz.data, forecasts.shape).flatten()
        data_filter = data_filter.flatten()

        log_error_ratio = np.log(forecasts_data[data_filter] / truths_data[data_filter])

        scale_factor = poly1d(
            polyfit(dz_data[data_filter], log_error_ratio, self.polyfit_deg)
        ).coef

        # Only retain the multiplicative coefficient as the scale factor.
        # This helps conceptually with the difference in altitude rescaling
        # where if the dz of the grid point and the site are the same, then no
        # adjustment will be made.
        return scale_factor[1]

    def _compute_scaled_dz(self, scale_factor: float, dz: np.ndarray) -> np.ndarray:
        """Compute the scaled difference in altitude.

        Args:
            scale_factor: A scale factor deduced from a polynomial fit.
            dz: The difference in altitude between the grid point and the site.

        Returns:
            Scaled difference in altitude at each site.
        """
        # Multiplication by -1 using negative exponent rule, so that this term can
        # be multiplied by the forecast during the application step.
        scaled_dz = np.exp(-1.0 * scale_factor * dz)

        # Compute lower and upper bounds for the scaled dz.
        # dz_lower_bound may not result in the lower bound for the scaled dz depending
        # upon the sign of the scale_factor term.
        scaled_dz_a = np.exp(-1.0 * scale_factor * self.dz_lower_bound)
        scaled_dz_b = np.exp(-1.0 * scale_factor * self.dz_upper_bound)
        scaled_dz_lower = np.amin([scaled_dz_a, scaled_dz_b])
        scaled_dz_upper = np.amax([scaled_dz_a, scaled_dz_b])

        return np.clip(scaled_dz.data, scaled_dz_lower, scaled_dz_upper)

    def _compute_scaled_dz_cube(
        self,
        forecast: Cube,
        dz: Cube,
        scale_factor: float,
    ) -> Cube:
        """Compute the scaled difference in altitude and ensure that the output cube
        has the correct metadata.

        Args:
            forecast: Forecast cube.
            dz: The difference in altitude between the grid point and the site.
            scale_factor: A scale factor deduced from a polynomial fit.

        Returns:
            Scaled difference in altitude cube with appropriate metadata.
        """
        scaled_dz = dz.copy()
        scaled_dz.rename("scaled_vertical_displacement")
        scaled_dz.units = "1"
        for coord_name in [
            "grid_attributes",
            "grid_attributes_key",
            "neighbour_selection_method",
        ]:
            scaled_dz.remove_coord(coord_name)
        scaled_dz.attributes.pop("model_grid_hash", None)

        scaled_dz.data = self._compute_scaled_dz(scale_factor, scaled_dz.data)

        fp_forecast_slice = next(forecast.slices_over("forecast_period"))

        fp_forecast_slice.coord("forecast_period").points = np.array(
            self.forecast_period * SECONDS_IN_HOUR,
            dtype=TIME_COORDS["forecast_period"].dtype,
        )
        scaled_dz.add_aux_coord(fp_forecast_slice.coord("forecast_period"))
        self._create_hour_coord(forecast, scaled_dz)
        return scaled_dz

    def _create_hour_coord(self, source_cube: Cube, target_cube: Cube):
        """Create a coordinate exclusively containing the hour of the forecast
        reference time. This is required as the date of the forecast reference time
        is not relevant when using a training dataset with the aim that the resulting
        scaling factor is applicable to future forecasts with the same hour for the
        forecast reference time. The auxiliary coordinate will be added to the
        target_cube.

        Args:
            source_cube: Cube containing the forecast reference time from which
                the hour will be extracted.
            target_cube: Cube to which an auxiliary coordinate will be added.
        """
        # Create forecast_reference_time_hour coordinate. Use the time coordinate and
        # the forecast_period argument provided in case the forecast_reference_time
        # coordinate is not always the same within all input forecasts.
        frt_hour = (
            source_cube.coord("time").cell(0).point
            - pd.Timedelta(hours=self.forecast_period)
        ).hour
        hour_coord = AuxCoord(
            np.array(frt_hour, np.int32),
            long_name="forecast_reference_time_hour",
            units="hours",
        )
        hour_coord.convert_units("seconds")
        hour_coord.points = hour_coord.points.astype(np.int32)
        target_cube.add_aux_coord(hour_coord)

    def process(self, forecasts: Cube, truths: Cube, neighbour_cube: Cube) -> Cube:
        """Fit a polynomial using the forecasts and truths to compute a scaled
        version of the difference of altitude between the grid point and the
        site location. There is expected to be overlap between the sites provided by
        the forecasts, truths and neighbour_cube. The polynomial will be fitted using
        the forecasts and truths and the resulting scale factor will be applied to
        all sites within the neighbour cube. The output scaled dz will contain sites
        matching the neighbour cube.

        A mathematical summary of the steps within this plugin are:

        1. Estimate a scale factor for the relationship between the difference in
        altitude between the grid point and the site, and the natural log of the
        forecast divided by the truth, where s is the scale factor in the equations
        below.

        .. math::

            dz = \\ln(forecast / truth) \\times s

        2. Rearranging this equation gives:

        .. math::

            truth = forecast / \\exp(s \\times dz)

        or alternatively:

        .. math::

            truth = forecast \\times \\exp(-s \\times dz)

        This plugin is aiming to estimate the :math:`\\exp(-s \\times dz)` component,
        which can later be used for multiplying by a forecast to estimate the truth.

        Args:
            forecasts: Forecast cube.
            truths: Truth cube.
            neighbour_cube: A neighbour cube containing the difference in altitude
                between the grid point and the site location. Note that the output
                will have the same sites as found within the neighbour cube.

        Returns:
            A scaled difference of altitude between the grid point and the
            site location.
        """
        method = iris.Constraint(
            neighbour_selection_method_name=self.neighbour_selection_method
        )
        index_constraint = iris.Constraint(
            grid_attributes_key=["vertical_displacement"]
        )
        dz_cube = neighbour_cube.extract(method & index_constraint)

        sites = list(
            set(forecasts.coord(self.site_id_coord).points)
            & set(truths.coord(self.site_id_coord).points)
            & set(dz_cube.coord(self.site_id_coord).points)
        )

        constr = iris.Constraint(coord_values={self.site_id_coord: sites})
        training_forecasts = forecasts.extract(constr)
        training_truths = truths.extract(constr)
        dz_training_cube = dz_cube.extract(constr)

        forecast_cube, truth_cube = filter_non_matching_cubes(
            training_forecasts, training_truths
        )

        constr = iris.Constraint(percentile=50.0)
        forecast_cube = forecast_cube.extract(constr)

        scale_factor = self._fit_polynomial(forecast_cube, truth_cube, dz_training_cube)
        scaled_dz_cube = self._compute_scaled_dz_cube(
            forecast_cube, dz_cube, scale_factor
        )

        return scaled_dz_cube


class ApplyDzRescaling(PostProcessingPlugin):

    """Apply rescaling of the forecast using the difference in altitude between the
    grid point and the site."""

    def __init__(self, site_id_coord: str = "wmo_id", frt_hour_leniency=1):
        """Initialise class.

        Args:
            site_id_coord:
                The name of the site ID coordinate. This defaults to 'wmo_id'.
            frt_hour_leniency:
                The forecast reference time hour for the forecast and the scaled_dz
                are expected to match. If no match is found and a leniency greater
                than zero is specified, the forecast reference time hour will be
                compared with e.g. a +/-1 hour leniency.
        """
        self.site_id_coord = site_id_coord
        self.frt_hour_leniency = frt_hour_leniency

    def _check_mismatched_sites(self, forecast: Cube, scaled_dz: Cube) -> None:
        """Check that the sites match for the forecast and the scaled_dz inputs.

        Args:
            forecast: Forecast to be adjusted using dz rescaling.
            scaled_dz: A scaled version of the difference in altitude between the
                grid point and the site.

        Raises:
            ValueError: Sites do not match between the forecast and the scaled dz.
        """
        if len(forecast.coord(self.site_id_coord).points) != len(
            scaled_dz.coord(self.site_id_coord).points
        ) or np.any(
            forecast.coord(self.site_id_coord).points
            != scaled_dz.coord(self.site_id_coord).points
        ):
            mismatched_sites = set(
                forecast.coord(self.site_id_coord).points
            ).symmetric_difference(scaled_dz.coord(self.site_id_coord).points)
            n_forecast_sites = len(forecast.coord(self.site_id_coord).points)
            n_scaled_dz_sites = len(scaled_dz.coord(self.site_id_coord).points)
            msg = (
                "The sites do not match between the forecast and the scaled "
                "version of the difference in altitude. "
                f"Forecast number of sites: {n_forecast_sites}. "
                f"Scaled_dz number of sites: {n_scaled_dz_sites}. "
                f"The mismatched sites are: {mismatched_sites}."
            )
            raise ValueError(msg)

    @staticmethod
    def _create_forecast_period_constraint(
        forecast: Cube, scaled_dz: Cube
    ) -> iris.Constraint:
        """Create a forecast period constraint to identify the most appropriate
        forecast period from the scaled_dz to extract. The most appropriate scaled dz
        is selected by choosing the nearest forecast period that is greater than or
        equal to the forecast period of the forecast.

        Args:
            forecast: Forecast to be adjusted using dz rescaling.
            scaled_dz: A scaled version of the difference in altitude between the
                grid point and the site.

        Returns:
            Forecast period constraint.

        Raises:
            ValueError: No scaled dz could be found for the forecast period provided.
        """
        fp_diff = (
            scaled_dz.coord("forecast_period").points
            - forecast.coord("forecast_period").points
        )

        if not any(fp_diff >= 0):
            (fp_hour,) = forecast.coord("forecast_period").points / SECONDS_IN_HOUR
            msg = (
                "There is no scaled version of the difference in altitude for "
                f"a forecast period greater than or equal to {fp_hour}"
            )
            raise ValueError(msg)

        fp_index = np.argmax(fp_diff >= 0)
        chosen_fp = scaled_dz.coord("forecast_period").points[fp_index]
        return iris.Constraint(forecast_period=chosen_fp)

    @staticmethod
    def _create_forecast_reference_time_constraint(
        forecast: Cube, leniency: int
    ) -> iris.Constraint:
        """Create a forecast reference time constraint based on the hour within the
        forecast reference time.

        Args:
            forecast: Forecast to be adjusted using dz rescaling.
            leniency: The leniency in hours to adjust the forecast reference time hour
                when looking for a match.

        Returns:
            Forecast reference time hour constraint.
        """
        # Define forecast_reference_time constraint
        frt_hour_in_seconds = (
            (forecast.coord("forecast_reference_time").cell(0).point.hour + leniency)
            % 24
        ) * SECONDS_IN_HOUR
        return iris.Constraint(forecast_reference_time_hour=frt_hour_in_seconds)

    def process(self, forecast: Cube, scaled_dz: Cube) -> Cube:
        """Apply rescaling of the forecast to account for differences in the altitude
        between the grid point and the site, as assessed using a training dataset.
        The most appropriate scaled dz is selected by choosing the nearest forecast
        period that is greater than or equal to the forecast period of the forecast.

        Args:
            forecast: Forecast to be adjusted using dz rescaling.
            scaled_dz: A scaled version of the difference in altitude between the
                grid point and the site.

        Returns:
            Altitude-corrected forecast.

        Raises:
            ValueError: No scaled dz could be found for the forecast period and
                forecast reference time hour provided.

        """
        self._check_mismatched_sites(forecast, scaled_dz)
        fp_constr = self._create_forecast_period_constraint(forecast, scaled_dz)
        frt_hour_leniency_range = sorted(
            list(range(-self.frt_hour_leniency, self.frt_hour_leniency + 1)), key=abs
        )
        for leniency in frt_hour_leniency_range:
            frt_constr = self._create_forecast_reference_time_constraint(
                forecast, leniency
            )
            scaled_dz_extracted = scaled_dz.extract(fp_constr & frt_constr)
            if scaled_dz_extracted is not None:
                break
        else:
            frt_hour = forecast.coord("forecast_reference_time").cell(0).point.hour
            (fp_hour,) = forecast.coord("forecast_period").points / SECONDS_IN_HOUR
            msg = (
                "There is no scaled version of the difference in altitude for "
                f"a forecast period greater than or equal to {fp_hour} and "
                f"a forecast reference time hour equal to {frt_hour} or within "
                f"the specified leniency {self.frt_hour_leniency}."
            )
            raise ValueError(msg)

        forecast.data = forecast.data * scaled_dz_extracted.data
        return forecast
