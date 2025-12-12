# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Class for Temporal Interpolation calculations."""

from datetime import datetime, timedelta
from typing import Any, List, Optional, Tuple, Union

import iris
import numpy as np
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from numpy import ndarray

from improver import BasePlugin
from improver.metadata.constants import FLOAT_DTYPE
from improver.metadata.constants.time_types import TIME_COORDS
from improver.metadata.forecast_times import unify_cycletime
from improver.metadata.utilities import enforce_time_point_standard
from improver.utilities.complex_conversion import complex_to_deg, deg_to_complex
from improver.utilities.cube_manipulation import MergeCubes
from improver.utilities.round import round_close
from improver.utilities.solar import DayNightMask, calc_solar_elevation
from improver.utilities.spatial import lat_lon_determine, transform_grid_to_lat_lon
from improver.utilities.temporal import iris_time_to_datetime


class TemporalInterpolation(BasePlugin):
    """
    Interpolate data to intermediate times between the validity times of two
    cubes. This can be used to fill in missing data (e.g. for radar fields) or
    to ensure data is available at the required intervals when model data is
    not available at these times.

    The plugin will return the interpolated times and the later of the two
    input times. This allows us to modify the input diagnostics if they
    represent accumulations.

    The IMPROVER convention is that period diagnostics have their time
    coordinate point at the end of the period. The later of the two inputs
    therefore covers the period that has been broken down into shorter periods
    by the interpolation and, if working with accumulations, must itself be
    modified. The result of this approach is that in a long run of
    lead-times, e.g. T+0 to T+120 all the lead-times will be available except
    T+0.

    If working with period maximums and minimums we cannot return values in
    the new periods that do not adhere to the inputs. For example, we might
    have a 3-hour maximum of 5 ms-1 between 03-06Z. The period before it might
    have a maximum of 11 ms-1. Upon splitting the 3-hour period into 1-hour
    periods the gradient might give us the following results:

    Inputs: 00-03Z: 11 ms-1, 03-06Z: 5 ms-1
    Outputs: 03-04Z: 9 ms-1, 04-05Z: 7 ms-1, 05-06Z: 5ms-1

    However these outputs are not in agreement with the original 3-hour period
    maximum of 5 ms-1 over the period 03-06Z. We enforce the maximum from the
    original period which results in:

    Inputs: 00-03Z: 10 ms-1, 03-06Z: 5 ms-1
    Outputs: 03-04Z: 5 ms-1, 04-05Z: 5 ms-1, 05-06Z: 5ms-1

    If instead the preceding period maximum was 2 ms-1 we would use the trend
    to produce lower maximums in the interpolated 1-hour periods, becoming:

    Inputs: 00-03Z: 2 ms-1, 03-06Z: 5 ms-1
    Outputs: 03-04Z: 3 ms-1, 04-05Z: 4 ms-1, 05-06Z: 5ms-1

    This interpretation of the gradient information is retained in the output
    as it is consistent with the original period maximum of 5 ms-1 between
    03-06Z. As such we can impart increasing trends into maximums over periods
    but not decreasing trends. The counter argument can be made when
    interpolating minimums in periods, allowing us only to introduce
    decreasing trends for these.

    We could use the cell methods to determine whether we are working with
    accumulations, maximums, or minimums. This should be denoted as a cell
    method associated with the time coordinate, e.g. for an accumulation it
    would be `time: sum`, whilst a maximum would have `time: max`. However
    we cannot guarantee these cell methods are present. As such the
    interpolation of periods here relies on the user supplying a suitable
    keyword argument that denotes the type of period being processed.
    """

    def __init__(
        self,
        interval_in_minutes: Optional[int] = None,
        times: Optional[List[datetime]] = None,
        interpolation_method: str = "linear",
        accumulation: bool = False,
        max: bool = False,
        min: bool = False,
        model_path: Optional[str] = None,
        scaling: str = "minmax",
        clipping_bounds: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """
        Initialise class.

        Args:
            interval_in_minutes:
                Specifies the interval in minutes at which to interpolate
                between the two input cubes. A number of minutes which does not
                divide up the interval equally will raise an exception.

                   | e.g. cube_t0 valid at 03Z, cube_t1 valid at 06Z,
                   | interval_in_minutes = 60 --> interpolate to 04Z and 05Z.

            times:
                A list of datetime objects specifying the times to which to
                interpolate.
            interpolation_method:
                Method of interpolation to use. Default is linear.
                Only methods in known_interpolation_methods can be used.
            accumulation:
                Set True if the diagnostic being temporally interpolated is a
                period accumulation. The output will be renormalised to ensure
                that the total across the period constructed from the shorter
                intervals matches the total across the period from the coarser
                intervals.
            max:
                Set True if the diagnostic being temporally interpolated is a
                period maximum. Trends between adjacent input periods will be used
                to provide variation across the interpolated periods where these
                are consistent with the inputs.
            min:
                Set True if the diagnostic being temporally interpolated is a
                period minimum. Trends between adjacent input periods will be used
                to provide variation across the interpolated periods where these
                are consistent with the inputs.
            model_path:
                Path to the TensorFlow Hub module for the Google FILM model.
                Required if interpolation_method is "google_film".
            scaling:
                Scaling method to apply to the data before interpolation when
                using "google_film" method. Supported methods are "log10" and
                "minmax". Default is "minmax".
            clipping_bounds:
                A tuple specifying the (min, max) bounds to which to clip
                the interpolated data when using "google_film" method.
                Default is (0.0, 1.0).

        Raises:
            ValueError: If neither interval_in_minutes nor times are set.
            ValueError: If both interval_in_minutes and times are both set.
            ValueError: If interpolation method not in known list.
            ValueError: If multiple period diagnostic kwargs are set True.
            ValueError: A period diagnostic is being interpolated with a method
                        not found in the period_interpolation_methods list.
            ValueError: If interpolation_method is "google_film" but model_path
                        is not provided.
        """
        if interval_in_minutes is None and times is None:
            raise ValueError(
                "TemporalInterpolation: One of "
                "'interval_in_minutes' or 'times' must be set. "
                "Currently both are none."
            )
        if interval_in_minutes is not None and times is not None:
            raise ValueError(
                "TemporalInterpolation: Only one of "
                "'interval_in_minutes' or 'times' must be set. "
                "Currently both are set."
            )
        self.interval_in_minutes = interval_in_minutes
        self.times = times
        known_interpolation_methods = ["linear", "solar", "daynight", "google_film"]
        if interpolation_method not in known_interpolation_methods:
            raise ValueError(
                "TemporalInterpolation: Unknown interpolation method {}. ".format(
                    interpolation_method
                )
            )
        self.interpolation_method = interpolation_method

        # GoogleFilm-specific parameters
        if interpolation_method == "google_film":
            if model_path is None:
                raise ValueError(
                    "model_path must be provided when using google_film interpolation method."
                )
        self.model_path = model_path
        self.scaling = scaling
        self.clipping_bounds = clipping_bounds

        self.period_inputs = False
        if np.sum([accumulation, max, min]) > 1:
            raise ValueError(
                "Only one type of period diagnostics may be specified: "
                f"accumulation = {accumulation}, max = {max}, "
                f"min = {min}"
            )
        self.accumulation = accumulation
        self.max = max
        self.min = min
        if any([accumulation, max, min]):
            self.period_inputs = True

            period_interpolation_methods = ["linear"]
            if self.interpolation_method not in period_interpolation_methods:
                raise ValueError(
                    "Period diagnostics can only be temporally interpolated "
                    f"using these methods: {period_interpolation_methods}.\n"
                    f"Currently selected method is: {self.interpolation_method}. "
                    "Note: google_film method does not support period diagnostics."
                )

    def construct_time_list(
        self, initial_time: datetime, final_time: datetime
    ) -> List[Tuple[str, List[datetime]]]:
        """
        A function to construct a list of datetime objects formatted
        appropriately for use by iris' interpolation method.

        Args:
            initial_time:
                The start of the period over which a time list is to be
                constructed.
            final_time:
                The end of the period over which a time list is to be
                constructed.

        Returns:
            A list containing a tuple that specifies the coordinate and a
            list of points along that coordinate to which to interpolate,
            as required by the iris interpolation method, e.g.::

                    [('time', [<datetime object 0>,
                               <datetime object 1>])]

        Raises:
            ValueError: If list of times provided falls outside the range
                        specified by the initial and final times.
            ValueError: If the interval_in_minutes does not divide the time
                        range up equally.
        """
        time_list = []
        if self.times is not None:
            self.times = sorted(self.times)
            if self.times[0] < initial_time or self.times[-1] > final_time:
                raise ValueError(
                    "List of times falls outside the range given by "
                    "initial_time and final_time. "
                )
            time_list = self.times
        elif self.interval_in_minutes is not None:
            if (final_time - initial_time).seconds % (
                60 * self.interval_in_minutes
            ) != 0:
                msg = (
                    "interval_in_minutes of {} does not"
                    " divide into the interval of"
                    " {} mins equally.".format(
                        self.interval_in_minutes,
                        int((final_time - initial_time).seconds / 60),
                    )
                )
                raise ValueError(msg)

            time_entry = initial_time
            while True:
                time_entry = time_entry + timedelta(minutes=self.interval_in_minutes)
                if time_entry >= final_time:
                    break
                time_list.append(time_entry)

        time_list.append(final_time)
        time_list = sorted(set(time_list))

        return [("time", time_list)]

    @staticmethod
    def enforce_time_coords_dtype(cube: Cube) -> Cube:
        """
        Enforce the data type of the time, forecast_reference_time and
        forecast_period within the cube, so that time coordinates do not
        become mis-represented. The units of the time and
        forecast_reference_time are enforced to be
        "seconds since 1970-01-01 00:00:00" with a datatype of int64.
        The units of forecast_period are enforced to be seconds with a datatype
        of int32. This functions modifies the cube in-place.

        Args:
            cube:
                The cube that will have the datatype and units for the
                time, forecast_reference_time and forecast_period coordinates
                enforced.

        Returns:
            Cube where the datatype and units for the
            time, forecast_reference_time and forecast_period coordinates
            have been enforced.
        """
        for coord_name in ["time", "forecast_reference_time", "forecast_period"]:
            coord_spec = TIME_COORDS[coord_name]
            if cube.coords(coord_name):
                coord = cube.coord(coord_name)
                coord.convert_units(coord_spec.units)
                coord.points = round_close(coord.points, dtype=coord_spec.dtype)
                if hasattr(coord, "bounds") and coord.bounds is not None:
                    coord.bounds = round_close(coord.bounds, dtype=coord_spec.dtype)
        return cube

    @staticmethod
    def calc_sin_phi(dtval: datetime, lats: ndarray, lons: ndarray) -> ndarray:
        """
        Calculate sin of solar elevation

        Args:
            dtval:
                Date and time.
            lats:
                Array 2d of latitudes for each point
            lons:
                Array 2d of longitudes for each point

        Returns:
            Array of sine of solar elevation at each point
        """
        day_of_year = (dtval - datetime(dtval.year, 1, 1)).days
        utc_hour = (dtval.hour * 60.0 + dtval.minute) / 60.0
        sin_phi = calc_solar_elevation(
            lats, lons, day_of_year, utc_hour, return_sine=True
        )
        return sin_phi

    @staticmethod
    def calc_lats_lons(cube: Cube) -> Tuple[ndarray, ndarray]:
        """
        Calculate the lats and lons of each point from a non-latlon cube,
        or output a 2d array of lats and lons, if the input cube has latitude
        and longitude coordinates.

        Args:
            cube:
                cube containing x and y axis

        Returns:
            - 2d Array of latitudes for each point.
            - 2d Array of longitudes for each point.
        """
        trg_crs = lat_lon_determine(cube)
        if trg_crs is not None:
            xycube = next(cube.slices([cube.coord(axis="y"), cube.coord(axis="x")]))
            lats, lons = transform_grid_to_lat_lon(xycube)
        else:
            lats_row = cube.coord("latitude").points
            lons_col = cube.coord("longitude").points
            lats = np.repeat(lats_row[:, np.newaxis], len(lons_col), axis=1)
            lons = np.repeat(lons_col[np.newaxis, :], len(lats_row), axis=0)
        return lats, lons

    def solar_interpolate(self, diag_cube: Cube, interpolated_cube: Cube) -> CubeList:
        """
        Temporal Interpolation code using solar elevation for
        parameters (e.g. solar radiation parameters like
        Downward Shortwave (SW) radiation or UV index)
        which are zero if the sun is below the horizon and
        scaled by the sine of the solar elevation angle if the sun is above the
        horizon.

        Args:
            diag_cube:
                cube containing diagnostic data valid at the beginning
                of the period and at the end of the period.
            interpolated_cube:
                cube containing Linear interpolation of
                diag_cube at interpolation times in time_list.

        Returns:
            A list of cubes interpolated to the desired times.
        """

        interpolated_cubes = CubeList()
        (lats, lons) = self.calc_lats_lons(diag_cube)
        prev_data = diag_cube[0].data
        next_data = diag_cube[1].data
        dtvals = iris_time_to_datetime(diag_cube.coord("time"))
        # Calculate sine of solar elevation for cube valid at the
        # beginning of the period.
        dtval_prev = dtvals[0]
        sin_phi_prev = self.calc_sin_phi(dtval_prev, lats, lons)
        # Calculate sine of solar elevation for cube valid at the
        # end of the period.
        dtval_next = dtvals[1]
        sin_phi_next = self.calc_sin_phi(dtval_next, lats, lons)
        # Length of time between beginning and end in seconds
        diff_step = (dtval_next - dtval_prev).seconds

        for single_time in interpolated_cube.slices_over("time"):
            # Calculate sine of solar elevation for cube at this
            # interpolated time.
            dtval_interp = iris_time_to_datetime(single_time.coord("time"))[0]
            sin_phi_interp = self.calc_sin_phi(dtval_interp, lats, lons)
            # Length of time between beginning and interpolated time in seconds
            diff_interp = (dtval_interp - dtval_prev).seconds
            # Set all values to 0.0, to be replaced
            # with values calculated through this solar method.
            single_time.data[:] = 0.0
            sun_up = np.where(sin_phi_interp > 0.0)
            # Solar value is calculated only for points where the sun is up
            # and is a weighted combination of the data using the sine of
            # solar elevation and the data in the diag_cube valid
            # at the beginning and end.

            # If the diag_cube containing data valid at the
            # beginning of the period and at the end of the period
            # has more than x and y coordinates
            # the calculation needs to adapted to accommodate this.
            if len(single_time.shape) > 2:
                prevv = prev_data[..., sun_up[0], sun_up[1]] / sin_phi_prev[sun_up]
                nextv = next_data[..., sun_up[0], sun_up[1]] / sin_phi_next[sun_up]
                single_time.data[..., sun_up[0], sun_up[1]] = sin_phi_interp[sun_up] * (
                    prevv + (nextv - prevv) * (diff_interp / diff_step)
                )
            else:
                prevv = prev_data[sun_up] / sin_phi_prev[sun_up]
                nextv = next_data[sun_up] / sin_phi_next[sun_up]
                single_time.data[sun_up] = sin_phi_interp[sun_up] * (
                    prevv + (nextv - prevv) * (diff_interp / diff_step)
                )
            # cube with new data added to interpolated_cubes cube List.
            interpolated_cubes.append(single_time)
        return interpolated_cubes

    @staticmethod
    def daynight_interpolate(interpolated_cube: Cube) -> CubeList:
        """
        Set linearly interpolated data to zero for parameters
        (e.g. solar radiation parameters) which are zero if the
        sun is below the horizon.

        Args:
            interpolated_cube:
                cube containing Linear interpolation of
                cube at interpolation times in time_list.

        Returns:
            A list of cubes interpolated to the desired times.
        """
        daynightplugin = DayNightMask()
        daynight_mask = daynightplugin(interpolated_cube)
        index = daynight_mask.data == daynightplugin.night

        # Reshape the time, y, x mask to match the input which may include addtional
        # dimensions, such as realization.
        dropped_crds = [
            crd
            for crd in interpolated_cube.coords(dim_coords=True)
            if crd not in daynight_mask.coords(dim_coords=True)
        ]
        if dropped_crds:
            cslices = interpolated_cube.slices_over(dropped_crds)
            masked_data = CubeList()
            for cslice in cslices:
                cslice.data[index] = 0.0
                masked_data.append(cslice)
            interpolated_cube = masked_data.merge_cube()
        else:
            interpolated_cube.data[index] = 0.0

        return CubeList(list(interpolated_cube.slices_over("time")))

    @staticmethod
    def add_bounds(cube_t0: Cube, interpolated_cube: Cube):
        """Calcualte bounds using the interpolated times and the time
        taken from cube_t0. This function is used rather than iris's guess
        bounds method as we want to use the earlier time cube to inform
        the lowest bound. The interpolated_cube `crd` is modified in
        place.

        Args:
            cube_t0:
                The input cube corresponding to the earlier time.
            interpolated_cube:
                The cube containing the interpolated times, which includes
                the data corresponding to the time of the later of the two
                input cubes.

        Raises:
            CoordinateNotFoundError: if time or forecast_period coordinates
                                     are not present on the input cubes.
        """
        for crd in ["time", "forecast_period"]:
            try:
                interpolated_times = np.concatenate(
                    [cube_t0.coord(crd).points, interpolated_cube.coord(crd).points]
                )
            except CoordinateNotFoundError:
                raise CoordinateNotFoundError(
                    f"Period diagnostic cube is missing expected coordinate: {crd}"
                )
            all_bounds = []
            for start, end in zip(interpolated_times[:-1], interpolated_times[1:]):
                all_bounds.append([start, end])
            interpolated_cube.coord(crd).bounds = all_bounds

    @staticmethod
    def _calculate_accumulation(
        cube_t0: Cube, period_reference: Cube, interpolated_cube: Cube
    ):
        """If the input is an accumulation we use the trapezium rule to
        calculate a new accumulation for each output period from the rates
        we converted the accumulations to prior to interpolating. We then
        renormalise to ensure the total accumulation across the period is
        unchanged by expressing it as a series of shorter periods.

        The interpolated cube is modified in place.

        Args:
            cube_t0:
                The input cube corresponding to the earlier time.
            period_reference:
                The input cube corresponding to the later time, with the
                values prior to conversion to rates.
            interpolated_cube:
                The cube containing the interpolated times, which includes
                the data corresponding to the time of the later of the two
                input cubes.
        """
        # Calculate an average rate for the period from the edges
        accumulation_edges = [cube_t0, *interpolated_cube.slices_over("time")]
        period_rates = np.array(
            [
                (a.data + b.data) / 2
                for a, b in zip(accumulation_edges[:-1], accumulation_edges[1:])
            ]
        )
        interpolated_cube.data = period_rates

        # Multiply the average rate by the length of each period to get a new
        # accumulation.
        new_periods = np.diff(interpolated_cube.coord("forecast_period").bounds)
        for _ in range(interpolated_cube.ndim - new_periods.ndim):
            new_periods = np.expand_dims(new_periods, axis=1)
        interpolated_cube.data = np.multiply(new_periods, interpolated_cube.data)

        # Renormalise the total of the new periods to ensure it matches the
        # total expressed in the longer original period.
        (time_coord,) = interpolated_cube.coord_dims("time")
        interpolated_total = np.sum(interpolated_cube.data, axis=time_coord)
        renormalisation = period_reference.data / interpolated_total
        interpolated_cube.data *= renormalisation
        interpolated_cube.data = interpolated_cube.data.astype(FLOAT_DTYPE)

    def process(self, cube_t0: Cube, cube_t1: Cube) -> CubeList:
        """
        Interpolate data to intermediate times between validity times of
        cube_t0 and cube_t1.

        Args:
            cube_t0:
                A diagnostic cube valid at the beginning of the period within
                which interpolation is to be permitted.
            cube_t1:
                A diagnostic cube valid at the end of the period within which
                interpolation is to be permitted.

        Returns:
            A list of cubes interpolated to the desired times.

        Raises:
            TypeError: If cube_t0 and cube_t1 are not of type iris.cube.Cube.
            ValueError: A mix of instantaneous and period diagnostics have
                        been used as inputs.
            ValueError: A period type has been declared but inputs are not
                        period diagnostics.
            ValueError: Period diagnostics with overlapping periods.
            CoordinateNotFoundError: The input cubes contain no time
                                     coordinate.
            ValueError: Cubes contain multiple validity times.
            ValueError: The input cubes are ordered such that the initial time
                        cube has a later validity time than the final cube.
        """
        if not isinstance(cube_t0, iris.cube.Cube) or not isinstance(
            cube_t1, iris.cube.Cube
        ):
            msg = (
                "Inputs to TemporalInterpolation are not of type "
                "iris.cube.Cube, first input is type "
                "{}, second input is type {}".format(type(cube_t0), type(cube_t1))
            )
            raise TypeError(msg)

        try:
            (initial_time,) = iris_time_to_datetime(cube_t0.coord("time"))
            (final_time,) = iris_time_to_datetime(cube_t1.coord("time"))
        except CoordinateNotFoundError:
            msg = "Cube provided to TemporalInterpolation contains no time coordinate."
            raise CoordinateNotFoundError(msg)
        except ValueError:
            msg = (
                "Cube provided to TemporalInterpolation contains multiple "
                "validity times, only one expected."
            )
            raise ValueError(msg)

        if initial_time > final_time:
            raise ValueError(
                "TemporalInterpolation input cubes "
                "ordered incorrectly"
                ", with the final time being before the initial "
                "time."
            )

        cube_t0_bounds = cube_t0.coord("time").has_bounds()
        cube_t1_bounds = cube_t1.coord("time").has_bounds()
        if cube_t0_bounds + cube_t1_bounds == 1:
            raise ValueError(
                "Period and non-period diagnostics cannot be combined for"
                " temporal interpolation."
            )

        if cube_t0_bounds and not self.period_inputs:
            raise ValueError(
                "Interpolation of period diagnostics should be done using "
                "the appropriate period specifier (accumulation, min or max)."
            )

        if self.period_inputs:
            # Declaring period type requires the inputs be period diagnostics.
            if not cube_t0_bounds:
                raise ValueError(
                    "A period method has been declared for temporal "
                    "interpolation (max, min, or accumulation). Period "
                    "diagnostics must be provided. The input cubes have no "
                    "time bounds."
                )

            cube_interval = (
                cube_t1.coord("time").points[0] - cube_t0.coord("time").points[0]
            )
            (period,) = np.diff(cube_t1.coord("time").bounds[0])
            if not cube_interval == period:
                raise ValueError(
                    "The diagnostic provided represents the period "
                    f"{period / 3600} hours. The interval between the "
                    f"diagnostics is {cube_interval / 3600} hours. Temporal "
                    "interpolation can only be applied to a period "
                    "diagnostic provided at intervals that match the "
                    "diagnostic period such that all points in time are "
                    "captured by only one of the inputs and do not overlap."
                )

        time_list = self.construct_time_list(initial_time, final_time)

        # If the target output time is the same as the time at which the
        # trailing input is valid, just return it unchanged.
        if (
            len(time_list[0][1]) == 1
            and time_list[0][1][0] == cube_t1.coord("time").cell(0).point
        ):
            return CubeList([cube_t1])

        # If the units of the two cubes are degrees, assume we are dealing with
        # directions. Convert the directions to complex numbers so
        # interpolations (esp. the 0/360 wraparound) are handled in a sane
        # fashion.
        if cube_t0.units == "degrees" and cube_t1.units == "degrees":
            cube_t0.data = deg_to_complex(cube_t0.data)
            cube_t1.data = deg_to_complex(cube_t1.data)

        # Convert accumulations into rates to allow interpolation using trends
        # in the data and to accommodate non-uniform output intervals. This also
        # accommodates cube_t0 and cube_t1 representing different periods of
        # accumulation, for example where the forecast period interval changes
        # in an NWP model's output.
        if self.accumulation:
            cube_t0.data /= np.diff(cube_t0.coord("forecast_period").bounds[0])[0]
            period_reference = cube_t1.copy()
            cube_t1.data /= np.diff(cube_t1.coord("forecast_period").bounds[0])[0]

        cubes = CubeList([cube_t0, cube_t1])
        cube = MergeCubes()(cubes)

        interpolated_cube = cube.interpolate(time_list, iris.analysis.Linear())
        if cube_t0.units == "degrees" and cube_t1.units == "degrees":
            interpolated_cube.data = complex_to_deg(interpolated_cube.data)

        if self.period_inputs:
            # Add bounds to the time coordinates of the interpolated outputs
            # if the inputs were period diagnostics.
            self.add_bounds(cube_t0, interpolated_cube)

            # Apply suitable constraints to the returned values.
            # - accumulations are renormalised to ensure the period total is
            #   unchanged when broken into shorter periods.
            # - period maximums are enforced to not exceed the original
            #   maximum that occurred across the whole longer period.
            # - period minimums are enforced to not be below the original
            #   minimum that occurred across the whole longer period.
            if self.accumulation:
                self._calculate_accumulation(
                    cube_t0, period_reference, interpolated_cube
                )
            elif self.max:
                interpolated_cube.data = np.minimum(
                    cube_t1.data, interpolated_cube.data
                )
            elif self.min:
                interpolated_cube.data = np.maximum(
                    cube_t1.data, interpolated_cube.data
                )

        self.enforce_time_coords_dtype(interpolated_cube)
        interpolated_cubes = CubeList()
        if self.interpolation_method == "solar":
            interpolated_cubes = self.solar_interpolate(cube, interpolated_cube)
        elif self.interpolation_method == "daynight":
            interpolated_cubes = self.daynight_interpolate(interpolated_cube)
        elif self.interpolation_method == "google_film":
            interpolated_cubes = GoogleFilmInterpolation(
                self.model_path,
                self.scaling,
                self.clipping_bounds,
            ).process(cube[0], cube[1], interpolated_cube)
        else:
            for single_time in interpolated_cube.slices_over("time"):
                interpolated_cubes.append(single_time)

        return interpolated_cubes


class ForecastPeriodGapFiller(BasePlugin):
    """Fill gaps in forecast periods using temporal interpolation.

    This plugin identifies gaps in a sequence of forecast periods and fills
    them using temporal interpolation. When cluster_sources are configured,
    it can also identify forecast periods that should be regenerated (e.g.,
    when transitioning between forecast sources) even if they exist in the
    input data.

    The plugin will:
    1. Sort input cubes by forecast period
    2. Identify missing forecast periods (gaps)
    3. Optionally identify periods to regenerate based on cluster sources
    4. Use TemporalInterpolation to fill gaps
    5. Return a complete CubeList with all forecast periods
    """

    def __init__(
        self,
        interval_in_minutes: Optional[int] = None,
        interpolation_method: str = "linear",
        cluster_sources_attribute: Optional[str] = None,
        interpolation_window_in_hours: Optional[int] = None,
        model_path: Optional[str] = None,
        scaling: str = "minmax",
        clipping_bounds: Union[Tuple[float, float], List[float]] = (0.0, 1.0),
        **kwargs,
    ) -> None:
        """Initialize the ForecastPeriodGapFiller.

        Args:
            interval_in_minutes:
                The expected interval between forecast periods in minutes.
                Used to identify gaps in the sequence.
            interpolation_method:
                Method of interpolation to use.
                Options: linear, solar, daynight, google_film.
            cluster_sources_attribute:
                Name of cube attribute containing cluster sources dictionary.
                When provided with interpolation_window_in_hours, enables
                identification of forecast periods to regenerate at source transitions.
            interpolation_window_in_hours:
                Time window (in hours) as +/- range around forecast source transitions.
            model_path:
                Path to TensorFlow Hub module for Google FILM model
                (if using google_film).
            scaling:
                Scaling method for google_film interpolation (log10 or minmax).
            clipping_bounds:
                Bounds for clipping google_film interpolated data. Can be a tuple
                or list of two floats.
            **kwargs:
                Additional arguments passed to TemporalInterpolation.
        """
        self.interval_in_minutes = interval_in_minutes
        self.interpolation_method = interpolation_method
        self.cluster_sources_attribute = cluster_sources_attribute
        self.interpolation_window_in_hours = interpolation_window_in_hours
        self.model_path = model_path
        self.scaling = scaling
        # Convert clipping_bounds to tuple if it's a list
        self.clipping_bounds = (
            tuple(clipping_bounds)
            if isinstance(clipping_bounds, list)
            else clipping_bounds
        )
        self.kwargs = kwargs

    def _get_forecast_periods(self, cubelist: CubeList) -> List[int]:
        """Extract forecast periods from cubes in hours.

        Args:
            cubelist: List of cubes to extract forecast periods from.

        Returns:
            Sorted list of unique forecast periods in hours.
        """
        periods = set()
        for cube in cubelist:
            period_seconds = cube.coord("forecast_period").points[0]
            period_hours = int(round(period_seconds / 3600))
            periods.add(period_hours)
        return sorted(periods)

    def identify_gaps(self, cubelist: CubeList) -> List[int]:
        """Identify missing forecast periods that need filling.

        Args:
            cubelist: List of input cubes.

        Returns:
            List of forecast periods (in hours) that are missing.

        Raises:
            ValueError: If interval_in_minutes is not set.
        """
        if self.interval_in_minutes is None:
            raise ValueError(
                "interval_in_minutes must be set to identify gaps in forecast periods."
            )

        existing_periods = self._get_forecast_periods(cubelist)
        if len(existing_periods) < 2:
            return []

        # Calculate expected interval in hours
        interval_hours = self.interval_in_minutes / 60

        # Find all periods that should exist
        min_period = existing_periods[0]
        max_period = existing_periods[-1]
        expected_periods = []
        current = min_period
        while current <= max_period:
            expected_periods.append(int(round(current)))
            current += interval_hours

        # Find missing periods
        missing = [p for p in expected_periods if p not in existing_periods]
        return missing

    def parse_cluster_sources(self, cube: Cube) -> dict:
        """Parse the cluster sources dictionary from a cube attribute.

        Args:
            cube:
                A cube containing the cluster sources attribute.

        Returns:
            Dictionary mapping realization indices to their forecast sources
            and periods. Format: {realization_index: {source_name: [periods]}}

        Raises:
            ValueError: If the cluster_sources_attribute is not found on the cube.
            ValueError: If the cluster sources dictionary is not properly formatted.
        """
        if self.cluster_sources_attribute is None:
            return {}

        try:
            cluster_sources = cube.attributes[self.cluster_sources_attribute]
        except KeyError:
            raise ValueError(
                f"Attribute '{self.cluster_sources_attribute}' not found "
                f"on cube. Available attributes: "
                f"{list(cube.attributes.keys())}"
            )

        # Parse JSON string if needed
        if isinstance(cluster_sources, str):
            import json

            try:
                cluster_sources = json.loads(cluster_sources)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse cluster sources JSON: {e}")

        # Validate dictionary structure
        if not isinstance(cluster_sources, dict):
            raise ValueError(
                f"Cluster sources attribute must be a dictionary, "
                f"got {type(cluster_sources)}"
            )

        for real_idx, sources in cluster_sources.items():
            if not isinstance(sources, dict):
                raise ValueError(
                    f"Sources for realization {real_idx} must be a dictionary, "
                    f"got {type(sources)}"
                )
            for source_name, periods in sources.items():
                if not isinstance(periods, list):
                    raise ValueError(
                        f"Periods for source {source_name} in realization "
                        f"{real_idx} must be a list, got {type(periods)}"
                    )

        return cluster_sources

    def identify_source_transitions(
        self, cluster_sources: dict, realization_index: int
    ) -> List[int]:
        """Identify forecast source transitions for a given realization.

        Args:
            cluster_sources:
                Dictionary mapping realization indices to their forecast sources
                and periods.
            realization_index:
                The realization index to check for transitions.

        Returns:
            List of forecast periods immediately before a source transition.
            Only includes transitions where the source actually changes.
        """
        real_key = str(realization_index)
        if real_key not in cluster_sources:
            return []

        sources_dict = cluster_sources[real_key]

        # Sort sources by their periods to find transitions
        source_period_list = []
        for source_name, periods in sources_dict.items():
            for period in periods:
                source_period_list.append((period, source_name))

        source_period_list.sort()

        # Find transitions
        transitions = []
        for i in range(len(source_period_list) - 1):
            period_before, source_before = source_period_list[i]
            _, source_after = source_period_list[i + 1]

            # Only record if source changes
            if source_before != source_after:
                # Store the period_before as the transition point.
                transitions.append(period_before)

        return transitions

    def identify_periods_to_regenerate(
        self, cubelist: CubeList
    ) -> List[Tuple[int, int, int]]:
        """Identify periods to regenerate based on cluster source transitions.

        Args:
            cubelist: List of input cubes.

        Returns:
            List of tuples (transition_period, expected_t0, expected_t1) where
            transition_period is the forecast period at the source
            transition, expected_t0 is (transition - window), and
            expected_t1 is (transition + window).
        """
        if (
            self.cluster_sources_attribute is None
            or self.interpolation_window_in_hours is None
        ):
            return []

        periods_to_regenerate = []

        # Check first cube for cluster sources
        if not cubelist:
            return []

        first_cube = cubelist[0]
        if self.cluster_sources_attribute not in first_cube.attributes:
            return []

        # Parse cluster sources using self method
        cluster_sources = self.parse_cluster_sources(first_cube)
        if not cluster_sources:
            return []

        # Get all realization indices
        if first_cube.coords("realization"):
            realization_indices = first_cube.coord("realization").points
        else:
            return []

        # Find transitions for each realization
        seen_transitions = set()
        for real_idx in realization_indices:
            transitions = self.identify_source_transitions(
                cluster_sources, int(real_idx)
            )
            for trans_period in transitions:
                if trans_period not in seen_transitions:
                    expected_t0 = trans_period - self.interpolation_window_in_hours
                    expected_t1 = trans_period + self.interpolation_window_in_hours
                    periods_to_regenerate.append(
                        (trans_period, expected_t0, expected_t1)
                    )
                    seen_transitions.add(trans_period)

        return periods_to_regenerate

    def _get_cubes_for_period(self, cubelist: CubeList, period_hours: int) -> Cube:
        """Get cube(s) for a specific forecast period.

        Args:
            cubelist: List of cubes to search.
            period_hours: Forecast period in hours.

        Returns:
            Cube at the specified forecast period.

        Raises:
            ValueError: If no cube found at the specified period.
        """
        for cube in cubelist:
            cube_period = cube.coord("forecast_period").points[0] / 3600
            if abs(cube_period - period_hours) < 0.01:
                return cube
        raise ValueError(f"No cube found for forecast period T+{period_hours}")

    def _validate_input(self, cubelist: CubeList) -> None:
        """Validate that the input cubelist meets requirements.

        Args:
            cubelist: List of cubes to validate.

        Raises:
            ValueError: If cubelist is empty or has fewer than 2 cubes.
            ValueError: If cubes don't have forecast_period coordinate.
        """
        if not cubelist or len(cubelist) < 2:
            raise ValueError(
                "ForecastPeriodGapFiller requires at least 2 cubes in the "
                "input CubeList."
            )

        # Verify all cubes have forecast_period
        for cube in cubelist:
            if not cube.coords("forecast_period"):
                raise ValueError(
                    "All cubes must have a forecast_period coordinate for gap filling."
                )

    def _create_gap_filling_tasks(
        self, missing_periods: List[int], sorted_cubelist: CubeList
    ) -> List[Tuple[str, int, int, int]]:
        """Create interpolation tasks for missing forecast periods.

        Args:
            missing_periods: List of forecast periods (in hours) that are missing.
            sorted_cubelist: Sorted list of cubes by forecast period.

        Returns:
            List of tuples (task_type, target_period, t0_period, t1_period)
            for gap filling tasks.
        """
        interpolation_tasks = []
        existing_periods = self._get_forecast_periods(sorted_cubelist)

        for period in missing_periods:
            # Find appropriate bounding cubes
            lower_periods = [p for p in existing_periods if p < period]
            upper_periods = [p for p in existing_periods if p > period]

            if lower_periods and upper_periods:
                t0_period = max(lower_periods)
                t1_period = min(upper_periods)
                interpolation_tasks.append(("gap", period, t0_period, t1_period))

        return interpolation_tasks

    def _create_regeneration_tasks(
        self,
        periods_to_regenerate: List[Tuple[int, int, int]],
        sorted_cubelist: CubeList,
    ) -> List[Tuple[str, int, int, int]]:
        """Create interpolation tasks for periods to regenerate.

        Args:
            periods_to_regenerate:
                List of tuples (transition_period, expected_t0, expected_t1).
            sorted_cubelist: Sorted list of cubes by forecast period.

        Returns:
            List of tuples (task_type, target_period, t0_period, t1_period)
            for regeneration tasks.
        """
        interpolation_tasks = []
        existing_periods = self._get_forecast_periods(sorted_cubelist)

        for trans_period, expected_t0, expected_t1 in periods_to_regenerate:
            # Check if the required boundary cubes exist
            if expected_t0 in existing_periods and expected_t1 in existing_periods:
                interpolation_tasks.append(
                    ("regenerate", trans_period, expected_t0, expected_t1)
                )

        return interpolation_tasks

    def _calculate_target_time(
        self, cube_t0: Cube, target_period: int, t0_period: int
    ) -> datetime:
        """Calculate the target time for interpolation.

        Args:
            cube_t0: The cube at the earlier forecast period.
            target_period: The target forecast period in hours.
            t0_period: The earlier forecast period in hours.

        Returns:
            The target time as a datetime object.
        """
        time_t0 = iris_time_to_datetime(cube_t0.coord("time"))[0]
        target_offset = (target_period - t0_period) * 3600
        target_time = time_t0 + timedelta(seconds=target_offset)
        return target_time

    def _interpolate_single_period(
        self,
        interpolator: TemporalInterpolation,
        sorted_cubelist: CubeList,
        target_period: int,
        t0_period: int,
        t1_period: int,
    ) -> CubeList:
        """Interpolate a single forecast period.

        Args:
            interpolator: The TemporalInterpolation plugin to use.
            sorted_cubelist: Sorted list of cubes by forecast period.
            target_period: The target forecast period in hours.
            t0_period: The earlier forecast period in hours.
            t1_period: The later forecast period in hours.

        Returns:
            CubeList containing the interpolated cube for the target period.
        """
        cube_t0 = self._get_cubes_for_period(sorted_cubelist, t0_period)
        cube_t1 = self._get_cubes_for_period(sorted_cubelist, t1_period)

        # Calculate target time
        target_time = self._calculate_target_time(cube_t0, target_period, t0_period)

        # Set interpolation times
        interpolator.times = [target_time]

        # Perform interpolation
        interpolated = interpolator.process(cube_t0, cube_t1)

        # Filter to only return the target period cube
        result_cubes = CubeList()
        for cube in interpolated:
            # Check if this is the target period (not the t1 boundary)
            cube_period = cube.coord("forecast_period").points[0] / 3600
            if abs(cube_period - target_period) < 0.01:
                result_cubes.append(cube)

        return result_cubes

    def _assemble_final_cubelist(
        self,
        sorted_cubelist: CubeList,
        result_cubes: CubeList,
        periods_to_exclude: set,
    ) -> CubeList:
        """Assemble the final cubelist by combining interpolated and original cubes.

        Args:
            sorted_cubelist: Original sorted list of cubes.
            result_cubes: CubeList of interpolated cubes.
            periods_to_exclude: Set of forecast periods to exclude from originals.

        Returns:
            Final sorted CubeList with all forecast periods.
        """
        # Add original cubes that aren't being regenerated
        for cube in sorted_cubelist:
            cube_period = cube.coord("forecast_period").points[0] / 3600
            period_hours = int(round(cube_period))
            if period_hours not in periods_to_exclude:
                result_cubes.append(cube)

        # Sort final result by forecast period
        result_cubes = CubeList(
            sorted(
                result_cubes,
                key=lambda c: c.coord("forecast_period").points[0],
            )
        )

        return result_cubes

    def process(self, *cubes: Union[Cube, CubeList]) -> Cube:
        """Fill gaps in forecast period sequence.

        Args:
            cubes:
                One or more cubes with potentially missing forecast periods.
                Can be:
                - A single Cube with a forecast_period dimension (will be sliced)
                - Multiple Cube arguments representing different forecast periods
                - A single CubeList containing multiple forecast periods
                All cubes should have the same validity time coordinate structure
                and dimensions (except for forecast_period and time).

        Returns:
            A single merged Cube with gaps filled using temporal interpolation.
            The cube will have time as a dimension coordinate.

        Raises:
            ValueError: If cubelist is empty or has fewer than 2 cubes.
            ValueError: If cubes don't have forecast_period coordinate.
        """
        # Handle variable arguments - convert to single CubeList
        if len(cubes) == 1:
            cubelist = cubes[0]
            # Convert single Cube to CubeList if necessary
            if isinstance(cubelist, Cube):
                # Slice over the appropriate coordinate
                if cubelist.coords("forecast_period", dim_coords=True):
                    # Cube has forecast_period as a dimension - slice it
                    cubelist = CubeList(cubelist.slices_over("forecast_period"))
                elif (
                    cubelist.coords("time", dim_coords=True)
                    and cubelist.coord("time").shape[0] > 1
                ):
                    # Cube has time as a dimension - slice it
                    cubelist = CubeList(cubelist.slices_over("time"))
                else:
                    # Single cube without sliceable dimension
                    cubelist = CubeList([cubelist])
        else:
            # Multiple cubes passed as separate arguments
            cubelist = CubeList(cubes)

        # Validate input
        self._validate_input(cubelist)

        # Sort cubelist by forecast period
        sorted_cubelist = CubeList(
            sorted(cubelist, key=lambda c: c.coord("forecast_period").points[0])
        )

        # Identify gaps and periods to regenerate
        missing_periods = self.identify_gaps(sorted_cubelist)
        periods_to_regenerate = self.identify_periods_to_regenerate(sorted_cubelist)

        # Create interpolation tasks
        interpolation_tasks = self._create_gap_filling_tasks(
            missing_periods, sorted_cubelist
        )
        interpolation_tasks.extend(
            self._create_regeneration_tasks(periods_to_regenerate, sorted_cubelist)
        )

        # If no interpolation needed, merge and return original
        if not interpolation_tasks:
            return MergeCubes()(sorted_cubelist)

        # Create TemporalInterpolation plugin
        interpolator = TemporalInterpolation(
            times=[],  # We'll provide explicit times
            interpolation_method=self.interpolation_method,
            model_path=self.model_path,
            scaling=self.scaling,
            clipping_bounds=self.clipping_bounds,
            **self.kwargs,
        )

        # Perform interpolations
        result_cubes = CubeList()
        periods_to_exclude = set()

        for task_type, target_period, t0_period, t1_period in interpolation_tasks:
            interpolated_cubes = self._interpolate_single_period(
                interpolator,
                sorted_cubelist,
                target_period,
                t0_period,
                t1_period,
            )
            result_cubes.extend(interpolated_cubes)

            if task_type == "regenerate":
                # Mark original for exclusion
                periods_to_exclude.add(target_period)

        # Assemble final cubelist
        final_cubelist = self._assemble_final_cubelist(
            sorted_cubelist, result_cubes, periods_to_exclude
        )

        # Merge cubes into a single cube with time as a coordinate
        return MergeCubes()(final_cubelist)


class GoogleFilmInterpolation:
    """Class to perform temporal interpolation using the Google FILM model.

    The model is expected to be a TensorFlow Hub module that takes as input two
    images and a time point between 0 and 1, and outputs an interpolated image.

    The input cubes are expected to have the same spatial dimensions and
    coordinate system. The output cube will have the same metadata as cube1.
    """

    def __init__(
        self,
        model_path: str,
        scaling: str = "minmax",
        clipping_bounds: Tuple[float, float] = (0.0, 1.0),
        cluster_sources_attribute: Optional[str] = None,
        interpolation_window_in_hours: Optional[int] = None,
    ) -> None:
        """
        Args:
            model_path:
                Path to the TensorFlow Hub module for the Google FILM model.
            scaling:
                Scaling method to apply to the data before interpolation.
                Supported methods are "log10" and "minmax".
            clipping_bounds:
                A tuple specifying the (min, max) bounds to which to clip
                the interpolated data. Use None for no bound in that direction.
            cluster_sources_attribute:
                Name of a cube attribute containing a dictionary that maps
                realization indices to their forecast sources and periods.
                When provided with interpolation_window_in_hours, enables
                interpolation across forecast source transitions.
            interpolation_window_in_hours:
                The time window (in hours) to use as a +/- range around forecast
                source transition points. Used with cluster_sources_attribute.
        """
        self.model_path = model_path
        self.scaling = scaling
        self.clipping_bounds = clipping_bounds
        self.cluster_sources_attribute = cluster_sources_attribute
        self.interpolation_window_in_hours = interpolation_window_in_hours
        self.cluster_sources_attribute = cluster_sources_attribute
        self.interpolation_window_in_hours = interpolation_window_in_hours

    def load_model(self, model_path: str) -> Any:
        """Load the TensorFlow Hub model.

        Args:
            model_path:
                Path to the TensorFlow Hub module for the Google FILM model.
        Returns:
            The loaded TensorFlow Hub model.
        """
        # Apply monkey patch before importing anything TensorFlow-related
        # We need to patch all possible import paths that tf_keras might use
        # Related to https://github.com/keras-team/tf-keras/issues/257
        try:
            import tensorflow as tf

            # Patch all the different ways tensorflow's __internal__ can be accessed
            if hasattr(tf.__internal__, "register_call_context_function"):
                func = tf.__internal__.register_call_context_function
                tf.__internal__.register_load_context_function = func
                # Also patch the compat.v2 path that tf_keras uses
                if hasattr(tf.compat, "v2"):
                    tf.compat.v2.__internal__.register_load_context_function = func
                # And the _api.v2.compat.v2 path
                import tensorflow._api.v2.compat.v2 as tf_api

                tf_api.__internal__.register_load_context_function = func
        except (ImportError, AttributeError):
            pass

        import tensorflow_hub as hub

        return hub.load(model_path)

    def apply_scaling(self, cube1: Cube, cube2: Cube, scaling: str) -> None:
        """Apply scaling to the input cubes before interpolation.
        Args:
            cube1:
                The first input cube.
            cube2:
                The second input cube.
            scaling:
                Scaling method to apply. Supported methods are "log10" and "minmax".
        """
        if scaling == "log10":
            cube1.data = np.log10(cube1.data + 1)
            cube2.data = np.log10(cube2.data + 1)
        elif scaling == "minmax":
            min_val = min(cube1.data.min(), cube2.data.min())
            max_val = max(cube1.data.max(), cube2.data.max())
            cube1.data = (cube1.data - min_val) / (max_val - min_val)
            cube2.data = (cube2.data - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"Unsupported scaling method: {scaling}")

    def reverse_scaling(
        self, cube: Cube, cube1: Cube, cube2: Cube, scaling: str
    ) -> None:
        """Reverse scaling on the interpolated cube after interpolation.

        Args:
            cube:
                The interpolated cube.
            cube1:
                The first input cube.
            cube2:
                The second input cube.
            scaling:
                Scaling method to reverse. Supported methods are "log10" and "minmax".
        """
        if scaling == "log10":
            cube.data = 10**cube.data - 1
        elif scaling == "minmax":
            min_val = min(cube1.data.min(), cube2.data.min())
            max_val = max(cube1.data.max(), cube2.data.max())
            cube.data = cube.data * (max_val - min_val) + min_val
        else:
            raise ValueError(f"Unsupported scaling method: {scaling}")

    def apply_clipping(self, interpolated: Cube) -> None:
        """Clip the interpolated cube data to be within the bounds of the
        input cubes.

        Args:
            interpolated:
                The interpolated cube.
        """
        interpolated.data = np.clip(
            interpolated.data, self.clipping_bounds[0], self.clipping_bounds[1]
        )

    def run_google_film(
        self, cube1: Cube, cube2: Cube, model: Any, time_point: float
    ) -> Cube:
        """Run the Google FILM model to interpolate between two cubes.

        Args:
            cube1:
                The first input cube.
            cube2:
                The second input cube.
            model:
                The loaded TensorFlow Hub model.
            time_point:
                A float between 0 and 1 indicating the interpolation point.

        Returns:
            The interpolated cube.
        """
        # Stack the data 3 times to simulate the RGB data the model expects.
        # Convert to float32 as required by the model
        image1 = np.stack([cube1.data] * 3, axis=-1).astype(np.float32)
        image2 = np.stack([cube2.data] * 3, axis=-1).astype(np.float32)

        inputs = {
            "time": np.array([[time_point]], dtype=np.float32),  # Shape (1, 1)
            "x0": np.expand_dims(
                image1, axis=0
            ),  # adding the batch dimension to the image
            "x1": np.expand_dims(
                image2, axis=0
            ),  # adding the batch dimension to the image
        }

        frame = model(inputs)
        # Handle both TensorFlow tensors and numpy arrays (for testing)
        result_data = frame["image"][0, ..., 0]
        if hasattr(result_data, "numpy"):
            result_data = result_data.numpy()
        interpolated = cube1.copy(data=result_data)

        return interpolated

    def process(
        self, cube1: Cube, cube2: Cube, template_interpolated_cube: Cube
    ) -> CubeList:
        """Perform temporal interpolation between two cubes using the Google FILM model.

        Args:
            cube1:
                The first input cube (at time t=0).
            cube2:
                The second input cube (at time t=1).
            template_interpolated_cube:
                A cube containing the linearly interpolated data with the correct
                metadata for the output times.

        Returns:
            A CubeList containing the interpolated cubes at the specified times.

        Raises:
            ValueError: If cube1 or cube2 do not have realization coordinates.
            ValueError: If cube1 and cube2 have different numbers of realizations.
        """
        # Validate that both cubes have realization coordinates
        if not cube1.coords("realization"):
            raise ValueError(
                "cube1 must have a realization coordinate for GoogleFilmInterpolation."
            )
        if not cube2.coords("realization"):
            raise ValueError(
                "cube2 must have a realization coordinate for GoogleFilmInterpolation."
            )

        # Validate that both cubes have the same number of realizations
        n_realizations_1 = len(cube1.coord("realization").points)
        n_realizations_2 = len(cube2.coord("realization").points)
        if n_realizations_1 != n_realizations_2:
            raise ValueError(
                f"Input cubes must have the same number of realizations. "
                f"cube1 has {n_realizations_1} realizations, "
                f"cube2 has {n_realizations_2} realizations."
            )

        model = self.load_model(self.model_path)

        # Store original data for reverting scaling
        cube1_orig = cube1.copy()
        cube2_orig = cube2.copy()

        self.apply_scaling(cube1, cube2, self.scaling)

        # Calculate time fractions for each target time
        t0 = cube1.coord("time").points[0]
        t1 = cube2.coord("time").points[0]
        time_range = t1 - t0

        interpolated_cubes = CubeList()
        for template_slice in template_interpolated_cube.slices_over("time"):
            # Get the target time from the template
            target_seconds = template_slice.coord("time").points[0]

            # Calculate fraction (0 to 1) between the two input times
            time_fraction = (target_seconds - t0) / time_range

            realization_slices = CubeList([])
            for cube1_slice, cube2_slice in zip(
                cube1.slices_over("realization"), cube2.slices_over("realization")
            ):
                temporary_interpolated_cube = self.run_google_film(
                    cube1_slice, cube2_slice, model, time_fraction
                )
                realization_slices.append(temporary_interpolated_cube)
            temporary_interpolated_cube = realization_slices.merge_cube()

            interpolated_cube = template_slice.copy()
            interpolated_cube.data = temporary_interpolated_cube.data

            # Apply clipping and reverse scaling
            self.apply_clipping(interpolated_cube)
            self.reverse_scaling(
                interpolated_cube, cube1_orig, cube2_orig, self.scaling
            )

            interpolated_cubes.append(interpolated_cube)

        return interpolated_cubes


class DurationSubdivision:
    """Subdivide a duration diagnostic, e.g. sunshine duration, into
    shorter periods, optionally applying a night mask to ensure that
    quantities defined only in the day or night are not spread into
    night or day periods respectively.

    This is a very simple approach. In the case of sunshine duration
    the duration is divided up evenly across the short periods defined
    by the fidelity argument. These are then optionally masked to zero
    for chosen periods (day or night). Values in the non-zeroed periods
    are then renormalised relative to the original period total, such
    that the total across the whole period ought to equal the original. This
    is not always possible as the night mask applied is simpler than e.g. the
    radiation scheme impact on a 3D orography. As such the renormalisation
    could yield durations longer than the fidelity period in each
    non-zeroed period as it tries to allocate e.g. 5 hours of sunlight
    across 4 non-zeroed hours. This is not physical, so the renormalisation
    is partnered with a clip that limits the duration allocated to the
    renormalised periods to not exceed their length. The result of this
    is that the original sunshine durations cannot be recovered for points
    that are affected. Instead the calculated night mask is limiting the
    accuracy to allow the subdivision to occur. This is the cost of this
    method.

    Note that this method cannot account for any weather impacts e.g. cloud
    that is affecting the sunshine duration in a period. If a 6-hour period is
    split into three 2-hour periods the split will be even regardless of
    when thick cloud might occur.
    """

    def __init__(
        self,
        target_period: int,
        fidelity: int,
        night_mask: bool = True,
        day_mask: bool = False,
    ):
        """Define the length of the target periods to be constructed and the
        intermediate fidelity. This fidelity is the length of the shorter
        periods into which the data is split and from which the target periods
        are constructed. A shorter fidelity period allows the time dependent
        day or night masks to be applied more accurately.

        Args:
            target_period:
                The time period described by the output cubes in seconds.
                The data will be reconstructed into non-overlapping periods.
                The target_period must be a factor of the original period.
            fidelity:
                The shortest increment in seconds into which the input periods are
                divided and to which the night mask is applied. The
                target periods are reconstructed from these shorter periods.
                Shorter fidelity periods better capture where the day / night
                discriminator falls.
            night_mask:
                If true, points that fall at night are zeroed and duration
                reallocated to day time periods as much as possible.
            day_mask:
                If true, points that fall in the day time are zeroed and
                duration reallocated to night time periods as much as possible.
        Raises:
            ValueError: If target_period and / or fidelity are not positive integers.
            ValueError: If day and night mask options are both set True.
        """
        for item in [target_period, fidelity]:
            if item <= 0:
                raise ValueError(
                    "Target period and fidelity must be a positive integer "
                    "numbers of seconds. Currently set to "
                    f"target_period: {target_period}, fidelity: {fidelity}"
                )

        self.target_period = target_period
        self.fidelity = fidelity
        if night_mask and day_mask:
            raise ValueError(
                "Only one or neither of night_mask and day_mask may be set to True"
            )
        elif not night_mask and not day_mask:
            self.mask_value = None
        else:
            self.mask_value = 0 if night_mask else 1

    @staticmethod
    def cube_period(cube: Cube) -> int:
        """Return the time period of the cube in seconds.

        Args:
            cube:
                The cube for which the period is to be returned.
        Return:
            period:
                Period of cube time coordinate in seconds.
        """
        (period,) = np.diff(cube.coord("time").bounds[0])
        return period

    def allocate_data(self, cube: Cube, period: int) -> Cube:
        """Allocate fractions of the original cube duration diagnostic to
        shorter fidelity periods with metadata that describes these shorter
        periods appropriately. The fidelity period cubes will be merged to
        form a cube with a longer time dimension. This cube will be returned
        and used elsewhere to construct the target period cubes.

        Args:
            cube:
                The original period cube from which duration data will be
                taken and divided up.
            period:
                The period of the input cube in seconds.
        Returns:
            A cube, with a time dimension, that contains the subdivided data.
        """
        # Split the whole period duration into allocations for each fidelity
        # period.
        intervals = period // self.fidelity
        interval_data = (cube.data / intervals).astype(cube.data.dtype)

        daynightplugin = DayNightMask()
        start_time, _ = cube.coord("time").bounds.flatten()

        interpolated_cubes = iris.cube.CubeList()

        for i in range(intervals):
            interval_cube = cube.copy(data=interval_data.copy())
            interval_start = start_time + i * self.fidelity
            interval_end = start_time + (i + 1) * self.fidelity

            interval_cube.coord("time").points = np.array(
                [interval_end], dtype=np.int64
            )
            interval_cube.coord("time").bounds = np.array(
                [[interval_start, interval_end]], dtype=np.int64
            )

            if self.mask_value is not None:
                daynight_mask = daynightplugin(interval_cube).data
                daynight_mask = np.broadcast_to(daynight_mask, interval_cube.shape)
                interval_cube.data[daynight_mask == self.mask_value] = 0.0
            interpolated_cubes.append(interval_cube)

        return interpolated_cubes.merge_cube()

    @staticmethod
    def renormalisation_factor(cube: Cube, fidelity_period_cube: Cube) -> np.ndarray:
        """Sum up the total of the durations distributed amongst the fidelity
        period cubes following the application of any masking. These are
        then used with the durations in the unsubdivided original data to
        calculate a factor to restore the correct totals; note that where
        clipping plays a role the original totals may not be restored.

        Args:
            cube:
                The original period cube of duration data.
            fidelity_period_cube:
                The cube of fidelity period durations (the original durations
                divided up into shorter fidelity periods).
        Returns:
            factor:
                An array of factors that can be used to multiply up the
                fidelity period durations such that when the are summed up
                they are equal to the original durations.
        """
        retotal = fidelity_period_cube.collapsed("time", iris.analysis.SUM)
        factor = cube.data / retotal.data
        # Masked points indicate divide by 0, set these points to 0. Also handle
        # a case in which there is no masking on the factor array.
        try:
            factor = factor.filled(0)
        except AttributeError:
            factor[factor == np.inf] = 0

        return factor

    def construct_target_periods(self, fidelity_period_cube: Cube) -> Cube:
        """Combine the short fidelity period cubes into cubes that describe
        the target period.

        Args:
            fidelity_period_cube:
                The short fidelity period cubes from which the target periods
                are constructed.
        Returns:
            A cube containing the target period data with a time dimension
            with an entry for each target period. These periods combined span
            the original cube's period.
        """
        new_period_cubes = iris.cube.CubeList()

        interval = timedelta(seconds=self.target_period)
        start_time = fidelity_period_cube.coord("time").cell(0).bound[0]
        end_time = fidelity_period_cube.coord("time").cell(-1).bound[-1]
        while start_time < end_time:
            period_constraint = iris.Constraint(
                time=lambda cell: start_time <= cell.bound[0] < start_time + interval
            )
            components = fidelity_period_cube.extract(period_constraint)
            component_cube = components.collapsed("time", iris.analysis.SUM)
            enforce_time_point_standard(component_cube)
            new_period_cubes.append(component_cube)
            start_time += interval
        # The cycle times are already the same. This code will recalculate
        # the forecasts periods relative to the cycletime for each of our
        # extracted shorter duration cubes.
        cycle_time = fidelity_period_cube.coord("forecast_reference_time").cell(0).point

        new_period_cubes = unify_cycletime(new_period_cubes, cycle_time)
        return new_period_cubes.merge_cube()

    def process(self, cube: Cube) -> Cube:
        """Create target period duration diagnostics from the original duration
        diagnostic data.

        Args:
            cube:
                The original duration diagnostic cube.
        Returns:
            A cube containing the target period data with a time dimension
            with an entry for each period. These periods combined span the
            original cube's period.
        Raises:
            ValueError: The target period is not a factor of the input period.
            ValueError: The fidelity period is not less than or equal to the
                        target period.
        """
        period = self.cube_period(cube)

        # If the input cube period matches the target period return it.
        if period == self.target_period:
            return cube

        if period / self.target_period % 1 != 0:
            raise ValueError(
                "The target period must be a factor of the original period "
                "of the input cube and the target period must be <= the input "
                "period. "
                f"Input period: {period}, target period: {self.target_period}"
            )
        if self.fidelity > self.target_period:
            raise ValueError(
                "The fidelity period must be less than or equal to the target period."
            )
        # Ensure that the cube is already self-consistent and does not include
        # any durations that exceed the period described. This is mostly to
        # handle grib packing errors for ECMWF data.
        cube.data = np.clip(cube.data, 0, period, dtype=cube.data.dtype)

        fidelity_period_cube = self.allocate_data(cube, period)
        factor = self.renormalisation_factor(cube, fidelity_period_cube)

        # Apply clipping to limit these values to the maximum possible
        # duration that can be contained within the period.
        fidelity_period_cube = fidelity_period_cube.copy(
            data=np.clip(fidelity_period_cube.data * factor, 0, self.fidelity)
        )

        return self.construct_target_periods(fidelity_period_cube)
