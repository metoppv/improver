# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Class for Temporal Interpolation calculations."""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import iris
import numpy as np
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from numpy import ndarray

from improver import BasePlugin
from improver.metadata.constants import FLOAT_DTYPE
from improver.metadata.constants.time_types import TIME_COORDS
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

        Raises:
            ValueError: If neither interval_in_minutes nor times are set.
            ValueError: If both interval_in_minutes and times are both set.
            ValueError: If interpolation method not in known list.
            ValueError: If multiple period diagnostic kwargs are set True.
            ValueError: A period diagnostic is being interpolated with a method
                        not found in the period_interpolation_methods list.
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
        known_interpolation_methods = ["linear", "solar", "daynight"]
        if interpolation_method not in known_interpolation_methods:
            raise ValueError(
                "TemporalInterpolation: Unknown interpolation method {}. ".format(
                    interpolation_method
                )
            )
        self.interpolation_method = interpolation_method
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
                    f"Currently selected method is: {self.interpolation_method}."
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
        else:
            for single_time in interpolated_cube.slices_over("time"):
                interpolated_cubes.append(single_time)

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
        interval_data = cube.data / intervals

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
            component_cube.coord("time").points = component_cube.coord("time").bounds[
                0
            ][-1]
            new_period_cubes.append(component_cube)
            start_time += interval

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
        """
        period = self.cube_period(cube)

        if period / self.target_period % 1 != 0:
            raise ValueError(
                "The target period must be a factor of the original period "
                "of the input cube and the target period must be <= the input "
                "period. "
                f"Input period: {period}, target period: {self.target_period}"
            )
        if self.fidelity > self.target_period:
            raise ValueError(
                "The fidelity period must be less than or equal to the "
                "target period."
            )
        # Ensure that the cube is already self-consistent and does not include
        # any durations that exceed the period described. This is mostly to
        # handle grib packing errors for ECMWF data.
        cube.data = np.clip(cube.data, 0, period)
        # If the input cube period matches the target period return it.
        if period == self.target_period:
            return cube

        fidelity_period_cube = self.allocate_data(cube, period)
        factor = self.renormalisation_factor(cube, fidelity_period_cube)

        # Apply clipping to limit these values to the maximum possible
        # duration that can be contained within the period.
        fidelity_period_cube = fidelity_period_cube.copy(
            data=np.clip(fidelity_period_cube.data * factor, 0, self.fidelity)
        )

        return self.construct_target_periods(fidelity_period_cube)
