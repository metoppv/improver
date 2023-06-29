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
"""
Functions to set up variable, multi-realization, percentile and probability
cubes for unit tests.  Standardises time units and spatial coordinates,
including coordinate order expected by IMPROVER plugins.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import iris
import numpy as np
from cf_units import Unit, date2num
from iris.coords import Coord, DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from numpy import ndarray
from numpy.ma.core import MaskedArray

from improver.grids import GRID_COORD_ATTRIBUTES
from improver.metadata.check_datatypes import check_mandatory_standards
from improver.metadata.constants.mo_attributes import MOSG_GRID_DEFINITION
from improver.metadata.constants.time_types import TIME_COORDS
from improver.metadata.forecast_times import forecast_period_coord

DIM_COORD_ATTRIBUTES = {
    "realization": {"units": "1"},
    "height": {"units": "m", "attributes": {"positive": "up"}},
    "pressure": {"units": "Pa", "attributes": {"positive": "down"}},
}


def construct_yx_coords(
    ypoints: int,
    xpoints: int,
    spatial_grid: str,
    grid_spacing: Optional[float] = None,
    domain_corner: Optional[Tuple[float, float]] = None,
) -> Tuple[DimCoord, DimCoord]:
    """
    Construct y/x spatial dimension coordinates

    Args:
        ypoints:
            Number of grid points required along the y-axis
        xpoints:
            Number of grid points required along the x-axis
        spatial_grid:
            Specifier to produce either a "latlon" or "equalarea" grid
        grid_spacing:
            Grid resolution (degrees for latlon or metres for equalarea). If not provided,
            defaults to 10 degrees for "latlon" grid or 2000 metres for "equalarea" grid
        domain_corner:
            Bottom left corner of grid domain (y,x) (degrees for latlon or metres for equalarea).
            If not provided, a grid is created centred around (0,0).

    Returns:
        Tuple containing y and x iris.coords.DimCoords
    """
    if spatial_grid not in GRID_COORD_ATTRIBUTES.keys():
        raise ValueError("Grid type {} not recognised".format(spatial_grid))

    if grid_spacing is None:
        grid_spacing = GRID_COORD_ATTRIBUTES[spatial_grid]["default_grid_spacing"]

    if domain_corner is None:
        domain_corner = _set_domain_corner(ypoints, xpoints, grid_spacing)
    y_array, x_array = _create_yx_arrays(ypoints, xpoints, domain_corner, grid_spacing)

    y_coord = DimCoord(
        y_array,
        GRID_COORD_ATTRIBUTES[spatial_grid]["yname"],
        units=GRID_COORD_ATTRIBUTES[spatial_grid]["units"],
        coord_system=GRID_COORD_ATTRIBUTES[spatial_grid]["coord_system"],
    )
    x_coord = DimCoord(
        x_array,
        GRID_COORD_ATTRIBUTES[spatial_grid]["xname"],
        units=GRID_COORD_ATTRIBUTES[spatial_grid]["units"],
        coord_system=GRID_COORD_ATTRIBUTES[spatial_grid]["coord_system"],
    )

    # add bounds on spatial coordinates
    if ypoints > 1:
        y_coord.guess_bounds()
    if xpoints > 1:
        x_coord.guess_bounds()

    return y_coord, x_coord


def _create_yx_arrays(
    ypoints: int, xpoints: int, domain_corner: Tuple[float, float], grid_spacing: float,
) -> Tuple[ndarray, ndarray]:
    """
    Creates arrays for constructing y and x DimCoords.

    Args:
        ypoints
        xpoints
        domain_corner
        grid_spacing

    Returns:
        Tuple containing arrays of y and x coordinate values
    """
    y_stop = domain_corner[0] + (grid_spacing * (ypoints - 1))
    x_stop = domain_corner[1] + (grid_spacing * (xpoints - 1))

    y_array = np.linspace(domain_corner[0], y_stop, ypoints, dtype=np.float32)
    x_array = np.linspace(domain_corner[1], x_stop, xpoints, dtype=np.float32)

    return y_array, x_array


def _set_domain_corner(
    ypoints: int, xpoints: int, grid_spacing: float
) -> Tuple[float, float]:
    """
    Set domain corner to create a grid around 0,0.

    Args:
        ypoints
        xpoints
        grid_spacing

    Returns:
        (y,x) values of the bottom left corner of the domain
    """
    y_start = 0 - ((ypoints - 1) * grid_spacing) / 2
    x_start = 0 - ((xpoints - 1) * grid_spacing) / 2

    return y_start, x_start


def _create_time_point(time: datetime) -> int:
    """Returns a coordinate point with appropriate units and datatype
    from a datetime.datetime instance.

    Args:
        time

    Returns:
        Returns coordinate point as datatype specified in TIME_COORDS["time"]
    """
    coord_spec = TIME_COORDS["time"]
    point = date2num(time, coord_spec.units, coord_spec.calendar)
    return np.around(point).astype(coord_spec.dtype)


def construct_scalar_time_coords(
    time: datetime,
    time_bounds: Optional[List[datetime]],
    frt: Optional[datetime] = None,
    blend_time: Optional[datetime] = None,
) -> List[Tuple[DimCoord, None]]:
    """
    Construct scalar time coordinates as aux_coord list

    Args:
        time:
            Single time point
        time_bounds:
            Lower and upper bound on time point, if required
        frt:
            Single forecast reference time point. Either this or blend_time is required.
        blend_time:
            Single blend time point. Either this or frt is required. Both may be supplied.

    Returns:
        List of iris.coords.DimCoord instances with the associated "None"
        dimension (format required by iris.cube.Cube initialisation).
    """
    # generate time coordinate points
    time_point_seconds = _create_time_point(time)
    if frt:
        reference_point_seconds = _create_time_point(frt)
    elif blend_time:
        reference_point_seconds = _create_time_point(blend_time)
    else:
        raise ValueError(
            "Cannot create forecast_period without either a forecast reference time "
            "or a blend time."
        )

    fp_coord_spec = TIME_COORDS["forecast_period"]
    if time_point_seconds < reference_point_seconds:
        raise ValueError("Cannot set up cube with negative forecast period")
    fp_point_seconds = (time_point_seconds - reference_point_seconds).astype(
        fp_coord_spec.dtype
    )

    # parse bounds if required
    if time_bounds is not None:
        lower_bound = _create_time_point(time_bounds[0])
        upper_bound = _create_time_point(time_bounds[1])
        bounds = (min(lower_bound, upper_bound), max(lower_bound, upper_bound))
        if time_point_seconds < bounds[0] or time_point_seconds > bounds[1]:
            raise ValueError(
                "Time point {} not within bounds {}-{}".format(
                    time, time_bounds[0], time_bounds[1]
                )
            )
        fp_bounds = np.array(
            [[bounds[0] - reference_point_seconds, bounds[1] - reference_point_seconds]]
        ).astype(fp_coord_spec.dtype)
    else:
        bounds = None
        fp_bounds = None

    # create coordinates
    time_coord = DimCoord(
        time_point_seconds, "time", bounds=bounds, units=TIME_COORDS["time"].units
    )
    coord_dims = [(time_coord, None)]
    if frt:
        frt_coord = DimCoord(
            reference_point_seconds,
            "forecast_reference_time",
            units=TIME_COORDS["forecast_reference_time"].units,
        )
        coord_dims.append((frt_coord, None))
    if blend_time:
        blend_time_coord = DimCoord(
            _create_time_point(blend_time),
            long_name="blend_time",
            units=TIME_COORDS["blend_time"].units,
        )
        coord_dims.append((blend_time_coord, None))
    fp_coord = DimCoord(
        fp_point_seconds, "forecast_period", bounds=fp_bounds, units=fp_coord_spec.units
    )
    coord_dims.append((fp_coord, None))

    return coord_dims


def _create_dimension_coord(
    coord_array: Optional[List[float]],
    data_length: int,
    coord_name: str,
    **kwargs: Any,
) -> DimCoord:
    """
    Creates dimension coordinate from coord_array if not None, otherwise creating an
    array of integers with an interval of 1
    """
    if coord_array is not None:
        if len(coord_array) != data_length:
            raise ValueError(
                "Cannot generate {} {}s with data of length "
                "{}".format(len(coord_array), coord_name, data_length)
            )

        coord_array = np.array(coord_array)

        if issubclass(coord_array.dtype.type, float):
            # option needed for realizations percentile & probability cube setup
            # and heights coordinate
            coord_array = coord_array.astype(np.float32)
    else:
        coord_array = np.arange(data_length).astype(np.int32)

    if coord_name in iris.std_names.STD_NAMES:
        dim_coord = DimCoord(coord_array, standard_name=coord_name, **kwargs)
    else:
        dim_coord = DimCoord(coord_array, long_name=coord_name, **kwargs)

    return dim_coord


def _construct_dimension_coords(
    data: Union[MaskedArray, ndarray],
    y_coord: DimCoord,
    x_coord: DimCoord,
    realizations: Union[List[float], ndarray],
    height_levels: Union[List[float], ndarray],
    pressure: bool,
) -> DimCoord:
    """ Create array of all dimension coordinates. These dimensions will be ordered:
    realization, height/pressure, y, x. """
    data_shape = data.shape
    ndims = len(data_shape)

    if ndims not in (2, 3, 4):
        raise ValueError(
            "Expected 2 to 4 dimensions on input data: got {}".format(ndims)
        )

    if realizations is not None and height_levels is not None and ndims != 4:
        raise ValueError(
            "Input data must have 4 dimensions to add both realization "
            "and height coordinates: got {}".format(ndims)
        )

    if height_levels is None and ndims == 4:
        raise ValueError("Height levels must be provided if data has 4 dimensions.")

    dim_coords = []

    if ndims == 4 or (height_levels is None and ndims == 3):
        coord_name = "realization"
        coord_units = DIM_COORD_ATTRIBUTES[coord_name]["units"]
        coord_length = data_shape[0]

        realization_coord = _create_dimension_coord(
            realizations, coord_length, coord_name, units=coord_units
        )
        dim_coords.append((realization_coord, 0))

    if height_levels is not None and ndims in (3, 4):
        # Determine the index of the height coord based on if a realization coord has been created
        i = len(dim_coords)
        coord_length = data_shape[i]

        if pressure:
            coord_name = "pressure"
        else:
            coord_name = "height"

        coord_units = DIM_COORD_ATTRIBUTES[coord_name]["units"]
        coord_attributes = DIM_COORD_ATTRIBUTES[coord_name]["attributes"]

        height_coord = _create_dimension_coord(
            height_levels,
            coord_length,
            coord_name,
            units=coord_units,
            attributes=coord_attributes,
        )
        dim_coords.append((height_coord, i))

    dim_coords.append((y_coord, len(dim_coords)))
    dim_coords.append((x_coord, len(dim_coords)))

    return dim_coords


def set_up_variable_cube(
    data: ndarray,
    name: str = "air_temperature",
    units: str = "K",
    spatial_grid: str = "latlon",
    time: datetime = datetime(2017, 11, 10, 4, 0),
    time_bounds: Optional[Tuple[datetime, datetime]] = None,
    frt: datetime = datetime(2017, 11, 10, 0, 0),
    realizations: Optional[Union[List[float], ndarray]] = None,
    include_scalar_coords: Optional[List[Coord]] = None,
    attributes: Optional[Dict[str, str]] = None,
    standard_grid_metadata: Optional[str] = None,
    grid_spacing: Optional[float] = None,
    domain_corner: Optional[Tuple[float, float]] = None,
    height_levels: Optional[Union[List[float], ndarray]] = None,
    pressure: bool = False,
) -> Cube:
    """
    Set up a cube containing a single variable field with:
    - x/y spatial dimensions (equal area or lat / lon)
    - optional leading "realization" dimension
    - optional "height" dimension
    - "time", "forecast_reference_time" and "forecast_period" scalar coords
    - option to specify additional scalar coordinates
    - configurable attributes

    Args:
        data:
            2D (y-x ordered) or 3D (realization-y-x ordered) array of data
            to put into the cube.
        name:
            Variable name (standard / long)
        units:
            Variable units
        spatial_grid:
            What type of x/y coordinate values to use.  Permitted values are
            "latlon" or "equalarea".
        time:
            Single cube validity time
        time_bounds:
            Lower and upper bound on time point, if required
        frt:
            Single cube forecast reference time
        realizations:
            List of forecast realizations.  If not present, taken from the
            leading dimension of the input data array (if 3D).
        include_scalar_coords:
            List of iris.coords.DimCoord or AuxCoord instances of length 1.
        attributes:
            Optional cube attributes.
        standard_grid_metadata:
            Recognised mosg__model_configuration for which to set up Met
            Office standard grid attributes.  Should be 'uk_det', 'uk_ens',
            'gl_det' or 'gl_ens'.
        grid_spacing:
            Grid resolution (degrees for latlon or metres for equalarea).
        domain_corner:
            Bottom left corner of grid domain (y,x) (degrees for latlon or metres for equalarea).
        height_levels:
            List of height levels in metres or pressure levels in Pa.
        pressure:
            Flag to indicate whether the height levels are specified as pressure, in Pa.
            If False, use height in metres.

    Returns:
        Cube containing a single variable field
    """
    # construct spatial dimension coordimates
    ypoints = data.shape[-2]
    xpoints = data.shape[-1]
    y_coord, x_coord = construct_yx_coords(
        ypoints,
        xpoints,
        spatial_grid,
        grid_spacing=grid_spacing,
        domain_corner=domain_corner,
    )

    dim_coords = _construct_dimension_coords(
        data, y_coord, x_coord, realizations, height_levels, pressure
    )

    # construct list of aux_coords_and_dims
    scalar_coords = construct_scalar_time_coords(time, time_bounds, frt)
    if include_scalar_coords is not None:
        for coord in include_scalar_coords:
            scalar_coords.append((coord, None))

    # set up attributes
    cube_attrs = {}
    if standard_grid_metadata is not None:
        cube_attrs.update(MOSG_GRID_DEFINITION[standard_grid_metadata])
    if attributes is not None:
        cube_attrs.update(attributes)

    # create data cube
    cube = iris.cube.Cube(
        data,
        units=units,
        attributes=cube_attrs,
        dim_coords_and_dims=dim_coords,
        aux_coords_and_dims=scalar_coords,
    )
    cube.rename(name)

    # don't allow unit tests to set up invalid cubes
    check_mandatory_standards(cube)

    return cube


def set_up_percentile_cube(
    data: ndarray, percentiles: Union[List[float], ndarray], **kwargs: Any,
) -> Cube:
    """
    Set up a cube containing percentiles of a variable with:
    - x/y spatial dimensions (equal area or lat / lon)
    - leading "percentile" dimension
    - "time", "forecast_reference_time" and "forecast_period" scalar coords
    - option to specify additional scalar coordinates
    - configurable attributes

    Args:
        data:
            3D (percentile-y-x ordered) array of data to put into the cube
        percentiles:
            List of int / float percentile values whose length must match the
            first dimension on the input data cube
        **kwargs:
            Additional keyword arguments passed to 'set_up_variable_cube' function

    Returns:
        Cube containing percentiles
    """
    cube = set_up_variable_cube(data, realizations=percentiles, **kwargs,)
    cube.coord("realization").rename("percentile")
    cube.coord("percentile").units = Unit("%")
    if len(percentiles) == 1:
        cube = next(cube.slices_over("percentile"))
    return cube


def set_up_probability_cube(
    data: ndarray,
    thresholds: Union[List[float], ndarray],
    variable_name: str = "air_temperature",
    threshold_units: str = "K",
    spp__relative_to_threshold: str = "greater_than",
    **kwargs: Any,
) -> Cube:
    """
    Set up a cube containing probabilities at thresholds with:
    - x/y spatial dimensions (equal area or lat / lon)
    - leading "threshold" dimension
    - "time", "forecast_reference_time" and "forecast_period" scalar coords
    - option to specify additional scalar coordinates
    - "spp__relative_to_threshold" attribute (default "greater_than")
    - default or configurable attributes
    - configurable cube data, name conforms to
    "probability_of_X_above(or below)_threshold" convention

    Args:
        data:
            3D (threshold-y-x ordered) array of data to put into the cube
        thresholds:
            List of int / float threshold values whose length must match the
            first dimension on the input data cube
        variable_name:
            Name of the underlying variable to which the probability field
            applies, eg "air_temperature".  NOT name of probability field.
        threshold_units:
            Units of the underlying variable / threshold.
        spp__relative_to_threshold:
            Value of the attribute "spp__relative_to_threshold" which is
            required for IMPROVER probability cubes.
        **kwargs:
            Additional keyword arguments passed to 'set_up_variable_cube' function

    Returns:
        Cube containing probabilities at thresholds
    """
    # create a "relative to threshold" attribute
    coord_attributes = {"spp__relative_to_threshold": spp__relative_to_threshold}

    if spp__relative_to_threshold in (
        "above",
        "greater_than",
        "greater_than_or_equal_to",
    ):
        name = "probability_of_{}_above_threshold".format(variable_name)
    elif spp__relative_to_threshold in ("below", "less_than", "less_than_or_equal_to"):
        name = "probability_of_{}_below_threshold".format(variable_name)
    else:
        msg = (
            "The spp__relative_to_threshold attribute MUST be set for "
            "IMPROVER probability cubes"
        )
        raise ValueError(msg)

    cube = set_up_variable_cube(
        data, name=name, units="1", realizations=thresholds, **kwargs,
    )
    threshold_name = variable_name.replace("_in_vicinity", "")
    cube.coord("realization").rename(threshold_name)
    cube.coord(threshold_name).var_name = "threshold"
    cube.coord(threshold_name).attributes.update(coord_attributes)
    cube.coord(threshold_name).units = Unit(threshold_units)
    if len(thresholds) == 1:
        cube = next(cube.slices_over(threshold_name))
    return cube


def add_coordinate(
    incube: Cube,
    coord_points: List,
    coord_name: str,
    coord_units: Optional[str] = None,
    dtype: Type = np.float32,
    order: Optional[List[int]] = None,
    is_datetime: bool = False,
    attributes: Optional[Dict[str, Any]] = None,
) -> Cube:
    """
    Function to duplicate a sample cube with an additional coordinate to create
    a cubelist. The cubelist is merged to create a single cube, which can be
    reordered to place the new coordinate in the required position.

    Args:
        incube:
            Cube to be duplicated.
        coord_points:
            Values for the coordinate.
        coord_name:
            Long name of the coordinate to be added.
        coord_units:
            Coordinate unit required.
        dtype:
            Datatype for coordinate points.
        order:
            Optional list of integers to reorder the dimensions on the new
            merged cube.  For example, if the new coordinate is required to
            be in position 1 on a 4D cube, use order=[1, 0, 2, 3] to swap the
            new coordinate position with that of the original leading
            coordinate.
        is_datetime:
            If "true", the leading coordinate points have been given as a
            list of datetime objects and need converting.  In this case the
            "coord_units" argument is overridden and the time points provided
            in seconds.  The "dtype" argument is overridden and set to int64.
        attributes:
            Optional coordinate attributes.

    Returns:
        Cube containing an additional dimension coordinate.
    """
    # if the coordinate already exists as a scalar coordinate, remove it
    cube = incube.copy()
    try:
        cube.remove_coord(coord_name)
    except CoordinateNotFoundError:
        pass

    # if new coordinate points are provided as datetimes, convert to seconds
    if is_datetime:
        coord_units = TIME_COORDS["time"].units
        dtype = TIME_COORDS["time"].dtype
        new_coord_points = [_create_time_point(val) for val in coord_points]
        coord_points = new_coord_points

    cubes = iris.cube.CubeList([])
    for val in coord_points:
        temp_cube = cube.copy()
        coord = DimCoord(
            np.array([val], dtype=dtype), units=coord_units, attributes=attributes,
        )
        coord.rename(coord_name)
        temp_cube.add_aux_coord(coord)

        # recalculate forecast period if time or frt have been updated
        if (
            coord_name in ["time", "forecast_reference_time"]
            and coord_units is not None
            and Unit(coord_units).is_time_reference()
        ):
            forecast_period = forecast_period_coord(
                temp_cube, force_lead_time_calculation=True
            )
            try:
                temp_cube.replace_coord(forecast_period)
            except CoordinateNotFoundError:
                temp_cube.add_aux_coord(forecast_period)

        cubes.append(temp_cube)

    new_cube = cubes.merge_cube()
    if order is not None:
        new_cube.transpose(order)

    return new_cube
