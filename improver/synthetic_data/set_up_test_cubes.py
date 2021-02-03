# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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

import iris
import numpy as np
from cf_units import Unit, date2num
from iris.coords import DimCoord
from iris.exceptions import CoordinateNotFoundError

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
    ypoints, xpoints, spatial_grid, grid_spacing=None, domain_corner=None
):
    """
    Construct y/x spatial dimension coordinates

    Args:
        ypoints (int):
            Number of grid points required along the y-axis
        xpoints (int):
            Number of grid points required along the x-axis
        spatial_grid (str):
            Specifier to produce either a "latlon" or "equalarea" grid
        grid_spacing (Optional[float]):
            Grid resolution (degrees for latlon or metres for equalarea). If not provided, defaults to 10 degrees
            for "latlon" grid or 2000 metres for "equalarea" grid
        domain_corner (Optional[Tuple[float, float]]):
            Bottom left corner of grid domain (y,x) (degrees for latlon or metres for equalarea). If not
            provided, a grid is created centred around (0,0).

    Returns:
        Tuple[iris.coords.DimCoord, iris.coords.DimCoord]:
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


def _create_yx_arrays(ypoints, xpoints, domain_corner, grid_spacing):
    """
    Creates arrays for constructing y and x DimCoords.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]:
            Tuple containing arrays of y and x coordinate values
    """
    y_stop = domain_corner[0] + (grid_spacing * (ypoints - 1))
    x_stop = domain_corner[1] + (grid_spacing * (xpoints - 1))

    y_array = np.linspace(domain_corner[0], y_stop, ypoints, dtype=np.float32)
    x_array = np.linspace(domain_corner[1], x_stop, xpoints, dtype=np.float32)

    return y_array, x_array


def _set_domain_corner(ypoints, xpoints, grid_spacing):
    """
    Set domain corner to create a grid around 0,0.

    Returns:
        Tuple[float, float]:
            (y,x) values of the bottom left corner of the domain
    """
    y_start = 0 - ((ypoints - 1) * grid_spacing) / 2
    x_start = 0 - ((xpoints - 1) * grid_spacing) / 2

    return y_start, x_start


def _create_time_point(time):
    """Returns a coordinate point with appropriate units and datatype
    from a datetime.datetime instance.

    Returns:
        Any:
            Returns coordinate point as datatype specified in TIME_COORDS["time"]
    """
    coord_spec = TIME_COORDS["time"]
    point = date2num(time, coord_spec.units, coord_spec.calendar)
    return np.around(point).astype(coord_spec.dtype)


def construct_scalar_time_coords(time, time_bounds, frt):
    """
    Construct scalar time coordinates as aux_coord list

    Args:
        time (datetime.datetime):
            Single time point
        time_bounds (Sequence[datetime.datetime] or None):
            Lower and upper bound on time point, if required
        frt (datetime.datetime):
            Single forecast reference time point

    Returns:
        List[Tuple[iris.coord.DimCoord, bool]]:
            List of iris.coords.DimCoord instances with the associated "None"
            dimension (format required by iris.cube.Cube initialisation).
    """
    # generate time coordinate points
    time_point_seconds = _create_time_point(time)
    frt_point_seconds = _create_time_point(frt)

    fp_coord_spec = TIME_COORDS["forecast_period"]
    if time_point_seconds < frt_point_seconds:
        raise ValueError("Cannot set up cube with negative forecast period")
    fp_point_seconds = (time_point_seconds - frt_point_seconds).astype(
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
            [[bounds[0] - frt_point_seconds, bounds[1] - frt_point_seconds]]
        ).astype(fp_coord_spec.dtype)
    else:
        bounds = None
        fp_bounds = None

    # create coordinates
    time_coord = DimCoord(
        time_point_seconds, "time", bounds=bounds, units=TIME_COORDS["time"].units
    )
    frt_coord = DimCoord(
        frt_point_seconds,
        "forecast_reference_time",
        units=TIME_COORDS["forecast_reference_time"].units,
    )
    fp_coord = DimCoord(
        fp_point_seconds, "forecast_period", bounds=fp_bounds, units=fp_coord_spec.units
    )

    coord_dims = [(time_coord, None), (frt_coord, None), (fp_coord, None)]
    return coord_dims


def _create_dimension_coord(coord_array, data_length, coord_name, **kwargs):
    """
    Creates dimension coordinate from coord_array if not None, otherwise creating an array of integers with an interval of 1
    """
    if coord_array is not None:
        if len(coord_array) != data_length:
            raise ValueError(
                "Cannot generate {} {}s with data of length "
                "{}".format(len(coord_array), coord_name, data_length)
            )

        coord_array = np.array(coord_array)

        if issubclass(coord_array.dtype.type, np.float):
            # option needed for realizations percentile & probability cube setup and heights coordinate
            coord_array = coord_array.astype(np.float32)
    else:
        coord_array = np.arange(data_length).astype(np.int32)

    if coord_name in iris.std_names.STD_NAMES:
        dim_coord = DimCoord(coord_array, standard_name=coord_name, **kwargs)
    else:
        dim_coord = DimCoord(coord_array, long_name=coord_name, **kwargs)

    return dim_coord


def _construct_dimension_coords(
    data, y_coord, x_coord, realizations, height_levels, pressure
):
    """ Create array of all dimension coordinates. These dimensions will be ordered: realization, height/pressure, y, x. """
    data_shape = data.shape
    ndims = len(data_shape)

    if ndims not in (2, 3, 4):
        raise ValueError(
            "Expected 2 to 4 dimensions on input data: got {}".format(ndims)
        )

    if realizations is not None and height_levels is not None and ndims != 4:
        raise ValueError(
            "Input data must have 4 dimensions to add both realization and height coordinates: got {}".format(
                ndims
            )
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
        # Determine the index of the height coord based on whether a realization coord has been created
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
    data,
    name="air_temperature",
    units="K",
    spatial_grid="latlon",
    time=datetime(2017, 11, 10, 4, 0),
    time_bounds=None,
    frt=datetime(2017, 11, 10, 0, 0),
    realizations=None,
    include_scalar_coords=None,
    attributes=None,
    standard_grid_metadata=None,
    grid_spacing=None,
    domain_corner=None,
    height_levels=None,
    pressure=False,
):
    """
    Set up a cube containing a single variable field with:
    - x/y spatial dimensions (equal area or lat / lon)
    - optional leading "realization" dimension
    - optional "height" dimension
    - "time", "forecast_reference_time" and "forecast_period" scalar coords
    - option to specify additional scalar coordinates
    - configurable attributes

    Args:
        data (numpy.ndarray):
            2D (y-x ordered) or 3D (realization-y-x ordered) array of data
            to put into the cube.
        name (Optional[str]):
            Variable name (standard / long)
        units (Optional[str]):
            Variable units
        spatial_grid (Optional[str]):
            What type of x/y coordinate values to use.  Permitted values are
            "latlon" or "equalarea".
        time (Optional[datetime.datetime]):
            Single cube validity time
        time_bounds (Optional[Sequence[datetime.datetime]]):
            Lower and upper bound on time point, if required
        frt (Optional[datetime.datetime]):
            Single cube forecast reference time
        realizations (Optional[List[numpy.ndarray]]):
            List of forecast realizations.  If not present, taken from the
            leading dimension of the input data array (if 3D).
        include_scalar_coords (Optional[List[iris.coords.DimCoord] or List[iris.coords.AuxCoord]]):
            List of iris.coords.DimCoord or AuxCoord instances of length 1.
        attributes (Optional[Dict[Any]]):
            Optional cube attributes.
        standard_grid_metadata (Optional[str]):
            Recognised mosg__model_configuration for which to set up Met
            Office standard grid attributes.  Should be 'uk_det', 'uk_ens',
            'gl_det' or 'gl_ens'.
        grid_spacing (Optional[float]):
            Grid resolution (degrees for latlon or metres for equalarea).
        domain_corner (Optional[Tuple[float, float]]):
            Bottom left corner of grid domain (y,x) (degrees for latlon or metres for equalarea).
        height_levels (Optional[List[float]]):
            List of height levels in metres or pressure levels in Pa.
        pressure (Optional[bool]):
            Flag to indicate whether the height levels are specified as pressure, in Pa. If False, use height in metres.

    Returns:
        iris.cube.Cube:
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
    data, percentiles, **kwargs,
):
    """
    Set up a cube containing percentiles of a variable with:
    - x/y spatial dimensions (equal area or lat / lon)
    - leading "percentile" dimension
    - "time", "forecast_reference_time" and "forecast_period" scalar coords
    - option to specify additional scalar coordinates
    - configurable attributes

    Args:
        data (numpy.ndarray):
            3D (percentile-y-x ordered) array of data to put into the cube
        percentiles (List[float] or numpy.ndarray):
            List of int / float percentile values whose length must match the
            first dimension on the input data cube
        **kwargs:
            Additional keyword arguments passed to 'set_up_variable_cube' function

    Returns:
        iris.cube.Cube:
            Cube containing percentiles
    """
    cube = set_up_variable_cube(data, realizations=percentiles, **kwargs,)
    cube.coord("realization").rename("percentile")
    cube.coord("percentile").units = Unit("%")
    if len(percentiles) == 1:
        cube = next(cube.slices_over("percentile"))
    return cube


def set_up_probability_cube(
    data,
    thresholds,
    variable_name="air_temperature",
    threshold_units="K",
    spp__relative_to_threshold="greater_than",
    **kwargs,
):
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
        data (numpy.ndarray):
            3D (threshold-y-x ordered) array of data to put into the cube
        thresholds (List[float] or numpy.ndarray):
            List of int / float threshold values whose length must match the
            first dimension on the input data cube
        variable_name (Optional[str]):
            Name of the underlying variable to which the probability field
            applies, eg "air_temperature".  NOT name of probability field.
        threshold_units (Optional[str]):
            Units of the underlying variable / threshold.
        spp__relative_to_threshold (Optional[str]):
            Value of the attribute "spp__relative_to_threshold" which is
            required for IMPROVER probability cubes.
        **kwargs:
            Additional keyword arguments passed to 'set_up_variable_cube' function

    Returns:
        iris.cube.Cube:
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
    cube.coord("realization").rename(variable_name)
    cube.coord(variable_name).var_name = "threshold"
    cube.coord(variable_name).attributes.update(coord_attributes)
    cube.coord(variable_name).units = Unit(threshold_units)
    if len(thresholds) == 1:
        cube = next(cube.slices_over(variable_name))
    return cube


def add_coordinate(
    incube,
    coord_points,
    coord_name,
    coord_units=None,
    dtype=np.float32,
    order=None,
    is_datetime=False,
    attributes=None,
):
    """
    Function to duplicate a sample cube with an additional coordinate to create
    a cubelist. The cubelist is merged to create a single cube, which can be
    reordered to place the new coordinate in the required position.

    Args:
        incube (iris.cube.Cube):
            Cube to be duplicated.
        coord_points (List[Any] or numpy.ndarray):
            Values for the coordinate.
        coord_name (str):
            Long name of the coordinate to be added.
        coord_units (Optional[str]):
            Coordinate unit required.
        dtype (Optional[type]):
            Datatype for coordinate points.
        order (Optional[List[int]]):
            Optional list of integers to reorder the dimensions on the new
            merged cube.  For example, if the new coordinate is required to
            be in position 1 on a 4D cube, use order=[1, 0, 2, 3] to swap the
            new coordinate position with that of the original leading
            coordinate.
        is_datetime (Optional[bool]):
            If "true", the leading coordinate points have been given as a
            list of datetime objects and need converting.  In this case the
            "coord_units" argument is overridden and the time points provided
            in seconds.  The "dtype" argument is overridden and set to int64.
        attributes (Optional[Dict[Any]]):
            Optional coordinate attributes.

    Returns:
        iris.cube.Cube:
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
        temp_cube.add_aux_coord(
            DimCoord(
                np.array([val], dtype=dtype),
                long_name=coord_name,
                units=coord_units,
                attributes=attributes,
            )
        )

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
