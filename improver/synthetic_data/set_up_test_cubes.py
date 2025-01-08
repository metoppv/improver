# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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
from iris.coords import AuxCoord, Coord, DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from numpy import ndarray
from numpy.ma.core import MaskedArray

from improver.grids import GRID_COORD_ATTRIBUTES
from improver.metadata.check_datatypes import check_mandatory_standards
from improver.metadata.constants.mo_attributes import MOSG_GRID_DEFINITION
from improver.metadata.constants.time_types import TIME_COORDS
from improver.metadata.forecast_times import forecast_period_coord
from improver.spotdata import UNIQUE_ID_ATTRIBUTE

DIM_COORD_ATTRIBUTES = {
    "realization": {"units": "1"},
    "height": {"units": "m", "attributes": {"positive": "up"}},
    "pressure": {"units": "Pa", "attributes": {"positive": "down"}},
}


def construct_yx_coords(
    ypoints: int,
    xpoints: int,
    spatial_grid: str,
    x_grid_spacing: Optional[float] = None,
    y_grid_spacing: Optional[float] = None,
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
        x_grid_spacing:
            Grid resolution along the x axis. Degrees for latlon or metres for equalarea.
             If not provided, defaults to 10 degrees for "latlon" grid or 2000 metres for
             "equalarea" grid
        y_grid_spacing:
             Grid resolution along the y axis. Degrees for latlon or metres for equalarea.
             If not provided, defaults to 10 degrees for "latlon" grid or 2000 metres for
             "equalarea" grid
        domain_corner:
            Bottom left corner of grid domain (y,x) (degrees for latlon or metres for equalarea).
            If not provided, a grid is created centred around (0,0).

    Returns:
        Tuple containing y and x iris.coords.DimCoords
    """
    if spatial_grid not in GRID_COORD_ATTRIBUTES.keys():
        raise ValueError("Grid type {} not recognised".format(spatial_grid))

    if x_grid_spacing is None:
        x_grid_spacing = GRID_COORD_ATTRIBUTES[spatial_grid]["default_grid_spacing"]
    if y_grid_spacing is None:
        y_grid_spacing = GRID_COORD_ATTRIBUTES[spatial_grid]["default_grid_spacing"]

    if domain_corner is None:
        domain_corner = _set_domain_corner(
            ypoints, xpoints, x_grid_spacing, y_grid_spacing
        )
    y_array, x_array = _create_yx_arrays(
        ypoints, xpoints, domain_corner, x_grid_spacing, y_grid_spacing
    )

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
    ypoints: int,
    xpoints: int,
    domain_corner: Tuple[float, float],
    x_grid_spacing: float,
    y_grid_spacing: float,
) -> Tuple[ndarray, ndarray]:
    """
    Creates arrays for constructing y and x DimCoords.

    Args:
        ypoints
        xpoints
        domain_corner
        x_grid_spacing
        y_grid_spacing

    Returns:
        Tuple containing arrays of y and x coordinate values
    """
    y_stop = domain_corner[0] + (y_grid_spacing * (ypoints - 1))
    x_stop = domain_corner[1] + (x_grid_spacing * (xpoints - 1))

    y_array = np.linspace(domain_corner[0], y_stop, ypoints, dtype=np.float32)
    x_array = np.linspace(domain_corner[1], x_stop, xpoints, dtype=np.float32)

    return y_array, x_array


def _set_domain_corner(
    ypoints: int, xpoints: int, x_grid_spacing: float, y_grid_spacing: float
) -> Tuple[float, float]:
    """
    Set domain corner to create a grid around 0,0.

    Args:
        ypoints
        xpoints
        x_grid_spacing
        y_grid_spacing

    Returns:
        (y,x) values of the bottom left corner of the domain
    """
    y_start = 0 - ((ypoints - 1) * y_grid_spacing) / 2
    x_start = 0 - ((xpoints - 1) * x_grid_spacing) / 2

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
    time_bounds: Optional[List[datetime]] = None,
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
            Single blend time point. Either this or frt is required. Both may not be supplied.

    Returns:
        List of iris.coords.DimCoord instances with the associated "None"
        dimension (format required by iris.cube.Cube initialisation).

    Raises:
        ValueError: if differing frt and blend_time are supplied
    """
    if blend_time and frt:
        if blend_time != frt:
            raise ValueError(
                "Refusing to create cube with different values for forecast_reference_time and "
                "blend_time"
            )
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
    coord_array: Optional[List[float]], data_length: int, coord_name: str, **kwargs: Any
) -> DimCoord:
    """
    Creates dimension coordinate from coord_array if not None, otherwise creating an
    array of integers with an interval of 1
    """
    if coord_array is not None:
        if len(coord_array) != data_length:
            raise ValueError(
                "Cannot generate {} {}s with data of length " "{}".format(
                    len(coord_array), coord_name, data_length
                )
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
    y_coord: Optional[DimCoord] = None,
    x_coord: Optional[DimCoord] = None,
    spot_index: Optional[DimCoord] = None,
    realizations: Optional[Union[List[float], ndarray]] = None,
    vertical_levels: Optional[Union[List[float], ndarray]] = None,
    pressure: bool = False,
    height: bool = False,
) -> DimCoord:
    """Create array of all dimension coordinates. The expected dimension order
    for gridded cubes is realization, height/pressure, y, x or realization,
    height/pressure, spot_index for site cubes. The returned coordinates will
    reflect this ordering.

    A realization coordinate will be created if the cube is
    (n_spatial_dims + 1) or (n_spatial_dims + 2), even if no values for the
    realizations argument are provided. To create a height coordinate, the
    vertical_levels must be provided.
    """

    data_shape = data.shape
    ndims = data.ndim
    n_spatial_dims = sum([item is not None for item in [y_coord, x_coord, spot_index]])

    if not n_spatial_dims <= ndims <= n_spatial_dims + 2:
        raise ValueError(
            f"Expected {n_spatial_dims} to {n_spatial_dims + 2} dimensions on "
            f"input data: got {ndims}"
        )

    if (
        realizations is not None
        and vertical_levels is not None
        and ndims != n_spatial_dims + 2
    ):
        raise ValueError(
            f"Input data must have {n_spatial_dims + 2} dimensions to add both realization "
            f"and height coordinates: got {ndims}"
        )

    if vertical_levels is None and ndims > n_spatial_dims + 1:
        raise ValueError(
            "Vertical levels must be provided if data has > "
            f"{n_spatial_dims + 1} dimensions."
        )

    dim_coords = []

    if ndims == n_spatial_dims + 2 or (
        vertical_levels is None and ndims == n_spatial_dims + 1
    ):
        coord_name = "realization"
        coord_units = DIM_COORD_ATTRIBUTES[coord_name]["units"]
        coord_length = data_shape[0]

        realization_coord = _create_dimension_coord(
            realizations, coord_length, coord_name, units=coord_units
        )
        dim_coords.append((realization_coord, 0))

    if (
        vertical_levels is not None
        and n_spatial_dims + 1 <= ndims <= n_spatial_dims + 2
    ):
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
            vertical_levels,
            coord_length,
            coord_name,
            units=coord_units,
            attributes=coord_attributes,
        )
        dim_coords.append((height_coord, i))

    if spot_index is not None:
        dim_coords.append((spot_index, len(dim_coords)))
    else:
        dim_coords.append((y_coord, len(dim_coords)))
        dim_coords.append((x_coord, len(dim_coords)))

    return dim_coords


def _yx_for_grid(data, spatial_grid, x_grid_spacing, y_grid_spacing, domain_corner):
    """Construct y and x coordinates for gridded data based upon the shape of
    the data array.
    """
    ypoints = data.shape[-2]
    xpoints = data.shape[-1]
    y_coord, x_coord = construct_yx_coords(
        ypoints,
        xpoints,
        spatial_grid,
        x_grid_spacing=x_grid_spacing,
        y_grid_spacing=y_grid_spacing,
        domain_corner=domain_corner,
    )
    return y_coord, x_coord


def set_up_spot_variable_cube(
    data: ndarray,
    latitudes: Optional[Union[List[float], ndarray]] = None,
    longitudes: Optional[Union[List[float], ndarray]] = None,
    altitudes: Optional[Union[List[float], ndarray]] = None,
    wmo_ids: Optional[Union[List[float], ndarray]] = None,
    unique_site_id: Optional[Union[List[str], ndarray]] = None,
    unique_site_id_key: Optional[str] = None,
    realizations: Optional[Union[List[float], ndarray]] = None,
    vertical_levels: Optional[Union[List[float], ndarray]] = None,
    pressure: bool = False,
    height: bool = False,
    *args,
    **kwargs,
):
    """
    Set up a spot cube containing a single variable field with:

    - latitude, longitude, altitude, WMO ID and unique site ID coordinates
      associated with the sites.
    - optional leading realization coordinate.
    - optional height or pressure levels.
    - additional options available by keywords passed to the _variable_cube
      function.

    Args:
        data:
            1D (length of no. of sites) or 2D (realization-sites ordered)
            array of data to put into the cube.
        latitudes:
            Optional list of latitude values of the same length as the number
            of sites.
        longitudes:
            Optional list of longitude values of the same length as the number
            of sites.
        altitudes:
            Optional list of altitude values of the same length as the number
            of sites.
        wmo_ids:
            Optional list of WMO IDs that identify the sites, these are stored
            as padded 5-digit strings. Same length as the number of sites.
        unique_site_id:
            Optional list of IDs that identify the sites, these are stored as
            padded 8-digit strings. Same length as the number of sites.
            If provided, unique_site_id_key must also be provided.
        unique_site_id_key:
            Optional string that names the unique_site_id coordinate.
        realizations:
            List of forecast realizations.  If not present, taken from the
            leading dimension of the input data array (if 2D).
        vertical_levels:
            List of vertical levels in height (metres) or pressure levels (Pa).
        pressure:
            Flag to indicate whether the vertical levels are specified as pressure, in Pa.
            If False, use height in metres.
        height:
            Flag to indicate whether the vertical levels are specified as height, in metres.

    Returns:
        Cube containing a single spot variable field
    """

    n_sites = data.shape[-1]
    spot_index = DimCoord(np.arange(n_sites), long_name="spot_index", units="1")

    altitudes = (
        altitudes if altitudes is not None else np.ones((n_sites), dtype=np.float32)
    )
    latitudes = (
        latitudes
        if latitudes is not None
        else np.linspace(50, 60, n_sites, dtype=np.float32)
    )
    longitudes = (
        longitudes
        if longitudes is not None
        else np.linspace(-5, 5, n_sites, dtype=np.float32)
    )
    wmo_ids = wmo_ids if wmo_ids is not None else range(n_sites)
    wmo_ids = [f"{int(item):05d}" for item in wmo_ids]

    alt_coord = AuxCoord(altitudes, "altitude", units="m")
    y_coord = AuxCoord(latitudes, "latitude", units="degrees")
    x_coord = AuxCoord(longitudes, "longitude", units="degrees")
    id_coord = AuxCoord(wmo_ids, long_name="wmo_id")
    unique_id_coord = None

    if unique_site_id is not None:
        if not unique_site_id_key:
            raise ValueError(
                "A unique_site_id_key must be provided if a unique_site_id is"
                " provided."
            )
        unique_site_id = [f"{int(item):08d}" for item in unique_site_id]
        unique_id_coord = AuxCoord(
            unique_site_id,
            long_name=unique_site_id_key,
            units="no_unit",
            attributes={UNIQUE_ID_ATTRIBUTE: "true"},
        )

    dim_coords = _construct_dimension_coords(
        data,
        spot_index=spot_index,
        realizations=realizations,
        vertical_levels=vertical_levels,
        pressure=pressure,
        height=height,
    )

    cube = _variable_cube(data, dim_coords, *args, **kwargs)

    (spatial_coord_index,) = cube.coord_dims("spot_index")
    crds = [y_coord, x_coord, alt_coord, id_coord]
    if unique_id_coord is not None:
        crds.append(unique_id_coord)

    for crd in crds:
        cube.add_aux_coord(crd, spatial_coord_index)

    # don't allow unit tests to set up invalid cubes
    check_mandatory_standards(cube)
    return cube


def set_up_variable_cube(
    data: ndarray,
    spatial_grid: str = "latlon",
    x_grid_spacing: Optional[float] = None,
    y_grid_spacing: Optional[float] = None,
    domain_corner: Optional[Tuple[float, float]] = None,
    realizations: Optional[Union[List[float], ndarray]] = None,
    vertical_levels: Optional[Union[List[float], ndarray]] = None,
    pressure: bool = False,
    height: bool = False,
    *args,
    **kwargs,
):
    """
    Set up a gridded cube containing a single variable field with:

    - a lat/lon or equal areas projection.
    - optional leading realization coordinate.
    - optional height or pressure levels.
    - additional options available by keywords passed to the _variable_cube
      function.

    Args:
        data:
            2D (y-x ordered) or 3D (realization-y-x ordered) array of data
            to put into the cube.
        spatial_grid:
            What type of x/y coordinate values to use.  Permitted values are
            "latlon" or "equalarea".
        x_grid_spacing:
            Grid resolution along the x axis (degrees for latlon or metres for equalarea).
        y_grid_spacing:
            Grid resolution along the y axis (degrees for latlon or metres for equalarea).
        domain_corner:
            Bottom left corner of grid domain (y,x) (degrees for latlon or metres for equalarea).
        realizations:
            List of forecast realizations.  If not present, taken from the
            leading dimension of the input data array (if 3D).
        vertical_levels:
            List of vertical levels in height (metres) or pressure levels (Pa).
        pressure:
            Flag to indicate whether the vertical levels are specified as pressure, in Pa.
            If False, use height in metres.
        height:
            Flag to indicate whether the vertical levels are specified as height, in metres.

    Returns:
        Cube containing a single gridded variable field
    """

    y_coord, x_coord = _yx_for_grid(
        data, spatial_grid, x_grid_spacing, y_grid_spacing, domain_corner
    )

    dim_coords = _construct_dimension_coords(
        data,
        y_coord=y_coord,
        x_coord=x_coord,
        realizations=realizations,
        vertical_levels=vertical_levels,
        pressure=pressure,
        height=height,
    )

    cube = _variable_cube(data, dim_coords, *args, **kwargs)

    # don't allow unit tests to set up invalid cubes
    check_mandatory_standards(cube)

    return cube


def _variable_cube(
    data: ndarray,
    dim_coords: Tuple[Union[AuxCoord, DimCoord]],
    name: str = "air_temperature",
    units: str = "K",
    time: datetime = datetime(2017, 11, 10, 4, 0),
    time_bounds: Optional[Tuple[datetime, datetime]] = None,
    frt: Optional[datetime] = None,
    blend_time: Optional[datetime] = None,
    include_scalar_coords: Optional[List[Coord]] = None,
    attributes: Optional[Dict[str, str]] = None,
    standard_grid_metadata: Optional[str] = None,
) -> Cube:
    """
    Set up a cube containing a single variable field with:
    - the provided dimension coordinates
    - "time", "forecast_reference_time" and "forecast_period" scalar coords
    - option to specify additional scalar coordinates
    - configurable attributes

    Args:
        data:
            2D (y-x ordered) or 3D (realization-y-x ordered) array of data
            to put into the cube.
        dim_coords:
            Dimension coordinates for the constructed cube, e.g. the x and
            y dimensions.
        name:
            Variable name (standard / long)
        units:
            Variable units
        time:
            Single cube validity time
        time_bounds:
            Lower and upper bound on time point, if required
        frt:
            Single cube forecast reference time. Default value is datetime(2017, 11, 10, 0, 0).
        blend_time:
            Single cube blend time
        include_scalar_coords:
            List of iris.coords.DimCoord or AuxCoord instances of length 1.
        attributes:
            Optional cube attributes.
        standard_grid_metadata:
            Recognised mosg__model_configuration for which to set up Met
            Office standard grid attributes.  Should be 'uk_det', 'uk_ens',
            'gl_det' or 'gl_ens'.

    Returns:
        Cube containing a single variable field
    """
    if not frt and not blend_time:
        frt = datetime(2017, 11, 10, 0, 0)

    # construct list of aux_coords_and_dims
    scalar_coords = construct_scalar_time_coords(time, time_bounds, frt, blend_time)
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

    return cube


def set_up_spot_percentile_cube(*args, **kwargs):
    """Set up a cube containing spot percentiles of a variable.

    Args:
        data:
            2D (percentile-sites ordered) array of data to put into the cube
    Returns:
        Cube containing spot percentiles.
    """
    function = set_up_spot_variable_cube
    return _percentile_cube(function, *args, **kwargs)


def set_up_percentile_cube(*args, **kwargs):
    """Set up a cube containing gridded percentiles of a variable.

    Args:
        data:
            3D (percentile-y-x ordered) array of data to put into the cube
    Returns:
        Cube containing gridded percentiles.
    """
    function = set_up_variable_cube
    return _percentile_cube(function, *args, **kwargs)


def _percentile_cube(
    function, data: ndarray, percentiles: Union[List[float], ndarray], **kwargs: Any
) -> Cube:
    """
    Set up a cube containing percentiles of a variable with:
    - leading "percentile" dimension
    - "time", "forecast_reference_time" and "forecast_period" scalar coords
    - option to specify additional scalar coordinates
    - configurable attributes

    Args:
        data:
            Array of data to put into the cube
        percentiles:
            List of int / float percentile values whose length must match the
            first dimension on the input data cube
        **kwargs:
            Additional keyword arguments passed to 'set_up_variable_cube' function

    Returns:
        Cube containing percentiles
    """
    cube = function(data, realizations=percentiles, **kwargs)
    cube.coord("realization").rename("percentile")
    cube.coord("percentile").units = Unit("%")
    if len(percentiles) == 1:
        cube = next(cube.slices_over("percentile"))
    return cube


def set_up_spot_probability_cube(*args, **kwargs):
    """Set up a cube containing spot probabilities at thresholds.

    Args:
        data:
            2D (threshold-sites) array of data to put into the cube
    Returns:
        Cube containing spot probabilities.
    """
    function = set_up_spot_variable_cube
    return _probability_cube(function, *args, **kwargs)


def set_up_probability_cube(*args, **kwargs):
    """Set up a cube containing gridded probabilities at thresholds.

    Args:
        data:
            3D (threshold-y-x ordered) array of data to put into the cube
    Returns:
        Cube containing gridded probabilities.
    """
    function = set_up_variable_cube
    return _probability_cube(function, *args, **kwargs)


def _probability_cube(
    function,
    data: ndarray,
    thresholds: Union[List[float], ndarray],
    variable_name: str = "air_temperature",
    threshold_units: str = "K",
    spp__relative_to_threshold: str = "greater_than",
    **kwargs: Any,
) -> Cube:
    """
    Set up a cube containing probabilities at thresholds with:
    - leading "threshold" dimension
    - "time", "forecast_reference_time" and "forecast_period" scalar coords
    - option to specify additional scalar coordinates
    - "spp__relative_to_threshold" attribute (default "greater_than")
    - default or configurable attributes
    - configurable cube data, name conforms to
    "probability_of_X_above(or below)_threshold" convention

    Args:
        data:
            Array of data to put into the cube
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

    cube = function(data, name=name, units="1", realizations=thresholds, **kwargs)
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
            np.array([val], dtype=dtype), units=coord_units, attributes=attributes
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
