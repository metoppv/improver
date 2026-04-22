# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""unit tests for standard_geopotential_height.StandardGeopotentialHeight"""

import math

import iris
import numpy as np
import pytest
from iris import coord_systems as cs
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeAttrsDict

from improver.constants import EARTH_SURFACE_GRAVITY_ACCELERATION, R_DRY_AIR
from improver.standard_geopotential_height.standard_geopotential_height import (
    StandardGeopotentialHeight,
)


def cube_from_dict(d: dict) -> Cube:
    """Generates a cube from a dictionary description

    Args:
        d: dictionary containing a description of a cube

    Returns: cube

    """
    # Rebuild data
    data = np.asarray(d["data"])

    cad = CubeAttrsDict(locals=d["attributes_locals"], globals=d["attributes_globals"])

    cube = iris.cube.Cube(
        data,
        var_name=d.get("name"),
        long_name=d.get("long_name"),
        units=d.get("units"),
        attributes=cad,
    )

    # Dimension coordinates
    for c in d["dim_coords"]:
        standard_name = c["name"]

        if standard_name in ["longitude", "latitude"]:
            coord = DimCoord(
                np.asarray(c["points"]),
                bounds=np.asarray(c["bounds"]) if c["bounds"] is not None else None,
                standard_name=c["name"],
                long_name=c["long_name"],
                units=c["units"],
                coord_system=cs.GeogCS(6371229.0),
            )
        else:
            laea = cs.LambertAzimuthalEqualArea(
                latitude_of_projection_origin=54.9,
                longitude_of_projection_origin=-2.5,
                false_easting=0.0,
                false_northing=0.0,
                ellipsoid=cs.GeogCS(
                    semi_major_axis=6378137.0, semi_minor_axis=6356752.314140356
                ),
            )
            if c["name"] == "pressure":
                c["name"] = "air_pressure"
            coord = DimCoord(
                np.asarray(c["points"]),
                bounds=np.asarray(c["bounds"]) if c["bounds"] is not None else None,
                standard_name=c["name"],
                long_name=c["long_name"],
                units=c["units"],
                coord_system=laea,
            )

        cube.add_dim_coord(coord, c["dim"])

    # Auxiliary coordinates
    for c in d["aux_coords"]:
        coord = AuxCoord(
            np.asarray(c["points"]),
            bounds=np.asarray(c["bounds"]) if c["bounds"] is not None else None,
            standard_name=c["name"],
            long_name=c["long_name"],
            units=c["units"],
        )
        cube.add_aux_coord(coord, c["dims"])

    cube.data = np.ma.array(cube.data)

    return cube


def generate_dfactors_input_dict() -> dict:
    """generates a 37 x 2 x 2 cube (i.e. cut-down) dictionary definition
    originally generated from the file:

    20260325T1200Z-PT0144H00M-geopotential_height_on_pressure_levels.nc

    Returns:
             dict
    """
    return {
        "attributes_globals": {
            "Conventions": "CF-1.7, UKMO-1.0",
            "ancillary_variables": "flag",
            "history": "2026-03-19T16:02:33Z: StaGE Decoupler",
            "institution": "Met Office",
            "mosg__forecast_run_duration": "PT168H",
            "mosg__grid_domain": "global",
            "mosg__grid_type": "standard",
            "mosg__grid_version": "1.7.0",
            "mosg__model_configuration": "gl_det",
            "source": "Met Office Unified Model",
            "title": "Global Model Forecast on Global 10 km Standard Grid",
            "um_version": "13.8",
        },
        "attributes_locals": {"least_significant_digit": np.int64(0)},
        "aux_coords": [
            {
                "bounds": None,
                "dims": [],
                "long_name": None,
                "name": "forecast_period",
                "points": [518400],
                "units": "seconds",
            },
            {
                "bounds": None,
                "dims": [],
                "long_name": None,
                "name": "forecast_reference_time",
                "points": [1773921600],
                "units": "seconds since 1970-01-01 00:00:00",
            },
            {
                "bounds": None,
                "dims": [],
                "long_name": None,
                "name": "time",
                "points": [1774440000],
                "units": "seconds since 1970-01-01 00:00:00",
            },
        ],
        "data": [
            [[96.0, 96.0], [96.0, 96.0]],
            [[320.0, 321.0], [320.0, 321.0]],
            [[549.0, 549.0], [549.0, 550.0]],
            [[783.0, 783.0], [783.0, 783.0]],
            [[1022.0, 1022.0], [1022.0, 1022.0]],
            [[1516.0, 1516.0], [1516.0, 1516.0]],
            [[2036.0, 2036.0], [2036.0, 2036.0]],
            [[2583.0, 2583.0], [2583.0, 2583.0]],
            [[3161.0, 3160.0], [3161.0, 3160.0]],
            [[3773.0, 3773.0], [3773.0, 3773.0]],
            [[4425.0, 4424.0], [4425.0, 4424.0]],
            [[5123.0, 5122.0], [5122.0, 5122.0]],
            [[5875.0, 5875.0], [5875.0, 5875.0]],
            [[6693.0, 6693.0], [6693.0, 6693.0]],
            [[7588.0, 7588.0], [7588.0, 7588.0]],
            [[8070.0, 8070.0], [8070.0, 8070.0]],
            [[8579.0, 8579.0], [8579.0, 8579.0]],
            [[9118.0, 9118.0], [9118.0, 9118.0]],
            [[9692.0, 9692.0], [9692.0, 9692.0]],
            [[10305.0, 10304.0], [10305.0, 10305.0]],
            [[10961.0, 10960.0], [10961.0, 10961.0]],
            [[11669.0, 11668.0], [11669.0, 11668.0]],
            [[12439.0, 12439.0], [12439.0, 12438.0]],
            [[13283.0, 13283.0], [13283.0, 13283.0]],
            [[14223.0, 14223.0], [14223.0, 14223.0]],
            [[15301.0, 15300.0], [15301.0, 15300.0]],
            [[16591.0, 16591.0], [16591.0, 16591.0]],
            [[18626.0, 18627.0], [18626.0, 18627.0]],
            [[20574.0, 20574.0], [20573.0, 20573.0]],
            [[21917.0, 21916.0], [21917.0, 21916.0]],
            [[23724.0, 23723.0], [23724.0, 23723.0]],
            [[26344.0, 26345.0], [26344.0, 26345.0]],
            [[30987.0, 30987.0], [30987.0, 30987.0]],
            [[35786.0, 35786.0], [35786.0, 35787.0]],
            [[42623.0, 42623.0], [42623.0, 42621.0]],
            [[47955.0, 47955.0], [47955.0, 47955.0]],
            [[54776.0, 54776.0], [54776.0, 54776.0]],
        ],
        "dim_coords": [
            {
                "bounds": None,
                "dim": 0,
                "long_name": "pressure",
                "name": "pressure",
                "points": [
                    100000.0,
                    97500.0,
                    95000.0,
                    92500.0,
                    90000.0,
                    85000.0,
                    80000.0,
                    75000.0,
                    70000.0,
                    65000.0,
                    60000.0,
                    55000.0,
                    50000.0,
                    45000.0,
                    40000.0,
                    37500.0,
                    35000.0,
                    32500.0,
                    30000.0,
                    27500.0,
                    25000.0,
                    22500.0,
                    20000.0,
                    17500.0,
                    15000.0,
                    12500.0,
                    10000.0,
                    7000.0,
                    5000.0,
                    4000.0,
                    3000.0,
                    2000.0,
                    1000.0,
                    500.0,
                    200.0,
                    100.0,
                    40.0,
                ],
                "units": "Pa",
            },
            {
                "bounds": [[0.0, 0.09375], [0.09375, 0.1875]],
                "dim": 1,
                "long_name": None,
                "name": "latitude",
                "points": [0.046875, 0.140625],
                "units": "degrees",
            },
            {
                "bounds": [[0.0, 0.140625], [0.140625, 0.28125]],
                "dim": 2,
                "long_name": None,
                "name": "longitude",
                "points": [0.0703125, 0.2109375],
                "units": "degrees",
            },
        ],
        "long_name": None,
        "name": "geopotential_height",
        "shape": (37, 2, 2),
        "units": "m",
    }


def pressure_to_layer(pressure: float) -> str:
    """Convert pressure to the name of an atmospheric layer

    Args:
        pressure (Pa)

    Returns:

    """
    # lower pressurw boundaries
    layer_to_pressure = {
        "Troposphere": 101325,
        "Stratosphere": 22632,
        "Mesosphere": 5475,
        "Thermosphere": 868,
    }

    layer_names = list(layer_to_pressure.keys())
    pressures = list(layer_to_pressure.values())

    if pressure > pressures[1]:
        return layer_names[0]

    for i, p in enumerate(pressures[:-1]):
        if pressures[i] >= pressure and pressure >= pressures[i + 1]:
            return layer_names[i]

    return layer_names[-1]


def masked_layer_count(cube: Cube) -> int:
    """Count the number of fully masked layers in a cube
    for testing purposes.

    Args:
        cube:

    Returns:
        the count
    """

    mask = cube.data.mask
    ndims = len(mask.shape)
    if ndims != 3:
        return 0
    n_layers = mask.shape[0]
    layer_count = 0
    for layer in range(n_layers):
        if np.all(mask[layer, :, :]):  # all True:
            layer_count += 1
    return layer_count


def reference_standard_geopotential_height_calculation(
    incube_arg: Cube,
    pressure_min_hpa: float = 10.0,
    pressure_max_hpa: float = 1000.0,
) -> Cube:
    """referenbce implementation for standard geopotential height

    Args:
        incube_arg: input pressure cube (Pa)
        pressure_min_hpa: lower pressure bound (hPa)
        pressure_max_hpa: upper pressure bound (hPa)

    Returns:
        cube containing standard_geopotential_height values
        masked where pressure is outside valid range for calculation
    """
    incube = incube_arg.copy()

    R = R_DRY_AIR  # 287.0
    g = EARTH_SURFACE_GRAVITY_ACCELERATION  # 9.81
    layer_to_constants = {
        "Troposphere": (-0.0065, 0, 288.15, 101325),
        "Stratosphere": (0.0000, 11_000, 216.65, 22632),
        "Mesosphere": (0.0010, 20_000, 216.65, 5475),
        "Thermosphere": (0.0028, 32000, 228.65, 868),
    }
    shape = incube.shape
    (dz, dy, dx) = shape
    pressures = incube.dim_coords[0].points
    for iz in range(dz):
        P = pressures[iz]
        layer = pressure_to_layer(P)
        beta, Z_b, T_b, P_b = layer_to_constants[layer]
        if beta == 0.0:
            Z = Z_b - ((R * T_b) / g) * math.log(P / P_b)
        else:
            Z = Z_b + (T_b / beta) * (((P / P_b) ** (-beta * R / g)) - 1)
        incube.data[iz, :, :] = Z

    output_cube = StandardGeopotentialHeight._add_masking_by_pressure(
        incube, pressure_min_hpa, pressure_max_hpa
    )

    return output_cube


def generate_dfactors_input_cube() -> Cube:
    """generates an input cube for testing D-factors computation
    Returns:
             cube
    """
    dfactors_cube_dict = generate_dfactors_input_dict()
    dfactors_cube = cube_from_dict(dfactors_cube_dict)
    return dfactors_cube


def test_input_and_processing_masking():
    """tests the masking set-up in the input cube is
    preserved in the output cube, where the D-factors
    calculation does additional masking.

    Returns:
            None
    """
    input_cube = generate_dfactors_input_cube()

    if np.isscalar(input_cube.data.mask):
        input_cube.data.mask = np.full(input_cube.shape, input_cube.data.mask)

    input_cube.data.mask[:, 1, 0] = True  # set up a single column of True

    assert np.ma.isMaskedArray(input_cube.core_data())

    pressure_max, pressure_min = (
        93000.0,
        600.0,
    )  # missing 7 layers at the top & bottom
    pressure_max_hpa = pressure_max / 100
    pressure_min_hpa = pressure_min / 100

    masked_layer_count_before = masked_layer_count(input_cube)
    assert masked_layer_count_before == 0  # expect 0 input layers to be masked

    improver_result = StandardGeopotentialHeight(
        pressure_max_hpa=pressure_max_hpa, pressure_min_hpa=pressure_min_hpa
    )(input_cube)
    masked_layer_count_after = masked_layer_count(improver_result)
    assert masked_layer_count_after == 7  # expect 7 output layers to be masked

    assert np.ma.isMaskedArray(improver_result.core_data())
    squashed_layer = np.all(
        improver_result.data.mask, axis=0
    )  # "and" the mask layers together
    expected_squashed_layer = np.array(
        [[False, False], [True, False]]
    )  # single column cross section
    assert np.all(squashed_layer == expected_squashed_layer)  # detect the single column


def test_input_and_process_mask():
    """tests the masking set-up in the input cube is
    preserved in the output cube, where the D-factors
    calculation does no additional masking.

    Returns:
           None
    """
    input_cube = generate_dfactors_input_cube()

    if np.isscalar(input_cube.data.mask):
        # make 3D so we can change cell by cell
        input_cube.data.mask = np.full(input_cube.shape, input_cube.data.mask)

    input_cube.data.mask[:, 1, 0] = True  # setting a single column to True

    assert np.ma.isMaskedArray(input_cube.core_data())

    masked_layer_count_before = masked_layer_count(input_cube)
    assert masked_layer_count_before == 0  # no masked layers before calculation

    improver_result = StandardGeopotentialHeight(
        pressure_max_hpa=200000, pressure_min_hpa=0.1
    )(input_cube)

    masked_layer_count_after = masked_layer_count(improver_result)
    assert masked_layer_count_after == 0  # no masked layers after calculation

    assert np.ma.isMaskedArray(improver_result.core_data())  # output is masked
    assert np.all(input_cube.data.mask == improver_result.data.mask)  # mask preserved


def test_no_mask():
    """The input cube to StandardGeopotentialHeight can contain
    a masked or non-masked array. This test passes a non-masked
    array to check if things still wo
    rk, but expects masked output.

    Returns:
            None
    """
    input_cube = generate_dfactors_input_cube()

    # unmask the input cube
    if np.ma.isMaskedArray(input_cube.core_data()):
        input_cube.data = input_cube.data.filled(np.nan)

    # is unmasking successful ?
    assert not np.ma.isMaskedArray(input_cube.core_data())

    improver_result = StandardGeopotentialHeight(
        pressure_max_hpa=200000, pressure_min_hpa=0.1
    )(input_cube)

    # is the output masked ?
    assert np.ma.isMaskedArray(improver_result.core_data())


def test_broadcast_to_template_two_pressure_dimensions():
    """test to achieve 100% code coverage
    i.e. passing function _broadcast_to_template
    parameters to cause an error
    i.e. 2 pressure dimensions

    Returns:
            None
    """
    data = np.random.rand(3, 4)

    cube = iris.cube.Cube(data, long_name="example_data", units="1")

    lat = DimCoord([-30, 0, 30], standard_name="latitude", units="degrees_north")

    lon = DimCoord([0, 90, 180, 270], standard_name="longitude", units="degrees_east")

    cube.add_dim_coord(lat, 0)
    cube.add_dim_coord(lon, 1)

    name = "air_presssure"
    pressure = AuxCoord(np.random.rand(3, 4), long_name=name, units="m")

    # Attach to BOTH latitude and longitude dimensions
    cube.add_aux_coord(pressure, (0, 1))

    geo = StandardGeopotentialHeight(pressure_max_hpa=200000, pressure_min_hpa=0.1)

    # pressure_coord = geo._get_pressure_coord(cube)
    pressures = [950, 900, 850, 800, 750, 700]
    with pytest.raises(
        ValueError, match="Pressure coordinate must span exactly one dimension"
    ):
        geo._broadcast_to_template(pressures, cube, name)


def test_broadcast_to_template_pressure_dims_zero():
    """test to achieve 100% code coverage
    by passing function _broadcast_to_template
    parameters to cover all code paths
    i.e. 0 "pressure" dimensions

    Returns:
        None

    """

    input_cube = generate_dfactors_input_cube()
    geo = StandardGeopotentialHeight(pressure_max_hpa=200000, pressure_min_hpa=0.1)

    pressure_coord = geo._get_pressure_coord(input_cube)
    # looking for "time" to fake finding 0 pressure dimensions
    # not meaningful semantically, but will trigger obscure code path
    geo._broadcast_to_template(pressure_coord.points[0:1], input_cube, "time")


def test_broadcast_to_template():
    """test to achieve 100% code coverage
    standard use of function _broadcast_to_template

    Returns:
        None

    """
    input_cube = generate_dfactors_input_cube()

    geo = StandardGeopotentialHeight(pressure_max_hpa=200000, pressure_min_hpa=0.1)
    pressure_coord = geo._get_pressure_coord(input_cube)
    geo._broadcast_to_template(pressure_coord.points, input_cube, pressure_coord)


def test_bad_pressure_units():
    """test to trigger a bad pressure unit error path

    Returns:
        None
    """
    input_cube = generate_dfactors_input_cube()
    input_cube.coord("air_pressure").units = "millibar2"
    with pytest.raises(ValueError, match="are not convertible to hPa"):
        StandardGeopotentialHeight(pressure_max_hpa=200000, pressure_min_hpa=0.1)(
            input_cube
        )


def test_no_pressure_coord():
    """test to achieve 100% code coverage
    by exercising error path for no pressure coordinate

    Returns:
            None
    """
    input_cube = generate_dfactors_input_cube()

    names_to_censor = ["air_pressure", "pressure"]
    for name in names_to_censor:
        present = name in (coord.name() for coord in input_cube.coords())
        if present:
            coord = input_cube.coord(name)
            coord.rename("not " + name)

    with pytest.raises(ValueError, match="cube must contain a pressure coordinate"):
        StandardGeopotentialHeight(pressure_max_hpa=200000, pressure_min_hpa=0.1)(
            input_cube
        )


def test_basic_dfactors_calculations():
    """This test compares the basic standard geopotential height calculation
    in Pmprover against an independent implmentation written for testing
    and assurance purposes.

    The calculation is performed within pressure bounds to ensure that the
    results that are computed within a valid range.

    Results outside this valid range are masked out. As well as testing the
    computation. the masking is also being tested.

    For reference the pressure levels (Pa) in the input cube are:

    [100000.0, 97500.0, 95000.0, 92500.0, 90000.0, 85000.0, 80000.0, 75000.0,
    70000.0, 65000.0, 60000.0, 55000.0, 50000.0, 45000.0, 40000.0, 37500.0,
    35000.0, 32500.0, 30000.0, 27500.0, 25000.0, 22500.0, 20000.0, 17500.0,
    15000.0, 12500.0, 10000.0, 7000.0, 5000.0, 4000.0, 3000.0, 2000.0, 1000.0,
    500.0, 200.0, 100.0, 40.0]

    For different values of the pressure bounds (hPa), different levels in the output
    cube will be masked-in/masked-oy.

    Returns:
             None
    """

    max_min_pairs_to_n_expected_masked_layers = {
        (100000.1, 39.9): 0,  # no levels masked
        (93000.0, 39.9): 3,  # 3 at top masked out
        (100000.1, 600.0): 4,  # 4 at bottom masked out
        (93000.0, 600.0): 7,  # 7 masked out - 3 at the top & 4 at the bottom
        (44000.0, 41000.0): 37,  # all levels masked out
        (41000.0, 44000.0): 37,  # all levels masked out (N.B. order swapped)
    }

    cube = generate_dfactors_input_cube()

    for i, (min_max_pair, n_expected_masked_layers) in enumerate(
        max_min_pairs_to_n_expected_masked_layers.items()
    ):
        hpa_max = min_max_pair[0] / 100.0
        hpa_min = min_max_pair[1] / 100.0

        improver_result = StandardGeopotentialHeight(
            pressure_max_hpa=hpa_max, pressure_min_hpa=hpa_min
        )(cube)

        # does improver return expected number of masked out layers?
        assert n_expected_masked_layers == masked_layer_count(improver_result)

        homebrew_result = reference_standard_geopotential_height_calculation(
            cube, pressure_max_hpa=hpa_max, pressure_min_hpa=hpa_min
        )

        # does test implementation return expected number of masked out layers?
        assert n_expected_masked_layers == masked_layer_count(homebrew_result)

        # do results match from Improver and test implementation
        assert np.ma.allclose(improver_result.data, homebrew_result.data)

        # do mask agree (should be tested implicitly by the line above)
        assert np.allclose(improver_result.data.mask, homebrew_result.data.mask)
