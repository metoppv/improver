# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import iris.quickplot as qplt
import pytest
import unittest.mock as mock

from improver.utilities.load import load_cube
from improver.api import apply_mask
from . import acceptance as acc
from improver_tests import assertGraphic, assertCML


@pytest.fixture(scope="session")
def load_cubes():
    """Load input and output cubes."""
    src_root = acc.kgo_root() / "apply-mask/"
    wind_speed = src_root / "wind_speed.nc"
    mask = src_root / "mask.nc"
    return (
        (wind_speed, mask),
        (load_cube(wind_speed), load_cube(mask)),
    )


@pytest.fixture
def cubes(load_cubes):
    paths, cubes = load_cubes
    return paths, [cube.copy() for cube in cubes]


@pytest.mark.parametrize("invert", [True, False])
def test_all(cubes, invert, request):
    _, cubes = cubes
    res_cube = apply_mask(*cubes, mask_name="land_binary_mask", invert_mask=invert)

    #assertCML(res_cube)

    qplt.pcolormesh(res_cube)
    qplt.plt.gca().coastlines()
    assertGraphic()


@pytest.mark.parametrize("invert", [True, False])
def test_cli(cubes, invert):
    paths, cubes = cubes
    run_cli = acc.run_cli("apply-mask")

    args = list(paths) + [
        "--mask-name",
        "land_binary_mask",
        "--invert-mask",
        f"{invert}",
    ]

    with mock.patch(f"{apply_mask.__module__}.{apply_mask.__name__}") as mock_app:
        run_cli(args)
    for i, cube in enumerate(cubes):
        assert mock_app.call_args[0][i] == cube
    assert mock_app.call_args[1] == {'mask_name': 'land_binary_mask', 'invert_mask': invert}