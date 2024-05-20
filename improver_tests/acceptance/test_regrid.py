# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the regrid CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
GLOBAL_UK_TITLE = "Global Model Forecast on UK 2 km Standard Grid"
MOGREPS_G_UK_TITLE = "MOGREPS-G Model Forecast on UK 2 km Standard Grid"
UKV_GLOBAL_CUTOUT_TITLE = "UKV Model Forecast on UK 10 km Grid"
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_regrid_basic(tmp_path):
    """Test basic regridding"""
    # KGO for this test (default: bilinear) is the same as test_regrid_bilinear_2
    kgo_dir = acc.kgo_root() / "regrid"
    kgo_path = kgo_dir / "basic/kgo.nc"
    input_path = kgo_dir / "global_cutout.nc"
    target_path = kgo_dir / "ukvx_grid.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, target_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_regrid_bilinear_2(tmp_path):
    """Test bilinear-2 regridding"""
    # KGO for this test is the same as test_regrid_basic
    # bilinear-2 regridding produces the same result as Iris/bilinear regridding
    kgo_dir = acc.kgo_root() / "regrid"
    kgo_path = kgo_dir / "basic/kgo.nc"
    input_path = kgo_dir / "global_cutout.nc"
    target_path = kgo_dir / "ukvx_grid.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        target_path,
        "--output",
        output_path,
        "--regrid-mode",
        "bilinear-2",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_regrid_nearest(tmp_path):
    """Test nearest neighbour regridding"""
    kgo_dir = acc.kgo_root() / "regrid"
    kgo_path = kgo_dir / "nearest/kgo.nc"
    input_path = kgo_dir / "global_cutout.nc"
    target_path = kgo_dir / "ukvx_grid.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        target_path,
        "--output",
        output_path,
        "--regrid-mode",
        "nearest",
        "--regridded-title",
        GLOBAL_UK_TITLE,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_regrid_extrapolate(tmp_path):
    """Test nearest neighbour regridding with extrapolation"""
    kgo_dir = acc.kgo_root() / "regrid"
    kgo_path = kgo_dir / "extrapolate/kgo.nc"
    input_path = kgo_dir / "ukvx_grid.nc"
    target_path = kgo_dir / "global_cutout.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        target_path,
        "--output",
        output_path,
        "--regrid-mode",
        "nearest",
        "--extrapolation-mode",
        "extrapolate",
        "--regridded-title",
        UKV_GLOBAL_CUTOUT_TITLE,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_regrid_nearest_landmask(tmp_path):
    """Test nearest neighbour regridding with land sea mask"""
    kgo_dir = acc.kgo_root() / "regrid/landmask"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../global_cutout.nc"
    landmask_path = kgo_dir / "glm_landmask.nc"
    target_path = kgo_dir / "ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        target_path,
        landmask_path,
        "--output",
        output_path,
        "--regrid-mode",
        "nearest-with-mask",
        "--regridded-title",
        GLOBAL_UK_TITLE,
        "--land-sea-mask-vicinity",
        "100000",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_regrid_check_landmask(tmp_path):
    """Test land sea mask output matches other test"""
    kgo_dir = acc.kgo_root() / "regrid"
    kgo_path = kgo_dir / "nearest/kgo.nc"
    input_path = kgo_dir / "global_cutout.nc"
    landmask_path = kgo_dir / "landmask/glm_landmask.nc"
    target_path = kgo_dir / "ukvx_grid.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        target_path,
        landmask_path,
        "--output",
        output_path,
        "--regrid-mode",
        "nearest-with-mask",
        "--regridded-title",
        GLOBAL_UK_TITLE,
        "--land-sea-mask-vicinity",
        "100000",
    ]
    with pytest.warns(UserWarning, match=".*land_binary_mask.*"):
        run_cli(args)
    # Don't recreate output as it is the same as other test
    acc.compare(output_path, kgo_path, recreate=False)


def test_args_error_landmask(tmp_path):
    """Test land sea mask specified but no regrid mode"""
    kgo_dir = acc.kgo_root() / "regrid/landmask"
    input_path = kgo_dir / "../global_cutout.nc"
    landmask_path = kgo_dir / "glm_landmask.nc"
    target_path = kgo_dir / "ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, target_path, landmask_path, "--output", output_path]
    with pytest.raises(ValueError, match=".*nearest-with-mask.*"):
        run_cli(args)


def test_args_error_no_landmask(tmp_path):
    """Test landmask mode but no land sea mask provided"""
    kgo_dir = acc.kgo_root() / "regrid/landmask"
    input_path = kgo_dir / "../global_cutout.nc"
    target_path = kgo_dir / "ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        target_path,
        "--output",
        output_path,
        "--regrid-mode",
        "nearest-with-mask",
    ]
    with pytest.raises(ValueError, match=".*input landmask.*"):
        run_cli(args)


@pytest.mark.slow
def test_regrid_nearest_landmask_multi_realization(tmp_path):
    """Test nearest neighbour with land sea mask and realizations"""
    kgo_dir = acc.kgo_root() / "regrid/landmask"
    kgo_path = kgo_dir / "kgo_multi_realization.nc"
    input_path = kgo_dir / "global_cutout_multi_realization.nc"
    landmask_path = kgo_dir / "engl_landmask.nc"
    target_path = kgo_dir / "ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        target_path,
        landmask_path,
        "--output",
        output_path,
        "--regrid-mode",
        "nearest-with-mask",
        "--regridded-title",
        MOGREPS_G_UK_TITLE,
        "--land-sea-mask-vicinity",
        "100000",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_regrid_nearest_2_multi_realization(tmp_path):
    """Test nearest-2 neighbour regridding"""
    kgo_dir = acc.kgo_root() / "regrid"
    kgo_path = kgo_dir / "nearest_2/kgo_multi_realization.nc"
    input_path = kgo_dir / "landmask/global_cutout_multi_realization.nc"
    target_path = kgo_dir / "landmask/ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        target_path,
        "--output",
        output_path,
        "--regrid-mode",
        "nearest-2",
        "--regridded-title",
        MOGREPS_G_UK_TITLE,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_regrid_bilinear_2_multi_realization(tmp_path):
    """Test bilinear-2 regridding"""
    kgo_dir = acc.kgo_root() / "regrid"
    kgo_path = kgo_dir / "bilinear_2/kgo_multi_realization.nc"
    input_path = kgo_dir / "landmask/global_cutout_multi_realization.nc"
    target_path = kgo_dir / "landmask/ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        target_path,
        "--output",
        output_path,
        "--regrid-mode",
        "bilinear-2",
        "--regridded-title",
        MOGREPS_G_UK_TITLE,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_regrid_nearest_landmask_2_multi_realization(tmp_path):
    """Test nearest neighbour-2 with land sea mask and realizations"""
    kgo_dir = acc.kgo_root() / "regrid"
    kgo_path = kgo_dir / "nearest_landmask_2/kgo_multi_realization.nc"
    input_path = kgo_dir / "landmask/global_cutout_multi_realization.nc"
    landmask_path = kgo_dir / "landmask/engl_landmask.nc"
    target_path = kgo_dir / "landmask/ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        target_path,
        landmask_path,
        "--output",
        output_path,
        "--regrid-mode",
        "nearest-with-mask-2",
        "--regridded-title",
        MOGREPS_G_UK_TITLE,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_regrid_bilinear_landmask_2_multi_realization(tmp_path):
    """Test bilinear-2 with land sea mask and realizations"""
    kgo_dir = acc.kgo_root() / "regrid"
    kgo_path = kgo_dir / "bilinear_landmask_2/kgo_multi_realization.nc"
    input_path = kgo_dir / "landmask/global_cutout_multi_realization.nc"
    landmask_path = kgo_dir / "landmask/engl_landmask.nc"
    target_path = kgo_dir / "landmask/ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        target_path,
        landmask_path,
        "--output",
        output_path,
        "--regrid-mode",
        "bilinear-with-mask-2",
        "--regridded-title",
        MOGREPS_G_UK_TITLE,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=0.05)
