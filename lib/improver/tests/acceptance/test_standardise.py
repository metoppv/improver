# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
Tests for the standardise CLI
"""

import pytest

from improver.tests.acceptance import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
GLOBAL_UK_TITLE = "Global Model Forecast on UK 2 km Standard Grid"
UKV_GLOBAL_TITLE = "UKV Model Forecast on Global 10 km Standard Grid"
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_regrid_basic(tmp_path):
    """Test basic regridding"""
    # KGO for this test is the same as test_regrid_check_landmask below
    kgo_dir = acc.kgo_root() / "standardise/regrid-basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "global_cutout.nc"
    target_path = kgo_dir / "ukvx_grid.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--target_grid_filepath", target_path,
            "--output_filepath", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_regrid_nearest(tmp_path):
    """Test nearest neighbour regridding"""
    kgo_dir = acc.kgo_root() / "standardise/regrid-nearest"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../regrid-basic/global_cutout.nc"
    target_path = kgo_dir / "../regrid-basic/ukvx_grid.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--target_grid_filepath", target_path,
            "--output_filepath", output_path,
            "--regrid_mode=nearest",
            "--regridded_title", GLOBAL_UK_TITLE]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_regrid_extrapolate(tmp_path):
    """Test nearest neighbour regridding with extrapolation"""
    kgo_dir = acc.kgo_root() / "standardise/regrid-extrapolate"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../regrid-basic/ukvx_grid.nc"
    target_path = kgo_dir / "../regrid-basic/global_cutout.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--target_grid_filepath", target_path,
            "--output_filepath", output_path,
            "--regrid_mode=nearest",
            "--extrapolation_mode", "extrapolate",
            "--regridded_title", UKV_GLOBAL_TITLE]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_regrid_json(tmp_path):
    """Test regridding with a JSON metadata file"""
    kgo_dir = acc.kgo_root() / "standardise/metadata"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../regrid-basic/global_cutout.nc"
    target_path = kgo_dir / "../regrid-basic/ukvx_grid.nc"
    metadata_path = kgo_dir / "metadata.json"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--target_grid_filepath", target_path,
            "--output_filepath", output_path,
            "--json_file", metadata_path,
            "--regridded_title", GLOBAL_UK_TITLE]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_change_metadata(tmp_path):
    """Test applying a JSON metadata file"""
    kgo_dir = acc.kgo_root() / "standardise/metadata"
    kgo_path = kgo_dir / "kgo_no_regrid.nc"
    input_path = kgo_dir / "input.nc"
    metadata_path = kgo_dir / "radar_metadata.json"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--output_filepath", output_path,
            "--new_name", "lwe_precipitation_rate",
            "--new_units", "m s-1",
            "--json_file", metadata_path,
            "--coords_to_remove", "height"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_fix_float64(tmp_path):
    """Test conversion of float64 data to float32"""
    kgo_dir = acc.kgo_root() / "standardise/float64"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "float64_data.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output_filepath", output_path, "--fix_float64"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_check_float64():
    """Test detection of float64 data"""
    kgo_dir = acc.kgo_root() / "standardise/float64"
    input_path = kgo_dir / "float64_data.nc"
    args = [input_path]
    with pytest.raises(TypeError, match=".*64 bit.*"):
        run_cli(args)


def test_no_output_path():
    """Test missing output path"""
    kgo_dir = acc.kgo_root() / "standardise/float64"
    input_path = kgo_dir / "float64_data.nc"
    args = [input_path, "--fix_float64"]
    with pytest.raises(ValueError, match=".*output.*"):
        run_cli(args)


@pytest.mark.slow
def test_regrid_nearest_landmask(tmp_path):
    """Test nearest neighbour regridding with land sea mask"""
    kgo_dir = acc.kgo_root() / "standardise/regrid-landmask"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../regrid-basic/global_cutout.nc"
    landmask_path = kgo_dir / "../regrid-landmask/glm_landmask.nc"
    target_path = kgo_dir / "../regrid-landmask/ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--target_grid_filepath", target_path,
            "--input_landmask_filepath", landmask_path,
            "--output_filepath", output_path,
            "--regrid_mode=nearest-with-mask",
            "--regridded_title", GLOBAL_UK_TITLE]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_regrid_check_landmask(tmp_path):
    """Test land sea mask output matches other test"""
    # KGO for this test is the same as test_basic above
    kgo_dir = acc.kgo_root() / "standardise/regrid-nearest"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../regrid-basic/global_cutout.nc"
    landmask_path = kgo_dir / "../regrid-landmask/glm_landmask.nc"
    target_path = kgo_dir / "../regrid-basic/ukvx_grid.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--target_grid_filepath", target_path,
            "--input_landmask_filepath", landmask_path,
            "--output_filepath", output_path,
            "--regrid_mode=nearest-with-mask",
            "--regridded_title", GLOBAL_UK_TITLE]
    with pytest.warns(UserWarning, match=".*land_binary_mask.*"):
        run_cli(args)
    # Don't recreate output as it is the same as other test
    acc.compare(output_path, kgo_path, recreate=False)


def test_args_error_landmask(tmp_path):
    """Test land sea mask specified but no regrid mode"""
    kgo_dir = acc.kgo_root() / "standardise/regrid-landmask"
    input_path = kgo_dir / "../regrid-basic/global_cutout.nc"
    landmask_path = kgo_dir / "../regrid-landmask/glm_landmask.nc"
    target_path = kgo_dir / "../regrid-landmask/ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--target_grid_filepath", target_path,
            "--input_landmask_filepath", landmask_path,
            "--output_filepath", output_path]
    with pytest.raises(ValueError, match=".*nearest-with-mask.*"):
        run_cli(args)


def test_args_error_no_target_with_landmask(tmp_path):
    """Test land sea mask specified but no target grid"""
    kgo_dir = acc.kgo_root() / "standardise/regrid-landmask"
    input_path = kgo_dir / "../regrid-basic/global_cutout.nc"
    landmask_path = kgo_dir / "../regrid-landmask/glm_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--input_landmask_filepath", landmask_path,
            "--output_filepath", output_path,
            "--regrid_mode=nearest-with-mask"]
    with pytest.raises(ValueError, match=".target_grid.*"):
        run_cli(args)


def test_args_error_no_landmask(tmp_path):
    """Test landmask mode but no land sea mask provided"""
    kgo_dir = acc.kgo_root() / "standardise/regrid-landmask"
    input_path = kgo_dir / "../regrid-basic/global_cutout.nc"
    target_path = kgo_dir / "../regrid-landmask/ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--target_grid_filepath", target_path,
            "--output_filepath", output_path,
            "--regrid_mode=nearest-with-mask"]
    with pytest.raises(ValueError, match=".*input landmask.*"):
        run_cli(args)


@pytest.mark.slow
def test_regrid_nearest_landmask_multi_realization(tmp_path):
    """Test nearest neighbour with land sea mask and realizations"""
    kgo_dir = acc.kgo_root() / "standardise/regrid-landmask"
    kgo_path = kgo_dir / "kgo_multi_realization.nc"
    input_path = kgo_dir / "../regrid-basic/global_cutout_multi_realization.nc"
    landmask_path = kgo_dir / "../regrid-landmask/glm_landmask.nc"
    target_path = kgo_dir / "../regrid-landmask/ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path,
            "--target_grid_filepath", target_path,
            "--input_landmask_filepath", landmask_path,
            "--output_filepath", output_path,
            "--regrid_mode=nearest-with-mask",
            "--regridded_title", GLOBAL_UK_TITLE]
    run_cli(args)
    acc.compare(output_path, kgo_path)
