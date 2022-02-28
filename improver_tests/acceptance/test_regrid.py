# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2022 Met Office.
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
