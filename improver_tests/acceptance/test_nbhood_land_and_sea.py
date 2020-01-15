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
Tests for the nbhood-land-and-sea CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic land-sea without topographic bands"""
    kgo_dir = acc.kgo_root() / "nbhood-land-and-sea/no_topographic_bands"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, mask_path,
            "--radii", "20000",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_radii_with_lead_times(tmp_path):
    """Test basic land-sea without topographic bands"""
    kgo_dir = acc.kgo_root() / "nbhood-land-and-sea/radii_no_topographic_bands"
    kgo_path = kgo_dir / "kgo.nc"
    imp_dir = acc.kgo_root() / "nbhood-land-and-sea/no_topographic_bands"
    input_path = imp_dir / "input.nc"
    mask_path = imp_dir / "ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, mask_path,
            "--radii", "18000,54000,90000,162000",
            "--lead-times", "0,36,72,144",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
@pytest.mark.parametrize("intermediate", (True, False))
def test_topographic_bands(tmp_path, intermediate):
    """Test land-sea with topographic bands"""
    kgo_dir = acc.kgo_root() / "nbhood-land-and-sea/topographic_bands"
    kgo_path = kgo_dir / "kgo.nc"
    land_kgo_path = kgo_dir / "kgo_land.nc"
    input_path = kgo_dir / "input.nc"
    bands_path = kgo_dir / "topographic_bands_land.nc"
    weights_path = kgo_dir / "weights_land.nc"
    output_path = tmp_path / "output.nc"
    land_output_path = tmp_path / "output_land.nc"
    if intermediate:
        im_args = ["--intermediate-output", land_output_path,
                   "--return-intermediate"]
    else:
        im_args = []
    args = [input_path, bands_path, weights_path,
            "--radii", "20000",
            *im_args,
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
    if intermediate:
        acc.compare(land_output_path, land_kgo_path)


def test_unnecessary_weights(tmp_path):
    """Test land-sea with additional unnecessary weights argument"""
    kgo_dir = acc.kgo_root() / "nbhood-land-and-sea/no_topographic_bands"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "ukvx_landmask.nc"
    weights_path = kgo_dir / "../topographic_bands/weights_land.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, mask_path, weights_path,
            "--radii", "20000",
            "--output", output_path]
    with pytest.raises(TypeError, match=".*weights cube.*"):
        run_cli(args)


def test_missing_weights(tmp_path):
    """Test land-sea with missing weights"""
    kgo_dir = acc.kgo_root() / "nbhood-land-and-sea/topographic_bands"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "topographic_bands_land.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, mask_path,
            "--radii", "20000",
            "--output", output_path]
    with pytest.raises(TypeError, match=".*weights cube.*"):
        run_cli(args)


def test_incorrect_weights(tmp_path):
    """Test land-sea with incorrect weights"""
    kgo_dir = acc.kgo_root() / "nbhood-land-and-sea/topographic_bands"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "topographic_bands_land.nc"
    weights_path = kgo_dir / "weights_any_surface.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, mask_path, weights_path,
            "--radii", "20000",
            "--output", output_path]
    with pytest.raises(ValueError, match=".*weights cube.*"):
        run_cli(args)


def test_topographic_sea(tmp_path):
    """Test topographic mask with sea"""
    kgo_dir = acc.kgo_root() / "nbhood-land-and-sea/topographic_bands"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "topographic_bands_any_surface.nc"
    weights_path = kgo_dir / "weights_land.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, mask_path, weights_path,
            "--radii", "20000",
            "--output", output_path]
    with pytest.raises(ValueError, match=".*mask cube.*"):
        run_cli(args)


@pytest.mark.parametrize("landsea", ["land", "sea"])
def test_landsea_only(tmp_path, landsea):
    """Test with land-only and sea-only masks"""
    kgo_dir = acc.kgo_root() / \
        f"nbhood-land-and-sea/no_topographic_bands/{landsea}_only"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, mask_path,
            "--radii", "20000",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_topographic_bands_probabilities(tmp_path):
    """Test topographic banding of probabilities"""
    kgo_dir = acc.kgo_root() / "nbhood-land-and-sea/topographic_bands"
    kgo_path = kgo_dir / "kgo_probs.nc"
    input_path = kgo_dir / "input_probs.nc"
    mask_path = kgo_dir / "../topographic_bands/topographic_bands_land.nc"
    weights_path = kgo_dir / "../topographic_bands/weights_land.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, mask_path, weights_path,
            "--radii", "20000",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_lead_time_radii_mismatch(tmp_path):
    """Tests that lead times and radii of different lengths errors."""
    kgo_dir = acc.kgo_root() / "nbhood-land-and-sea/no_topographic_bands"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, mask_path,
            "--radii", "20000,20001",
            "--lead-times", "1",
            "--output", output_path]
    with pytest.raises(RuntimeError, match=".*list of radii.*"):
        run_cli(args)
