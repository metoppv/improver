# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the nbhood-land-and-sea CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize(
    "kgo_name, shape", (("kgo.nc", "square"), ("kgo_circular.nc", "circular")),
)
def test_basic(tmp_path, kgo_name, shape):
    """Test basic land-sea without topographic bands"""
    kgo_dir = acc.kgo_root() / "nbhood-land-and-sea/no_topographic_bands"
    kgo_path = kgo_dir / kgo_name
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        mask_path,
        "--neighbourhood-shape",
        shape,
        "--radii",
        "20000",
        "--output",
        output_path,
    ]
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
    args = [
        input_path,
        mask_path,
        "--radii",
        "18000,54000,90000,162000",
        "--lead-times",
        "0,36,72,144",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_unnecessary_weights(tmp_path):
    """Test land-sea with additional unnecessary weights argument"""
    kgo_dir = acc.kgo_root() / "nbhood-land-and-sea/no_topographic_bands"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "ukvx_landmask.nc"
    weights_path = kgo_dir / "../topographic_bands/weights_land.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        mask_path,
        weights_path,
        "--neighbourhood-shape",
        "square",
        "--radii",
        "20000",
        "--output",
        output_path,
    ]
    with pytest.raises(TypeError, match=".*weights cube.*"):
        run_cli(args)


def test_missing_weights(tmp_path):
    """Test land-sea with missing weights"""
    kgo_dir = acc.kgo_root() / "nbhood-land-and-sea/topographic_bands"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "topographic_bands_land.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        mask_path,
        "--neighbourhood-shape",
        "square",
        "--radii",
        "20000",
        "--output",
        output_path,
    ]
    with pytest.raises(TypeError, match=".*weights cube.*"):
        run_cli(args)


def test_incorrect_weights(tmp_path):
    """Test land-sea with incorrect weights"""
    kgo_dir = acc.kgo_root() / "nbhood-land-and-sea/topographic_bands"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "topographic_bands_land.nc"
    weights_path = kgo_dir / "weights_any_surface.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        mask_path,
        weights_path,
        "--neighbourhood-shape",
        "square",
        "--radii",
        "20000",
        "--output",
        output_path,
    ]
    with pytest.raises(ValueError, match=".*weights cube.*"):
        run_cli(args)


def test_topographic_sea(tmp_path):
    """Test topographic mask with sea"""
    kgo_dir = acc.kgo_root() / "nbhood-land-and-sea/topographic_bands"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "topographic_bands_any_surface.nc"
    weights_path = kgo_dir / "weights_land.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        mask_path,
        weights_path,
        "--neighbourhood-shape",
        "square",
        "--radii",
        "20000",
        "--output",
        output_path,
    ]
    with pytest.raises(ValueError, match=".*mask cube.*"):
        run_cli(args)


@pytest.mark.parametrize("landsea", ["land", "sea"])
def test_landsea_only(tmp_path, landsea):
    """Test with land-only and sea-only masks"""
    kgo_dir = (
        acc.kgo_root() / f"nbhood-land-and-sea/no_topographic_bands/{landsea}_only"
    )
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        mask_path,
        "--neighbourhood-shape",
        "square",
        "--radii",
        "20000",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
@pytest.mark.parametrize(
    "kgo_name, shape",
    (("kgo_probs.nc", "square"), ("kgo_probs_circular.nc", "circular")),
)
def test_topographic_bands_probabilities(tmp_path, kgo_name, shape):
    """Test topographic banding of probabilities"""
    kgo_dir = acc.kgo_root() / "nbhood-land-and-sea/topographic_bands"
    kgo_path = kgo_dir / kgo_name
    input_path = kgo_dir / "input_probs.nc"
    mask_path = kgo_dir / "../topographic_bands/topographic_bands_land.nc"
    weights_path = kgo_dir / "../topographic_bands/weights_land.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        mask_path,
        weights_path,
        "--neighbourhood-shape",
        shape,
        "--radii",
        "20000",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_lead_time_radii_mismatch(tmp_path):
    """Tests that lead times and radii of different lengths errors."""
    kgo_dir = acc.kgo_root() / "nbhood-land-and-sea/no_topographic_bands"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        mask_path,
        "--neighbourhood-shape",
        "square",
        "--radii",
        "20000,20001",
        "--lead-times",
        "1",
        "--output",
        output_path,
    ]
    with pytest.raises(RuntimeError, match=".*list of radii.*"):
        run_cli(args)
