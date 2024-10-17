# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the generate-topography-bands-mask and
generate-topography-bands-weights CLIs.
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]

# Parameterisation to test generation of masks and weights with appropriate
# classes and KGO directories
MASK_WEIGHT = ["generate-topography-bands-mask", "generate-topography-bands-weights"]
BASIC_MULTI = ["basic", "multi_realization"]


@pytest.mark.parametrize("maskweight", MASK_WEIGHT)
@pytest.mark.parametrize("realizations", BASIC_MULTI)
def test_basic(tmp_path, maskweight, realizations):
    """Test basic generation of topography bands"""
    kgo_dir = acc.kgo_root() / f"{maskweight}/{realizations}"
    kgo_path = kgo_dir / "kgo.nc"
    orography_path = kgo_dir / "input_orog.nc"
    landmask_path = kgo_dir / "input_land.nc"
    output_path = tmp_path / "output.nc"
    args = [orography_path, landmask_path, "--output", output_path]
    acc.run_cli(maskweight)(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize("maskweight", MASK_WEIGHT)
def test_bounds_json(tmp_path, maskweight):
    """Test generation of topography bands mask with bounds"""
    kgo_dir = acc.kgo_root() / f"{maskweight}/basic"
    kgo_path = kgo_dir / "kgo_from_json_bounds.nc"
    orography_path = kgo_dir / "input_orog.nc"
    landmask_path = kgo_dir / "input_land.nc"
    bounds_path = kgo_dir / "bounds.json"
    output_path = tmp_path / "output.nc"
    args = [
        orography_path,
        landmask_path,
        "--bands-config",
        bounds_path,
        "--output",
        output_path,
    ]
    acc.run_cli(maskweight)(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize("maskweight", MASK_WEIGHT)
def test_basic_nolandsea(tmp_path, maskweight):
    """Test basic generation of topography bands mask without land/sea"""
    kgo_dir = acc.kgo_root() / f"{maskweight}/basic_no_landsea_mask"
    kgo_path = kgo_dir / "kgo.nc"
    orography_path = kgo_dir / "../basic/input_orog.nc"
    output_path = tmp_path / "output.nc"
    args = [orography_path, "--output", output_path]
    acc.run_cli(maskweight)(args)
    acc.compare(output_path, kgo_path)
