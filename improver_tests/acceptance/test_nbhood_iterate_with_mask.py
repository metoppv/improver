# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the nbhood-iterate-with-mask CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.slow
def test_basic(tmp_path):
    """Test basic iterate with mask"""
    kgo_dir = acc.kgo_root() / "nbhood-iterate-with-mask/basic"
    kgo_path = kgo_dir / "kgo_basic.nc"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "mask.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        mask_path,
        "--coord-for-masking",
        "topographic_zone",
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
    (("kgo_collapsed.nc", "square"), ("kgo_collapsed_circular.nc", "circular")),
)
def test_collapse_bands(tmp_path, kgo_name, shape):
    """Test with collapsing orographic bands"""
    kgo_dir = acc.kgo_root() / "nbhood-iterate-with-mask/basic_collapse_bands"
    kgo_path = kgo_dir / kgo_name
    input_path = kgo_dir / "thresholded_input.nc"
    mask_path = kgo_dir / "orographic_bands_mask.nc"
    weights_path = kgo_dir / "orographic_bands_weights.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        mask_path,
        weights_path,
        "--coord-for-masking",
        "topographic_zone",
        "--neighbourhood-shape",
        shape,
        "--radii",
        "10000",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
