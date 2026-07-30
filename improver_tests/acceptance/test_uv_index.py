# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the uv-index CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic UV index calculation"""
    kgo_dir = acc.kgo_root() / "uv-index/basic"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [
        kgo_dir
        / ("20181210T0600Z-PT0000H00M-radiation_flux_in_uv_downward_at_surface.nc")
    ]
    output_path = tmp_path / "output.nc"
    args = [
        *input_paths,
        "--output",
        output_path,
        "--model-id-attr",
        "mosg__model_configuration",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_non_default_scale_factor(tmp_path):
    """Test UV index calculation with non-default scale factor"""
    kgo_dir = acc.kgo_root() / "uv-index"
    kgo_path = kgo_dir / "non_default_scale_factor/non_default_scale_factor_kgo.nc"
    input_paths = [
        kgo_dir
        / (
            "basic/20181210T0600Z-PT0000H00M-radiation_flux_in_uv_downward_at_surface.nc"
        )
    ]
    output_path = tmp_path / "output.nc"
    args = [
        *input_paths,
        "--output",
        output_path,
        "--scale-factor",
        "0.1",
        "--model-id-attr",
        "mosg__model_configuration",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
