# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the convection-ratio CLI"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)

ALL_PARAMS = ["lwe_convective_precipitation_rate", "lwe_stratiform_precipitation_rate"]


def test_basic(tmp_path):
    """Test basic convection-ratio processing"""
    kgo_dir = acc.kgo_root() / f"{CLI}/basic"
    kgo_path = kgo_dir / "kgo.nc"
    param_paths = [kgo_dir / f"{p}.nc" for p in ALL_PARAMS]
    output_path = tmp_path / "output.nc"
    args = [
        *param_paths,
        "--output",
        output_path,
        "--model-id-attr",
        "mosg__model_configuration",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_too_many_files(tmp_path):
    """Test convection-ratio rejects more than two files"""
    kgo_dir = acc.kgo_root() / f"{CLI}/basic"
    param_paths = [kgo_dir / f"{p}.nc" for p in ALL_PARAMS + ["kgo.nc"]]
    output_path = tmp_path / "output.nc"
    args = [*param_paths, "--output", output_path]
    with pytest.raises(IOError):
        run_cli(args)
