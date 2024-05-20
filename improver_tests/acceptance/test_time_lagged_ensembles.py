# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the time-lagged-ensembles CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
T2M = "temperature_at_surface"
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.slow
def test_basic(tmp_path):
    """Test basic time lagging"""
    kgo_dir = acc.kgo_root() / "time-lagged-ens/same_validity"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [
        kgo_dir / f"20180924T1300Z-PT{hr:04}H00M-{T2M}.nc" for hr in range(5, 11)
    ]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_validity_error(tmp_path):
    """Test validity times mismatched"""
    kgo_dir = acc.kgo_root() / "time-lagged-ens/mixed_validity"
    input_paths = [
        kgo_dir / f"20180924T1300Z-PT0001H00M-{T2M}.nc",
        kgo_dir / f"20180924T1900Z-PT0006H00M-{T2M}.nc",
    ]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--output", output_path]
    with pytest.raises(ValueError, match=".*validity times.*"):
        run_cli(args)


def test_single_cube(tmp_path):
    """Test time lagging a single input cube"""
    kgo_dir = acc.kgo_root() / "time-lagged-ens/same_validity"
    kgo_path = kgo_dir / "kgo_single_cube.nc"
    input_paths = [kgo_dir / f"20180924T1300Z-PT0005H00M-{T2M}.nc"]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_renumbered_realizations(tmp_path):
    """Test renumbering non-unique realization numbers"""
    kgo_dir = acc.kgo_root() / "time-lagged-ens/renumbered_realizations"
    kgo_path = kgo_dir / "kgo.nc"
    input_dir = kgo_dir / "../same_validity"
    input_paths = [
        input_dir / f"20180924T1300Z-PT0005H00M-{T2M}.nc",
        input_dir / f"20180924T1300Z-PT0005H00M-{T2M}.nc",
    ]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
