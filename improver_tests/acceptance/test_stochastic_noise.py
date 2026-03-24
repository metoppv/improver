# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the StochasticNoise plugin
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_basic(tmp_path):
    """Test basic stochastic noise addition."""
    kgo_dir = acc.kgo_root() / "stochastic_noise"
    kgo_path = kgo_dir / "unscaled/kgo.nc"
    dependence_template_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        dependence_template_path,
        "--ssft-init-params",
        "{'win_size': (100, 100), 'overlap': 0.3, 'war_thr': 0.1}",
        "--ssft-generate-params",
        "{'overlap': 0.3, 'seed': 0}",
        "--db-threshold",
        "0.03",
        "--db-threshold-units",
        "mm/hr",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_scale_non_positive_noise(tmp_path):
    """Test stochastic noise addition with scale_non_positive_noise=True"""
    kgo_dir = acc.kgo_root() / "stochastic_noise"
    kgo_path = kgo_dir / "scaled/kgo.nc"
    dependence_template_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        dependence_template_path,
        "--ssft-init-params",
        "{'win_size': (100, 100), 'overlap': 0.3, 'war_thr': 0.1}",
        "--ssft-generate-params",
        "{'overlap': 0.3, 'seed': 0}",
        "--db-threshold",
        "0.03",
        "--db-threshold-units",
        "mm/hr",
        "--scale-non-positive-noise",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
