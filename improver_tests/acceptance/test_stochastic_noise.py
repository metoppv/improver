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
    pytest.importorskip("pysteps")
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
    acc.compare(output_path, kgo_path, atol=1e-9, rtol=1e-9)


def test_scale_non_positive_noise(tmp_path):
    pytest.importorskip("pysteps")
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
    acc.compare(output_path, kgo_path, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("specify_fallback", [False, True])
def test_dry_realizations(tmp_path, specify_fallback):
    """Test stochastic noise addition with scale_non_positive_noise=True
    and non_positive_noise_floor set, and where the input realizations are completely
    dry. For precipitation, non-positive values correspond to dry grid points
    (zero precipitation), so this checks that the fallback/noise-floor logic still
    produces a valid dry-field output."""
    kgo_dir = acc.kgo_root() / "stochastic_noise"
    kgo_path = kgo_dir / "dry" / "kgo.nc"
    if specify_fallback:
        kgo_path = kgo_dir / "dry" / "kgo_with_fallback.nc"
    dependence_template_path = kgo_dir / "dry" / "input.nc"
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
        "--non-positive-noise-floor",
        "-10",
        "--output",
        output_path,
    ]
    if specify_fallback:
        args += ["--non-positive-fallback-range", "(-200.0, -100.0)"]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("by_source", [False, True])
def test_positive_regions(tmp_path, by_source):
    """Test when stochastic noise is added to positive (wet) regions."""
    pytest.importorskip("pysteps")
    if by_source:
        kgo_dir = acc.kgo_root() / "stochastic_noise" / "wet_by_source"
        dependence_template_path = kgo_dir / "input_with_cluster_sources.nc"
    else:
        kgo_dir = acc.kgo_root() / "stochastic_noise" / "wet"
        dependence_template_path = acc.kgo_root() / "stochastic_noise" / "input.nc"

    kgo_path = kgo_dir / "kgo.nc"

    output_path = tmp_path / "output.nc"
    args = [
        dependence_template_path,
        "--ssft-init-params",
        "{'win_size': (100, 100), 'overlap': 0.3, 'war_thr': 0.1}",
        "--ssft-generate-params",
        "{'overlap': 0.3, 'seed': 0}",
        "--db-threshold",
        "0.03",
        "--scale-non-positive-noise",
        "--db-threshold-units",
        "mm/hr",
        "--positive-region-noise-amplitude",
        "0.5",
        "--output",
        output_path,
    ]
    if by_source:
        args += ["--apply-noise-to-positive-values-by-source", "uk_ens"]
    else:
        args += ["--apply-noise-to-positive-values"]

    run_cli(args)
    acc.compare(output_path, kgo_path, atol=1e-9, rtol=1e-9)
