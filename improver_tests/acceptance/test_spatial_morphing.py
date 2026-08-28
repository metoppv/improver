# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the spatial-morphing CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize(
    "transitions,kgo,quantile_mapping",
    [
        ("no_implied_transitions.json", "no_implied_transitions_kgo.nc", False),
        ("transitions.json", "kgo.nc", False),
        ("transitions.json", "kgo_with_qm.nc", True),
    ],
)
def test_spatial_morphing(tmp_path, transitions, kgo, quantile_mapping):
    """Test using spatial morphing between two forecast sources with a cluster cube
    and a transitions file. This tests the situation where 1) there is no
    transition between the two sources so one of the input sources is returned,
    2) there is a transition between the two sources, and 3) there is a transition
    between the two sources and quantile mapping is applied. When using a linear
    morphing method, the quantile mapping has no effect, although it does have an
    effect when using the Google FILM morphing method."""
    kgo_dir = acc.kgo_root() / "spatial-morphing"
    kgo_path = kgo_dir / kgo
    cluster_path = kgo_dir / "clustering_result.nc"

    output_path = tmp_path / "output.nc"
    args = [
        kgo_dir / "source_a.nc",
        kgo_dir / "source_b.nc",
        cluster_path,
        "--forecast-period",
        "10800",
        "--cluster-number",
        "7",
        "--transitions",
        kgo_dir / transitions,
        "--morphing-method",
        "linear",
        "--output",
        output_path,
    ]
    if quantile_mapping:
        args.extend(["--apply-quantile-mapping=True"])
        args.extend(["--occurrence-threshold", "0.00003"])

    run_cli(args)
    acc.compare(output_path, kgo_path)
