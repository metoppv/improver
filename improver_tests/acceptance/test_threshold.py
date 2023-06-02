# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
Tests for the threshold CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize(
    "extra_args,kgo_subdir",
    (
        (["--threshold-values", "280"], "basic"),
        (
            ["--threshold-values", "280", "--comparison-operator", "<="],
            "below_threshold",
        ),
        (["--threshold-values", "270,280,290"], "multiple_thresholds"),
        (
            ["--threshold-values", "6.85", "--threshold-units", "celsius"],
            "threshold_units",
        ),
        (["--threshold-values", "280", "--fuzzy-factor", "0.99"], "fuzzy_factor"),
        (
            [
                "--threshold-config",
                acc.kgo_root() / "threshold" / "fuzzy_bounds" / "threshold_config.json",
            ],
            "fuzzy_factor",
        ),
        (
            [
                "--threshold-values",
                "6.85",
                "--threshold-units",
                "celsius",
                "--fuzzy-factor",
                "0.2",
            ],
            "threshold_units_fuzzy_factor",
        ),
        (
            [
                "--threshold-config",
                acc.kgo_root() / "threshold" / "json" / "threshold_config.json",
            ],
            "basic",
        ),
    ),
)
def test_args(tmp_path, extra_args, kgo_subdir):
    """Test thresholding with different argument combinations using temperature data"""
    cli_dir = acc.kgo_root() / "threshold"
    kgo_dir = cli_dir / kgo_subdir
    kgo_path = kgo_dir / "kgo.nc"
    input_path = cli_dir / "basic" / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", output_path]
    if extra_args:
        args += extra_args
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_collapse_realization(tmp_path):
    """Test thresholding with collapsing realizations. Ensure that in this case,
    using unmasked data, that no warning is raised in relation to collapsing the
    coordinate."""
    kgo_dir = acc.kgo_root() / "threshold/coord_collapse"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../basic/input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--output",
        output_path,
        "--threshold-values",
        "280",
        "--collapse-coord",
        "realization",
    ]
    with pytest.warns(None) as record:
        run_cli(args)
    for msg in record:
        assert "Blending masked data without spatial" not in str(msg.message)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize(
    "extra_arg,kgo", (([], "kgo.nc"), (["--fill-masked", "inf"], "kgo_mask_filled.nc"))
)
def test_collapse_realization_masked_data(tmp_path, extra_arg, kgo):
    """Test thresholding and collapsing realizations where the data being
    thresholded is masked."""
    kgo_dir = acc.kgo_root() / "threshold/masked_collapse"
    kgo_path = kgo_dir / kgo
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--output",
        output_path,
        "--threshold-values",
        "500",
        "--collapse-coord",
        "realization",
    ]
    args += extra_arg
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize(
    "extra_args,kgo",
    (
        (["--vicinity", "10000"], "kgo.nc"),
        (
            ["--vicinity", "10000,20000", "--collapse-coord", "realization"],
            "kgo_multiple_vicinities.nc",
        ),
        (
            ["--vicinity", "10000", "--collapse-coord", "realization"],
            "kgo_collapsed.nc",
        ),
        (
            [
                acc.kgo_root() / "threshold" / "vicinity" / "landmask.nc",
                "--vicinity",
                "10000",
            ],
            "kgo_landmask.nc",
        ),
        (
            [
                acc.kgo_root() / "threshold" / "vicinity" / "landmask.nc",
                "--vicinity",
                "10000",
                "--collapse-coord",
                "realization",
            ],
            "kgo_landmask_collapsed.nc",
        ),
    ),
)
def test_vicinity(tmp_path, extra_args, kgo):
    """Test thresholding with vicinity"""
    kgo_dir = acc.kgo_root() / "threshold/vicinity"
    kgo_path = kgo_dir / kgo
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path]
    if extra_args:
        args += extra_args
    args += [
        "--output",
        output_path,
        "--threshold-values",
        "0.03,0.1,1.0",
        "--threshold-units",
        "mm hr-1",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_vicinity_masked(tmp_path):
    """Test thresholding with vicinity and masked precipitation"""
    kgo_dir = acc.kgo_root() / "threshold/vicinity"
    kgo_path = kgo_dir / "kgo_masked.nc"
    input_path = kgo_dir / "masked_precip.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--output",
        output_path,
        "--threshold-values",
        "0.03,0.1,1.0",
        "--threshold-units",
        "mm hr-1",
        "--vicinity",
        "10000",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_landmask_without_vicinity():
    """Test supplying a land-mask triggers an error"""
    kgo_dir = acc.kgo_root() / "threshold/vicinity"
    input_path = kgo_dir / "input.nc"
    args = [
        input_path,
        acc.kgo_root() / "threshold" / "vicinity" / "landmask.nc",
        "--threshold-values",
        "0.03",
    ]
    with pytest.raises(
        ValueError, match="Cannot apply land-mask cube without in-vicinity processing"
    ):
        run_cli(args)
