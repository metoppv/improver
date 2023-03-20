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
Tests for the spot-extract CLI
"""

import pytest
from iris.exceptions import CoordinateNotFoundError

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


UK_SPOT_TITLE = "IMPROVER UK Spot Values"


@pytest.mark.parametrize(
    "neighbour_cube,extra_args,kgo_file",
    (
        ("all_methods_uk.nc", [], "nearest_uk_temperatures.nc"),
        ("all_methods_uk.nc", ["--similar-altitude"], "mindz_uk_temperatures.nc"),
        ("all_methods_uk_unique_ids.nc", [], "nearest_uk_temperatures_unique_ids.nc"),
    ),
)
def test_nearest_uk(tmp_path, neighbour_cube, extra_args, kgo_file):
    """Test spot extraction using nearest location"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / f"inputs/{neighbour_cube}"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    kgo_path = kgo_dir / f"outputs/{kgo_file}"
    output_path = tmp_path / "output.nc"
    args = [
        diag_path,
        neighbour_path,
        "--output",
        output_path,
        *extra_args,
        "--new-title",
        UK_SPOT_TITLE,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_lapse_rate_adjusted_uk(tmp_path):
    """Test spot extraction with lapse rate adjustment"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    lapse_path = kgo_dir / "inputs/ukvx_lapse_rate.nc"
    kgo_path = kgo_dir / "outputs/lapse_rate_adjusted_uk_temperatures.nc"
    output_path = tmp_path / "output.nc"
    args = [
        diag_path,
        lapse_path,
        neighbour_path,
        "--output",
        output_path,
        "--apply-lapse-rate-correction",
        "--new-title",
        UK_SPOT_TITLE,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_lapse_adjusting_multiple_percentile_input(tmp_path):
    """Test adjusting multiple percentiles from multiple percentile input"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/enukx_temperature_percentiles.nc"
    lapse_path = kgo_dir / "inputs/enukx_lapse_rate.nc"
    kgo_path = kgo_dir / "outputs/lapse_adjusted_multiple_percentile_kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        diag_path,
        lapse_path,
        neighbour_path,
        "--output",
        output_path,
        "--apply-lapse-rate-correction",
        "--new-title",
        UK_SPOT_TITLE,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_fixed_lapse_rate_adjusting(tmp_path):
    """Test adjusting multiple percentiles from multiple percentile input
    using a fixed lapse rate."""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/enukx_temperature_percentiles.nc"
    kgo_path = kgo_dir / "outputs/fixed_lapse_rate_adjusted_multiple_percentile_kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        diag_path,
        neighbour_path,
        "--output",
        output_path,
        "--apply-lapse-rate-correction",
        "--fixed-lapse-rate",
        "-6E-3",
        "--new-title",
        UK_SPOT_TITLE,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_global_extract_on_uk_grid(tmp_path):
    """Test attempting to extract global sites from a UK-only grid"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_global.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    output_path = tmp_path / "output.nc"
    args = [diag_path, neighbour_path, "--output", output_path]
    with pytest.raises(ValueError, match=".*same grid.*"):
        run_cli(args)


def test_nearest_minimum_dz_unavailable(tmp_path):
    """Test attempting to extract with an unavailable neighbour selection method"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/nearest_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    output_path = tmp_path / "output.nc"
    args = [diag_path, neighbour_path, "--output", output_path, "--similar-altitude"]
    with pytest.raises(ValueError, match=".*neighbour_selection_method.*"):
        run_cli(args)


def test_lapse_rate_mismatch(tmp_path):
    """Test lapse rate adjustment mismatch between datasets"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    lapse_path = kgo_dir / "inputs/ukvx_lapse_rate_2m.nc"
    output_path = tmp_path / "output.nc"
    args = [
        diag_path,
        lapse_path,
        neighbour_path,
        "--output",
        output_path,
        "--apply-lapse-rate-correction",
        "--new-title",
        UK_SPOT_TITLE,
    ]
    with pytest.raises(ValueError, match=".*height.*not adjusted.*"):
        run_cli(args)


def test_lapse_rate_wrong_height(tmp_path):
    """Test lapse rate adjustment height inconsistent between files"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    lapse_path = kgo_dir / "inputs/ukvx_lapse_rate_no_height.nc"
    output_path = tmp_path / "output.nc"
    args = [
        diag_path,
        lapse_path,
        neighbour_path,
        "--output",
        output_path,
        "--apply-lapse-rate-correction",
        "--new-title",
        UK_SPOT_TITLE,
    ]
    with pytest.raises(CoordinateNotFoundError, match=".*single valued height.*"):
        run_cli(args)


def test_lapse_rate_non_temperature(tmp_path):
    """Test attempting to apply lapse rate to non-temperature data"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_pmsl.nc"
    lapse_path = kgo_dir / "inputs/ukvx_lapse_rate.nc"
    output_path = tmp_path / "output.nc"
    args = [
        diag_path,
        lapse_path,
        neighbour_path,
        "--output",
        output_path,
        "--apply-lapse-rate-correction",
    ]
    with pytest.raises(ValueError, match=".*not air temperature.*"):
        run_cli(args)


def test_multiple_constraints(tmp_path):
    """Test use of multiple constraints"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    kgo_path = kgo_dir / "outputs/mindz_land_constraint_uk_temperatures.nc"
    output_path = tmp_path / "output.nc"
    args = [
        diag_path,
        neighbour_path,
        "--output",
        output_path,
        "--similar-altitude",
        "--land-constraint",
        "--new-title",
        UK_SPOT_TITLE,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_percentile_thresholded_input(tmp_path):
    """Test extracting percentiles from thresholded input"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    threshold_path = kgo_dir / "inputs/enukx_temperature_thresholds.nc"
    kgo_path = kgo_dir / "outputs/extract_percentile_kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        threshold_path,
        neighbour_path,
        "--output",
        output_path,
        "--extract-percentiles",
        "50",
        "--new-title",
        UK_SPOT_TITLE,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_percentile_percentile_input(tmp_path):
    """Test extracting percentiles from percentile input"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    threshold_path = kgo_dir / "inputs/enukx_temperature_percentiles.nc"
    kgo_path = kgo_dir / "outputs/extract_percentile_kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        threshold_path,
        neighbour_path,
        "--output",
        output_path,
        "--extract-percentiles",
        "50",
        "--new-title",
        UK_SPOT_TITLE,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_percentile_resampling(tmp_path):
    """Test extracting percentiles that are not provided by percentile
    input. This invokes percentile resampling to produce the requested
    outputs."""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    threshold_path = kgo_dir / "inputs/enukx_temperature_percentiles.nc"
    kgo_path = kgo_dir / "outputs/extract_resampled_percentiles.nc"
    output_path = tmp_path / "output.nc"
    args = [
        threshold_path,
        neighbour_path,
        "--output",
        output_path,
        "--extract-percentiles",
        "45,50,55",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_percentile_deterministic(tmp_path):
    """Test extracting percentiles from deterministic input"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    kgo_path = kgo_dir / "outputs/nearest_uk_temperatures.nc"
    output_path = tmp_path / "output.nc"
    args = [
        diag_path,
        neighbour_path,
        "--output",
        output_path,
        "--extract-percentiles",
        "50",
        "--new-title",
        UK_SPOT_TITLE,
    ]
    with pytest.warns(
        UserWarning, match="Diagnostic cube is not a known probabilistic type."
    ):
        run_cli(args)
    acc.compare(output_path, kgo_path)


def test_percentile_deterministic_quiet(tmp_path):
    """Test extracting percentiles from deterministic input. In this case the
    --suppress-warnings flag is enabled. This excludes the warning raised when
    spot-extract is set to extract percentiles and used with deterministic data.
    This is intended to reduce output in logs when this warning is expected and
    thus not useful."""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    kgo_path = kgo_dir / "outputs/nearest_uk_temperatures.nc"
    output_path = tmp_path / "output.nc"
    args = [
        diag_path,
        neighbour_path,
        "--output",
        output_path,
        "--extract-percentiles",
        "50",
        "--new-title",
        UK_SPOT_TITLE,
        "--suppress-warnings",
    ]
    with pytest.warns(None) as collected_warns:
        run_cli(args)

    msg = "Diagnostic cube is not a known probabilistic type."
    assert all([msg not in str(warning.message) for warning in collected_warns])
    acc.compare(output_path, kgo_path)


def test_multiple_percentile_thresholded_input(tmp_path):
    """Test extracting multiple percentiles from thresholded input"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    threshold_path = kgo_dir / "inputs/enukx_temperature_thresholds.nc"
    kgo_path = kgo_dir / "outputs/extract_multiple_percentiles_kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        threshold_path,
        neighbour_path,
        "--output",
        output_path,
        "--extract-percentiles",
        "25, 50, 75",
        "--new-title",
        UK_SPOT_TITLE,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_multiple_percentile_percentile_input(tmp_path):
    """Test extracting multiple percentiles from percentile input"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    threshold_path = kgo_dir / "inputs/enukx_temperature_percentiles.nc"
    kgo_path = kgo_dir / "outputs/extract_multiple_percentiles_kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        threshold_path,
        neighbour_path,
        "--output",
        output_path,
        "--extract-percentiles",
        "25, 50, 75",
        "--new-title",
        UK_SPOT_TITLE,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_disordered_percentiles_percentile_input(tmp_path):
    """Test extracting percentiles from percentile input where the
    requested percentile list is not monotonically increasing."""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    threshold_path = kgo_dir / "inputs/enukx_temperature_percentiles.nc"
    kgo_path = kgo_dir / "outputs/extract_multiple_percentiles_kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        threshold_path,
        neighbour_path,
        "--output",
        output_path,
        "--extract-percentiles",
        "50,75,25",
        "--new-title",
        UK_SPOT_TITLE,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_percentile_realization_input(tmp_path):
    """Test extracting percentiles from realization input"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    realization_path = kgo_dir / "inputs/enukx_temperature_realizations.nc"
    kgo_path = kgo_dir / "outputs/extract_percentile_kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        realization_path,
        neighbour_path,
        "--output",
        output_path,
        "--extract-percentiles",
        "50",
        "--new-title",
        UK_SPOT_TITLE,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_multiple_percentile_realization_input(tmp_path):
    """Test extracting multiple percentiles from realization input"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    realization_path = kgo_dir / "inputs/enukx_temperature_realizations.nc"
    kgo_path = kgo_dir / "outputs/extract_multiple_percentiles_kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        realization_path,
        neighbour_path,
        "--output",
        output_path,
        "--extract-percentiles",
        "25, 50, 75",
        "--new-title",
        UK_SPOT_TITLE,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_invalid_lapse_rate(tmp_path):
    """Test use of an invalid lapse rate adjustment cube"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    lapse_path = diag_path
    output_path = tmp_path / "output.nc"
    args = [
        diag_path,
        lapse_path,
        neighbour_path,
        "--output",
        output_path,
        "--apply-lapse-rate-correction",
        "--new-title",
        UK_SPOT_TITLE,
    ]
    with pytest.raises(ValueError, match=".*lapse rate.*"):
        run_cli(args)


def test_no_lapse_rate_data(tmp_path):
    """Test missing lapse rate adjustment data"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    output_path = tmp_path / "output.nc"
    args = [
        diag_path,
        neighbour_path,
        "--output",
        output_path,
        "--apply-lapse-rate-correction",
        "--new-title",
        UK_SPOT_TITLE,
    ]
    msg = "A lapse rate cube or fixed lapse rate was not provided"
    with pytest.warns(UserWarning, match=msg):
        run_cli(args)


def test_percentile_from_threshold_with_realizations(tmp_path):
    """Test requesting a percentile from a cube with thresholds where the realizations
    need collapsing first"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/enukx_precipacc_realizations_thresholds.nc"
    kgo_path = kgo_dir / "outputs/with_realization_collapse.nc"
    output_path = tmp_path / "output.nc"
    args = [
        diag_path,
        neighbour_path,
        "--output",
        output_path,
        "--realization-collapse",
        "--extract-percentiles",
        "50",
        "--new-title",
        "MOGREPS-UK Spot Values",
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_multi_time_input(tmp_path):
    """Test extracting from a cube with a time and threshold coordinate. Note
    that utilities.load.load_cube reverses the order of the leading dimensions
    on load. As such the KGO has the threshold and time coordinates in a
    different order to the input, but this is unrelated to spot-extract."""

    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/enukx_temperature_thresholds_multi_time.nc"
    kgo_path = kgo_dir / "outputs/multi_time_kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        diag_path,
        neighbour_path,
        "--output",
        output_path,
        "--new-title",
        UK_SPOT_TITLE,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
