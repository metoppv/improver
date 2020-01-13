# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize(
    "extra_args,kgo_file",
    (([], "nearest_uk_temperatures.nc"),
     (["--similar-altitude"], "mindz_uk_temperatures.nc")))
def test_nearest_uk(tmp_path, extra_args, kgo_file):
    """Test spot extraction using nearest location"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    kgo_path = kgo_dir / f"outputs/{kgo_file}"
    output_path = tmp_path / "output.nc"
    args = [neighbour_path, diag_path, "--output", output_path, *extra_args]
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
    args = [neighbour_path, diag_path, lapse_path, "--output", output_path,
            "--apply-lapse-rate-correction"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_global_extract_on_uk_grid(tmp_path):
    """Test attempting to extract global sites from a UK-only grid"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_global.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    output_path = tmp_path / "output.nc"
    args = [neighbour_path, diag_path, "--output", output_path]
    with pytest.raises(ValueError, match=".*same grid.*"):
        run_cli(args)


def test_nearest_minimum_dz_unavailable(tmp_path):
    """Test attempting to extract global sites from a UK-only grid"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/nearest_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    output_path = tmp_path / "output.nc"
    args = [neighbour_path, diag_path, "--output", output_path,
            "--similar-altitude"]
    with pytest.raises(ValueError, match=".*neighbour_selection_method.*"):
        run_cli(args)


def test_lapse_rate_mismatch(tmp_path):
    """Test lapse rate adjustment mismatch between datasets"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    lapse_path = kgo_dir / "inputs/ukvx_lapse_rate_2m.nc"
    kgo_path = kgo_dir / "outputs/nearest_uk_temperatures.nc"
    output_path = tmp_path / "output.nc"
    args = [neighbour_path, diag_path, lapse_path, "--output", output_path,
            "--apply-lapse-rate-correction"]
    with pytest.warns(UserWarning, match=".*height.*not adjusted.*"):
        run_cli(args)
    acc.compare(output_path, kgo_path)


def test_lapse_rate_wrong_height(tmp_path):
    """Test lapse rate adjustment height inconsistent between files"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    lapse_path = kgo_dir / "inputs/ukvx_lapse_rate_no_height.nc"
    output_path = tmp_path / "output.nc"
    args = [neighbour_path, diag_path, lapse_path, "--output", output_path,
            "--apply-lapse-rate-correction"]
    with pytest.raises(ValueError, match=".*single valued height.*"):
        run_cli(args)


def test_new_spot_title(tmp_path):
    """Test spot extraction with external JSON metadata"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    kgo_path = kgo_dir / "outputs/nearest_uk_temperatures_amended_metadata.nc"
    output_path = tmp_path / "output.nc"
    args = [neighbour_path, diag_path,
            "--new-title", "IMPROVER Spot Values",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_lapse_rate_non_temperature(tmp_path):
    """Test attempting to apply lapse rate to non-temperature data"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_pmsl.nc"
    lapse_path = kgo_dir / "inputs/ukvx_lapse_rate.nc"
    output_path = tmp_path / "output.nc"
    args = [neighbour_path, diag_path, lapse_path, "--output", output_path,
            "--apply-lapse-rate-correction"]
    with pytest.raises(ValueError, match=".*not air temperature.*"):
        run_cli(args)


def test_multiple_constraints(tmp_path):
    """Test use of multiple constraints"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    kgo_path = kgo_dir / "outputs/mindz_land_constraint_uk_temperatures.nc"
    output_path = tmp_path / "output.nc"
    args = [neighbour_path, diag_path, "--output", output_path,
            "--similar-altitude", "--land-constraint"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_percentile_thresholded_input(tmp_path):
    """Test extracting percentiles from thresholded input"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    threshold_path = kgo_dir / "inputs/enukx_temperature_thresholds.nc"
    kgo_path = kgo_dir / "outputs/extract_percentile_kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [neighbour_path, threshold_path, "--output", output_path,
            "--extract-percentiles", "50"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_percentile_percentile_input(tmp_path):
    """Test extracting percentiles from percentile input"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    threshold_path = kgo_dir / "inputs/enukx_temperature_percentiles.nc"
    kgo_path = kgo_dir / "outputs/extract_percentile_kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [neighbour_path, threshold_path, "--output", output_path,
            "--extract-percentiles", "50"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_percentile_unavailable(tmp_path):
    """Test extracting an unavailable percentile from percentile input"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    threshold_path = kgo_dir / "inputs/enukx_temperature_percentiles.nc"
    output_path = tmp_path / "output.nc"
    args = [neighbour_path, threshold_path, "--output", output_path,
            "--extract-percentiles", "45"]
    with pytest.raises(ValueError, match=".*percentile.*"):
        run_cli(args)


def test_percentile_deterministic(tmp_path):
    """Test extracting percentiles from deterministic input"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    kgo_path = kgo_dir / "outputs/nearest_uk_temperatures.nc"
    output_path = tmp_path / "output.nc"
    args = [neighbour_path, diag_path, "--output", output_path,
            "--extract-percentiles", "50"]
    with pytest.warns(UserWarning) as collected_warns:
        run_cli(args)
    assert len(collected_warns) == 1
    assert ("Diagnostic cube is not a known probabilistic type."
            in collected_warns[0].message.args[0])
    acc.compare(output_path, kgo_path)


def test_percentile_deterministic_quiet(tmp_path):
    """Test extracting percentiles from deterministic input"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    kgo_path = kgo_dir / "outputs/nearest_uk_temperatures.nc"
    output_path = tmp_path / "output.nc"
    args = [neighbour_path, diag_path, "--output", output_path,
            "--extract-percentiles", "50", "--suppress-warnings"]
    with pytest.warns(None) as collected_warns:
        run_cli(args)
    # check that no warning is collected
    assert len(collected_warns) == 0
    acc.compare(output_path, kgo_path)


def test_multiple_percentile_thresholded_input(tmp_path):
    """Test extracting multiple percentiles from thresholded input"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    threshold_path = kgo_dir / "inputs/enukx_temperature_thresholds.nc"
    kgo_path = kgo_dir / "outputs/extract_multiple_percentiles_kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [neighbour_path, threshold_path, "--output", output_path,
            "--extract-percentiles", "25, 50, 75"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_multiple_percentile_percentile_input(tmp_path):
    """Test extracting multiple percentiles from percentile input"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    threshold_path = kgo_dir / "inputs/enukx_temperature_percentiles.nc"
    kgo_path = kgo_dir / "outputs/extract_multiple_percentiles_kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [neighbour_path, threshold_path, "--output", output_path,
            "--extract-percentiles", "25, 50, 75"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_percentile_realization_input(tmp_path):
    """Test extracting percentiles from realization input"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    realization_path = kgo_dir / "inputs/enukx_temperature_realizations.nc"
    kgo_path = kgo_dir / "outputs/extract_percentile_kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [neighbour_path, realization_path, "--output", output_path,
            "--extract-percentiles", "50"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_multiple_percentile_realization_input(tmp_path):
    """Test extracting multiple percentiles from realization input"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    realization_path = kgo_dir / "inputs/enukx_temperature_realizations.nc"
    kgo_path = kgo_dir / "outputs/extract_multiple_percentiles_kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [neighbour_path, realization_path, "--output", output_path,
            "--extract-percentiles", "25, 50, 75"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_invalid_lapse_rate(tmp_path):
    """Test use of an invalid lapse rate adjustment cube"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    lapse_path = diag_path
    output_path = tmp_path / "output.nc"
    args = [neighbour_path, diag_path, lapse_path, "--output", output_path,
            "--apply-lapse-rate-correction"]
    with pytest.raises(ValueError, match=".*lapse rate.*"):
        run_cli(args)


def test_no_lapse_rate_data(tmp_path):
    """Test missing lapse rate adjustment data"""
    kgo_dir = acc.kgo_root() / "spot-extract"
    neighbour_path = kgo_dir / "inputs/all_methods_uk.nc"
    diag_path = kgo_dir / "inputs/ukvx_temperature.nc"
    output_path = tmp_path / "output.nc"
    args = [neighbour_path, diag_path, "--output", output_path,
            "--apply-lapse-rate-correction"]
    with pytest.warns(UserWarning, match=".*lapse rate.*"):
        run_cli(args)
