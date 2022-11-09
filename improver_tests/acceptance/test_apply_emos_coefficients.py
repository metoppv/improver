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
"""Tests for the apply-emos-coefficients CLI"""

import pytest

from improver.constants import LOOSE_TOLERANCE

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_normal(tmp_path):
    """Test diagnostic with assumed normal distribution"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/normal"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    emos_est_path = kgo_dir / "normal_coefficients.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        "--random-seed",
        "0",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


def test_truncated_normal(tmp_path):
    """Test diagnostic with assumed truncated normal distribution"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/truncated_normal"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    emos_est_path = kgo_dir / "truncated_normal_coefficients.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        "--random-seed",
        "0",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


def test_normal_point_by_point_sites(tmp_path):
    """Test using a normal distribution when coefficients have been calculated
    independently at each site (initial guess and minimisation)."""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/sites/point_by_point"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / ".." / "realization_input.nc"
    emos_est_path = kgo_dir / "coefficients.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        "--random-seed",
        "0",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


def test_normal_sites_subhourly(tmp_path):
    """Test using a normal distribution for site forecasts that differ
    in terms of spatial extent, forecast reference time and forecast period
    from the coefficients."""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/sites/offset"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "offset_input.nc"
    emos_est_path = kgo_dir / "coefficients.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        "--random-seed",
        "0",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


def test_realizations_input_land_sea(tmp_path):
    """Test realizations as input with a land sea mask"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/land_sea"
    kgo_path = kgo_dir / "realizations_kgo.nc"
    input_path = kgo_dir / "../normal/input.nc"
    emos_est_path = kgo_dir / "../normal/normal_coefficients.nc"
    land_sea_path = kgo_dir / "landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        land_sea_path,
        "--random-seed",
        "0",
        "--land-sea-mask-name",
        "land_binary_mask",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


def test_realizations_as_predictor(tmp_path):
    """Implementation of test using non-default predictor realizations"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/realizations"
    kgo_path = kgo_dir / "realizations_kgo.nc"
    input_path = kgo_dir / "../normal/input.nc"
    emos_est_path = kgo_dir / "realizations_coefficients.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        "--predictor",
        "realizations",
        "--random-seed",
        "0",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


def test_probabilities(tmp_path):
    """Test using probabilities as input"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/probabilities"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    emos_est_path = kgo_dir / "../normal/normal_coefficients.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        "--realizations-count",
        "18",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE, rtol=LOOSE_TOLERANCE)


def test_probabilities_input_land_sea(tmp_path):
    """Test probabilities as input with a land sea mask"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/land_sea"
    kgo_path = kgo_dir / "probabilities_kgo.nc"
    input_path = kgo_dir / "../probabilities/input.nc"
    emos_est_path = kgo_dir / "../normal/normal_coefficients.nc"
    land_sea_path = kgo_dir / "landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        land_sea_path,
        "--realizations-count",
        "18",
        "--land-sea-mask-name",
        "land_binary_mask",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


def test_probabilities_error(tmp_path):
    """Test using probabilities as input without num_realizations"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/probabilities"
    input_path = kgo_dir / "input.nc"
    emos_est_path = kgo_dir / "../normal/normal_coefficients.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        "--output",
        output_path,
    ]
    with pytest.raises(ValueError, match=".*provided as probabilities.*"):
        run_cli(args)


def test_percentiles(tmp_path):
    """Test using percentiles as input"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/percentiles"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    emos_est_path = kgo_dir / "../normal/normal_coefficients.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        "--realizations-count",
        "18",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE, rtol=LOOSE_TOLERANCE)


def test_percentiles_input_land_sea(tmp_path):
    """Test percentiles as input with a land sea mask"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/land_sea"
    kgo_path = kgo_dir / "percentiles_kgo.nc"
    input_path = kgo_dir / "../percentiles/input.nc"
    emos_est_path = kgo_dir / "../normal/normal_coefficients.nc"
    land_sea_path = kgo_dir / "landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        land_sea_path,
        "--realizations-count",
        "18",
        "--land-sea-mask-name",
        "land_binary_mask",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


def test_alternative_percentiles(tmp_path):
    """Test using percentiles as input with an alternative set of
    percentiles specified."""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/alternative_percentiles"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "../percentiles/input.nc"
    emos_est_path = kgo_dir / "../normal/normal_coefficients.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        "--realizations-count",
        "18",
        "--percentiles",
        "25,50,75",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE, rtol=LOOSE_TOLERANCE)


def test_percentiles_error(tmp_path):
    """Test using percentiles as input"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/percentiles"
    input_path = kgo_dir / "input.nc"
    emos_est_path = kgo_dir / "../normal/normal_coefficients.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        "--output",
        output_path,
    ]
    with pytest.raises(ValueError):
        run_cli(args)


def test_rebadged_percentiles(tmp_path):
    """Test using realizations rebadged as percentiles as input"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/percentiles"
    kgo_path = kgo_dir / "kgo.nc"
    emos_est_path = kgo_dir / "../normal/normal_coefficients.nc"
    output_path = tmp_path / "output.nc"
    args = [
        kgo_dir / "../rebadged_percentiles/input.nc",
        emos_est_path,
        "--realizations-count",
        "18",
        "--output",
        output_path,
    ]
    run_cli(args)
    # The known good output in this case is the same as when passing in
    # percentiles directly, apart from a difference in the coordinates, such
    # that the percentile input will have a percentile coordinate, whilst the
    # rebadged percentile input will result in a realization coordinate.
    acc.compare(
        output_path,
        kgo_path,
        exclude_vars=["realization", "percentile"],
        atol=LOOSE_TOLERANCE,
        rtol=LOOSE_TOLERANCE,
    )


def test_percentiles_in_probabilities_out(tmp_path):
    """Test using percentiles as input whilst providing a probability
    template cube, so that probabilities are output."""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/percentiles"
    kgo_path = kgo_dir / "../probabilities/kgo.nc"
    input_path = kgo_dir / "input.nc"
    emos_est_path = kgo_dir / "../normal/normal_coefficients.nc"
    prob_template_path = kgo_dir / "../probabilities/input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        prob_template_path,
        "--realizations-count",
        "18",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE, rtol=LOOSE_TOLERANCE)


def test_percentile_sites_additional_predictor(tmp_path):
    """Test using percentile site forecasts with a static additional
    predictor."""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/sites/additional_predictor"
    kgo_path = kgo_dir / "percentile_kgo.nc"
    input_path = kgo_dir / ".." / "percentile_input.nc"
    emos_est_path = kgo_dir / "coefficients.nc"
    additional_predictor_path = kgo_dir / "altitude.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        additional_predictor_path,
        "--realizations-count",
        "19",
        "--random-seed",
        "0",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


def test_percentile_sites_period_diagnostic(tmp_path):
    """Test using percentile site forecasts where the diagnostic supplied
    represents a period e.g. daytime max temperature."""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/sites/percentile_period"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / ".." / "percentile_period_input.nc"
    emos_est_path = kgo_dir / "coefficients.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        "--realizations-count",
        "18",
        "--random-seed",
        "0",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


def test_perc_in_prob_out_sites_additional_predictor(tmp_path):
    """Test using percentile site forecasts with a static additional
    predictor and a probability template to generate a probability
    site forecast."""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/sites/additional_predictor"
    kgo_path = kgo_dir / "probability_kgo.nc"
    input_path = kgo_dir / ".." / "percentile_input.nc"
    emos_est_path = kgo_dir / "coefficients.nc"
    additional_predictor_path = kgo_dir / "altitude.nc"
    prob_template = kgo_dir / "probability_template.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        additional_predictor_path,
        prob_template,
        "--realizations-count",
        "19",
        "--random-seed",
        "0",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


def test_no_coefficients(tmp_path):
    """Test no coefficients provided"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/normal"
    kgo_path = kgo_dir / "kgo_with_comment.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--random-seed",
        "0",
        "--output",
        output_path,
    ]
    with pytest.warns(UserWarning, match=".*no coefficients provided.*"):
        run_cli(args)
    # Check output matches input excluding the comment attribute.
    acc.compare(
        output_path,
        input_path,
        recreate=False,
        atol=LOOSE_TOLERANCE,
        rtol=LOOSE_TOLERANCE,
        exclude_attributes="comment",
    )
    # Check output matches kgo.
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


def test_no_coefficients_percentiles(tmp_path):
    """Test returning alternative percentiles when no coefficients are provided"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/subsetted_percentiles"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        "--random-seed",
        "0",
        "--percentiles",
        "25,50,75",
        "--output",
        output_path,
    ]
    with pytest.warns(UserWarning, match=".*no coefficients provided.*"):
        run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


def test_no_coefficients_with_prob_template(tmp_path):
    """Test no coefficients provided with a probability template."""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/sites/additional_predictor"
    input_path = kgo_dir / ".." / "percentile_input.nc"
    prob_template = kgo_dir / "probability_template.nc"
    kgo_path = kgo_dir / "probability_template_kgo.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        prob_template,
        "--realizations-count",
        "19",
        "--random-seed",
        "0",
        "--output",
        output_path,
    ]
    with pytest.warns(
        UserWarning, match=".*no coefficients provided.*probability template.*"
    ):
        run_cli(args)
    # Check output matches the probability template excluding the comment attribute.
    acc.compare(
        output_path,
        prob_template,
        recreate=False,
        atol=LOOSE_TOLERANCE,
        rtol=LOOSE_TOLERANCE,
        exclude_attributes="comment",
    )
    # Check output matches kgo.
    acc.compare(
        output_path, kgo_path, atol=LOOSE_TOLERANCE, rtol=LOOSE_TOLERANCE,
    )


def test_wrong_coefficients(tmp_path):
    """Test wrong coefficients provided"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/normal"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        input_path,
        "--random-seed",
        "0",
        "--output",
        output_path,
    ]
    with pytest.raises(ValueError, match="Multiple items have been provided.*"):
        run_cli(args)


def test_matching_validity_times(tmp_path):
    """Test passing validity times when the forecast validity time matches
    one of the validity times within the list."""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/normal"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    emos_est_path = kgo_dir / "normal_coefficients.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        "--validity-times",
        "1500,1800,2100",
        "--random-seed",
        "0",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


def test_mismatching_validity_times(tmp_path):
    """Test passing validity times when the forecast validity time does not match
    any of the validity times within the list."""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/normal"
    kgo_path = kgo_dir / "kgo_with_comment.nc"
    input_path = kgo_dir / "input.nc"
    emos_est_path = kgo_dir / "normal_coefficients.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        "--validity-times",
        "0600,0900,1200",
        "--random-seed",
        "0",
        "--output",
        output_path,
    ]
    run_cli(args)
    # Check output matches input excluding the comment attribute.
    acc.compare(
        output_path,
        input_path,
        recreate=False,
        atol=LOOSE_TOLERANCE,
        rtol=LOOSE_TOLERANCE,
        exclude_attributes="comment",
    )
    # Check output matches kgo.
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


def test_mismatching_validity_times_percentiles(tmp_path):
    """Test passing validity times when the forecast validity time does not match
    any of the validity times within the list. The desired percentiles are supplied."""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/subsetted_percentiles"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    emos_est_path = (
        acc.kgo_root() / "apply-emos-coefficients/normal/normal_coefficients.nc"
    )
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        "--validity-times",
        "1200,1500,1800",
        "--random-seed",
        "0",
        "--percentiles",
        "25,50,75",
        "--output",
        output_path,
    ]
    run_cli(args)
    # Check output matches kgo.
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)
