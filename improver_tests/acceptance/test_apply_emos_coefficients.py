# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
    input_path = kgo_dir / ".." / "input.nc"
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
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path, atol=LOOSE_TOLERANCE)


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


def test_no_coefficients(tmp_path):
    """Test no coefficients provided"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/normal"
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
    acc.compare(
        output_path,
        input_path,
        recreate=False,
        atol=LOOSE_TOLERANCE,
        rtol=LOOSE_TOLERANCE,
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
    from iris.exceptions import ConstraintMismatchError

    with pytest.raises(ConstraintMismatchError, match="Got 0 cubes for constraint.*"):
        run_cli(args)


def test_wrong_land_sea_mask(tmp_path):
    """Test wrong land_sea_mask provided"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/normal"
    emos_est_path = kgo_dir / "normal_coefficients.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [
        input_path,
        emos_est_path,
        input_path,
        "--random-seed",
        "0",
        "--output",
        output_path,
    ]
    with pytest.raises(ValueError, match=".*land_sea_mask.*"):
        run_cli(args)


def test_wrong_forecast_coefficients(tmp_path):
    """Test forecast cube being a coefficients cube"""
    kgo_dir = acc.kgo_root() / "apply-emos-coefficients/normal"
    emos_est_path = kgo_dir / "normal_coefficients.nc"
    output_path = tmp_path / "output.nc"
    args = [
        emos_est_path,
        "--random-seed",
        "0",
        "--output",
        output_path,
    ]
    from iris.exceptions import MergeError

    with pytest.raises(MergeError, match="failed to merge into a single cube.*"):
        run_cli(args)
