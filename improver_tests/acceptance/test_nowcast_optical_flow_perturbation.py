# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
Tests for the nowcast-optical-flow-perturbation CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


KGO_DIR = acc.kgo_root() / "nowcast-feature-branch/nowcast-optical-flow-perturbation"
FORECAST = "20190101T1600Z-PT0000H00M"
CURRENT = "20190101T1615Z"


def test_basic(tmp_path):
    """Test optical flow given whole input forecast"""
    kgo_path = KGO_DIR / "kgo.nc"
    obs_path = KGO_DIR / f"{CURRENT}_current_obs.nc"
    forecast_path = KGO_DIR / f"{FORECAST}-precip_rate.nc"
    advection_path = KGO_DIR / f"{FORECAST}-precipitation_advection_velocity.nc"
    orogenh_path = KGO_DIR / f"{FORECAST}-orographic_enhancement.nc"

    input_paths = [obs_path, forecast_path, advection_path, orogenh_path]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_slice(tmp_path):
    """Test optical flow given slice of input forecast"""
    kgo_path = KGO_DIR / "kgo.nc"
    obs_path = KGO_DIR / f"{CURRENT}_current_obs.nc"
    forecast_path = KGO_DIR / f"{CURRENT}_forecast_slice.nc"
    advection_path = KGO_DIR / f"{FORECAST}-precipitation_advection_velocity.nc"
    orogenh_path = KGO_DIR / f"{FORECAST}-orographic_enhancement.nc"

    input_paths = [obs_path, forecast_path, advection_path, orogenh_path]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_forecast_period(tmp_path):
    """Test different forecast period can be extracted and causes
    mismatched time error"""
    obs_path = KGO_DIR / f"{CURRENT}_current_obs.nc"
    forecast_path = KGO_DIR / f"{FORECAST}-precip_rate.nc"
    advection_path = KGO_DIR / f"{FORECAST}-precipitation_advection_velocity.nc"
    orogenh_path = KGO_DIR / f"{FORECAST}-orographic_enhancement.nc"

    input_paths = [obs_path, forecast_path, advection_path, orogenh_path]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--forecast-period", "30", "--output", output_path]
    with pytest.raises(ValueError, match=".*validity time must match.*"):
        run_cli(args)
