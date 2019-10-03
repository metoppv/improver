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
"""Tests for the combine CLI"""

import pytest

from improver.cli import combine
from improver.tests import acceptance as acc


@pytest.mark.acc
@acc.skip_if_kgo_missing
def test_basic(tmp_path):
    """Test basic combine operation"""
    kgo_dir = acc.kgo_root() / "combine/basic"
    kgo_path = kgo_dir / "kgo_cloud.nc"
    output_path = tmp_path / "output.nc"
    args = ["--operation=max",
            "--new-name=cloud_area_fraction",
            str(kgo_dir / "low_cloud.nc"),
            str(kgo_dir / "medium_cloud.nc"),
            str(output_path)]
    combine.main(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.acc
@acc.skip_if_kgo_missing
def test_metadata(tmp_path):
    """Test combining with a separate metadata file"""
    kgo_dir = acc.kgo_root() / "combine/metadata"
    kgo_path = kgo_dir / "kgo_prob_precip.nc"
    precip_meta = kgo_dir / "prob_precip.json"
    output_path = tmp_path / "output.nc"
    new_name = "probability_of_total_precipitation_rate_between_thresholds"
    args = ["--operation=-",
            f"--new-name={new_name}",
            f"--metadata_jsonfile={precip_meta}",
            str(kgo_dir / "precip_prob_0p1.nc"),
            str(kgo_dir / "precip_prob_1p0.nc"),
            str(output_path)]
    combine.main(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.acc
@acc.skip_if_kgo_missing
@pytest.mark.parametrize("minmax", ("min", "max"))
def test_minmax_temperatures(tmp_path, minmax):
    """Test combining minimum and maximum temperatures"""
    kgo_dir = acc.kgo_root() / "combine/bounds"
    kgo_path = kgo_dir / f"kgo_{minmax}.nc"
    timebound_meta = kgo_dir / "time_bound.json"
    temperatures = kgo_dir.glob(f"*temperature_at_screen_level_{minmax}.nc")
    output_path = tmp_path / "output.nc"
    args = [f"--operation={minmax}",
            f"--metadata_jsonfile={timebound_meta}",
            *[str(t) for t in temperatures],
            str(output_path)]
    combine.main(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.acc
@acc.skip_if_kgo_missing
def test_combine_accumulation(tmp_path):
    """Test combining precipitation accumulations"""
    kgo_dir = acc.kgo_root() / "combine/accum"
    kgo_path = kgo_dir / "kgo_accum.nc"
    rains = kgo_dir.glob("*rainfall_accumulation.nc")
    timebound_meta = kgo_dir / "time_bound.json"
    output_path = tmp_path / "output.nc"
    args = [f"--metadata_jsonfile={timebound_meta}",
            *[str(r) for r in rains],
            str(output_path)]
    combine.main(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.acc
@acc.skip_if_kgo_missing
def test_mean_temperature(tmp_path):
    """Test combining mean temperature"""
    kgo_dir = acc.kgo_root() / "combine/bounds"
    kgo_path = kgo_dir / "kgo_mean.nc"
    timebound_meta = kgo_dir / "time_bound.json"
    temperatures = kgo_dir.glob("*temperature_at_screen_level.nc")
    output_path = tmp_path / "output.nc"
    args = ["--operation=mean",
            f"--metadata_jsonfile={timebound_meta}",
            *[str(t) for t in temperatures],
            str(output_path)]
    combine.main(args)
    acc.compare(output_path, kgo_path)
