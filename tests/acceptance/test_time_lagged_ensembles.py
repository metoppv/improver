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
Tests for the time-lagged-ensembles CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
T2M = "temperature_at_surface"
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.slow
def test_basic(tmp_path):
    """Test basic time lagging"""
    kgo_dir = acc.kgo_root() / "time-lagged-ens/same_validity"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = [kgo_dir / f"20180924T1300Z-PT{l:04}H00M-{T2M}.nc"
                   for l in range(5, 11)]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_validity_error(tmp_path):
    """Test validity times mismatched"""
    kgo_dir = acc.kgo_root() / "time-lagged-ens/mixed_validity"
    input_paths = [kgo_dir / f"20180924T1300Z-PT0001H00M-{T2M}.nc",
                   kgo_dir / f"20180924T1900Z-PT0006H00M-{T2M}.nc"]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--output", output_path]
    with pytest.raises(ValueError, match=".*validity times.*"):
        run_cli(args)


def test_single_cube(tmp_path):
    """Test time lagging a single input cube"""
    kgo_dir = acc.kgo_root() / "time-lagged-ens/same_validity"
    kgo_path = kgo_dir / "kgo_single_cube.nc"
    input_paths = [kgo_dir / f"20180924T1300Z-PT0005H00M-{T2M}.nc"]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_renumbered_realizations(tmp_path):
    """Test renumbering non-unique realization numbers"""
    kgo_dir = acc.kgo_root() / "time-lagged-ens/renumbered_realizations"
    kgo_path = kgo_dir / "kgo.nc"
    input_dir = kgo_dir / "../same_validity"
    input_paths = [input_dir / f"20180924T1300Z-PT0005H00M-{T2M}.nc",
                   input_dir / f"20180924T1300Z-PT0005H00M-{T2M}.nc"]
    output_path = tmp_path / "output.nc"
    args = [*input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
