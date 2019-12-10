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
Tests for the weighted-blending CLI
"""

import pytest

from improver.tests.acceptance import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
PRECIP = "lwe_precipitation_rate"
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.slow
def test_basic_nonlin(tmp_path):
    """Test basic non linear weights"""
    kgo_dir = acc.kgo_root() / "weighted_blending/basic_nonlin"
    kgo_path = kgo_dir / "kgo.nc"
    input_dir = kgo_dir / "../basic_lin"
    input_paths = sorted((input_dir.glob("multiple_probabilities_rain_*H.nc")))
    output_path = tmp_path / "output.nc"
    args = ["--wts_calc_method", "nonlinear",
            "forecast_reference_time",
            "--cval", "0.85",
            *input_paths,
            output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_basic_lin(tmp_path):
    """Test basic linear weights"""
    kgo_dir = acc.kgo_root() / "weighted_blending/basic_lin"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = sorted((kgo_dir.glob("multiple_probabilities_rain_*H.nc")))
    output_path = tmp_path / "output.nc"
    args = ["forecast_reference_time",
            "--y0val", "20.0",
            "--ynval", "2.0",
            *input_paths,
            output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_bothoptions_fail(tmp_path):
    """Test linear and non linear options together fails"""
    kgo_dir = acc.kgo_root() / "weighted_blending/basic_lin"
    input_paths = sorted((kgo_dir.glob("multiple_probabilities_rain_*H.nc")))
    output_path = tmp_path / "output.nc"
    args = ["--wts_calc_method", "linear nonlinear",
            "time",
            *input_paths,
            output_path]
    with pytest.raises(SystemExit):
        run_cli(args)


def test_invalid_lin_nonlin(tmp_path):
    """Test linear and non linear options together fails"""
    kgo_dir = acc.kgo_root() / "weighted_blending/basic_lin"
    input_paths = sorted((kgo_dir.glob("multiple_probabilities_rain_*H.nc")))
    output_path = tmp_path / "output.nc"
    args = ["time",
            "--ynval", "1",
            "--cval", "0.5",
            *input_paths,
            output_path]
    with pytest.raises(SystemExit):
        run_cli(args)


def test_invalid_nonlin_lin(tmp_path):
    """Test linear and non linear options together fails"""
    kgo_dir = acc.kgo_root() / "weighted_blending/basic_lin"
    input_paths = sorted((kgo_dir.glob("multiple_probabilities_rain_*H.nc")))
    output_path = tmp_path / "output.nc"
    args = ["--wts_calc_method", "nonlinear",
            "time",
            "--ynval", "1",
            "--y0val", "0",
            *input_paths,
            output_path]
    with pytest.raises(SystemExit):
        run_cli(args)


@pytest.mark.xfail(reason="takes ~5 minutes to run")
def test_percentile(tmp_path):
    """Test percentile blending"""
    kgo_dir = acc.kgo_root() / "weighted_blending/percentiles"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = ["--wts_calc_method", "nonlinear",
            "forecast_reference_time",
            "--cval", "1.0",
            input_path,
            output_path]
    pytest.fail()
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_cycletime(tmp_path):
    """Test cycletime blending"""
    kgo_dir = acc.kgo_root() / "weighted_blending/cycletime"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = ["forecast_reference_time",
            "--y0val", "1.0",
            "--ynval", "4.0",
            "--cycletime", "20171129T0900Z",
            input_path,
            output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_model(tmp_path):
    """Test multi-model blending"""
    kgo_dir = acc.kgo_root() / "weighted_blending/model"
    kgo_path = kgo_dir / "kgo.nc"
    attr_path = kgo_dir / "../attributes.json"
    ukv_path = kgo_dir / "ukv_input.nc"
    enuk_path = kgo_dir / "enuk_input.nc"
    output_path = tmp_path / "output.nc"
    args = ["model_configuration",
            "--ynval", "1",
            "--y0val", "1",
            "--model_id_attr", "mosg__model_configuration",
            "--attributes_dict", attr_path,
            ukv_path, enuk_path,
            output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_realization_collapse(tmp_path):
    """Test realization collapsing"""
    kgo_dir = acc.kgo_root() / "weighted_blending/realizations"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = ["realization",
            "--ynval=1",
            "--y0val=1",
            input_path,
            output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_weights_dict(tmp_path):
    """Test use of weights dictionary"""
    kgo_dir = acc.kgo_root() / "weighted_blending/weights_from_dict"
    kgo_path = kgo_dir / "kgo.nc"
    ukv_path = kgo_dir / "../model/ukv_input.nc"
    enuk_path = kgo_dir / "../model/enuk_input.nc"
    dict_path = kgo_dir / "input_dict.json"
    attr_path = kgo_dir / "../attributes.json"
    output_path = tmp_path / "output.nc"
    args = ["--wts_calc_method", "dict",
            "--wts_dict", dict_path,
            "--weighting_coord", "forecast_period",
            "model_configuration",
            "--model_id_attr", "mosg__model_configuration",
            "--attributes_dict", attr_path,
            ukv_path, enuk_path,
            output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.xfail(reason="takes ~5 minutes to run")
def test_percentile_weights_dict(tmp_path):
    """Test percentile blending with weights dictionary"""
    kgo_dir = acc.kgo_root() / "weighted_blending/percentile_weights_from_dict"
    kgo_path = kgo_dir / "kgo.nc"
    ukv_path = kgo_dir / "ukv_input.nc"
    enuk_path = kgo_dir / "enuk_input.nc"
    dict_path = kgo_dir / "../weights_from_dict/input_dict.json"
    output_path = tmp_path / "output.nc"
    args = ["--wts_calc_method", "dict",
            "--wts_dict", dict_path,
            "--weighting_coord", "forecast_period",
            "model_configuration",
            "--model_id_attr", "mosg__model_configuration",
            ukv_path, enuk_path,
            output_path]
    pytest.fail()
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_accum_cycle_blend(tmp_path):
    """Test blending accumulation across cycle times"""
    kgo_dir = acc.kgo_root() / "weighted_blending/accum_cycle_blend"
    kgo_path = kgo_dir / "kgo.nc"
    input_paths = sorted(kgo_dir.glob("ukv_prob_accum_PT?H.nc"))
    output_path = tmp_path / "output.nc"
    args = ["forecast_reference_time",
            "--y0val=1",
            "--ynval=1",
            *input_paths,
            output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path, rtol=None)


def test_non_mo_model(tmp_path):
    """Test blending non-Met Office models"""
    kgo_dir = acc.kgo_root() / "weighted_blending/non_mo_model"
    kgo_path = kgo_dir / "kgo.nc"
    det_path = kgo_dir / "non_mo_det.nc"
    ens_path = kgo_dir / "non_mo_ens.nc"
    attr_path = kgo_dir / "../non_mo_attributes.json"
    output_path = tmp_path / "output.nc"
    args = ["model_configuration",
            "--y0val", "1",
            "--ynval", "1",
            det_path,
            ens_path,
            output_path,
            "--model_id_attr", "non_mo_model_config",
            "--attributes_dict", attr_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_nowcast_cycle_blending(tmp_path):
    """Test blending nowcast cycles"""
    kgo_dir = acc.kgo_root() / "weighted_blending/spatial_weights"
    kgo_path = kgo_dir / "kgo/cycle.nc"
    input_files = [kgo_dir /
                   f"nowcast_data/20181129T1000Z-PT{l:04}H00M-{PRECIP}.nc"
                   for l in range(2, 5)]
    output_path = tmp_path / "output.nc"
    args = ["forecast_reference_time",
            "--y0val", "1",
            "--ynval", "1",
            "--spatial_weights_from_mask",
            *input_files,
            output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_spatial_model_blending(tmp_path):
    """Test spatial model blending"""
    kgo_dir = acc.kgo_root() / "weighted_blending/spatial_weights"
    kgo_path = kgo_dir / "kgo/model.nc"
    input_files = [(kgo_dir /
                    f"{t}_data/20181129T1000Z-PT0002H00M-{PRECIP}.nc")
                   for t in ("nowcast", "ukvx")]
    attr_path = kgo_dir / "../attributes.json"
    output_path = tmp_path / "output.nc"
    args = ["model_configuration",
            "--y0val", "1",
            "--ynval", "1",
            "--spatial_weights_from_mask",
            "--model_id_attr", "mosg__model_configuration",
            "--attributes_dict", attr_path,
            *input_files,
            output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_nowcast_cycle_no_fuzzy(tmp_path):
    """Test blending nowcast cycles"""
    kgo_dir = acc.kgo_root() / "weighted_blending/spatial_weights"
    kgo_path = kgo_dir / "kgo/cycle_no_fuzzy.nc"
    input_files = [kgo_dir /
                   f"nowcast_data/20181129T1000Z-PT{l:04}H00M-{PRECIP}.nc"
                   for l in range(2, 5)]
    output_path = tmp_path / "output.nc"
    args = ["forecast_reference_time",
            "--y0val", "1",
            "--ynval", "1",
            "--spatial_weights_from_mask",
            "--fuzzy_length", "1",
            *input_files,
            output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_spatial_model_no_fuzzy(tmp_path):
    """Test spatial model blending"""
    kgo_dir = acc.kgo_root() / "weighted_blending/spatial_weights"
    kgo_path = kgo_dir / "kgo/model_no_fuzzy.nc"
    input_files = [(kgo_dir /
                    f"{t}_data/20181129T1000Z-PT0002H00M-{PRECIP}.nc")
                   for t in ("nowcast", "ukvx")]
    attr_path = kgo_dir / "../attributes.json"
    output_path = tmp_path / "output.nc"
    args = ["model_configuration",
            "--y0val", "1",
            "--ynval", "1",
            "--spatial_weights_from_mask",
            "--fuzzy_length", "1",
            "--model_id_attr", "mosg__model_configuration",
            "--attributes_dict", attr_path,
            *input_files,
            output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_three_model_blending(tmp_path):
    """Test blending three models"""
    kgo_dir = acc.kgo_root() / "weighted_blending/three_models"
    kgo_path = kgo_dir / "kgo.nc"
    input_files = [(kgo_dir /
                    f"{t}/20190101T0400Z-PT{l:04}H00M-precip_rate.nc")
                   for t, l in (("enukxhrly", 4), ("nc", 1), ("ukvx", 2))]
    attr_path = kgo_dir / "../attributes.json"
    dict_path = kgo_dir / "blending-weights-preciprate.json"
    output_path = tmp_path / "output.nc"
    args = ["model_configuration",
            "--spatial_weights_from_mask",
            "--wts_calc_method", "dict",
            "--weighting_coord", "forecast_period",
            "--cycletime", "20190101T0300Z",
            "--model_id_attr", "mosg__model_configuration",
            "--attributes_dict", attr_path,
            "--wts_dict", dict_path,
            *input_files,
            output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
