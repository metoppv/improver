# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the phase-change-level CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("model_id_attr", (True, False))
@pytest.mark.parametrize(
    "phase_type,kgo_name,horiz_interp",
    [
        ("snow-sleet", "snow_sleet", "True"),
        ("sleet-rain", "sleet_rain", "True"),
        ("hail-rain", "hail_rain", "True"),
        ("sleet-rain", "sleet_rain_unfilled", "False"),
    ],
)
def test_phase_change(tmp_path, phase_type, kgo_name, horiz_interp, model_id_attr):
    """Testing:
        snow/sleet level
        sleet/rain level
        hail/rain level
        sleet/rain level leaving below orography points unfilled.
        Tests are for with and without the provision of the model_id_attr attribute.
    """
    pytest.importorskip("stratify")
    test_dir = acc.kgo_root() / CLI
    kgo_name = "{}_kgo.nc".format(kgo_name)
    output_path = tmp_path / "output.nc"
    input_paths = [
        test_dir / x
        for x in ("wet_bulb_temperature.nc", "wbti.nc", "orog.nc", "land_mask.nc")
    ]
    args = [
        *input_paths,
        "--phase-change",
        phase_type,
        "--horizontal-interpolation",
        horiz_interp,
        "--output",
        output_path,
    ]
    if model_id_attr:
        args += ["--model-id-attr", "mosg__model_configuration"]
        kgo_dir = test_dir / "with_id_attr"
    else:
        kgo_dir = test_dir / "without_id_attr"
    kgo_path = kgo_dir / kgo_name
    run_cli(args)
    acc.compare(output_path, kgo_path)
