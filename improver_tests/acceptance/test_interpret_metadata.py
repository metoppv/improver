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
"""
Tests for the interpret-metadata CLI
"""

import pytest
from clize.errors import UnknownOption

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)

KGO_compliant = """
cube name : probability_of_air_temperature_above_threshold
This is a gridded probabilities file
It contains probabilities of air temperature greater than or equal to thresholds
It has undergone some significant post-processing
It contains data from MOGREPS-UK
"""

KGO_non_compliant = """
cube name : weather_code
Non-compliant :
Attributes dict_keys(['institution', 'title', 'weather_code', 'weather_code_meaning', 'Conventions']) missing one or more mandatory values ['source']
No mosg__model_configuration on blended file
"""

KGO_verbose = """
cube name : probability_of_air_temperature_above_threshold
This is a gridded probabilities file
    Source: name, coordinates
It contains probabilities of air temperature greater than or equal to thresholds
    Source: name, threshold coordinate (probabilities only)
It has undergone some significant post-processing
    Source: title attribute
It contains data from MOGREPS-UK
    Source: model ID attribute
"""
ALL_KGOS = [KGO_compliant, KGO_non_compliant, KGO_verbose]


@pytest.mark.parametrize(
    "inputs,kgos,options",
    (
        (["temperature_realizations.nc"], [KGO_compliant], []),
        (["temperature_realizations.nc"], [KGO_verbose], ["--verbose"]),
        (
            ["non_compliant_weather_codes.nc", "temperature_realizations.nc"],
            [KGO_non_compliant, KGO_compliant],
            [],
        ),
        (
            ["non_compliant_weather_codes.nc", "temperature_realizations.nc"],
            [KGO_non_compliant],
            ["--failures-only"],
        ),
    ),
)
def test_interpretation(capsys, inputs, kgos, options):
    """Test metadata interpretation. Four tests are run:
    - A single compliant file
    - A single compliant file with verbose output
    - Multiple files, the first of which is non-compliant
    - Using the --failures-only option to only print output for non-compliant files.

    capsys is a pytest fixture that captures standard output/error for testing.
    """
    kgo_dir = acc.kgo_root() / "interpret_metadata"
    input_path = [kgo_dir / input for input in inputs]
    args = [*input_path, *options]

    if "non_compliant_weather_codes.nc" in inputs:
        with pytest.raises(ValueError, match=".*not metadata compliant.*"):
            run_cli(args)
    else:
        run_cli(args)
    captured = capsys.readouterr()

    for kgo in kgos:
        assert kgo in captured.out
    excluded_kgos = list(set(ALL_KGOS) ^ set(kgos))
    for kgo in excluded_kgos:
        assert kgo not in captured.out


def test_no_output_option_accepted():
    """Test that this CLI will not accept an output option, as it does not
    return a cube. This is to highlight  unexpected behaviour following
    future changes to CLI decorators."""
    kgo_dir = acc.kgo_root() / "interpret_metadata"
    input_path = kgo_dir / "temperature_realizations.nc"
    args = [input_path, "--output", "/dev/null"]
    with pytest.raises(UnknownOption, match=".*Unknown option '--output'.*"):
        run_cli(args)
