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
Tests for the blend-with-vicinity-and-rename CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
PRECIP = "lwe_precipitation_rate"
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


ATTRIBUTES_PATH = acc.kgo_root() / "blend-with-vicinity-and-rename/attributes.json"
BLEND_WEIGHTS_PATH = (
    acc.kgo_root() / "blend-with-vicinity-and-rename/blending_weights.json"
)
SOURCE_FILES = ["ncuk.nc", "ukvx.nc", "enukx.nc"]
SOURCE_DIR = acc.kgo_root() / "blend-with-vicinity-and-rename"


@pytest.mark.parametrize(
    "input_files,kgo_path",
    ((SOURCE_FILES, "with_nowcast"), (SOURCE_FILES[1:], "without_nowcast")),
)
def test_basic(tmp_path, input_files, kgo_path):
    """Test blend-with-vicinity-and-rename for the case where the vicinity on the
    input cubes matches (without_nowcast) and where it doesn't (with nowcast)"""
    kgo_dir = acc.kgo_root() / "blend-with-vicinity-and-rename" / kgo_path
    kgo_path = kgo_dir / "kgo.nc"
    source_files = [SOURCE_DIR / f for f in input_files]
    output_path = tmp_path / "output.nc"
    args = [
        "--new-name",
        "lwe_thickness_of_precipitation_amount_in_variable_vicinity",
        "--vicinity-radius",
        "10000",
        "--coordinate",
        "model_configuration",
        "--cycletime",
        "20230405T1100Z",
        "--spatial-weights-from-mask",
        "--model-id-attr",
        "mosg__model_configuration",
        "--record-run-attr",
        "mosg__model_run",
        "--weighting-coord",
        "forecast_period",
        "--weighting-config",
        BLEND_WEIGHTS_PATH,
        "--weighting-method",
        "dict",
        "--attributes-config",
        ATTRIBUTES_PATH,
        "--least-significant-digit",
        "3",
        *source_files,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
