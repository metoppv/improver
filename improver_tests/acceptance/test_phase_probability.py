# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the phase-probability CLI
"""

import pytest

from improver_tests.acceptance import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.parametrize("gtype", ("equalarea", "latlon", "spot"))
@pytest.mark.parametrize("ptype", ("deterministic", "percentiles"))
@pytest.mark.parametrize(
    "kgo_name,input_file",
    [
        ("snow_kgo", "snow_sleet"),
        ("rain_kgo", "sleet_rain"),
        ("hail_kgo", "hail_rain"),
    ],
)
def test_phase_probabilities(tmp_path, kgo_name, input_file, ptype, gtype):
    """Test phase probability calculations for snow->sleet, sleet->rain and hail->rain.
    Parameterisation covers gridded and spot forecasts, and for both inputs
    that include a percentile coordinate and those that do not."""

    # Excessive testing, only need to demonstrate latlon grid works.
    if ptype == "percentiles" and gtype == "latlon":
        pytest.skip("Nope")

    kgo_dir = acc.kgo_root() / f"{CLI}/{gtype}"
    kgo_path = kgo_dir / f"{kgo_name}_{ptype}.nc"
    output_path = tmp_path / "output.nc"
    input_paths = [
        kgo_dir / "altitudes.nc",
        kgo_dir / f"{input_file}_{ptype}.nc",
    ]
    args = [*input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
