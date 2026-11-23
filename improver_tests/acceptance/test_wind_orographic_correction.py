# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Acceptance tests for the wind-orographic-correction CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)
DATASET = "wind-orographic-correction"
KGO_DEFAULT = "kgo_default.nc"
KGO_TARGET_LEVELS = "kgo_target_height_levels.nc"
KGO_WITH_REALIZATION = "kgo_with_realization.nc"


def _required_path(path):
    """Return the path, raising FileNotFoundError if it does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"Required acceptance file missing: {path}")
    return path


def _no_realizations_dir():
    """Return the test-data directory for forecasts without realizations."""
    return acc.kgo_root() / DATASET / "no_realizations"


def _with_realizations_dir():
    """Return the test-data directory for ensemble forecasts."""
    return acc.kgo_root() / DATASET / "with_realizations"


def _input_paths(kgo_dir):
    """Return paths to the wind profile and four terrain ancillaries."""
    wind_profile = _required_path(kgo_dir / "input.nc")
    high_res_orog = _required_path(kgo_dir / "high_res_orog.nc")
    model_orog = _required_path(kgo_dir / "model_orog.nc")
    model_orog_stddev = _required_path(kgo_dir / "model_orog_stddev.nc")
    model_silhouette_roughness = _required_path(
        kgo_dir / "model_silhouette_roughness.nc"
    )
    return [
        wind_profile,
        high_res_orog,
        model_orog,
        model_orog_stddev,
        model_silhouette_roughness,
    ]


def _input_paths_with_realization(kgo_dir):
    """Return input paths for the ensemble test case."""
    return _input_paths(kgo_dir)


def _kgo_path(kgo_dir, filename):
    """Return the path to the known good output (KGO), checking it exists."""
    return _required_path(kgo_dir / filename)


@pytest.mark.slow
@pytest.mark.parametrize(
    "input_paths_getter,kgo_dir_getter,kgo_filename,cli_options",
    [
        pytest.param(
            _input_paths,
            _no_realizations_dir,
            KGO_DEFAULT,
            (),
            id="default-target-heights",
        ),  # Omitted targets keep the same height levels as the input forecast.
        pytest.param(
            _input_paths,
            _no_realizations_dir,
            KGO_TARGET_LEVELS,
            ("--target-height-levels=15,25",),
            id="explicit-target-heights",
        ),  # Interpolate to the requested heights before applying correction.
        pytest.param(
            _input_paths,
            _no_realizations_dir,
            KGO_DEFAULT,
            ("--target-height-levels=none",),
            id="none-target-heights-lowercase",
        ),  # Explicit lowercase 'none' keeps the same height levels as input.
        pytest.param(
            _input_paths,
            _no_realizations_dir,
            KGO_DEFAULT,
            ("--target-height-levels=NONE",),
            id="none-target-heights-uppercase",
        ),  # Explicit uppercase 'NONE' keeps the same height levels as input.
        pytest.param(
            _input_paths,
            _no_realizations_dir,
            KGO_TARGET_LEVELS,
            ("--target-height-levels=25, 15",),
            id="whitespace-and-unsorted-target-heights",
        ),  # Whitespace is accepted, and target heights are sorted.
        pytest.param(
            _input_paths_with_realization,
            _with_realizations_dir,
            KGO_WITH_REALIZATION,
            (),
            id="realization-input",
        ),  # Each ensemble realization is corrected independently.
    ],
)
def test_successful_wind_orographic_correction_matches_kgo(
    tmp_path,
    input_paths_getter,
    kgo_dir_getter,
    kgo_filename,
    cli_options,
):
    """Test wind-orographic-correction output matches the KGO."""
    kgo_dir = kgo_dir_getter()
    kgo_path = _kgo_path(kgo_dir, kgo_filename)
    input_paths = input_paths_getter(kgo_dir)
    output_path = tmp_path / "output.nc"

    args = [*input_paths, *cli_options, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize("target_value", ["not-a-number", "15,,25"])
def test_invalid_target_height_levels(tmp_path, target_value):
    """Test invalid target-height values raise ValueError."""
    kgo_dir = _no_realizations_dir()
    input_paths = _input_paths(kgo_dir)
    output_path = tmp_path / "output.nc"

    args = [
        *input_paths,
        f"--target-height-levels={target_value}",
        "--output",
        output_path,
    ]
    with pytest.raises(ValueError, match="could not convert string to float"):
        run_cli(args)
