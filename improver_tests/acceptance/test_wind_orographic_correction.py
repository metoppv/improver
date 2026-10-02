# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the wind-orographic-correction CLI."""

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
    """Return path if present; otherwise raise a clear setup error."""
    if not path.exists():
        raise FileNotFoundError(f"Required acceptance file missing: {path}")
    return path


def _no_realizations_dir():
    return acc.kgo_root() / DATASET / "no_realizations"


def _with_realizations_dir():
    return acc.kgo_root() / DATASET / "with_realizations"


def _input_paths(kgo_dir):
    """Resolve canonical wind-orographic-correction input filenames."""
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
    """Resolve canonical inputs from the with_realizations folder."""
    return _input_paths(kgo_dir)


def _kgo_path(kgo_dir, filename):
    """Resolve canonical expected output file."""
    return _required_path(kgo_dir / filename)


@pytest.mark.slow
def test_basic(tmp_path):
    """Test unresolved-orography correction with default target heights."""
    kgo_dir = _no_realizations_dir()
    kgo_path = _kgo_path(kgo_dir, KGO_DEFAULT)
    input_paths = _input_paths(kgo_dir)
    output_path = tmp_path / "output.nc"

    args = [*input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_target_height_levels(tmp_path):
    """Test unresolved-orography correction on explicit target heights."""
    kgo_dir = _no_realizations_dir()
    kgo_path = _kgo_path(kgo_dir, KGO_TARGET_LEVELS)
    input_paths = _input_paths(kgo_dir)
    output_path = tmp_path / "output.nc"

    args = [
        *input_paths,
        "--target-height-levels=15,25",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
@pytest.mark.parametrize("none_value", ["none", "NONE"])
def test_target_height_levels_none_string(tmp_path, none_value):
    """Test explicit none values behave like omitted target heights."""
    kgo_dir = _no_realizations_dir()
    kgo_path = _kgo_path(kgo_dir, KGO_DEFAULT)
    input_paths = _input_paths(kgo_dir)
    output_path = tmp_path / "output.nc"

    args = [
        *input_paths,
        f"--target-height-levels={none_value}",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
def test_target_height_levels_whitespace_and_unsorted(tmp_path):
    """Test unsorted target heights with whitespace map to expected output."""
    kgo_dir = _no_realizations_dir()
    kgo_path = _kgo_path(kgo_dir, KGO_TARGET_LEVELS)
    input_paths = _input_paths(kgo_dir)
    output_path = tmp_path / "output.nc"

    args = [
        *input_paths,
        "--target-height-levels=25, 15",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize("target_value", ["not-a-number", "15,,25"])
def test_invalid_target_height_levels(tmp_path, target_value):
    """Test malformed target-height strings are rejected."""
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


@pytest.mark.slow
def test_with_realization_input(tmp_path):
    """Test unresolved-orography correction with realization dimension input."""
    kgo_dir = _with_realizations_dir()
    kgo_path = _kgo_path(kgo_dir, KGO_WITH_REALIZATION)
    input_paths = _input_paths_with_realization(kgo_dir)
    output_path = tmp_path / "output.nc"

    args = [*input_paths, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
