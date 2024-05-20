# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Tests for the merge CLI
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_single_file(tmp_path):
    """Test merging with one file"""
    kgo_dir = acc.kgo_root() / "merge"
    kgo_path = kgo_dir / "single_file_kgo.nc"
    input_path = kgo_dir / "orographic_enhancement_T3.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_multiple_files(tmp_path):
    """Test merging multiple files"""
    kgo_dir = acc.kgo_root() / "merge"
    kgo_path = kgo_dir / "multiple_file_kgo.nc"
    input_path_t3 = kgo_dir / "orographic_enhancement_T3.nc"
    input_path_t4 = kgo_dir / "orographic_enhancement_T4.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path_t3, input_path_t4, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
