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
Tests for the nbhood-iterate-with-mask CLI
"""

import pytest

from improver.tests.acceptance import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


@pytest.mark.slow
def test_basic(tmp_path):
    """Test basic iterate with mask"""
    kgo_dir = acc.kgo_root() / "nbhood-iterate-with-mask/basic"
    kgo_path = kgo_dir / "kgo_basic.nc"
    input_path = kgo_dir / "input.nc"
    mask_path = kgo_dir / "mask.nc"
    output_path = tmp_path / "output.nc"
    args = ["topographic_zone", input_path, mask_path, output_path,
            "--radius=20000"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
@pytest.mark.parametrize("intermediate", (True, False))
def test_collapse_bands(tmp_path, intermediate):
    """Test with collapsing orographic bands"""
    kgo_dir = acc.kgo_root() / "nbhood-iterate-with-mask/basic_collapse_bands"
    kgo_path = kgo_dir / "kgo_collapsed.nc"
    kgo_intermediate_path = kgo_dir / \
        "../collapse_store_intermediate/kgo_pre_collapse.nc"
    input_path = kgo_dir / "thresholded_input.nc"
    mask_path = kgo_dir / "orographic_bands_mask.nc"
    weights_path = kgo_dir / "orographic_bands_weights.nc"
    output_path = tmp_path / "output.nc"
    output_intermediate_path = tmp_path / "intermediate.nc"
    if intermediate:
        im_args = ["--intermediate_filepath", output_intermediate_path]
    else:
        im_args = []
    args = ["topographic_zone", input_path, mask_path, output_path,
            "--radius", "10000", "--collapse_dimension",
            "--weights_for_collapsing_dim", weights_path, *im_args]
    run_cli(args)
    acc.compare(output_path, kgo_path)
    if intermediate:
        acc.compare(output_intermediate_path, kgo_intermediate_path)
