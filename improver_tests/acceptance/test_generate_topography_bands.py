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
Tests for the generate-topography-bands-mask and
generate-topography-bands-weights CLIs.
"""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]

# Parameterisation to test generation of masks and weights with appropriate
# classes and KGO directories
MASK_WEIGHT = ["generate-topography-bands-mask",
               "generate-topography-bands-weights"]
BASIC_MULTI = ["basic", "multi_realization"]


@pytest.mark.parametrize("maskweight", MASK_WEIGHT)
@pytest.mark.parametrize("realizations", BASIC_MULTI)
def test_basic(tmp_path, maskweight, realizations):
    """Test basic generation of topography bands"""
    kgo_dir = acc.kgo_root() / f"{maskweight}/{realizations}"
    kgo_path = kgo_dir / "kgo.nc"
    orography_path = kgo_dir / "input_orog.nc"
    landmask_path = kgo_dir / "input_land.nc"
    output_path = tmp_path / "output.nc"
    args = [orography_path, landmask_path,
            "--output", output_path]
    acc.run_cli(maskweight)(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize("maskweight", MASK_WEIGHT)
def test_bounds_json(tmp_path, maskweight):
    """Test generation of topography bands mask with bounds"""
    kgo_dir = acc.kgo_root() / f"{maskweight}/basic"
    kgo_path = kgo_dir / "kgo_from_json_bounds.nc"
    orography_path = kgo_dir / "input_orog.nc"
    landmask_path = kgo_dir / "input_land.nc"
    bounds_path = kgo_dir / "bounds.json"
    output_path = tmp_path / "output.nc"
    args = [orography_path,
            landmask_path,
            "--bands-config", bounds_path,
            "--output", output_path]
    acc.run_cli(maskweight)(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize("maskweight", MASK_WEIGHT)
def test_basic_nolandsea(tmp_path, maskweight):
    """Test basic generation of topography bands mask without land/sea"""
    kgo_dir = acc.kgo_root() / f"{maskweight}/basic_no_landsea_mask"
    kgo_path = kgo_dir / "kgo.nc"
    orography_path = kgo_dir / "../basic/input_orog.nc"
    output_path = tmp_path / "output.nc"
    args = [orography_path, "--output", output_path]
    acc.run_cli(maskweight)(args)
    acc.compare(output_path, kgo_path)
