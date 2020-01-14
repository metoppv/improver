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
Tests for the neighbour-finding CLI
"""

import json

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)

UK_GLOBAL = [("uk", "ukvx"), ("global", "global")]


@pytest.mark.parametrize("domain,model", UK_GLOBAL)
def test_nearest(tmp_path, domain, model):
    """Test basic neighbour finding"""
    kgo_dir = acc.kgo_root() / "neighbour-finding"
    kgo_path = kgo_dir / f"outputs/nearest_{domain}_kgo.nc"
    sites_path = kgo_dir / f"inputs/{domain}_sites.json"
    orography_path = kgo_dir / f"inputs/{model}_orography.nc"
    landmask_path = kgo_dir / f"inputs/{model}_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [orography_path, landmask_path,
            sites_path,
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.parametrize("domain,model", UK_GLOBAL)
def test_nearest_land(tmp_path, domain, model):
    """Test neighbour finding with a land constraint"""
    kgo_dir = acc.kgo_root() / "neighbour-finding"
    kgo_path = kgo_dir / f"outputs/nearest_land_{domain}_kgo.nc"
    sites_path = kgo_dir / f"inputs/{domain}_sites.json"
    orography_path = kgo_dir / f"inputs/{model}_orography.nc"
    landmask_path = kgo_dir / f"inputs/{model}_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [orography_path, landmask_path,
            sites_path,
            "--land-constraint",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
@pytest.mark.parametrize("domain,model", UK_GLOBAL)
def test_nearest_minimum_dz(tmp_path, domain, model):
    """Test neighbour finding with minimum altitude difference"""
    kgo_dir = acc.kgo_root() / "neighbour-finding"
    kgo_path = kgo_dir / f"outputs/nearest_minimum_dz_{domain}_kgo.nc"
    sites_path = kgo_dir / f"inputs/{domain}_sites.json"
    orography_path = kgo_dir / f"inputs/{model}_orography.nc"
    landmask_path = kgo_dir / f"inputs/{model}_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [orography_path, landmask_path,
            sites_path,
            "--similar-altitude",
            "--search-radius", "50000",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
@pytest.mark.parametrize("domain,model", UK_GLOBAL)
def test_nearest_land_minimum_dz(tmp_path, domain, model):
    """Test neighbour finding with land constraint plus min altitude diff"""
    kgo_dir = acc.kgo_root() / "neighbour-finding"
    kgo_path = kgo_dir / \
        f"outputs/nearest_land_constraint_minimum_dz_{domain}_kgo.nc"
    sites_path = kgo_dir / f"inputs/{domain}_sites.json"
    orography_path = kgo_dir / f"inputs/{model}_orography.nc"
    landmask_path = kgo_dir / f"inputs/{model}_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [orography_path, landmask_path,
            sites_path,
            "--land-constraint",
            "--similar-altitude",
            "--search-radius", "50000",
            "--output", output_path]
    if domain == "uk":
        args += ["--node-limit", "100"]
    run_cli(args)
    acc.compare(output_path, kgo_path)


@pytest.mark.slow
@pytest.mark.parametrize("domain,model", UK_GLOBAL)
def test_all_methods(tmp_path, domain, model):
    """Test neighbour finding using all methods"""
    kgo_dir = acc.kgo_root() / "neighbour-finding"
    kgo_path = kgo_dir / f"outputs/all_methods_{domain}_kgo.nc"
    sites_path = kgo_dir / f"inputs/{domain}_sites.json"
    orography_path = kgo_dir / f"inputs/{model}_orography.nc"
    landmask_path = kgo_dir / f"inputs/{model}_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [orography_path, landmask_path,
            sites_path,
            "--all-methods",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_alternative_coordinate_system(tmp_path):
    """Test use of an alternative coordinate system"""
    kgo_dir = acc.kgo_root() / "neighbour-finding"
    # same KGO as test_nearest global domain
    kgo_path = kgo_dir / "outputs/nearest_global_kgo.nc"
    sites_path = kgo_dir / "inputs/LAEA_grid_sites.json"
    orography_path = kgo_dir / "inputs/global_orography.nc"
    landmask_path = kgo_dir / "inputs/global_landmask.nc"
    output_path = tmp_path / "output.nc"
    coord_opts = {"central_latitude": 54.9,
                  "central_longitude": -2.5,
                  "false_easting": 0.0,
                  "false_northing": 0.0,
                  "globe": {"semimajor_axis": 6378137.0,
                            "semiminor_axis": 6356752.314140356}}
    coord_opts_json = json.dumps(coord_opts)
    args = [orography_path, landmask_path,
            sites_path,
            "--site-coordinate-system", "LambertAzimuthalEqualArea",
            "--site-coordinate-options", coord_opts_json,
            "--site-x-coordinate", "projection_x_coordinate",
            "--site-y-coordinate", "projection_y_coordinate",
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_incompatible_constraints(tmp_path):
    """Test use of incompatible options"""
    kgo_dir = acc.kgo_root() / "neighbour-finding"
    sites_path = kgo_dir / "inputs/uk_sites.json"
    orography_path = kgo_dir / "inputs/ukvx_orography.nc"
    landmask_path = kgo_dir / "inputs/ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [orography_path, landmask_path,
            sites_path,
            "--land-constraint",
            "--all-methods",
            "--output", output_path]
    with pytest.raises(ValueError, match=".*all_methods.*"):
        run_cli(args)


def test_invalid_site(tmp_path):
    """Test invalid global site coordinates"""
    kgo_dir = acc.kgo_root() / "neighbour-finding"
    kgo_path = kgo_dir / "outputs/nearest_global_invalid_site_kgo.nc"
    sites_path = kgo_dir / "inputs/global_sites_invalid.json"
    orography_path = kgo_dir / "inputs/global_orography.nc"
    landmask_path = kgo_dir / "inputs/global_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [orography_path, landmask_path,
            sites_path,
            "--output", output_path]
    with pytest.warns(UserWarning, match=".*outside the grid.*"):
        run_cli(args)
    acc.compare(output_path, kgo_path)


def test_coord_beyond_bounds(tmp_path):
    """Test coordinates up to 360 degrees"""
    kgo_dir = acc.kgo_root() / "neighbour-finding"
    kgo_path = kgo_dir / "outputs/nearest_global_kgo.nc"
    sites_path = kgo_dir / "inputs/global_sites_360.json"
    orography_path = kgo_dir / "inputs/global_orography.nc"
    landmask_path = kgo_dir / "inputs/global_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [orography_path, landmask_path,
            sites_path,
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path, exclude_vars=["longitude"])


def test_unset_wmo_ids(tmp_path):
    """Test sites with missing WMO ID numbers"""
    kgo_dir = acc.kgo_root() / "neighbour-finding"
    kgo_path = kgo_dir / "outputs/nearest_uk_kgo_some_unset_wmo_ids.nc"
    sites_path = kgo_dir / "inputs/uk_sites_missing_wmo_ids.json"
    orography_path = kgo_dir / "inputs/ukvx_orography.nc"
    landmask_path = kgo_dir / "inputs/ukvx_landmask.nc"
    output_path = tmp_path / "output.nc"
    args = [orography_path, landmask_path,
            sites_path,
            "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
