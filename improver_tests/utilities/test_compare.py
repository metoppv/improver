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
"""Unit tests for the compare plugin."""

import shutil

import netCDF4 as nc
import numpy as np
import pytest

from improver.utilities import compare

LAT = "latitude"
LON = "longitude"
BOUNDS = "bounds"
DEWPOINT = "dewpoint"
CAT = "category"


@pytest.fixture(scope='function', name='dummy_nc')
def dummy_netcdf_dataset(tmp_path):
    """
    Create an example netcdf dataset for use in testing comparison functions
    """
    expected_nc = tmp_path / "expected.nc"
    dset = nc.Dataset(expected_nc, mode='w', format='NETCDF4_CLASSIC')
    dset.createDimension(LAT, 10)
    dset.createDimension(LON, 12)
    dset.createDimension(BOUNDS, 2)
    lat = dset.createVariable(LAT, np.float64, dimensions=(LAT,))
    lat_bounds = dset.createVariable(
        f"{LAT}_{BOUNDS}", np.float64, dimensions=(LAT, BOUNDS))
    lon = dset.createVariable(LON, np.float64, dimensions=(LON,))
    lon_bounds = dset.createVariable(
        f"{LON}_{BOUNDS}", np.float64, dimensions=(LON, BOUNDS))
    dpoint = dset.createVariable(DEWPOINT, np.float32, dimensions=(LAT, LON))
    categ = dset.createVariable(CAT, np.int8, dimensions=(LAT, LON))

    dset.setncattr("grid_id_string", "example")
    dset.setncattr("float_number", 1.5)
    dset.setncattr("whole_number", 4)

    lat_data = np.linspace(2.5, 7, 10)
    lat[:] = lat_data
    lon_data = np.linspace(12, 18.5, 12)
    lon[:] = lon_data
    lat_bounds_data = np.transpose(
        np.array([lat_data - 0.25, lat_data + 0.25]))
    lat_bounds[:] = lat_bounds_data
    lon_bounds_data = np.transpose(
        np.array([lon_data - 0.25, lon_data + 0.25]))
    lon_bounds[:] = lon_bounds_data

    lat_fill = np.linspace(0.0, np.pi, 10)
    lat_fill = np.broadcast_to(lat_fill[:, np.newaxis], (10, 12))
    lon_fill = np.linspace(0.0, np.pi, 12)
    lon_fill = np.broadcast_to(lon_fill[np.newaxis, :], (10, 12))
    grid_fill = lat_fill * lon_fill
    dewpoint_grid = 273.15 + np.cos(grid_fill)
    dpoint[:] = dewpoint_grid
    dpoint.setncattr("units", "K")
    dpoint.setncattr("standard_name", "dew_point_temperature")
    category_grid = np.where(dewpoint_grid > 273.15, -1, 1)
    category_discrete = category_grid.astype(np.int8)
    categ[:] = category_discrete
    dpoint.setncattr("long_name", "threshold categorisation")
    dset.close()

    actual_nc = tmp_path / "actual.nc"
    shutil.copy(expected_nc, actual_nc)

    # This yield provides the fixture objects to the test function
    yield (actual_nc, expected_nc)

    # Cleanup is handled between yield and the fixture function return
    actual_nc.unlink()
    expected_nc.unlink()
    return (actual_nc, expected_nc)


def test_compare_identical_netcdfs(dummy_nc):
    """Check that comparing identical netCDFs does not raise any exceptions"""
    actual_nc, expected_nc = dummy_nc
    for tol in (1.0, 1e-5, 1e-10, 0.0):
        compare.compare_netcdfs(actual_nc, expected_nc, atol=tol, rtol=tol)


def test_compare_netcdf_attrs(dummy_nc):
    """Check that comparing attributes identifies the changed attribute"""
    actual_nc, expected_nc = dummy_nc
    expected_ds = nc.Dataset(expected_nc, mode='r')
    actual_ds = nc.Dataset(actual_nc, mode='a')

    messages_reported = []

    def message_collector(message):
        messages_reported.append(message)

    # Check that attributes initially match - message_collector is not called
    compare.compare_attributes(
        "root", actual_ds, expected_ds, message_collector)
    assert len(messages_reported) == 0

    # Check modifying an attribute
    actual_ds.setncattr("float_number", 3.2)
    compare.compare_attributes(
        "root", actual_ds, expected_ds, message_collector)
    assert len(messages_reported) == 1
    assert "float_number" in messages_reported[0]
    assert "3.2" in messages_reported[0]
    assert "1.5" in messages_reported[0]

    # Reset that attribute back to original value
    actual_ds.setncattr("float_number", 1.5)
    messages_reported = []

    # Check adding another attribute
    actual_ds.setncattr("extra", "additional")
    compare.compare_attributes(
        "longer name", actual_ds, expected_ds, message_collector)
    assert len(messages_reported) == 1
    assert "longer name" in messages_reported[0]
    # The difference message should mention the attribute which was added
    assert "extra" in messages_reported[0]

    # Remove added attribute
    actual_ds.delncattr("extra")
    messages_reported = []

    # Check removing an attribute
    actual_ds.delncattr("float_number")
    compare.compare_attributes(
        "root", actual_ds, expected_ds, message_collector)
    assert len(messages_reported) == 1
    assert "float_number" in messages_reported[0]


def test_compare_data_floats_equal(dummy_nc):
    """Check comparison of floating point data considered exactly equal"""
    actual_nc, expected_nc = dummy_nc
    expected_ds = nc.Dataset(expected_nc, mode='a')
    actual_ds = nc.Dataset(actual_nc, mode='a')

    messages_reported = []

    def message_collector(message):
        messages_reported.append(message)

    # Check that data originally matches exactly (zero tolerances)
    compare.compare_data(DEWPOINT, actual_ds[DEWPOINT], expected_ds[DEWPOINT],
                         0.0, 0.0, message_collector)
    assert len(messages_reported) == 0

    # Check that NaNs in same position compare equal rather than the
    # floating point "NaNs are always unequal" usual convention
    expected_dp = expected_ds[DEWPOINT]
    expected_dp[0, :] = np.nan
    expected_ds.sync()
    actual_dp = actual_ds[DEWPOINT]
    actual_dp[0, :] = np.nan
    actual_ds.sync()
    compare.compare_data(DEWPOINT, actual_ds[DEWPOINT], expected_ds[DEWPOINT],
                         0.0, 0.0, message_collector)
    assert len(messages_reported) == 0


def test_compare_data_floats_relative(dummy_nc):
    """Check comparison of floating point data with relative tolerances"""
    actual_nc, expected_nc = dummy_nc
    expected_ds = nc.Dataset(expected_nc, mode='a')
    actual_ds = nc.Dataset(actual_nc, mode='a')

    messages_reported = []

    def message_collector(message):
        messages_reported.append(message)

    # Check that modifying the data is picked up by relative tolerance
    actual_dp = actual_ds[DEWPOINT]
    # relative change is a little smaller than 1e-3
    actual_dp[1, :] = np.array(actual_dp[1, :]) * (1.0 + 8e-4)
    # 1e-3 relative tolerance -> no problem
    compare.compare_data(DEWPOINT, actual_ds[DEWPOINT], expected_ds[DEWPOINT],
                         1e-3, 0.0, message_collector)
    assert len(messages_reported) == 0
    # 5e-4 relative tolerance -> problem reported
    compare.compare_data(DEWPOINT, actual_ds[DEWPOINT], expected_ds[DEWPOINT],
                         5e-4, 0.0, message_collector)
    assert len(messages_reported) == 1
    assert DEWPOINT in messages_reported[0]
    assert 'tolerance' in messages_reported[0]
    # no relative tolerance -> problem reported
    compare.compare_data(DEWPOINT, actual_ds[DEWPOINT], expected_ds[DEWPOINT],
                         0.0, 0.0, message_collector)
    assert len(messages_reported) == 2


def test_compare_data_floats_absolute(dummy_nc):
    """Check comparison of floating point data with absolute tolerances"""
    actual_nc, expected_nc = dummy_nc
    expected_ds = nc.Dataset(expected_nc, mode='r')
    actual_ds = nc.Dataset(actual_nc, mode='a')

    messages_reported = []

    def message_collector(message):
        messages_reported.append(message)

    # Check that modifying the data is picked up by absolute tolerance
    actual_dp = actual_ds[DEWPOINT]
    # absolute change is a little smaller than 1e-2
    actual_dp[1, :] = np.array(actual_dp[1, :]) + 8e-3
    # 1e-2 absolute tolerance -> no problem
    compare.compare_data(DEWPOINT, actual_ds[DEWPOINT], expected_ds[DEWPOINT],
                         0.0, 1e-2, message_collector)
    assert len(messages_reported) == 0
    # 5e-3 absolute tolerance -> problem reported
    compare.compare_data(DEWPOINT, actual_ds[DEWPOINT], expected_ds[DEWPOINT],
                         0.0, 5e-3, message_collector)
    assert len(messages_reported) == 1
    assert DEWPOINT in messages_reported[0]
    assert 'tolerance' in messages_reported[0]
    # no relative tolerance -> problem reported
    compare.compare_data(DEWPOINT, actual_ds[DEWPOINT], expected_ds[DEWPOINT],
                         0.0, 0.0, message_collector)
    assert len(messages_reported) == 2


def test_compare_data_integers(dummy_nc):
    """Check comparison of integers"""
    actual_nc, expected_nc = dummy_nc
    expected_ds = nc.Dataset(expected_nc, mode='r')
    actual_ds = nc.Dataset(actual_nc, mode='a')

    messages_reported = []

    def message_collector(message):
        messages_reported.append(message)

    actual_dp = actual_ds[CAT]
    actual_dp[:] = np.array(actual_dp[:]) + 1

    # 2 absolute tolerance -> no problem
    compare.compare_data(CAT, actual_ds[CAT], expected_ds[CAT],
                         0, 2, message_collector)
    assert len(messages_reported) == 0
    # 1 absolute tolerance -> no problem
    compare.compare_data(CAT, actual_ds[CAT], expected_ds[CAT],
                         0, 1, message_collector)
    assert len(messages_reported) == 0
    # 0 relative and absolute tolerance -> problem reported
    compare.compare_data(CAT, actual_ds[CAT], expected_ds[CAT],
                         0, 0, message_collector)
    assert len(messages_reported) == 1
    assert "tolerance" in messages_reported[0]
    assert CAT in messages_reported[0]


def test_compare_data_shape(dummy_nc):
    """Check differing data shapes are reported"""
    actual_nc, expected_nc = dummy_nc
    expected_ds = nc.Dataset(expected_nc, mode='r')
    actual_ds = nc.Dataset(actual_nc, mode='a')

    messages_reported = []

    def message_collector(message):
        messages_reported.append(message)

    # super loose tolerance, but shapes don't match
    compare.compare_data("lonlat", actual_ds[LON], expected_ds[LAT],
                         100.0, 100.0, message_collector)
    assert len(messages_reported) == 1
    assert "shape" in messages_reported[0]
