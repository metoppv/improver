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
"""Unit tests for plugin wind_downscaling.RoughnessCorrection."""


import datetime
import unittest

import iris
import numpy as np
from cf_units import Unit
from iris.tests import IrisTest

from improver.constants import RMDI
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.wind_calculations.wind_downscaling import RoughnessCorrection


def _make_flat_cube(data, name, unit):
    """Create a "flat" y/x cube with required 2D shape, scalar time coordinates
    and 2 km grid bounds"""
    flat_cube = set_up_variable_cube(
        data,
        name=name,
        units=unit,
        spatial_grid="equalarea",
        domain_corner=(-1036000, -1158000),
        x_grid_spacing=2000,
        y_grid_spacing=2000,
    )
    for axis in ["x", "y"]:
        points = flat_cube.coord(axis=axis).points
        flat_cube.coord(axis=axis).bounds = np.array(
            [points - 1000.0, points + 1000.0]
        ).T
    return flat_cube


def make_ancil_cube(data, name, unit, shape=None):
    """Create an ancillary cube (constant with time)"""
    data = np.array(data, dtype=np.float32)
    if shape is not None:
        data = data.reshape(shape)
    cube = _make_flat_cube(data, name, unit)
    for coord in ["time", "forecast_reference_time", "forecast_period"]:
        cube.remove_coord(coord)
    return cube


def _add_model_levels(flat_cube, data):
    """Add a model level coordinate to a point cube and insert 1D height data"""
    cube = add_coordinate(flat_cube, np.arange(len(data)), "model_level_number", 1)
    cube.data = np.array(data).reshape((len(data), 1, 1))
    return cube


def make_point_height_ancil_cube(heights_data):
    """Create a multi-level height ancillary for one spatial point"""
    flat_cube = make_ancil_cube(1, None, None, shape=(1, 1))
    cube = _add_model_levels(flat_cube, heights_data)
    return cube


def make_point_data_cube(data, name, unit):
    """Create a multi-level data cube for one spatial point"""
    flat_cube = _make_flat_cube(np.ones((1, 1), dtype=np.float32), name, unit)
    cube = _add_model_levels(flat_cube, data)
    return cube


def make_data_cube(data, name, unit, shape, heights):
    """Create a multi-point data cube from 1D data on height levels"""
    flat_cube = _make_flat_cube(np.ones(shape, dtype=np.float32), name, unit)
    cube = add_coordinate(flat_cube, heights, "height", "m")
    data_3d = []
    for point in data:
        data_3d.append(np.broadcast_to(np.array([point]), shape))
    cube.data = np.array(data_3d, dtype=np.float32)
    return cube


class TestMultiPoint:

    """Test (typically) 3x1 or 3x3 point tests.

    It constructs cubes for the ancillary fields:
    Silhouette roughness (AoS), standard deviation of model height grid
    cell (Sigma), vegetative roughness (z_0), post-processing grid
    orography (pporog) and model orography (modelorog). If no values
    are supplied, the grids that are set up have equal values at all
    x-y points: AoS = 0.2, Sigma = 20, z_0 = 0.2, pporog = 250,
    modelorog = 230.

    """

    def __init__(
        self, shape=(3, 3), AoS=None, Sigma=None, z_0=0.2, pporog=None, modelorog=None
    ):
        """Set up multi-point tests.

        Args:
            shape (tuple):
                Required data shape.
            AoS (float or 1D or 2D numpy.ndarray):
                Silhouette roughness field
            Sigma (float or 1D or 2D numpy.ndarray):
                Standard deviation field of height in grid cell
            z_0 (float or 1D or 2D numpy.ndarray):
                Vegetative roughness field
            pporog (float or 1D or 2D numpy.ndarray):
                Unsmoothed orography field on post-processing grid
            modelorog (float or 1D or 2D numpy.ndarray):
                Model orography field on post-processing grid

        """
        self.shape = shape
        if AoS is None:
            AoS = np.full(shape, 0.2, dtype=np.float32)
        if Sigma is None:
            Sigma = np.full(shape, 20.0, dtype=np.float32)
        if pporog is None:
            pporog = np.full(shape, 250.0, dtype=np.float32)
        if modelorog is None:
            modelorog = np.full(shape, 230.0, dtype=np.float32)
        self.w_cube = None
        self.aos_cube = make_ancil_cube(AoS, "silhouette_roughness", 1, shape=shape)
        self.s_cube = make_ancil_cube(
            Sigma, "standard_deviation_of_height_in_grid_cell", "m", shape=shape
        )
        if z_0 is None:
            self.z0_cube = None
        elif isinstance(z_0, float):
            z_0 = np.full(shape, z_0, dtype=np.float32)
            self.z0_cube = make_ancil_cube(z_0, "vegetative_roughness_length", "m")
        elif isinstance(z_0, list):
            self.z0_cube = make_ancil_cube(
                np.array(z_0), "vegetative_roughness_length", "m", shape=shape
            )
        self.poro_cube = make_ancil_cube(pporog, "surface_altitude", "m", shape=shape)
        self.moro_cube = make_ancil_cube(
            modelorog, "surface_altitude", "m", shape=shape
        )

    def run_hc_rc(self, wind, dtime=1, height=None, aslist=False):
        """Function to set up a wind cube from the supplied np.array.

        Set up the wind and call the RoughnessCorrection class. If the
        supplied array is 1D, it is assumed to be the height profile
        and the values are copied to all x-y points and all time steps.
        If the supplied array is 2D, it is assumed that the supplied
        array is a function of height x time. The point is copied to
        all x-y points. The first dimension should be the height
        dimension. If a 3D array is supplied, the order should be
        height x time x x-y-grid. If a height is supplied, it needs to
        agree with the first (zeroth) dimension of the supplied wind
        array.

        Args:
            wind (2D or 3D numpy.ndarray)
                Multi-level wind target data
            dtime (int):
                Number of time dimension values, default 1
            height (float):
                Value for height in metres for zeroth slice of wind,
                default None
            aslist (bool):
                Make wind cube into a CubeList of height slices or not,
                default False
        """
        if aslist:
            self.w_cube = iris.cube.CubeList()
            for windfield in wind:
                windfield = np.array(windfield)
                self.w_cube.append(
                    make_data_cube(windfield, "wind_speed", "m s-1", self.shape, height)
                )

        else:
            if dtime == 1:
                self.w_cube = make_data_cube(
                    wind, "wind_speed", "m s-1", self.shape, height
                )
            else:
                cube = make_data_cube(wind, "wind_speed", "m s-1", self.shape, height)
                time_points = []
                for i in range(dtime):
                    offset = datetime.timedelta(seconds=i * 3600)
                    time_points.append(cube.coord("time").cell(0).point + offset)
                self.w_cube = add_coordinate(
                    cube, time_points, "time", is_datetime=True, order=[1, 0, 2, 3]
                )

        plugin = RoughnessCorrection(
            self.aos_cube,
            self.s_cube,
            self.poro_cube,
            self.moro_cube,
            1500.0,
            self.z0_cube,
        )
        return plugin(self.w_cube)


class TestSinglePoint:
    """Test a single 1x1 grid.

    A cube is a single 1x x 1y grid, however, the z dimension is not 1.
    It constructs 1x1 cubes for the ancillary fields Silhouette
    roughness (AoS) and standard deviation of model height grid cell
    (Sigma), vegetative roughness (z_0), post-processing grid orography
    (pporog) and model orography(modelorog). If no values are supplied,
    the values are: AoS = 0.2, Sigma = 20, z_0 = 0.2, pporog = 250,
    modelorog = 230.

    The height level grid (heightlevels) can be supplied as an 1D
    array. If nothing is supplied, the height level grid is [0.2, 3,
    13, 33, 133, 333, 1133].

    """

    def __init__(
        self,
        AoS=0.2,
        Sigma=20.0,
        z_0=0.2,
        pporog=250.0,
        modelorog=230.0,
        heightlevels=np.array([0.2, 3.0, 13.0, 33.0, 133.0, 333.0, 1133.0]),
    ):
        """Set up the single point test for RoughnessCorrection.

        Args:
            AoS (float):
                Silhouette roughness field
            Sigma (float):
                Standard deviation field of height in grid cell
            z_0 (float):
                Vegetative roughness field
            pporog (float):
                Unsmoothed orography on post-processing grid
            modelorog (float):
                Model orography on post-processing grid
            heightlevels (1D numpy.ndarray):
                Height level array

        """
        self.w_cube = None
        self.aos_cube = make_ancil_cube(AoS, "silhouette_roughness", 1, shape=(1, 1))
        self.s_cube = make_ancil_cube(
            Sigma, "standard_deviation_of_height_in_grid_cell", "m", shape=(1, 1)
        )
        if z_0 is None:
            self.z0_cube = None
        else:
            self.z0_cube = make_ancil_cube(
                z_0, "vegetative_roughness_length", "m", shape=(1, 1)
            )
        self.poro_cube = make_ancil_cube(pporog, "surface_altitude", "m", shape=(1, 1))
        self.moro_cube = make_ancil_cube(
            modelorog, "surface_altitude", "m", shape=(1, 1)
        )
        if heightlevels is not None:
            self.hl_cube = make_point_height_ancil_cube(heightlevels)
        else:
            self.hl_cube = None

    def run_hc_rc(self, wind, height=None):
        """Test single point height correction and roughness correction.

        Make an iris cube of the supplied wind and set up the height
        axis in m.

        Args:
            wind (list or 1D numpy.ndarray):
                Array of wind speeds
            height (float):
                Value for height in metres for zeroth slice of wind,
                default None.
        """
        self.w_cube = make_point_data_cube(wind, "wind_speed", "m s-1")
        plugin = RoughnessCorrection(
            self.aos_cube,
            self.s_cube,
            self.poro_cube,
            self.moro_cube,
            1500.0,
            self.z0_cube,
            self.hl_cube,
        )
        return plugin(self.w_cube)


class Test1D(IrisTest):

    """Class to test 1 x-y point cubes.

    This class tests the correct behaviour if np.nan or RMDI are
    passed, as well as testing the general behaviour of points that
    should not have a height corretion (equal height in model and pp
    orography) and the correct behaviour of doing roughness correction,
    depending on whether or not a vegetative roughness (z_0) cube is
    provided.

    Section 0 are tests where RMDI or np.nan values are passed.
    Section 1 are sensible single point tests.

    """

    uin = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0], dtype=np.float32)
    hls = np.array([0.2, 3, 13, 33, 133, 333, 1133], dtype=np.float32)

    def test_section0a(self):
        """Test AoS is RMDI, point should not do anything, uin = uout."""
        landpointtests_hc_rc = TestSinglePoint(AoS=RMDI, heightlevels=self.hls,)
        land_hc_rc = landpointtests_hc_rc.run_hc_rc(self.uin)
        self.assertArrayEqual(landpointtests_hc_rc.w_cube, land_hc_rc)

    def test_section0b(self):
        """Test AoS is np.nan, point should not do anything, uin = uout."""
        landpointtests_hc_rc = TestSinglePoint(AoS=np.nan, heightlevels=self.hls,)
        land_hc_rc = landpointtests_hc_rc.run_hc_rc(self.uin)
        self.assertArrayEqual(landpointtests_hc_rc.w_cube, land_hc_rc)

    def test_section0c(self):
        """Test Sigma is RMDI, point should not do anything, uin = uout."""
        landpointtests_hc_rc = TestSinglePoint(Sigma=RMDI, heightlevels=self.hls,)
        land_hc_rc = landpointtests_hc_rc.run_hc_rc(self.uin)
        self.assertArrayEqual(landpointtests_hc_rc.w_cube, land_hc_rc)

    def test_section0d(self):
        """Test Sigma is np.nan, point should not do anything, uin = uout."""
        landpointtests_hc_rc = TestSinglePoint(Sigma=np.nan, heightlevels=self.hls,)
        land_hc_rc = landpointtests_hc_rc.run_hc_rc(self.uin)
        self.assertArrayEqual(landpointtests_hc_rc.w_cube, land_hc_rc)

    def test_section0e(self):
        """Test z_0 is RMDI, point should not do RC.

        modeloro = pporo, so point should not do HC, uin = uout.

        """
        landpointtests_hc_rc = TestSinglePoint(
            z_0=RMDI, pporog=230.0, heightlevels=self.hls,
        )
        land_hc_rc = landpointtests_hc_rc.run_hc_rc(self.uin)
        self.assertArrayEqual(landpointtests_hc_rc.w_cube, land_hc_rc)

    def test_section0f(self):
        """Test z_0 is np.nan, point should not do RC.

        modeloro = pporo, so point should not do HC, uin = uout.

        """
        landpointtests_hc_rc = TestSinglePoint(
            z_0=np.nan, pporog=230.0, heightlevels=self.hls,
        )
        land_hc_rc = landpointtests_hc_rc.run_hc_rc(self.uin)
        self.assertArrayEqual(landpointtests_hc_rc.w_cube, land_hc_rc)

    def test_section0g(self):
        """Test z_0 is RMDI, point should not do RC.

        modeloro < pporo, so point should do positive HC, uin < uout.

        """
        landpointtests_hc_rc = TestSinglePoint(z_0=RMDI, heightlevels=self.hls,)
        land_hc_rc = landpointtests_hc_rc.run_hc_rc(self.uin)
        self.assertTrue((land_hc_rc.data > landpointtests_hc_rc.w_cube.data).all())

    def test_section0h(self):
        """Test pporog is RMDI (QUESTION: or should this fail???)

        RC could be done for this point, HC cannot.
        uin >= uout
        and since z_0=height[0]
        uout[0] = 0

        """
        landpointtests_hc_rc = TestSinglePoint(pporog=RMDI, heightlevels=self.hls,)
        land_hc_rc = landpointtests_hc_rc.run_hc_rc(self.uin)
        self.assertTrue(
            (land_hc_rc.data <= landpointtests_hc_rc.w_cube.data).all()
            and land_hc_rc.data[0] == 0
        )

    def test_section0i(self):
        """Test pporog is np.nan (QUESTION: or should this fail???)

        RC could be done for this point, HC cannot.
        uin >= uout
        and since z_0=height[0]
        uout[0] = 0

        """
        landpointtests_hc_rc = TestSinglePoint(pporog=np.nan, heightlevels=self.hls,)
        land_hc_rc = landpointtests_hc_rc.run_hc_rc(self.uin)
        self.assertTrue(
            (land_hc_rc.data <= landpointtests_hc_rc.w_cube.data).all()
            and land_hc_rc.data[0] == 0
        )

    def test_section0j(self):
        """Test modelorog is RMDI (QUESTION: or should this fail???).

        RC could be done for this point, HC cannot.
        uin >= uout
        and since z_0=height[0]
        uout[0] = 0

        """
        landpointtests_hc_rc = TestSinglePoint(modelorog=RMDI, heightlevels=self.hls,)
        land_hc_rc = landpointtests_hc_rc.run_hc_rc(self.uin)
        self.assertTrue(
            (land_hc_rc.data <= landpointtests_hc_rc.w_cube.data).all()
            and land_hc_rc.data[0] == 0
        )

    def test_section0k(self):
        """Test fail for RMDI in height grid.

        height grid is RMDI at that location somewhere in z-direction,
        should fail with ValueError.

        """
        hls = [0.2, 3, 13, RMDI, 133, 333, 1133]
        landpointtests_hc_rc = TestSinglePoint(heightlevels=hls)
        with self.assertRaises(ValueError):
            _ = landpointtests_hc_rc.run_hc_rc(self.uin)

    def test_section0l(self):
        """Test fail for np.nan in height grid.

        height grid is np.nan at that location somewhere in z-direction,
        should fail with ValueError.

        """
        hls = [0.2, 3, 13, np.nan, 133, 333, 1133]
        landpointtests_hc_rc = TestSinglePoint(heightlevels=hls)
        with self.assertRaises(ValueError):
            _ = landpointtests_hc_rc.run_hc_rc(self.uin)

    def test_section0m(self):
        """Test fail for RMDI in uin.

        uin is RMDI at that location somewhere in z-direction,
        should fail with ValueError.

        """
        uin = [20.0, 20.0, 20.0, RMDI, RMDI, 20.0, 0.0]
        landpointtests_hc_rc = TestSinglePoint(heightlevels=self.hls)
        with self.assertRaises(ValueError):
            _ = landpointtests_hc_rc.run_hc_rc(uin)

    def test_section0n(self):
        """Test fail for np.nan in uin.

        uin is np.nan at that location somewhere in z-direction,
        should fail with ValueError.

        """
        uin = [20.0, 20.0, 20.0, np.nan, 20.0, 20.0, 20.0]
        landpointtests_hc_rc = TestSinglePoint(heightlevels=self.hls)
        with self.assertRaises(ValueError):
            _ = landpointtests_hc_rc.run_hc_rc(uin)

    def test_section1a(self):
        """Test HC only, HC = 0.

        z_0 passed as None, hence RC not performed.
        modelorg = pporog, hence HC = 0.
        uin = uout

        """
        landpointtests_hc = TestSinglePoint(z_0=None, modelorog=250.0)
        land_hc_rc = landpointtests_hc.run_hc_rc(self.uin)
        self.assertArrayEqual(landpointtests_hc.w_cube, land_hc_rc)

    def test_section1b(self):
        """Test HC only.

        z_0 passed as None, hence RC not performed.
        modelorg < pporog, hence positive HC.
        uin <= uout, at least one height has uin < uout.

        """
        landpointtests_hc = TestSinglePoint(z_0=None)
        land_hc_rc = landpointtests_hc.run_hc_rc(self.uin)
        self.assertTrue(
            (land_hc_rc.data >= landpointtests_hc.w_cube.data).all()
            and (land_hc_rc.data > landpointtests_hc.w_cube.data).any()
        )

    def test_section1c(self):
        """Test RC and HC, HC=0.

        z_0 passed, hence RC performed.
        modelorg == pporog, hence no HC.
        uin >= uout, at least one height has uin > uout, uout[0] = 0.

        """
        landpointtests_rc = TestSinglePoint(modelorog=250.0)
        land_hc_rc = landpointtests_rc.run_hc_rc(self.uin)
        self.assertTrue(
            (land_hc_rc.data <= landpointtests_rc.w_cube.data).all()
            and (land_hc_rc.data < landpointtests_rc.w_cube.data).any()
            and land_hc_rc.data[0] == 0
        )

    def test_section1d(self):
        """Test RC and HC.

        z_0 passed, hence RC performed.
        modelorg >> pporog, hence negative HC.
        uin >= uout, at least one height has uin > uout
        z_0 = height[0] hence RC[0] results in 0.
        uout[0] RC is 0. HC is negative, negative speeds not allowed.
        Must be 0.

        """
        landpointtests_hc_rc = TestSinglePoint(pporog=230.0, modelorog=250.0)
        land_hc_rc = landpointtests_hc_rc.run_hc_rc(self.uin)
        self.assertTrue(
            (land_hc_rc.data <= landpointtests_hc_rc.w_cube.data).all()
            and (land_hc_rc.data < landpointtests_hc_rc.w_cube.data).any()
            and (land_hc_rc.data >= 0).all()
            and land_hc_rc.data[0] == 0
        )

    def test_section1e(self):
        """Test RC and HC, but sea point masked out (AoS).

        Sea point according to (AoS=0) => masked out.
        z_0 passed, hence RC performed in theory.
        uin = uout.

        """
        landpointtests_hc_rc = TestSinglePoint(AoS=0.0)
        land_hc_rc = landpointtests_hc_rc.run_hc_rc(self.uin)
        self.assertArrayEqual(landpointtests_hc_rc.w_cube, land_hc_rc)

    def test_section1f(self):
        """Test RC and HC, but sea point masked out (Sigma).

        Sea point according to (Sigma=0) => masked out
        z_0 passed, hence RC performed in theory.
        uin = uout.

        """
        landpointtests_hc_rc = TestSinglePoint(Sigma=0.0)
        land_hc_rc = landpointtests_hc_rc.run_hc_rc(self.uin)
        self.assertArrayEqual(landpointtests_hc_rc.w_cube, land_hc_rc)

    def test_section1g(self):
        """Test that code returns float32 precision."""
        landpointtests_hc_rc = TestSinglePoint()
        land_hc_rc = landpointtests_hc_rc.run_hc_rc(self.uin)
        self.assertEqual(land_hc_rc.dtype, np.float32)


class Test2D(IrisTest):

    """Test multi-point wind corrections.

    Section 2 are multiple point, multiple time tests
    Section 3 are tests that should fail because the grids are not all
    the same or units are wrong.

    """

    uin = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
    hls = [0.2, 3, 13, 33, 133, 333, 1133]

    def test_section2a(self):
        """Test multiple points.

        All points should have equal u profile hence all points in a
        slice over height should be equal.

        """
        hlvs = 10
        uin = np.ones(hlvs) * 20
        heights = ((np.arange(hlvs) + 1) ** 2.0) * 12.0
        multip_hc_rc = TestMultiPoint()
        land_hc_rc = multip_hc_rc.run_hc_rc(uin, dtime=1, height=heights)
        hidx = land_hc_rc.shape.index(hlvs)
        for field in land_hc_rc.slices_over(hidx):
            self.assertTrue((field.data == field.data[0, 0]).all())

    def test_section2b(self):
        """Test a mix of sea and land points over multiple timesteps.

        p1: sea point, no correction
        p2: land point, equal height, RC (<=uin), no HC
        p3: land point, model<pp orog: HC+, so p3>=p2 everywhere
        Two time steps tested, should be equal.

        """
        uin = np.ones(10) * 20
        heights = ((np.arange(10) + 1) ** 2.0) * 12
        multip_hc_rc = TestMultiPoint(
            shape=(3, 1),
            AoS=[0, 0.2, 0.2],
            pporog=[0, 250, 250],
            modelorog=[0, 250, 230],
        )
        land_hc_rc = multip_hc_rc.run_hc_rc(uin, dtime=2, height=heights)
        tidx = land_hc_rc.shape.index(2)
        time1 = land_hc_rc.data.take(0, axis=tidx)
        time2 = land_hc_rc.data.take(1, axis=tidx)
        # Check on time.
        self.assertArrayEqual(time1, time2)
        xidxnew = land_hc_rc.shape.index(3)
        xidxold = multip_hc_rc.w_cube.data.shape.index(3)
        landp1new = land_hc_rc.data.take(0, axis=xidxnew)
        landp1old = multip_hc_rc.w_cube.data.take(0, axis=xidxold)
        # Check on p1.
        self.assertArrayEqual(landp1new, landp1old)
        landp2new = land_hc_rc.data.take(1, axis=xidxnew)
        landp2old = multip_hc_rc.w_cube.data.take(1, axis=xidxold)
        # Check on p2.
        self.assertTrue(
            (landp2new <= landp2old).all() and (landp2new < landp2old).any()
        )
        landp3new = land_hc_rc.data.take(2, axis=xidxnew)
        # Check on p3.
        self.assertTrue(
            (landp2new <= landp3new).all() and (landp2new < landp3new).any()
        )

    def test_section2c(self):
        """As test 2b, but passing the two time steps in a list.

        timesteps are a list rather than a 4D cube. This should raise
        an error.

        """
        uin = np.ones(10) * 20
        heights = ((np.arange(10) + 1) ** 2.0) * 12
        multip_hc_rc = TestMultiPoint(
            shape=(3, 1),
            AoS=[0, 0.2, 0.2],
            pporog=[0, 250, 250],
            modelorog=[0, 250, 230],
        )
        msg = "wind input is not a cube, but <class 'iris.cube.CubeList'>"
        with self.assertRaisesRegex(TypeError, msg):
            _ = multip_hc_rc.run_hc_rc([uin, uin], dtime=2, height=heights, aslist=True)

    def test_section2d(self):
        """Test whether output is float32."""
        hlvs = 10
        uin = (np.ones(hlvs) * 20).astype(np.float32)
        heights = (((np.arange(hlvs) + 1) ** 2.0) * 12.0).astype(np.float32)
        multip_hc_rc = TestMultiPoint()
        land_hc_rc = multip_hc_rc.run_hc_rc(uin, dtime=1, height=heights)
        self.assertEqual(land_hc_rc.dtype, np.float32)

    def test_section3a(self):
        """As test 1c, however with manipulated z_0 cube.

        All ancillary fields have 1x1 dim, z_0 is on a different grid.
        This should fail with ValueError("ancillary grids are not
        consistent").

        """
        landpointtests_rc = TestSinglePoint(z_0=0.2, pporog=250.0, modelorog=250.0)
        z0_data = np.array(
            [landpointtests_rc.z0_cube.data, landpointtests_rc.z0_cube.data]
        )
        landpointtests_rc.z0_cube = make_ancil_cube(
            z0_data, "vegetative_roughness_length", "m", shape=(1, 2)
        )
        msg = "ancillary grids are not consistent"
        with self.assertRaisesRegex(ValueError, msg):
            _ = landpointtests_rc.run_hc_rc(self.uin)

    def test_section3b(self):
        """As test 3a, however with manipulated modelorog cube instead.

        This should fail with ValueError("ancillary grids are not
        consistent").

        """
        landpointtests_rc = TestSinglePoint(z_0=0.2, pporog=250.0, modelorog=250.0)
        moro_data = np.array(
            [landpointtests_rc.moro_cube.data, landpointtests_rc.moro_cube.data]
        )
        landpointtests_rc.moro_cube = make_ancil_cube(
            moro_data, "surface_altitude", "m", shape=(1, 2)
        )
        msg = "ancillary grids are not consistent"
        with self.assertRaisesRegex(ValueError, msg):
            _ = landpointtests_rc.run_hc_rc(self.uin)

    def test_section3c(self):
        """As test 3a, however with manipulated z_0 units.

        This should fail with a wrong units error.

        """
        landpointtests_rc = TestSinglePoint(z_0=0.2, pporog=250.0, modelorog=250.0)
        landpointtests_rc.z0_cube.units = Unit("s")
        msg = "z0 ancil has unexpected unit: should be {} " "is {}"
        with self.assertRaisesRegex(
            ValueError, msg.format(Unit("m"), landpointtests_rc.z0_cube.units)
        ):
            _ = landpointtests_rc.run_hc_rc(self.uin)


if __name__ == "__main__":
    unittest.main()
