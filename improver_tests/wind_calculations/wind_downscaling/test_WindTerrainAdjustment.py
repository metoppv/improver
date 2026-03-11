# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for plugin wind_downscaling.WindTerrainAdjustment."""

import datetime
import unittest

import iris
import numpy as np
from cf_units import Unit

from improver.constants import RMDI
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.wind_calculations.wind_downscaling import WindTerrainAdjustment


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
    Silhouette roughness (silhouette_roughness), standard deviation of model height grid
    cell (orog_stddev), vegetative roughness (roughness_length_z0), post-processing grid
    orography (orog_pp) and model orography (orog_model). If no values
    are supplied, the grids that are set up have equal values at all
    x-y points: silhouette_roughness = 0.2, orog_stddev = 20, roughness_length_z0 = 0.2, orog_pp = 250,
    orog_model = 230.

    """

    def __init__(
        self,
        shape=(3, 3),
        silhouette_roughness=None,
        orog_stddev=None,
        roughness_length_z0=0.2,
        orog_pp=None,
        orog_model=None,
    ):
        """Set up multi-point tests.

        Args:
            shape (tuple):
                Required data shape.
            silhouette_roughness (float or 1D or 2D numpy.ndarray):
                Silhouette roughness field
            orog_stddev (float or 1D or 2D numpy.ndarray):
                Standard deviation field of height in grid cell
            roughness_length_z0 (float or 1D or 2D numpy.ndarray):
                Vegetative roughness field
            orog_pp (float or 1D or 2D numpy.ndarray):
                Unsmoothed orography field on post-processing grid
            orog_model (float or 1D or 2D numpy.ndarray):
                Model orography field on post-processing grid

        """
        self.shape = shape
        if silhouette_roughness is None:
            silhouette_roughness = np.full(shape, 0.2, dtype=np.float32)
        if orog_stddev is None:
            orog_stddev = np.full(shape, 20.0, dtype=np.float32)
        if orog_pp is None:
            orog_pp = np.full(shape, 250.0, dtype=np.float32)
        if orog_model is None:
            orog_model = np.full(shape, 230.0, dtype=np.float32)
        self.w_cube = None
        self.silhouette_roughness_cube = make_ancil_cube(
            silhouette_roughness, "silhouette_roughness", 1, shape=shape
        )
        self.s_cube = make_ancil_cube(
            orog_stddev, "standard_deviation_of_height_in_grid_cell", "m", shape=shape
        )
        if roughness_length_z0 is None:
            self.z0_cube = None
        elif isinstance(roughness_length_z0, float):
            roughness_length_z0 = np.full(shape, roughness_length_z0, dtype=np.float32)
            self.z0_cube = make_ancil_cube(
                roughness_length_z0, "vegetative_roughness_length", "m"
            )
        elif isinstance(roughness_length_z0, list):
            self.z0_cube = make_ancil_cube(
                np.array(roughness_length_z0),
                "vegetative_roughness_length",
                "m",
                shape=shape,
            )
        self.orog_pp_cube = make_ancil_cube(
            orog_pp, "surface_altitude", "m", shape=shape
        )
        self.orog_model_cube = make_ancil_cube(
            orog_model, "surface_altitude", "m", shape=shape
        )

    def run_corrections(
        self, wind, dtime=1, height=None, aslist=False, mode="hc_and_rc"
    ):
        """Function to set up a wind cube from the supplied np.array.

        Set up the wind and call the WindTerrainAdjustment class. If the
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
            mode (str):
                Whether to do height correction (hc), roughness correction
                (rc) or both (hc_and_rc), default hc_and_rc.
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

        plugin = WindTerrainAdjustment(
            self.silhouette_roughness_cube,
            self.s_cube,
            self.orog_pp_cube,
            self.orog_model_cube,
            1500.0,
            self.z0_cube,
            mode=mode,
        )
        return plugin(self.w_cube)


class TestSinglePoint:
    """Test a single 1x1 grid.

    A cube is a single 1x x 1y grid, however, the z dimension is not 1.
    It constructs 1x1 cubes for the ancillary fields Silhouette
    roughness (silhouette_roughness) and standard deviation of model height grid cell
    (orog_stddev), vegetative roughness (roughness_length_z0), post-processing grid orography
    (orog_pp) and model orography(orog_model). If no values are supplied,
    the values are: silhouette_roughness = 0.2, orog_stddev = 20, roughness_length_z0 = 0.2, orog_pp = 250,
    orog_model = 230.

    The height level grid (heightlevels) can be supplied as an 1D
    array. If nothing is supplied, the height level grid is [0.2, 3,
    13, 33, 133, 333, 1133].

    """

    def __init__(
        self,
        silhouette_roughness=0.2,
        orog_stddev=20.0,
        roughness_length_z0=0.2,
        orog_pp=250.0,
        orog_model=230.0,
        heightlevels=np.array([0.2, 3.0, 13.0, 33.0, 133.0, 333.0, 1133.0]),
    ):
        """Set up the single point test for WindTerrainAdjustment.

        Args:
            silhouette_roughness (float):
                Silhouette roughness field
            orog_stddev (float):
                Standard deviation field of height in grid cell
            roughness_length_z0 (float):
                Vegetative roughness field
            orog_pp (float):
                Unsmoothed orography on post-processing grid
            orog_model (float):
                Model orography on post-processing grid
            heightlevels (1D numpy.ndarray):
                Height level array

        """
        self.w_cube = None
        self.silhouette_roughness_cube = make_ancil_cube(
            silhouette_roughness, "silhouette_roughness", 1, shape=(1, 1)
        )
        self.s_cube = make_ancil_cube(
            orog_stddev, "standard_deviation_of_height_in_grid_cell", "m", shape=(1, 1)
        )
        if roughness_length_z0 is None:
            self.z0_cube = None
        else:
            self.z0_cube = make_ancil_cube(
                roughness_length_z0, "vegetative_roughness_length", "m", shape=(1, 1)
            )
        self.orog_pp_cube = make_ancil_cube(
            orog_pp, "surface_altitude", "m", shape=(1, 1)
        )
        self.orog_model_cube = make_ancil_cube(
            orog_model, "surface_altitude", "m", shape=(1, 1)
        )
        if heightlevels is not None:
            self.hl_cube = make_point_height_ancil_cube(heightlevels)
        else:
            self.hl_cube = None

    def run_corrections(self, wind, height=None, mode="hc_and_rc"):
        """Test single point height correction and roughness correction.

        Make an iris cube of the supplied wind and set up the height
        axis in m.

        Args:
            wind (list or 1D numpy.ndarray):
                Array of wind speeds
            height (float):
                Value for height in metres for zeroth slice of wind,
                default None.
            mode (str):
                Whether to do height correction (hc), roughness correction
                (rc) or both (hc_and_rc), default hc_and_rc.
        """
        self.w_cube = make_point_data_cube(wind, "wind_speed", "m s-1")
        plugin = WindTerrainAdjustment(
            silhouette_roughness_cube=self.silhouette_roughness_cube,
            orog_stddev_cube=self.s_cube,
            orog_pp_cube=self.orog_pp_cube,
            orog_model_cube=self.orog_model_cube,
            res_model=1500.0,
            z0_cube=self.z0_cube,
            height_levels_cube=self.hl_cube,
            mode=mode,
        )
        return plugin(self.w_cube)


class Test1D(unittest.TestCase):
    """
    Tests 1D wind correction behaviour for RC, HC, and combined modes.

    Covers:
    - Error handling for invalid inputs (nan, RMDI, bad height grids, missing z0).
    - Mask logic (sea points, zero/invalid ancillaries -> no correction).
    - RC and HC behaviour individually (positive, negative, and zero corrections).
    - Equivalence of combined mode 'hc_and_rc' with sequential RC -> HC.
    - Output dtype (float32).
    """

    uin = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0], dtype=np.float32)
    hls = np.array([0.2, 3, 13, 33, 133, 333, 1133], dtype=np.float32)

    def test_hc_and_rc_equivalent_to_rc_then_hc(self):
        """
        Test that the combined mode 'hc_and_rc' produces the same result as
        applying RC first and then HC sequentially.
        """
        land = TestSinglePoint(
            roughness_length_z0=0.1,
            silhouette_roughness=0.5,
            orog_stddev=50.0,
            orog_pp=250.0,
            orog_model=200.0,
            heightlevels=self.hls,
        )

        # mode 'hc_and_rc'
        combined = land.run_corrections(self.uin, mode="hc_and_rc")

        # mode 'rc' followed by mode 'hc'
        after_rc_cube = land.run_corrections(self.uin, mode="rc")
        after_rc = after_rc_cube.data
        sequential_cube = land.run_corrections(after_rc, mode="hc")
        sequential = sequential_cube.data

        np.testing.assert_allclose(combined.data, sequential, rtol=1e-6, atol=1e-7)

    def test_rc_raises_error_when_z0_none(self):
        """Test that RC raises ValueError when roughness_length_z0 is None."""
        landpointtests_hc = TestSinglePoint(roughness_length_z0=None, orog_model=250.0)
        expected_msg = (
            r"Roughness correction \(RC\) requested via mode=.*?, "
            r"but no z0_cube was supplied\. Provide a roughness-length cube or use mode='hc'\."
        )
        with self.assertRaisesRegex(ValueError, expected_msg):
            _ = landpointtests_hc.run_corrections(self.uin, mode="rc")

    def test_rc_and_hc_raises_error_when_z0_none(self):
        """Test that HC+RC raises ValueError when roughness_length_z0 is None."""
        landpointtests_hc = TestSinglePoint(roughness_length_z0=None, orog_model=250.0)
        expected_msg = (
            r"Roughness correction \(RC\) requested via mode=.*?, "
            r"but no z0_cube was supplied\. Provide a roughness-length cube or use mode='hc'\."
        )
        with self.assertRaisesRegex(ValueError, expected_msg):
            _ = landpointtests_hc.run_corrections(self.uin, mode="hc_and_rc")

    def test_invalid_mode_raises_value_error(self):
        """Test that an invalid mode raises ValueError with the correct message."""
        landpointtests = TestSinglePoint(roughness_length_z0=1.0, orog_model=250.0)
        invalid_mode = "unsupported"
        expected_msg = (
            r"mode must be one of \('hc_and_rc', 'hc', 'rc'\), got 'unsupported'"
        )
        with self.assertRaisesRegex(ValueError, expected_msg):
            _ = landpointtests.run_corrections(self.uin, mode=invalid_mode)

    def test_rc_no_correction_when_silhouette_roughness_is_RMDI(self):
        """Test that RC performs no correction when silhouette_roughness is RMDI (uin == uout)."""
        landpointtests_hc_rc = TestSinglePoint(
            silhouette_roughness=RMDI, heightlevels=self.hls
        )
        land_hc_rc = landpointtests_hc_rc.run_corrections(self.uin, mode="rc")
        np.testing.assert_array_equal(landpointtests_hc_rc.w_cube, land_hc_rc)

    def test_rc_no_correction_when_silhouette_roughness_is_nan(self):
        """Test that RC performs no correction when silhouette_roughness is np.nan (uin == uout)."""
        landpointtests_hc_rc = TestSinglePoint(
            silhouette_roughness=np.nan, heightlevels=self.hls
        )
        land_hc_rc = landpointtests_hc_rc.run_corrections(self.uin, mode="rc")
        np.testing.assert_array_equal(landpointtests_hc_rc.w_cube, land_hc_rc)

    def test_rc_no_correction_when_orog_stddev_is_RMDI(self):
        """Test that RC performs no correction when orog_stddev is RMDI (uin == uout)."""
        land = TestSinglePoint(
            orog_stddev=RMDI,
            heightlevels=self.hls,
        )
        result = land.run_corrections(self.uin, mode="rc")
        np.testing.assert_array_equal(land.w_cube, result)

    def test_rc_no_correction_when_orog_stddev_is_nan(self):
        """Test that RC performs no correction when orog_stddev is NaN (uin == uout)."""
        land = TestSinglePoint(
            orog_stddev=np.nan,
            heightlevels=self.hls,
        )
        result = land.run_corrections(self.uin, mode="rc")
        np.testing.assert_array_equal(land.w_cube, result)

    def test_hc_no_correction_when_orog_stddev_is_nan(self):
        """Test that HC performs no correction when orog_stddev is NaN (uin == uout)."""
        land = TestSinglePoint(
            orog_stddev=np.nan,
            heightlevels=self.hls,
        )
        result = land.run_corrections(self.uin, mode="hc")
        np.testing.assert_array_equal(land.w_cube, result)

    def test_hc_no_correction_when_orography_equal(self):
        """Test that HC performs no correction when orog_pp equals orog_model (uin == uout)."""
        land = TestSinglePoint(
            orog_pp=230.0,
            orog_model=230.0,  # model orog = pp orog
            heightlevels=self.hls,
        )
        result = land.run_corrections(self.uin, mode="hc")
        np.testing.assert_array_equal(land.w_cube, result)

    def test_hc_positive_when_model_orog_less_than_pp(self):
        """Test that HC is positive when model orography is lower than PP orography."""
        land = TestSinglePoint(
            orog_pp=250.0,
            orog_model=200.0,  # model orog < pp orog
            heightlevels=self.hls,
        )
        result = land.run_corrections(self.uin, mode="hc")
        self.assertTrue((result.data > land.w_cube.data).all())

    def test_hc_negative_values_clamped_to_zero_near_surface(self):
        """Test that HC clamping prevents negative wind speeds near the surface.

        When orog_model > orog_pp, HC is negative. Negative wind speeds must be
        clamped to zero at the lowest height level.
        """
        land = TestSinglePoint(
            orog_pp=230.0,
            orog_model=250.0,  # model orog > pp orog, so HC is negative
            heightlevels=self.hls,
        )
        result = land.run_corrections(self.uin, mode="hc")
        self.assertTrue(
            (result.data >= 0).all()  # clamping: no negative winds
            and (result.data < land.w_cube.data).any()  # reduces at least 1 level
        )

    def test_error_when_height_grid_contains_RMDI(self):
        """Raise ValueError when the height grid contains RMDI."""
        hls = [0.2, 3, 13, RMDI, 133, 333, 1133]
        land = TestSinglePoint(heightlevels=hls)
        with self.assertRaisesRegex(
            ValueError, r"Height grid contains invalid points\."
        ):
            _ = land.run_corrections(self.uin)

    def test_error_when_height_grid_contains_nan(self):
        """Raise ValueError when the height grid contains NaN."""
        hls = [0.2, 3, 13, np.nan, 133, 333, 1133]
        land = TestSinglePoint(heightlevels=hls)
        with self.assertRaisesRegex(
            ValueError, r"Height grid contains invalid points\."
        ):
            _ = land.run_corrections(self.uin)

    def test_rc_error_when_wind_input_contains_RMDI(self):
        """Test that RC errors when uin contains RMDI."""
        uin = [20.0, 20.0, 20.0, RMDI, RMDI, 20.0, 0.0]
        land = TestSinglePoint(heightlevels=self.hls)
        with self.assertRaises(ValueError):
            _ = land.run_corrections(uin, mode="rc")

    def test_rc_error_when_wind_input_contains_nan(self):
        """Test that RC errors when uin contains NaN."""
        uin = [20.0, 20.0, 20.0, np.nan, 20.0, 20.0, 20.0]
        land = TestSinglePoint(heightlevels=self.hls)
        with self.assertRaises(ValueError):
            _ = land.run_corrections(uin, mode="rc")

    def test_hc_error_when_wind_input_contains_nan(self):
        """Test that HC errors when uin contains NaN."""
        uin = [20.0, 20.0, 20.0, np.nan, 20.0, 20.0, 20.0]
        land = TestSinglePoint(heightlevels=self.hls)
        with self.assertRaises(ValueError):
            _ = land.run_corrections(uin, mode="hc")

    def test_rc_no_correction_for_sea_point(self):
        """Test that RC performs no correction when silhouette_roughness=0 (sea point)."""
        land = TestSinglePoint(silhouette_roughness=0.0, heightlevels=self.hls)
        result = land.run_corrections(self.uin, mode="rc")
        np.testing.assert_array_equal(land.w_cube, result)

    def test_hc_no_correction_for_sea_point(self):
        """Test that HC performs no correction when silhouette_roughness=0 (sea point)."""
        land = TestSinglePoint(silhouette_roughness=0.0, heightlevels=self.hls)
        result = land.run_corrections(self.uin, mode="hc")
        np.testing.assert_array_equal(land.w_cube, result)

    def test_rc_no_correction_when_orog_stddev_zero_sea_point(self):
        """Test that RC performs no correction when orog_stddev=0 (sea point)."""
        land = TestSinglePoint(orog_stddev=0.0, heightlevels=self.hls)
        result = land.run_corrections(self.uin, mode="rc")
        np.testing.assert_array_equal(land.w_cube, result)

    def test_hc_no_correction_when_orog_stddev_zero_sea_point(self):
        """Test that HC performs no correction when orog_stddev=0 (sea point)."""
        land = TestSinglePoint(orog_stddev=0.0, heightlevels=self.hls)
        result = land.run_corrections(self.uin, mode="hc")
        np.testing.assert_array_equal(land.w_cube, result)

    def test_rc_output_is_float32(self):
        """Test that RC code returns float32 precision."""
        landpointtests_hc_rc = TestSinglePoint()
        land_hc_rc = landpointtests_hc_rc.run_corrections(self.uin, mode="rc")
        self.assertEqual(land_hc_rc.dtype, np.float32)

    def test_hc_output_is_float32(self):
        """Test that HC code returns float32 precision."""
        landpointtests_hc_rc = TestSinglePoint()
        land_hc_rc = landpointtests_hc_rc.run_corrections(self.uin, mode="hc")
        self.assertEqual(land_hc_rc.dtype, np.float32)


class Test2D(unittest.TestCase):
    """Tests multi-point and multi-time wind corrections.

    Covers:
    - multiple point + multiple time tests
    - tests that should fail (grids are not all the same or units are wrong)
    """

    uin = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
    hls = [0.2, 3, 13, 33, 133, 333, 1133]

    def test_uniform_profile_preserved_across_all_points(self):
        """Test multiple points.

        All points should have equal u profile hence all points in a
        slice over height should be equal.

        """
        hlvs = 10
        uin = np.ones(hlvs) * 20
        heights = ((np.arange(hlvs) + 1) ** 2.0) * 12.0
        multip_hc_rc = TestMultiPoint()
        land_hc_rc = multip_hc_rc.run_corrections(uin, dtime=1, height=heights)
        hidx = land_hc_rc.shape.index(hlvs)
        for field in land_hc_rc.slices_over(hidx):
            self.assertTrue((field.data == field.data[0, 0]).all())

    def test_mixed_sea_and_land_points_over_multiple_timesteps(self):
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
            silhouette_roughness=[0, 0.2, 0.2],
            orog_pp=[0, 250, 250],
            orog_model=[0, 250, 230],
        )
        land_hc_rc = multip_hc_rc.run_corrections(uin, dtime=2, height=heights)
        tidx = land_hc_rc.shape.index(2)
        time1 = land_hc_rc.data.take(0, axis=tidx)
        time2 = land_hc_rc.data.take(1, axis=tidx)
        # Check on time.
        np.testing.assert_array_equal(time1, time2)
        xidxnew = land_hc_rc.shape.index(3)
        xidxold = multip_hc_rc.w_cube.data.shape.index(3)
        landp1new = land_hc_rc.data.take(0, axis=xidxnew)
        landp1old = multip_hc_rc.w_cube.data.take(0, axis=xidxold)
        # Check on p1.
        np.testing.assert_array_equal(landp1new, landp1old)
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

    def test_error_when_passing_list_instead_of_cube(self):
        """Test that passing timesteps as a list instead of a Cube raises a TypeError."""
        uin = np.ones(10) * 20
        heights = ((np.arange(10) + 1) ** 2.0) * 12
        multip_hc_rc = TestMultiPoint(
            shape=(3, 1),
            silhouette_roughness=[0, 0.2, 0.2],
            orog_pp=[0, 250, 250],
            orog_model=[0, 250, 230],
        )
        msg = "Wind input is not a cube, but <class 'iris.cube.CubeList'>"
        with self.assertRaisesRegex(TypeError, msg):
            _ = multip_hc_rc.run_corrections(
                [uin, uin], dtime=2, height=heights, aslist=True
            )

    def test_output_dtype_float32_in_multipoint_case(self):
        """Test whether output is float32."""
        hlvs = 10
        uin = (np.ones(hlvs) * 20).astype(np.float32)
        heights = (((np.arange(hlvs) + 1) ** 2.0) * 12.0).astype(np.float32)
        multip_hc_rc = TestMultiPoint()
        land_hc_rc = multip_hc_rc.run_corrections(uin, dtime=1, height=heights)
        self.assertEqual(land_hc_rc.dtype, np.float32)

    def test_error_when_z0_grid_inconsistent(self):
        """Test that a roughness_length_z0 cube on an inconsistent grid raises an ancillary-grid error.

        All ancillary fields have 1x1 dim, roughness_length_z0 is on a different grid.
        This should fail with ValueError.

        """
        landpointtests_rc = TestSinglePoint(
            roughness_length_z0=0.2, orog_pp=250.0, orog_model=250.0
        )
        z0_data = np.array(
            [landpointtests_rc.z0_cube.data, landpointtests_rc.z0_cube.data]
        )
        landpointtests_rc.z0_cube = make_ancil_cube(
            z0_data, "vegetative_roughness_length", "m", shape=(1, 2)
        )
        msg = "Ancillary grids are not consistent."
        with self.assertRaisesRegex(ValueError, msg):
            _ = landpointtests_rc.run_corrections(self.uin)

    def test_error_when_orog_model_grid_inconsistent(self):
        """Test that a manipulated orog_model cube on an inconsistent grid raises an ancillary-grid error.

        This should fail with ValueError.
        """
        landpointtests_rc = TestSinglePoint(
            roughness_length_z0=0.2, orog_pp=250.0, orog_model=250.0
        )
        moro_data = np.array(
            [
                landpointtests_rc.orog_model_cube.data,
                landpointtests_rc.orog_model_cube.data,
            ]
        )
        landpointtests_rc.orog_model_cube = make_ancil_cube(
            moro_data, "surface_altitude", "m", shape=(1, 2)
        )
        msg = "Ancillary grids are not consistent."
        with self.assertRaisesRegex(ValueError, msg):
            _ = landpointtests_rc.run_corrections(self.uin)

    def test_error_when_z0_has_incorrect_units(self):
        """Test that a roughness_length_z0 cube with incorrect units raises a ValueError.

        This should fail with a wrong units error.
        """
        landpointtests_rc = TestSinglePoint(
            roughness_length_z0=0.2, orog_pp=250.0, orog_model=250.0
        )
        landpointtests_rc.z0_cube.units = Unit("s")
        msg = "z0 ancillary has unexpected unit: expected {}, got {}"
        with self.assertRaisesRegex(
            ValueError, msg.format(Unit("m"), landpointtests_rc.z0_cube.units)
        ):
            _ = landpointtests_rc.run_corrections(self.uin)


if __name__ == "__main__":
    unittest.main()
