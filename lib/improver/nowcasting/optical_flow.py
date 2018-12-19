# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
This module defines the optical flow velocity calculation and extrapolation
classes for advection nowcasting.
"""
import warnings
import numpy as np

import scipy.linalg
import scipy.ndimage
import scipy.signal

import iris
from iris.coords import AuxCoord
from iris.exceptions import CoordinateNotFoundError, InvalidCubeError

from improver.utilities.cube_checker import check_for_x_and_y_axes
from improver.utilities.cube_metadata import (
    amend_metadata, add_history_attribute)
from improver.utilities.spatial import check_if_grid_is_equal_area


def check_input_coords(cube, require_time=False):
    """
    Checks an input cube has precisely two non-scalar dimension coordinates
    (spatial x/y), or raises an error.  If "require_time" is set to True,
    raises an error if no scalar time coordinate is present.

    Args:
        cube (iris.cube.Cube):
            Cube to be checked
        require_time (bool):
            Flag to check for a scalar time coordinate

    Raises:
        InvalidCubeError if coordinate requirements are not met
    """
    # check that cube has both x and y axes
    try:
        check_for_x_and_y_axes(cube)
    except ValueError as msg:
        raise InvalidCubeError(msg)

    # check that cube data has only two non-scalar dimensions
    data_shape = np.array(cube.shape)
    non_scalar_coords = np.sum(np.where(data_shape > 1, 1, 0))
    if non_scalar_coords > 2:
        raise InvalidCubeError('Cube has {:d} (more than 2) non-scalar '
                               'coordinates'.format(non_scalar_coords))

    if require_time:
        try:
            _ = cube.coord("time")
        except CoordinateNotFoundError:
            raise InvalidCubeError('Input cube has no time coordinate')


class AdvectField(object):
    """
    Class to advect a 2D spatial field given velocities along the two vector
    dimensions
    """

    def __init__(self, vel_x, vel_y, metadata_dict=None):
        """
        Initialises the plugin.  Velocities are expected to be on a regular
        grid (such that grid spacing in metres is the same at all points in
        the domain).

        Args:
            vel_x (iris.cube.Cube):
                Cube containing a 2D array of velocities along the x
                coordinate axis
            vel_y (iris.cube.Cube):
                Cube containing a 2D array of velocities along the y
                coordinate axis

        Keyword Args:
            metadata_dict (dict):
                Dictionary containing information for amending the metadata
                of the output cube. Please see the
                :func:`improver.utilities.cube_metadata.amend_metadata`
                for information regarding the allowed contents of the metadata
                dictionary.
        """

        # check each input velocity cube has precisely two non-scalar
        # dimension coordinates (spatial x/y)
        check_input_coords(vel_x)
        check_input_coords(vel_y)

        # check input velocity cubes have the same spatial coordinates
        if (vel_x.coord(axis="x") != vel_y.coord(axis="x") or
                vel_x.coord(axis="y") != vel_y.coord(axis="y")):
            raise InvalidCubeError("Velocity cubes on unmatched grids")

        vel_x.convert_units('m s-1')
        vel_y.convert_units('m s-1')

        self.vel_x = vel_x
        self.vel_y = vel_y

        self.x_coord = vel_x.coord(axis="x")
        self.y_coord = vel_x.coord(axis="y")

        # Initialise metadata dictionary.
        if metadata_dict is None:
            metadata_dict = {}
        self.metadata_dict = metadata_dict

    def __repr__(self):
        """Represent the plugin instance as a string."""
        result = ('<AdvectField: vel_x={}, vel_y={}, '
                  'metadata_dict={}>'.format(
                      self.vel_x.name(), self.vel_y.name(),
                      self.metadata_dict))
        return result

    @staticmethod
    def _increment_output_array(indata, outdata, cond, xdest_grid, ydest_grid,
                                xsrc_grid, ysrc_grid, x_weight, y_weight):
        """
        Calculate and add contribution to the advected array from one source
        grid point, for all points where boolean condition "cond" is valid.

        Args:
            indata (numpy.ndarray):
                2D numpy array of source data to be advected
            outdata (numpy.ndarray):
                2D numpy array for advected output, modified in place by
                this method (is both input and output).
            cond (numpy.ndarray):
                2D boolean mask of points to be processed
            xdest_grid (numpy.ndarray):
                Integer x-coordinates of all points on destination grid
            ydest_grid (numpy.ndarray):
                Integer y-coordinates of all points on destination grid
            xsrc_grid (numpy.ndarray):
                Integer x-coordinates of all points on source grid
            ysrc_grid (numpy.ndarray):
                Integer y-coordinates of all points on source grid
            x_weight (numpy.ndarray):
                Fractional contribution to destination grid of source data
                advected along the x-axis.  Positive definite.
            y_weight (numpy.ndarray):
                Fractional contribution to destination grid of source data
                advected along the y-axis.  Positive definite.
        """
        xdest = xdest_grid[cond]
        ydest = ydest_grid[cond]
        xsrc = xsrc_grid[cond]
        ysrc = ysrc_grid[cond]
        outdata[ydest, xdest] += (
            indata[ysrc, xsrc]*x_weight[ydest, xdest]*y_weight[ydest, xdest])

    def _advect_field(self, data, grid_vel_x, grid_vel_y, timestep):
        """
        Performs a dimensionless grid-based extrapolation of spatial data
        using advection velocities via a backwards method.  Points where data
        cannot be extrapolated (ie the source is out of bounds) are given a
        fill value of np.nan and masked.

        Args:
            data (numpy.ndarray or numpy.ma.MaskedArray):
                2D numpy data array to be advected
            grid_vel_x (numpy.ndarray):
                Velocity in the x direction (in grid points per second)
            grid_vel_y (numpy.ndarray):
                Velocity in the y direction (in grid points per second)
            timestep (int):
                Advection time step in seconds

        Returns:
            adv_field (numpy.ma.MaskedArray):
                2D float array of advected data values with masked "no data"
                regions
        """
        # Cater for special case where timestep (integer) is 0
        if timestep == 0:
            return data

        # Initialise advected field with np.nan
        adv_field = np.full(data.shape, np.nan, dtype=np.float32)

        # Set up grids of data coordinates (meshgrid inverts coordinate order)
        ydim, xdim = data.shape
        (xgrid, ygrid) = np.meshgrid(np.arange(xdim),
                                     np.arange(ydim))

        # For each grid point on the output field, trace its (x,y) "source"
        # location backwards using advection velocities.  The source location
        # is generally fractional: eg with advection velocities of 0.5 grid
        # squares per second, the value at [2, 2] is represented by the value
        # that was at [1.5, 1.5] 1 second ago.
        xsrc_point_frac = -grid_vel_x * timestep + xgrid.astype(np.float32)
        ysrc_point_frac = -grid_vel_y * timestep + ygrid.astype(np.float32)

        # For all the points where fractional source coordinates are within
        # the bounds of the field, set the output field to 0
        def point_in_bounds(x, y, nx, ny):
            """Check point (y, x) lies within defined bounds"""
            return (x >= 0.) & (x < nx) & (y >= 0.) & (y < ny)

        cond_pt = point_in_bounds(xsrc_point_frac, ysrc_point_frac, xdim, ydim)
        adv_field[cond_pt] = 0

        # Find the integer points surrounding the fractional source coordinates
        xsrc_point_lower = xsrc_point_frac.astype(int)
        ysrc_point_lower = ysrc_point_frac.astype(int)
        x_points = [xsrc_point_lower, xsrc_point_lower + 1]
        y_points = [ysrc_point_lower, ysrc_point_lower + 1]

        # Calculate the distance-weighted fractional contribution of points
        # surrounding the source coordinates
        x_weight_upper = xsrc_point_frac - xsrc_point_lower.astype(float)
        y_weight_upper = ysrc_point_frac - ysrc_point_lower.astype(float)
        x_weights = np.array([1. - x_weight_upper, x_weight_upper],
                             dtype=np.float32)
        y_weights = np.array([1. - y_weight_upper, y_weight_upper],
                             dtype=np.float32)

        # Check whether the input data is masked - if so substitute NaNs for
        # the masked data.  Note there is an implicit type conversion here: if
        # data is of integer type this unmasking will convert it to float.
        if isinstance(data, np.ma.MaskedArray):
            data = np.where(data.mask, np.nan, data.data)

        # Advect data from each of the four source points onto the output grid
        for xpt, xwt in zip(x_points, x_weights):
            for ypt, ywt in zip(y_points, y_weights):
                cond = point_in_bounds(xpt, ypt, xdim, ydim) & cond_pt
                self._increment_output_array(
                    data, adv_field, cond, xgrid, ygrid, xpt, ypt, xwt, ywt)

        # Replace NaNs with a mask
        adv_field = np.ma.masked_where(~np.isfinite(adv_field), adv_field)

        return adv_field

    def process(self, cube, timestep):
        """
        Extrapolates input cube data and updates validity time.  The input
        cube should have precisely two non-scalar dimension coordinates
        (spatial x/y), and is expected to be in a projection such that grid
        spacing is the same (or very close) at all points within the spatial
        domain.  The input cube should also have a "time" coordinate.

        Args:
            cube (iris.cube.Cube):
                The 2D cube containing data to be advected
            timestep (datetime.timedelta):
                Advection time step

        Returns:
            advected_cube (iris.cube.Cube):
                New cube with updated time and extrapolated data.  New data
                are filled with np.nan and masked where source data were
                out of bounds (ie where data could not be advected from outside
                the cube domain).
        """
        # check that the input cube has precisely two non-scalar dimension
        # coordinates (spatial x/y) and a scalar time coordinate
        check_input_coords(cube, require_time=True)

        # check spatial coordinates match those of plugin velocities
        if (cube.coord(axis="x") != self.x_coord or
                cube.coord(axis="y") != self.y_coord):
            raise InvalidCubeError("Input data grid does not match advection "
                                   "velocities")

        # derive velocities in "grid squares per second"
        def grid_spacing(coord):
            """Calculate grid spacing along a given spatial axis"""
            new_coord = coord.copy()
            new_coord.convert_units('m')
            return np.float32(np.diff((new_coord).points)[0])

        grid_vel_x = self.vel_x.data / grid_spacing(cube.coord(axis="x"))
        grid_vel_y = self.vel_y.data / grid_spacing(cube.coord(axis="y"))

        # raise a warning if data contains unmasked NaNs
        nan_count = np.count_nonzero(~np.isfinite(cube.data))
        if nan_count > 0:
            warnings.warn("input data contains unmasked NaNs")

        # perform advection and create output cube
        advected_data = self._advect_field(cube.data, grid_vel_x, grid_vel_y,
                                           timestep.total_seconds())
        advected_cube = cube.copy(data=advected_data)

        # increment output cube time and add a "forecast_period" coordinate
        original_datetime, = \
            (cube.coord("time").units).num2date(cube.coord("time").points)
        new_datetime = original_datetime + timestep
        new_time = (cube.coord("time").units).date2num(new_datetime)

        if np.int64(new_time) == new_time:
            new_time = np.int64(new_time)
        else:
            new_time = np.float64(new_time)

        advected_cube.coord("time").points = new_time

        forecast_period_seconds = timestep.total_seconds()

        forecast_period_seconds = np.float32(forecast_period_seconds)

        forecast_period_coord = AuxCoord(forecast_period_seconds,
                                         standard_name="forecast_period",
                                         units="s")

        try:
            advected_cube.remove_coord("forecast_period")
        except CoordinateNotFoundError:
            pass
        advected_cube.add_aux_coord(forecast_period_coord)

        # Modify the source attribute to describe the advected field as a
        # Nowcast
        if "institution" in advected_cube.attributes.keys():
            advected_cube.attributes["source"] = (
                "{} Nowcast".format(advected_cube.attributes["institution"]))
        else:
            advected_cube.attributes["source"] = "Nowcast"
        add_history_attribute(advected_cube, "Nowcast")

        advected_cube = amend_metadata(advected_cube, **self.metadata_dict)
        return advected_cube


class OpticalFlow(object):
    """
    Class to calculate advection velocities along two orthogonal spatial axes
    from time-separated fields using an optical flow algorithm
    """

    def __init__(self, data_smoothing_method='box', iterations=100,
                 metadata_dict=None):
        """
        Initialise the class with smoothing parameters for estimating gridded
        u- and v- velocities via optical flow.

        Keyword Args:
            data_smoothing_method (str):
                Smoothing method to be used on input fields before estimating
                partial derivatives.  Can be square 'box' (as used in STEPS) or
                circular 'kernel' (used in post-calculation smoothing).
            iterations (int):
                Number of iterations to perform in post-calculation smoothing.
                The value for good convergence is 20 (Bowler et al. 2004).
            metadata_dict (dict):
                Dictionary containing information for amending the metadata
                of the output cube. Please see the
                :func:`improver.utilities.cube_metadata.amend_metadata`
                for information regarding the allowed contents of the metadata
                dictionary. This metadata_dict is used to amend both of the
                resulting u and v cubes.

        Raises:
            ValueError:
                If iterations < 20

        References:
            Bowler, N., Pierce, C. and Seed, A. 2004: Development of a
            precipitation nowcasting algorithm based upon optical flow
            techniques. Journal of Hydrology, 288, 74-91.
        """

        if iterations < 20:
            raise ValueError('Got {} iterations; minimum requirement 20 '
                             'iterations'.format(iterations))

        # Set parameters for input data smoothing.  14 km is suitable for input
        # fields separated by a 15 minute time step - this is updated if
        # necessary by the "process" function.
        self.data_smoothing_radius_km = 14.
        self.data_smoothing_method = data_smoothing_method

        # Set parameters for velocity calculation and "smart smoothing"
        self.iterations = iterations
        self.point_weight = 0.1

        # Initialise input data fields and shape
        self.data1 = None
        self.data2 = None
        self.shape = None

        # Initialise metadata dictionary.
        if metadata_dict is None:
            metadata_dict = {}
        self.metadata_dict = metadata_dict

    def __repr__(self):
        """Represent the plugin instance as a string."""
        result = ('<OpticalFlow: data_smoothing_radius_km: {}, '
                  'data_smoothing_method: {}, iterations: {}, '
                  'point_weight: {}, metadata_dict: {}>')
        return result.format(
            self.data_smoothing_radius_km, self.data_smoothing_method,
            self.iterations, self.point_weight, self.metadata_dict)

    @staticmethod
    def interp_to_midpoint(data, axis=None):
        """
        Interpolates to the x-y mid-point resulting in a new grid with
        dimensions reduced in length by one.  If axis is not None, the
        interpolation is performed only over the one spatial axis
        specified.  If the input array has an axis of length 1, the
        attempt to interpolate returns an empty array: [].

        Args:
            data (np.ndarray):
                2D gridded data (dimensions M x N)
            axis (int or None):
                Optional (0 or 1): average over 2 adjacent points along the
                specified axis, rather than all 4 corners
        Returns:
            midpoints (np.ndarray):
                2D gridded interpolated average (dimensions M-1 x N-1 if
                axis=None; M-1 x N if axis=0; M x N-1 if axis=1)
        """
        if axis is None:
            midpoints = 0.25*(data[1:, :-1] + data[:-1, 1:] +
                              data[1:, 1:] + data[:-1, :-1])
        elif axis == 0:
            midpoints = 0.5*(data[:-1, :] + data[1:, :])
        elif axis == 1:
            midpoints = 0.5*(data[:, :-1] + data[:, 1:])
        return midpoints

    def _partial_derivative_spatial(self, axis=0):
        """
        Calculate the average over the two class data fields of one spatial
        derivative, averaged over the other spatial dimension.  Pad with zeros
        in both dimensions, then smooth to the original grid shape.

        Args:
            axis (int):
                Axis over which to calculate the spatial derivative (0 or 1)

        Returns:
            (np.ndarray):
                Smoothed spatial derivative
        """
        outdata = []
        for data in [self.data1, self.data2]:
            diffs = self.interp_to_midpoint(
                np.diff(data, axis=axis), axis=1-axis)
            outdata.append(diffs)
        smoothed_diffs = np.zeros(
            [self.shape[0]+1, self.shape[1]+1], dtype=np.float32)
        smoothed_diffs[1:-1, 1:-1] = 0.5*(outdata[0] + outdata[1])
        return self.interp_to_midpoint(smoothed_diffs)

    def _partial_derivative_temporal(self):
        """
        Calculate the partial derivative of two fields over time.  Take the
        difference between time-separated fields data1 and data2, average
        over the two spatial dimensions, regrid to a zero-padded output
        array, and smooth to the original grid shape.

        Returns:
            (np.ndarray):
                Smoothed temporal derivative
        """
        tdiff = self.data2 - self.data1
        smoothed_diffs = np.zeros(
            [self.shape[0]+1, self.shape[1]+1], dtype=np.float32)
        smoothed_diffs[1:-1, 1:-1] = self.interp_to_midpoint(tdiff)
        return self.interp_to_midpoint(smoothed_diffs)

    def _make_subboxes(self, field):
        """
        Generate a list of non-overlapping "boxes" of size self.boxsize**2
        from the input field, along with weights based on data values at times
        1 and 2.  The final boxes in the list will be smaller if the size of
        the data field is not an exact multiple of "boxsize".

        Args:
            field (np.ndarray):
                Input field (partial derivative)

        Returns:
            (tuple) : tuple containing:
                **boxes** (list):
                    List of np.ndarrays of size boxsize*boxsize containing
                    slices of data from input field.
                **weights** (np.ndarray):
                    1D numpy array containing weights values associated with
                    each listed box.
        """
        boxes = []
        weights = []
        weighting_factor = 0.5 / self.boxsize**2.
        for i in range(0, field.shape[0], self.boxsize):
            for j in range(0, field.shape[1], self.boxsize):
                boxes.append(field[i:i+self.boxsize, j:j+self.boxsize])
                weight = weighting_factor*(
                    (self.data1[i:i+self.boxsize, j:j+self.boxsize]).sum() +
                    (self.data2[i:i+self.boxsize, j:j+self.boxsize]).sum())
                weight = 1. - np.exp(-1.*weight/0.8)
                weights.append(weight)
        weights = np.array(weights, dtype=np.float32)
        weights[weights < 0.01] = 0
        return boxes, weights

    def _box_to_grid(self, box_data):
        """
        Regrids calculated displacements from "box grid" (on which OFC
        equations are solved) to input data grid.

        Args:
            box_data (np.ndarray):
                Displacement of subbox on box grid

        Returns:
            grid_data (np.ndarray):
                Displacement on original data grid
        """
        grid_data = np.repeat(np.repeat(box_data, self.boxsize, axis=0),
                              self.boxsize, axis=1)
        grid_data = grid_data[:self.shape[0],
                              :self.shape[1]].astype(np.float32)
        return grid_data

    @staticmethod
    def makekernel(radius):
        """
        Make a pseudo-circular kernel of radius "radius" to smooth an input
        field (used in self.smoothing() with method='kernel').  The output
        array is zero-padded, so a radius of 1 gives the kernel:
        ::

            [[ 0.  0.  0.]
             [ 0.  1.  0.]
             [ 0.  0.  0.]]

        which has no effect on the input field.  The smallest valid radius
        of 2 gives the kernel:
        ::

            [[ 0.      0.      0.      0.      0.    ]
             [ 0.      0.0625  0.125   0.0625  0.    ]
             [ 0.      0.125   0.25    0.125   0.    ]
             [ 0.      0.0625  0.125   0.0625  0.    ]
             [ 0.      0.      0.      0.      0.    ]]

        Args:
            radius (int):
                Kernel radius or half box size for smoothing

        Returns:
            kernel_2d (np.ndarray):
                Kernel to use for generating a smoothed field.

        """
        kernel_1d = 1 - np.abs(np.linspace(-1, 1, radius*2+1))
        kernel_2d = kernel_1d.reshape(radius*2+1, 1) * \
            kernel_1d.reshape(1, radius*2+1)
        kernel_2d /= kernel_2d.sum()
        return kernel_2d

    def smooth(self, field, radius, method='box'):
        """
        Smoothing method using a square ('box') or circular kernel.  Kernel
        smoothing with a radius of 1 has no effect.

        Args:
            field (np.ndarray):
                Input field to be smoothed
            radius (int):
                Kernel radius or half box size for smoothing
            method (str):
                Method to use: 'box' (as in STEPS) or 'kernel'

        Returns:
            smoothed_field (np.ndarray):
                Smoothed data on input-shaped grid
        """
        if method == 'kernel':
            kernel = self.makekernel(radius)
            smoothed_field = scipy.signal.convolve2d(
                field, kernel, mode='same', boundary="symm")
        elif method == 'box':
            smoothed_field = scipy.ndimage.filters.uniform_filter(
                field, size=radius*2+1, mode='nearest')
        # Ensure the dtype does not change.
        smoothed_field = smoothed_field.astype(field.dtype)
        return smoothed_field

    def _smart_smooth(self, vel_point, vel_iter, weights):
        """
        Performs a single iteration of "smart smoothing" over a point and its
        neighbours as implemented in STEPS.  This smoothing (through the
        "weights" argument) ignores advection displacements which are
        identically zero, as these are assumed to occur only where there is no
        data structure from which to calculate displacements.

        Args:
            vel_point (np.ndarray):
                Original unsmoothed data
            vel_iter (np.ndarray):
                Latest iteration of smart-smoothed displacement
            weights (np.ndarray):
                Weight of each grid point for averaging

        Returns:
            vel (np.ndarray):
                Next iteration of smart-smoothed displacement
        """
        # define kernel for neighbour weighting
        neighbour_kernel = (np.array([[0.5, 1, 0.5],
                                      [1.0, 0, 1.0],
                                      [0.5, 1, 0.5]])/6.).astype(np.float32)

        # smooth input data and weights fields
        vel_neighbour = scipy.ndimage.convolve(weights*vel_iter,
                                               neighbour_kernel)
        neighbour_weights = scipy.ndimage.convolve(weights, neighbour_kernel)

        # initialise output data from latest iteration
        vel = scipy.ndimage.convolve(vel_iter, neighbour_kernel)

        # create "point" and "neighbour" validity masks using original and
        # kernel-smoothed weights
        pmask = abs(weights) > 0
        nmask = abs(neighbour_weights) > 0

        # where neighbouring points have weight, set up a "background" of
        # weighted average neighbouring values
        vel[nmask] = vel_neighbour[nmask] / neighbour_weights[nmask]

        # where a point has weight, calculate a weighted sum of the original
        # (uniterated) point value and its smoothed neighbours
        nweight = 1.0 - self.point_weight
        pweight = self.point_weight * weights
        norm = nweight * neighbour_weights + pweight

        vel[pmask] = (vel_neighbour[pmask] * nweight +
                      vel_point[pmask] * pweight[pmask]) / norm[pmask]
        return vel

    def _smooth_advection_fields(self, box_data, weights):
        """
        Performs iterative "smart smoothing" of advection displacement fields,
        accounting for zeros and reducting their weight in the final output.
        Then regrid from "box grid" (on which OFC equations are solved) to
        input data grid, and perform one final pass simple kernel smoothing.
        This is equivalent to applying the smoothness constraint defined in
        Bowler et al. 2004, equations 9-11.

        Args:
            box_data (np.ndarray):
                Displacements on box grid (modified by this function)
            weights (np.ndarray):
                Weights for smart smoothing

        Returns:
            grid_data (np.ndarray):
                Smoothed displacement vectors on input data grid

        References:
            Bowler, N., Pierce, C. and Seed, A. 2004: Development of a
            precipitation nowcasting algorithm based upon optical flow
            techniques. Journal of Hydrology, 288, 74-91.
        """
        v_orig = np.copy(box_data)

        # iteratively smooth umat and vmat
        for _ in range(self.iterations):
            box_data = self._smart_smooth(v_orig, box_data, weights)

        # reshape smoothed box velocity arrays to match input data grid
        grid_data = self._box_to_grid(box_data)

        # smooth regridded velocities to remove box edge discontinuities
        # this will fail if self.boxsize < 3
        kernelsize = int(self.boxsize/3)
        grid_data = self.smooth(grid_data, kernelsize, method='kernel')
        return grid_data

    @staticmethod
    def solve_for_uv(deriv_xy, deriv_t):
        """
        Solve the system of linear simultaneous equations for u and v using
        matrix inversion (equation 19 in STEPS document).  This is frequently
        singular, eg in the presence of too many zeroes.  In these cases,
        the function returns displacements of 0.

        Args:
            deriv_xy (np.ndarray):
                2-column matrix containing partial field derivatives d/dx
                (first column) and d/dy (second column)
            deriv_t (np.ndarray):
                1-column matrix containing partial field derivatives d/dt

        Returns:
            velocity (np.ndarray):
                2-column matrix (u, v) containing scalar displacement values
        """
        deriv_t = deriv_t.reshape([deriv_t.size, 1])
        m_to_invert = (deriv_xy.transpose()).dot(deriv_xy)
        try:
            m_inverted = np.linalg.inv(m_to_invert)
        except np.linalg.LinAlgError:
            # if matrix is not invertible, set velocities to zero
            velocity = np.array([0, 0])
        else:
            scale = (deriv_xy.transpose()).dot(deriv_t)
            velocity = -m_inverted.dot(scale)[:, 0]
        return velocity

    @staticmethod
    def extreme_value_check(umat, vmat, weights):
        """
        Checks for displacement vectors that exceed 1/3 of the dimensions
        of the input data matrix.  Replaces these extreme values and their
        smoothing weights with zeros.  Modifies ALL input arrays in place.

        Args:
            umat (np.ndarray):
                Displacement vectors in the x direction
            vmat (np.ndarray):
                Displacement vectors in the y direction
            weights (np.ndarray):
                Weights for smart smoothing
        """
        flag = (np.abs(umat) + np.abs(vmat)) > vmat.shape[0]/3.
        umat[flag] = 0
        vmat[flag] = 0
        weights[flag] = 0

    def calculate_displacement_vectors(self, partial_dx, partial_dy,
                                       partial_dt):
        """
        This implements the OFC algorithm, assuming all points in a box with
        "self.boxsize" sidelength have the same displacement components.

        Args:
            partial_dx (np.ndarray):
                2D array of partial input field derivatives d/dx
            partial_dy (np.ndarray):
                2D array of partial input field derivatives d/dy
            partial_dt (np.ndarray):
                2D array of partial input field derivatives d/dt

        Returns:
            (tuple) : tuple containing:
                **umat** (np.ndarray):
                    2D array of displacements in the x-direction
                **vmat** (np.ndarray):
                    2D array of displacements in the y-direction
        """

        # (a) Generate lists of subboxes over which velocity is constant
        dx_boxed, box_weights = self._make_subboxes(partial_dx)
        dy_boxed, _ = self._make_subboxes(partial_dy)
        dt_boxed, _ = self._make_subboxes(partial_dt)

        # (b) Solve optical flow displacement calculation on each subbox
        velocity = ([], [])
        for deriv_x, deriv_y, deriv_t in zip(dx_boxed, dy_boxed, dt_boxed):

            # Flatten arrays to create the system of linear simultaneous
            # equations to solve for this subbox
            deriv_x = deriv_x.flatten()
            deriv_y = deriv_y.flatten()
            deriv_t = deriv_t.flatten()
            # deriv_xy must be float64 in order to work OK.
            deriv_xy = (
                np.array([deriv_x, deriv_y], dtype=np.float64)).transpose()

            # Solve equations for u and v through matrix inversion
            u, v = self.solve_for_uv(deriv_xy, deriv_t)
            velocity[0].append(u)
            velocity[1].append(v)

        # (c) Reshape displacement arrays to match array of subbox points
        newshape = [int((self.shape[0]-1)/self.boxsize) + 1,
                    int((self.shape[1]-1)/self.boxsize) + 1]
        umat = np.array(velocity[0], dtype=np.float32).reshape(newshape)
        vmat = np.array(velocity[1], dtype=np.float32).reshape(newshape)
        weights = box_weights.reshape(newshape)

        # (d) Check for extreme advection displacements (over a significant
        #     proportion of the domain size) and set to zero
        self.extreme_value_check(umat, vmat, weights)

        # (e) smooth and reshape displacement arrays to match input data grid
        umat = self._smooth_advection_fields(umat, weights)
        vmat = self._smooth_advection_fields(vmat, weights)

        return umat, vmat

    @staticmethod
    def _zero_advection_velocities_warning(
            vel_comp, rain_mask, zero_vel_threshold=0.1):
        """
        Raise warning if more than a fixed threshold (default 10%) of cells
        where there is rain within the domain have zero advection velocities.

        Args:
            vel_comp (np.ndarray):
                Advection velocity that will be checked to assess the
                proportion of zeroes present in this field.
            rain_mask (tuple):
                Array indices where there is rain.

        Keyword Args:
            zero_vel_threshold (float):
                Fractional value to specify the proportion of zero values
                that the advection field should contain at a maximum.
                For example, if zero_vel_threshold=0.1 then up to 10% of
                the advection velocities can be zero before a warning will be
                raised.

        Warns:
            Warning: If the proportion of zero advection velocities is
                above the threshold specified by zero_vel_threshold.

        """
        zeroes_in_rain = np.count_nonzero(vel_comp[rain_mask] == 0)
        rain_pixels = vel_comp[rain_mask].size

        if zeroes_in_rain > rain_pixels*zero_vel_threshold:
            msg = ("{:.1f}% of rain cells within the domain have zero "
                   "advection velocities. It is expected that greater "
                   "than {:.1f}% of these advection velocities will be "
                   "non-zero.".format(zeroes_in_rain*100./rain_pixels,
                                      (1-zero_vel_threshold)*100))
            warnings.warn(msg)

    def process_dimensionless(self, data1, data2, xaxis, yaxis,
                              smoothing_radius):
        """
        Calculates dimensionless advection displacements between two input
        fields.

        Args:
            data1 (np.ndarray):
                2D input data array from time 1
            data2 (np.ndarray):
                2D input data array from time 2
            xaxis (int):
                Index of x coordinate axis
            yaxis (int):
                Index of y coordinate axis
            smoothing_radius (int):
                Radius (in grid squares) over which to smooth the input data

        Returns:
            (tuple) : tuple containing:
                **ucomp** (np.ndarray):
                    Advection displacement (grid squares) in the x direction
                **vcomp** (np.ndarray):
                    Advection displacement (grid squares) in the y direction
        """
        # Smooth input data
        self.shape = data1.shape
        self.data1 = self.smooth(data1, smoothing_radius,
                                 method=self.data_smoothing_method)
        self.data2 = self.smooth(data2, smoothing_radius,
                                 method=self.data_smoothing_method)

        # Calculate partial derivatives of the smoothed input fields
        partial_dx = self._partial_derivative_spatial(axis=xaxis)
        partial_dy = self._partial_derivative_spatial(axis=yaxis)
        partial_dt = self._partial_derivative_temporal()

        # Calculate advection displacements
        ucomp, vcomp = self.calculate_displacement_vectors(
            partial_dx, partial_dy, partial_dt)

        # Check for zeros where there should be valid displacements
        rain_mask = np.where((data1 > 0) | (data2 > 0))
        for vel_comp in [ucomp, vcomp]:
            self._zero_advection_velocities_warning(vel_comp, rain_mask)
        return ucomp, vcomp

    def process(self, cube1, cube2, boxsize=30):
        """
        Extracts data from input cubes, performs dimensionless advection
        displacement calculation, and creates new cubes with advection
        velocities in metres per second.  Each input cube should have precisely
        two non-scalar dimension coordinates (spatial x/y), and are expected to
        be in a projection such that grid spacing is the same (or very close)
        at all points within the spatial domain.  Each input cube must also
        have a scalar "time" coordinate.

        Args:
            cube1 (iris.cube.Cube):
                2D cube from (earlier) time 1
            cube2 (iris.cube.Cube):
                2D cube from (later) time 2

        Kwargs:
            boxsize (int):
                The side length of the square box over which to solve the
                optical flow constraint.  This should be greater than the
                data smoothing radius.

        Returns:
            (tuple) : tuple containing:
                **ucube** (iris.cube.Cube):
                    2D cube of advection velocities in the x-direction
                **vcube** (iris.cube.Cube):
                    2D cube of advection velocities in the y-direction
        """
        # clear existing parameters
        self.data_smoothing_radius = None
        self.boxsize = None

        # check the nature of the input cubes, and raise a warning if they are
        # not both precipitation
        if cube1.name() != cube2.name():
            msg = 'Input cubes contain different data types {} and {}'
            raise ValueError(msg.format(cube1.name(), cube2.name()))

        data_name = cube1.name().lower()
        if "rain" not in data_name and "precipitation" not in data_name:
            msg = ('Input data are of non-precipitation type {}.  Plugin '
                   'parameters have not been tested and may not be appropriate'
                   ' for this variable.')
            warnings.warn(msg.format(cube1.name()))

        # check cubes have exactly two spatial dimension coordinates and a
        # scalar time coordinate
        check_input_coords(cube1, require_time=True)
        check_input_coords(cube2, require_time=True)

        # check cube dimensions match
        if (cube1.coord(axis="x") != cube2.coord(axis="x") or
                cube1.coord(axis="y") != cube2.coord(axis="y")):
            raise InvalidCubeError("Input cubes on unmatched grids")

        # check grids are equal area
        check_if_grid_is_equal_area(cube1)
        check_if_grid_is_equal_area(cube2)

        # check time difference is positive
        time1 = (cube1.coord("time").units).num2date(
            cube1.coord("time").points[0])
        time2 = (cube2.coord("time").units).num2date(
            cube2.coord("time").points[0])
        cube_time_diff = time2 - time1
        if cube_time_diff.total_seconds() <= 0:
            msg = "Expected positive time difference cube2 - cube1: got {} s"
            raise InvalidCubeError(msg.format(cube_time_diff.total_seconds()))

        # if time difference is greater 15 minutes, increase data smoothing
        # radius so that larger advection displacements can be resolved
        if cube_time_diff.total_seconds() > 900:
            data_smoothing_radius_km = self.data_smoothing_radius_km * (
                cube_time_diff.total_seconds()/900.)
        else:
            data_smoothing_radius_km = self.data_smoothing_radius_km

        # calculate smoothing radius in grid square units
        new_coord = cube1.coord(axis='x').copy()
        new_coord.convert_units('km')
        grid_length_km = np.float32(np.diff((new_coord).points)[0])
        data_smoothing_radius = \
            int(data_smoothing_radius_km / grid_length_km)

        # Fail verbosely if data smoothing radius is too small and will
        # trigger silent failures downstream
        if data_smoothing_radius < 3:
            msg = ("Input data smoothing radius {} too small (minimum 3 "
                   "grid squares)")
            raise ValueError(msg.format(data_smoothing_radius))

        # Fail if self.boxsize is less than data smoothing radius
        self.boxsize = boxsize
        if self.boxsize < data_smoothing_radius:
            msg = ("Box size {} too small (should not be less than data "
                   "smoothing radius {})")
            raise ValueError(
                msg.format(self.boxsize, data_smoothing_radius))

        # extract 2-dimensional data arrays
        data1 = next(cube1.slices([cube1.coord(axis='y'),
                                   cube1.coord(axis='x')])).data
        data2 = next(cube2.slices([cube2.coord(axis='y'),
                                   cube2.coord(axis='x')])).data

        # if input arrays have no non-zero values, set velocities to zero here
        # and raise a warning
        if (np.allclose(data1, np.zeros(data1.shape)) or
                np.allclose(data2, np.zeros(data2.shape))):
            msg = ("No non-zero data in input fields: setting optical flow "
                   "velocities to zero")
            warnings.warn(msg)
            ucomp = np.zeros(data1.shape, dtype=np.float32)
            vcomp = np.zeros(data2.shape, dtype=np.float32)
        else:
            # calculate dimensionless displacement between the two input fields
            ucomp, vcomp = self.process_dimensionless(data1, data2, 1, 0,
                                                      data_smoothing_radius)
            # convert displacements to velocities in metres per second
            for vel in [ucomp, vcomp]:
                vel *= np.float32(1000.*grid_length_km)
                vel /= cube_time_diff.total_seconds()

        # create velocity output cubes based on metadata from later input cube
        x_coord = cube2.coord(axis="x")
        y_coord = cube2.coord(axis="y")
        t_coord = cube2.coord("time")

        ucube = iris.cube.Cube(
            ucomp, long_name="precipitation_advection_x_velocity",
            units="m s-1", dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])
        ucube.add_aux_coord(t_coord)
        ucube = amend_metadata(ucube, **self.metadata_dict)

        vcube = iris.cube.Cube(
            vcomp, long_name="precipitation_advection_y_velocity",
            units="m s-1", dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])
        vcube.add_aux_coord(t_coord)
        vcube = amend_metadata(vcube, **self.metadata_dict)
        return ucube, vcube
