# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
from typing import List, Optional, Tuple

import iris
import numpy as np
from iris.cube import Cube, CubeList
from iris.exceptions import (
    CoordinateCollapseError,
    CoordinateNotFoundError,
    InvalidCubeError,
)
from numpy import ndarray
from scipy import ndimage, signal

from improver import BasePlugin
from improver.nowcasting.pysteps_advection import PystepsExtrapolate
from improver.nowcasting.utilities import ApplyOrographicEnhancement
from improver.utilities.cube_checker import check_for_x_and_y_axes
from improver.utilities.cube_manipulation import collapsed
from improver.utilities.spatial import (
    calculate_grid_spacing,
    check_if_grid_is_equal_area,
)


def generate_optical_flow_components(
    cube_list: CubeList, ofc_box_size: int, smart_smoothing_iterations: int
) -> Tuple[Cube, Cube]:
    """
    Calculate the mean optical flow components between the cubes in cube_list

    Args:
        cube_list:
            Cubelist from which to calculate optical flow velocities
        ofc_box_size:
            Size of square 'box' (in grid spaces) within which to solve
            the optical flow equations
        smart_smoothing_iterations:
            Number of iterations to perform in enforcing smoothness constraint
            for optical flow velocities

    Returns:
        - Cube of x-advection velocities u_mean
        - Cube of y-advection velocities v_mean
    """
    cube_list.sort(key=lambda x: x.coord("time").points[0])
    time_coord = cube_list[-1].coord("time")

    ofc_plugin = OpticalFlow(iterations=smart_smoothing_iterations)
    u_cubes = iris.cube.CubeList([])
    v_cubes = iris.cube.CubeList([])
    for older_cube, newer_cube in zip(cube_list[:-1], cube_list[1:]):
        ucube, vcube = ofc_plugin(older_cube, newer_cube, boxsize=ofc_box_size)
        u_cubes.append(ucube)
        v_cubes.append(vcube)

    # average optical flow velocity components
    def _calculate_time_average(wind_cubes, time_coord):
        """Average input cubelist over time"""
        cube = wind_cubes.merge_cube()
        try:
            mean = collapsed(cube, "time", iris.analysis.MEAN)
        except CoordinateCollapseError:
            # collapse will fail if there is only one time point
            return cube
        mean.coord("time").points = time_coord.points
        mean.coord("time").units = time_coord.units
        return mean

    u_mean = _calculate_time_average(u_cubes, time_coord)
    v_mean = _calculate_time_average(v_cubes, time_coord)

    return u_mean, v_mean


def generate_advection_velocities_from_winds(
    cubes: CubeList, background_flow: CubeList, orographic_enhancement: Cube
) -> CubeList:
    """Generate advection velocities as perturbations from a non-zero background
    flow

    Args:
        cubes:
            Two rainfall observations separated by a time difference
        background_flow:
            u- and v-components of a non-zero background flow field
        orographic_enhancement:
            Field containing orographic enhancement data valid for both
            input cube times

    Returns:
        u- and v- advection velocities
    """
    cubes.sort(key=lambda x: x.coord("time").points[0])

    lead_time_seconds = (
        cubes[1].coord("time").cell(0).point - cubes[0].coord("time").cell(0).point
    ).total_seconds()
    lead_time_minutes = int(lead_time_seconds / 60)

    # advect earlier cube forward to match time of later cube, using steering flow
    advected_cube = PystepsExtrapolate(lead_time_minutes, lead_time_minutes)(
        cubes[0], *background_flow, orographic_enhancement
    )[-1]

    # calculate velocity perturbations required to match forecast to observation
    cube_list = ApplyOrographicEnhancement("subtract")(
        [advected_cube, cubes[1]], orographic_enhancement
    )
    perturbations = OpticalFlow(data_smoothing_radius_km=8.0, iterations=20)(
        *cube_list, boxsize=18
    )

    # sum perturbations and original flow field to get advection velocities
    total_advection = _perturb_background_flow(background_flow, perturbations)
    return total_advection


def _perturb_background_flow(
    background: List[Cube], adjustment: List[Cube]
) -> CubeList:
    """Add a background flow to a flow adjustment field.  The returned cubelist
    has the units of the adjustment field.

    Args:
        background
        adjustment

    Returns:
        The adjusted CubeList.
    """
    for flow, adj in zip(background, adjustment):
        flow.convert_units(adj.units)
        perturbed_field = np.where(
            np.isfinite(adj.data), adj.data + flow.data, flow.data
        )
        adj.data = perturbed_field.astype(adj.dtype)
    return iris.cube.CubeList(adjustment)


def check_input_coords(cube: Cube, require_time: bool = False) -> None:
    """
    Checks an input cube has precisely two non-scalar dimension coordinates
    (spatial x/y), or raises an error.  If "require_time" is set to True,
    raises an error if no scalar time coordinate is present.

    Args:
        cube:
            Cube to be checked
        require_time:
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
        raise InvalidCubeError(
            "Cube has {:d} (more than 2) non-scalar "
            "coordinates".format(non_scalar_coords)
        )

    if require_time:
        try:
            _ = cube.coord("time")
        except CoordinateNotFoundError:
            raise InvalidCubeError("Input cube has no time coordinate")


class OpticalFlow(BasePlugin):
    """
    Class to calculate advection velocities along two orthogonal spatial axes
    from time-separated fields using an optical flow algorithm

    References:
        Bowler, N., Pierce, C. and Seed, A. 2004: Development of a
        precipitation nowcasting algorithm based upon optical flow
        techniques. Journal of Hydrology, 288, 74-91.

        Friedrich, Martina M. 2017: STEPS investigation summary. Internal
        Met Office Document.
    """

    def __init__(
        self,
        data_smoothing_method: str = "box",
        data_smoothing_radius_km: float = 14.0,
        iterations: int = 100,
    ) -> None:
        """
        Initialise the class with smoothing parameters for estimating gridded
        u- and v- velocities via optical flow.

        Args:
            data_smoothing_method:
                Smoothing method to be used on input fields before estimating
                partial derivatives.  Can be square 'box' (as used in STEPS) or
                circular 'kernel' (used in post-calculation smoothing).
            data_smoothing_radius:
                The radius, in km, of the kernel used to smooth the input data fields
                before calculating optical flow.  14 km is suitable for precipitation
                rate data separated by a 15 minute time step.  If the time step is
                greater than 15 minutes, this radius is increased by the "process"
                method.
            iterations:
                Number of iterations to perform in post-calculation smoothing.
                The minimum value for good convergence is 20 (Bowler et al. 2004).

        Raises:
            ValueError:
                If iterations < 20
        """
        if iterations < 20:
            raise ValueError(
                "Got {} iterations; minimum requirement 20 "
                "iterations".format(iterations)
            )

        self.data_smoothing_radius_km = data_smoothing_radius_km
        self.data_smoothing_method = data_smoothing_method

        # Set parameters for velocity calculation and "smart smoothing"
        self.iterations = iterations
        self.point_weight = 0.1

        # Initialise input data fields and shape
        self.data1 = None
        self.data2 = None
        self.shape = None

    def __repr__(self) -> str:
        """Represent the plugin instance as a string."""
        result = (
            "<OpticalFlow: data_smoothing_radius_km: {}, "
            "data_smoothing_method: {}, iterations: {}, "
            "point_weight: {}>"
        )
        return result.format(
            self.data_smoothing_radius_km,
            self.data_smoothing_method,
            self.iterations,
            self.point_weight,
        )

    @staticmethod
    def _check_input_cubes(cube1: Cube, cube2: Cube) -> None:
        """Check that input cubes have appropriate and matching dimensions"""
        # check the nature of the input cubes, and raise a warning if they are
        # not both precipitation
        if cube1.name() != cube2.name():
            msg = "Input cubes contain different data types {} and {}"
            raise ValueError(msg.format(cube1.name(), cube2.name()))

        data_name = cube1.name().lower()
        if "rain" not in data_name and "precipitation" not in data_name:
            msg = (
                "Input data are of non-precipitation type {}.  Plugin "
                "parameters have not been tested and may not be appropriate"
                " for this variable."
            )
            warnings.warn(msg.format(cube1.name()))

        # check cubes have exactly two spatial dimension coordinates and a
        # scalar time coordinate
        check_input_coords(cube1, require_time=True)
        check_input_coords(cube2, require_time=True)

        # check cube dimensions match
        if cube1.coord(axis="x") != cube2.coord(axis="x") or cube1.coord(
            axis="y"
        ) != cube2.coord(axis="y"):
            raise InvalidCubeError("Input cubes on unmatched grids")

        # check grids are equal area
        check_if_grid_is_equal_area(cube1)
        check_if_grid_is_equal_area(cube2)

    @staticmethod
    def _get_advection_time(cube1: Cube, cube2: Cube) -> None:
        """Get time over which the advection has occurred, in seconds, using the
        difference in time or forecast reference time between input cubes"""
        time_diff_seconds = (
            cube2.coord("time").cell(0).point - cube1.coord("time").cell(0).point
        ).total_seconds()
        time_diff_seconds = int(time_diff_seconds)

        if time_diff_seconds == 0:
            # second cube should be an observation; first cube should have a
            # non-zero forecast period which describes the advection time
            if (
                cube2.coords("forecast_period")
                and cube2.coord("forecast_period").points[0] != 0
            ):
                raise InvalidCubeError(
                    "Second input cube must be a current observation"
                )

            # get the time difference from the first cube's forecast period
            fp_coord = cube1.coord("forecast_period").copy()
            fp_coord.convert_units("seconds")
            (time_diff_seconds,) = fp_coord.points

        if time_diff_seconds <= 0:
            error_msg = "Expected positive time difference cube2 - cube1: got {} s"
            raise InvalidCubeError(error_msg.format(time_diff_seconds))

        return time_diff_seconds

    def _get_smoothing_radius(
        self, time_diff_seconds: float, grid_length_km: float
    ) -> float:
        """Calculate appropriate data smoothing radius in grid squares.
        If time difference is greater 15 minutes, increase data smoothing
        radius in km so that larger advection displacements can be resolved.
        """
        if time_diff_seconds > 900:
            data_smoothing_radius_km = self.data_smoothing_radius_km * (
                time_diff_seconds / 900.0
            )
        else:
            data_smoothing_radius_km = self.data_smoothing_radius_km

        # calculate smoothing radius in integer grid squares
        data_smoothing_radius = int(data_smoothing_radius_km / grid_length_km)

        # fail verbosely if data smoothing radius is too small and will
        # trigger silent failures downstream
        if data_smoothing_radius < 3:
            msg = "Input data smoothing radius {} too small (minimum 3 grid squares)"
            raise ValueError(msg.format(data_smoothing_radius))

        return data_smoothing_radius

    @staticmethod
    def interp_to_midpoint(data: ndarray, axis: Optional[int] = None) -> ndarray:
        """
        Interpolates to the x-y mid-point resulting in a new grid with
        dimensions reduced in length by one.  If axis is not None, the
        interpolation is performed only over the one spatial axis
        specified.  If the input array has an axis of length 1, the
        attempt to interpolate returns an empty array: [].

        Args:
            data:
                2D gridded data (dimensions M x N)
            axis:
                Optional (0 or 1): average over 2 adjacent points along the
                specified axis, rather than all 4 corners

        Returns:
            2D gridded interpolated average (dimensions M-1 x N-1 if
            axis=None; M-1 x N if axis=0; M x N-1 if axis=1)
        """
        if axis is None:
            midpoints = 0.25 * (
                data[1:, :-1] + data[:-1, 1:] + data[1:, 1:] + data[:-1, :-1]
            )
        elif axis == 0:
            midpoints = 0.5 * (data[:-1, :] + data[1:, :])
        elif axis == 1:
            midpoints = 0.5 * (data[:, :-1] + data[:, 1:])
        return midpoints

    def _partial_derivative_spatial(self, axis: int = 0) -> ndarray:
        """
        Calculate the average over the two class data fields of one spatial
        derivative, averaged over the other spatial dimension.  Pad with zeros
        in both dimensions, then smooth to the original grid shape.

        Args:
            axis:
                Axis over which to calculate the spatial derivative (0 or 1)

        Returns:
            Smoothed spatial derivative
        """
        outdata = []
        for data in [self.data1, self.data2]:
            diffs = self.interp_to_midpoint(np.diff(data, axis=axis), axis=1 - axis)
            outdata.append(diffs)
        smoothed_diffs = np.zeros(
            [self.shape[0] + 1, self.shape[1] + 1], dtype=np.float32
        )
        smoothed_diffs[1:-1, 1:-1] = 0.5 * (outdata[0] + outdata[1])
        return self.interp_to_midpoint(smoothed_diffs)

    def _partial_derivative_temporal(self) -> ndarray:
        """
        Calculate the partial derivative of two fields over time.  Take the
        difference between time-separated fields data1 and data2, average
        over the two spatial dimensions, regrid to a zero-padded output
        array, and smooth to the original grid shape.

        Returns:
            Smoothed temporal derivative
        """
        tdiff = self.data2 - self.data1
        smoothed_diffs = np.zeros(
            [self.shape[0] + 1, self.shape[1] + 1], dtype=np.float32
        )
        smoothed_diffs[1:-1, 1:-1] = self.interp_to_midpoint(tdiff)
        return self.interp_to_midpoint(smoothed_diffs)

    def _make_subboxes(self, field: ndarray) -> Tuple[List[ndarray], ndarray]:
        """
        Generate a list of non-overlapping "boxes" of size self.boxsize**2
        from the input field, along with weights based on data values at times
        1 and 2.  The final boxes in the list will be smaller if the size of
        the data field is not an exact multiple of "boxsize".

        Note that the weights calculated below are valid for precipitation
        rates in mm/hr. This is a result of the constant 0.8 that is used,
        noting that in the source paper a value of 0.75 is used; see equation
        8. in Bowler et al. 2004.

        Args:
            field:
                Input field (partial derivative)

        Returns:
            - List of numpy.ndarrays of size boxsize*boxsize containing
              slices of data from input field.
            - 1D numpy array containing weights values associated with
              each listed box.
        """
        boxes = []
        weights = []
        weighting_factor = 0.5 / self.boxsize ** 2.0
        for i in range(0, field.shape[0], self.boxsize):
            for j in range(0, field.shape[1], self.boxsize):
                boxes.append(field[i : i + self.boxsize, j : j + self.boxsize])
                weight = weighting_factor * (
                    (self.data1[i : i + self.boxsize, j : j + self.boxsize]).sum()
                    + (self.data2[i : i + self.boxsize, j : j + self.boxsize]).sum()
                )
                weight = 1.0 - np.exp(-1.0 * weight / 0.8)
                weights.append(weight)
        weights = np.array(weights, dtype=np.float32)
        weights[weights < 0.01] = 0
        return boxes, weights

    def _box_to_grid(self, box_data: ndarray) -> ndarray:
        """
        Regrids calculated displacements from "box grid" (on which OFC
        equations are solved) to input data grid.

        Args:
            box_data:
                Displacement of subbox on box grid

        Returns:
            Displacement on original data grid
        """
        grid_data = np.repeat(
            np.repeat(box_data, self.boxsize, axis=0), self.boxsize, axis=1
        )
        grid_data = grid_data[: self.shape[0], : self.shape[1]].astype(np.float32)
        return grid_data

    @staticmethod
    def makekernel(radius: int) -> ndarray:
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
            radius:
                Kernel radius or half box size for smoothing

        Returns:
            Kernel to use for generating a smoothed field.

        """
        kernel_1d = 1 - np.abs(np.linspace(-1, 1, radius * 2 + 1))
        kernel_2d = kernel_1d.reshape(radius * 2 + 1, 1) * kernel_1d.reshape(
            1, radius * 2 + 1
        )
        kernel_2d /= kernel_2d.sum()
        return kernel_2d

    def smooth(self, field: ndarray, radius: int, method: str = "box") -> ndarray:
        """
        Smoothing method using a square ('box') or circular kernel.  Kernel
        smoothing with a radius of 1 has no effect.

        Smoothing with the "box" argument is equivalent to the method
        in equation 7 in Bowler et al. 2004.

        Args:
            field:
                Input field to be smoothed
            radius:
                Kernel radius or half box size for smoothing
            method:
                Method to use: 'box' (as in STEPS) or 'kernel'

        Returns:
            Smoothed data on input-shaped grid
        """
        if method == "kernel":
            kernel = self.makekernel(radius)
            smoothed_field = signal.convolve2d(
                field, kernel, mode="same", boundary="symm"
            )
        elif method == "box":
            smoothed_field = ndimage.filters.uniform_filter(
                field, size=radius * 2 + 1, mode="nearest"
            )
        # Ensure the dtype does not change.
        smoothed_field = smoothed_field.astype(field.dtype)
        return smoothed_field

    def _smart_smooth(
        self, vel_point: ndarray, vel_iter: ndarray, weights: ndarray
    ) -> ndarray:
        """
        Performs a single iteration of "smart smoothing" over a point and its
        neighbours as implemented in STEPS.  This smoothing (through the
        "weights" argument) ignores advection displacements which are
        identically zero, as these are assumed to occur only where there is no
        data structure from which to calculate displacements.

        Args:
            vel_point:
                Original unsmoothed data
            vel_iter:
                Latest iteration of smart-smoothed displacement
            weights:
                Weight of each grid point for averaging

        Returns:
            Next iteration of smart-smoothed displacement
        """
        # define kernel for neighbour weighting
        neighbour_kernel = (
            np.array([[0.5, 1, 0.5], [1.0, 0, 1.0], [0.5, 1, 0.5]]) / 6.0
        ).astype(np.float32)

        # smooth input data and weights fields
        vel_neighbour = ndimage.convolve(weights * vel_iter, neighbour_kernel)
        neighbour_weights = ndimage.convolve(weights, neighbour_kernel)

        # initialise output data from latest iteration
        vel = ndimage.convolve(vel_iter, neighbour_kernel)

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

        vel[pmask] = (
            vel_neighbour[pmask] * nweight + vel_point[pmask] * pweight[pmask]
        ) / norm[pmask]
        return vel

    def _smooth_advection_fields(self, box_data: ndarray, weights: ndarray) -> ndarray:
        """
        Performs iterative "smart smoothing" of advection displacement fields,
        accounting for zeros and reducting their weight in the final output.
        Then regrid from "box grid" (on which OFC equations are solved) to
        input data grid, and perform one final pass simple kernel smoothing.
        This is equivalent to applying the smoothness constraint defined in
        Bowler et al. 2004, equations 9-11.

        Args:
            box_data:
                Displacements on box grid (modified by this function)
            weights:
                Weights for smart smoothing

        Returns:
            Smoothed displacement vectors on input data grid
        """
        v_orig = np.copy(box_data)

        # iteratively smooth umat and vmat
        for _ in range(self.iterations):
            box_data = self._smart_smooth(v_orig, box_data, weights)

        # reshape smoothed box velocity arrays to match input data grid
        grid_data = self._box_to_grid(box_data)

        # smooth regridded velocities to remove box edge discontinuities
        # this will fail if self.boxsize < 3
        kernelsize = int(self.boxsize / 3)
        grid_data = self.smooth(grid_data, kernelsize, method="kernel")
        return grid_data

    @staticmethod
    def solve_for_uv(deriv_xy: ndarray, deriv_t: ndarray) -> ndarray:
        """
        Solve the system of linear simultaneous equations for u and v using
        matrix inversion (equation 19 in STEPS investigation summary document
        by Martina M. Friedrich 2017 (available internally at the Met Office)).
        This is frequently singular, eg in the presence of too many zeroes.
        In these cases, the function returns displacements of 0.

        Args:
            deriv_xy:
                2-column matrix containing partial field derivatives d/dx
                (first column) and d/dy (second column)
            deriv_t:
                1-column matrix containing partial field derivatives d/dt

        Returns:
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
    def extreme_value_check(umat: ndarray, vmat: ndarray, weights: ndarray) -> None:
        """
        Checks for displacement vectors that exceed 1/3 of the dimensions
        of the input data matrix.  Replaces these extreme values and their
        smoothing weights with zeros.  Modifies ALL input arrays in place.

        Args:
            umat:
                Displacement vectors in the x direction
            vmat:
                Displacement vectors in the y direction
            weights:
                Weights for smart smoothing
        """
        flag = (np.abs(umat) + np.abs(vmat)) > vmat.shape[0] / 3.0
        umat[flag] = 0
        vmat[flag] = 0
        weights[flag] = 0

    def calculate_displacement_vectors(
        self, partial_dx: ndarray, partial_dy: ndarray, partial_dt: ndarray
    ) -> Tuple[ndarray, ndarray]:
        """
        This implements the OFC algorithm, assuming all points in a box with
        "self.boxsize" sidelength have the same displacement components.

        Args:
            partial_dx:
                2D array of partial input field derivatives d/dx
            partial_dy:
                2D array of partial input field derivatives d/dy
            partial_dt:
                2D array of partial input field derivatives d/dt

        Returns:
            - 2D array of displacements in the x-direction
            - 2D array of displacements in the y-direction
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
            deriv_xy = (np.array([deriv_x, deriv_y], dtype=np.float64)).transpose()

            # Solve equations for u and v through matrix inversion
            u, v = self.solve_for_uv(deriv_xy, deriv_t)
            velocity[0].append(u)
            velocity[1].append(v)

        # (c) Reshape displacement arrays to match array of subbox points
        newshape = [
            int((self.shape[0] - 1) / self.boxsize) + 1,
            int((self.shape[1] - 1) / self.boxsize) + 1,
        ]
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
        vel_comp: ndarray, rain_mask: Tuple[int, ...], zero_vel_threshold: float = 0.1
    ) -> None:
        """
        Raise warning if more than a fixed threshold (default 10%) of cells
        where there is rain within the domain have zero advection velocities.

        Args:
            vel_comp:
                Advection velocity that will be checked to assess the
                proportion of zeroes present in this field.
            rain_mask:
                Array indices where there is rain.
            zero_vel_threshold:
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

        if zeroes_in_rain > rain_pixels * zero_vel_threshold:
            msg = (
                "{:.1f}% of rain cells within the domain have zero "
                "advection velocities. It is expected that greater "
                "than {:.1f}% of these advection velocities will be "
                "non-zero.".format(
                    zeroes_in_rain * 100.0 / rain_pixels, (1 - zero_vel_threshold) * 100
                )
            )
            warnings.warn(msg)

    def process_dimensionless(
        self,
        data1: ndarray,
        data2: ndarray,
        xaxis: int,
        yaxis: int,
        smoothing_radius: int,
    ) -> Tuple[ndarray, ndarray]:
        """
        Calculates dimensionless advection displacements between two input
        fields.

        Args:
            data1:
                2D input data array from time 1
            data2:
                2D input data array from time 2
            xaxis:
                Index of x coordinate axis
            yaxis:
                Index of y coordinate axis
            smoothing_radius:
                Radius (in grid squares) over which to smooth the input data

        Returns:
            - Advection displacement (grid squares) in the x direction
            - Advection displacement (grid squares) in the y direction
        """
        # Smooth input data
        self.shape = data1.shape
        self.data1 = self.smooth(
            data1, smoothing_radius, method=self.data_smoothing_method
        )
        self.data2 = self.smooth(
            data2, smoothing_radius, method=self.data_smoothing_method
        )

        # Calculate partial derivatives of the smoothed input fields
        partial_dx = self._partial_derivative_spatial(axis=xaxis)
        partial_dy = self._partial_derivative_spatial(axis=yaxis)
        partial_dt = self._partial_derivative_temporal()

        # Calculate advection displacements
        ucomp, vcomp = self.calculate_displacement_vectors(
            partial_dx, partial_dy, partial_dt
        )

        # Check for zeros where there should be valid displacements
        rain_mask = np.where((data1 > 0) | (data2 > 0))
        for vel_comp in [ucomp, vcomp]:
            self._zero_advection_velocities_warning(vel_comp, rain_mask)
        return ucomp, vcomp

    def process(self, cube1: Cube, cube2: Cube, boxsize: int = 30) -> Tuple[Cube, Cube]:
        """
        Extracts data from input cubes, performs dimensionless advection
        displacement calculation, and creates new cubes with advection
        velocities in metres per second.  Each input cube should have precisely
        two non-scalar dimension coordinates (spatial x/y), and are expected to
        be in a projection such that grid spacing is the same (or very close)
        at all points within the spatial domain.  Each input cube must also
        have a scalar "time" coordinate.

        Args:
            cube1:
                2D cube that advection will be FROM / advection start point.
                This may be an earlier observation or an extrapolation forecast
                for the current time.
            cube2:
                2D cube that advection will be TO / advection end point.
                This will be the most recent observation.
            boxsize:
                The side length of the square box over which to solve the
                optical flow constraint.  This should be greater than the
                data smoothing radius.

        Returns:
            - 2D cube of advection velocities in the x-direction
            - 2D cube of advection velocities in the y-direction
        """
        # clear existing parameters
        self.data_smoothing_radius = None
        self.boxsize = None

        # check input cubes have appropriate and matching contents and dimensions
        self._check_input_cubes(cube1, cube2)

        # get time over which advection displacement has occurred
        time_diff_seconds = self._get_advection_time(cube1, cube2)

        # if time difference is greater 15 minutes, increase data smoothing
        # radius so that larger advection displacements can be resolved
        grid_length_km = calculate_grid_spacing(cube1, "km")
        data_smoothing_radius = self._get_smoothing_radius(
            time_diff_seconds, grid_length_km
        )

        # fail if self.boxsize is less than data smoothing radius
        self.boxsize = boxsize
        if self.boxsize < data_smoothing_radius:
            msg = (
                "Box size {} too small (should not be less than data "
                "smoothing radius {})"
            )
            raise ValueError(msg.format(self.boxsize, data_smoothing_radius))

        # convert units to mm/hr as these avoid the need to manipulate tiny
        # decimals
        cube1 = cube1.copy()
        cube2 = cube2.copy()

        try:
            cube1.convert_units("mm/hr")
            cube2.convert_units("mm/hr")
        except ValueError as err:
            msg = (
                "Input data are in units that cannot be converted to mm/hr "
                "which are the required units for use with optical flow."
            )
            raise ValueError(msg) from err

        # extract 2-dimensional data arrays
        data1 = next(cube1.slices([cube1.coord(axis="y"), cube1.coord(axis="x")])).data
        data2 = next(cube2.slices([cube2.coord(axis="y"), cube2.coord(axis="x")])).data

        # fill any mask with 0 values so fill_values are not spread into the
        # domain when smoothing the fields.
        if np.ma.is_masked(data1):
            data1 = data1.filled(0)
        if np.ma.is_masked(data2):
            data2 = data2.filled(0)

        # if input arrays have no non-zero values, set velocities to zero here
        # and raise a warning
        if np.allclose(data1, np.zeros(data1.shape)) or np.allclose(
            data2, np.zeros(data2.shape)
        ):
            msg = (
                "No non-zero data in input fields: setting optical flow "
                "velocities to zero"
            )
            warnings.warn(msg)
            ucomp = np.zeros(data1.shape, dtype=np.float32)
            vcomp = np.zeros(data2.shape, dtype=np.float32)
        else:
            # calculate dimensionless displacement between the two input fields
            ucomp, vcomp = self.process_dimensionless(
                data1, data2, 1, 0, data_smoothing_radius
            )
            # convert displacements to velocities in metres per second
            for vel in [ucomp, vcomp]:
                vel *= np.float32(1000.0 * grid_length_km)
                vel /= time_diff_seconds

        # create velocity output cubes based on metadata from later input cube
        ucube = iris.cube.Cube(
            ucomp,
            long_name="precipitation_advection_x_velocity",
            units="m s-1",
            dim_coords_and_dims=[
                (cube2.coord(axis="y"), 0),
                (cube2.coord(axis="x"), 1),
            ],
            aux_coords_and_dims=[(cube2.coord("time"), None)],
        )
        vcube = ucube.copy(vcomp)
        vcube.rename("precipitation_advection_y_velocity")

        return ucube, vcube
