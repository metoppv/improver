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
classes for advection nowcasting of precipitation fields.
"""
import numpy as np

import scipy.linalg
import scipy.ndimage
import scipy.ndimage.filters
import scipy.signal

from iris.exceptions import InvalidCubeError
from iris.exceptions import CoordinateNotFoundError

from improver.utilities.cube_checker import check_for_x_and_y_axes


class AdvectField(object):
    """
    Class to advect a 2D spatial field given velocities along the two vector
    dimensions
    """

    def __init__(self, vel_x, vel_y):
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
        """

        # check each input velocity cube has precisely two non-scalar
        # dimension coordinates (spatial x/y)
        self._check_input_coords(vel_x)
        self._check_input_coords(vel_y)

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

    @staticmethod
    def _check_input_coords(cube, require_time=False):
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

    def _advect_field(self, data, grid_vel_x, grid_vel_y, timestep,
                      fill_value):
        """
        Performs a dimensionless grid-based extrapolation of spatial data
        using advection velocities via a backwards method.

        Args:
            data (numpy.ndarray):
                2D numpy data array to be advected
            grid_vel_x (numpy.ndarray):
                Velocity in the x direction (in grid points per second)
            grid_vel_y (numpy.ndarray):
                Velocity in the y direction (in grid points per second)
            timestep (int):
                Advection time step in seconds
            fill_value (float):
                Default output value for spatial points where data cannot be
                extrapolated (source is out of bounds)

        Returns:
            adv_field (numpy.ndarray):
                2D float array of advected data values
        """

        # Initialise advected field with default fill_value
        adv_field = np.full(data.shape, fill_value)

        # Set up grids of data coordinates (meshgrid inverts coordinate order)
        ydim, xdim = data.shape
        (xgrid, ygrid) = np.meshgrid(np.arange(xdim),
                                     np.arange(ydim))

        # For each grid point on the output field, trace its (x,y) "source"
        # location backwards using advection velocities.  The source location
        # is generally fractional: eg with advection velocities of 0.5 grid
        # squares per second, the value at [2, 2] is represented by the value
        # that was at [1.5, 1.5] 1 second ago.
        xsrc_point_frac = -grid_vel_x * timestep + xgrid.astype(float)
        ysrc_point_frac = -grid_vel_y * timestep + ygrid.astype(float)

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
        x_weights = [1. - x_weight_upper, x_weight_upper]
        y_weights = [1. - y_weight_upper, y_weight_upper]

        # Advect data from each of the four source points onto the output grid
        for xpt, xwt in zip(x_points, x_weights):
            for ypt, ywt in zip(y_points, y_weights):
                cond = point_in_bounds(xpt, ypt, xdim, ydim) & cond_pt
                self._increment_output_array(data, adv_field, cond, xgrid,
                                             ygrid, xpt, ypt, xwt, ywt)

        return adv_field

    def process(self, cube, timestep, fill_value=0.0):
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
            fill_value (float):
                Default output value for spatial points where data cannot be
                extrapolated (source is out of bounds)

        Returns:
            advected_cube (iris.cube.Cube):
                New cube with updated time and extrapolated data
        """
        # check that the input cube has precisely two non-scalar dimension
        # coordinates (spatial x/y) and a scalar time coordinate
        self._check_input_coords(cube, require_time=True)

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
            return float(np.diff((new_coord).points)[0])

        grid_vel_x = self.vel_x.data / grid_spacing(cube.coord(axis="x"))
        grid_vel_y = self.vel_y.data / grid_spacing(cube.coord(axis="y"))

        # perform advection and create output cube
        advected_data = self._advect_field(cube.data, grid_vel_x, grid_vel_y,
                                           timestep.total_seconds(),
                                           fill_value)
        advected_cube = cube.copy(data=advected_data)

        # increment output cube time
        original_datetime, = \
            (cube.coord("time").units).num2date(cube.coord("time").points)
        new_datetime = original_datetime + timestep
        new_time = (cube.coord("time").units).date2num(new_datetime)
        advected_cube.coord("time").points = new_time

        return advected_cube


class OpticalFlow(object):
    """
    Class to calculate advection velocities along two orthogonal spatial axes
    from time-separated fields using an optical flow algorithm
    """

    def __init__(self, data1=None, data2=None, kernel=7, myboxsize=30,
                 iterations=100, smethod=2, pointweight=0.1):
        """
        On initialization, the u- and v- component of the advection velocity field
        are calculated, given the two input fields.

        input:
            data1: 2D np.array
                first time slice
            data2: 2D np.array
                second time slice
            kernel: integer
                kernel size (radius) for use to smooth the data for partial
                derivative estimaten. If box smoothing is used, half box size
            myboxsize: integer
                side length of box size in which points are evaluated to have
                the same velocity
            iterations: integer
                number of smart smoothing iterations to perform
            smethod: 1 or 2
                smoothing method to be used for smoothing for the calculation
                of the partial derivatives, 1 is box smoothing as in steps
                2 is kernel smoothing
            pointweight: float
                weight given to the velocity of the point (box)
                when doing the smart smoothing. 0.1 is the original steps value
        """
        # need to calculate a suitable kernel size given dt (what is expected
        # movement between slices)
        # and expected velocities
        if data1 is not None:
            d1n2 = self.smoothing(data1, kernel, method=smethod)
            d2n2 = self.smoothing(data2, kernel, method=smethod)
            xdif_t = self.totx(d1n2, d2n2, 0)
            ydif_t = self.totx(d1n2, d2n2, 1)
            tdif_t = self.totx(d1n2, d2n2, 2)
            self.ucomp, self.vcomp = self.makeofc(
                xdif_t, ydif_t, tdif_t, myboxsize, d1n2, d2n2, iterations,
                pweight=pointweight)
        else:
            self.ucomp = None
            self.vcomp = None

    @staticmethod
    def mdiffx(xfield):
        """
        This function implements
        x_i+1,j+1 = 0.5[(x_i+1,j - x_i,j) + (x_i+1,j+1 -x_i+1,j+1)]
        """
        d_1 = np.diff(xfield, axis=0)
        d_2 = d_1[:, 1:]
        d_1 = d_1[:, :-1]
        d_3 = np.zeros(xfield.shape)
        d_3[1:, 1:] = (d_1 + d_2)*0.5
        return d_3

    def mdifftx(self, xfield, yfield):
        """
        This function makes (time) average of the x derivative (averaged in y),
        where the two input fields are time slices. Returned is time average
        """
        d_1 = self.mdiffx(xfield)
        d_2 = self.mdiffx(yfield)
        return (d_1 + d_2)*0.5

    def mdiffty(self, xfield, yfield):
        """
        This function returns analogous to mdifftx time averaged y derivative
        (averaged in x) where the two input fields are the time slices.
        """
        x_1 = xfield.transpose()
        y_1 = yfield.transpose()
        intm = self.mdifftx(x_1, y_1)
        return intm.transpose()

    @staticmethod
    def mdifftt(xfield, yfield):
        """ this corresponds to Eq. 31 - 34 in steps document """
        d_1 = yfield - xfield
        d_2 = d_1[1:, :-1]
        d_3 = d_1[:-1, 1:]
        d_4 = d_1[1:, 1:]
        d_1 = d_1[:-1, :-1]
        d_5 = np.zeros(xfield.shape)
        d_5[1:, 1:] = (d_1+d_2+d_3+d_4)*0.25
        return d_5

    def totx(self, xfield, yfield, xty):
        """ this corresponds to Eq. 35 - 37 in steps document """
        if xty == 0:  # this is for x derivative
            m_1 = self.mdifftx(xfield, yfield)
        elif xty == 1:  # this is for y derivative
            m_1 = self.mdiffty(xfield, yfield)
        elif xty == 2:  # this is for t derivative
            m_1 = self.mdifftt(xfield, yfield)
        m_0 = np.zeros([xfield.shape[0]+1, xfield.shape[1]+1])
        m_0[:-1, :-1] = m_1
        m_2 = m_0[1:, :-1]
        m_3 = m_0[:-1, 1:]
        m_4 = m_0[1:, 1:]
        m_1 = m_0[:-1, :-1]
        return (m_1+m_2+m_3+m_4)*0.25

    @staticmethod
    def makeIandDmat(d_u, d_v, d_t):
        """ this corresponds to Eq. 17 in steps document """
        d_u = d_u.flatten()
        d_v = d_v.flatten()
        d_t = d_t.flatten()
        II_mat = (np.array([d_u, d_v])).transpose()
        return II_mat, d_t

    @staticmethod
    def calcU(II_mat, dd):
        """ this corresponds to Eq. 19 in steps document """
        # there is a problem here if I have many fewer pixels with intensity
        # in here than pixels with
        dd = dd.reshape([dd.size, 1])
        IItrans = II_mat.transpose()
        m_1 = IItrans.dot(II_mat)
        m1inv = np.linalg.inv(m_1)
        m_2 = IItrans.dot(dd)
        myreturn = -m1inv.dot(m_2)
        return myreturn[0, 0], myreturn[1, 0]

    @staticmethod
    def makesubboxes(xfield, yfield, tfield, d_1, d_2, boxsize):
        """
        This function makes a list of subboxes with dimension boxsize*boxsize
        of inputfields x,y and t.
        It returns the lists of these subboxes, where the faster varying index
        is the second index in the original fields.
        The original input fields are used to calculate the weight for each
        such subbox.
        """
        xdif_tb = []
        ydif_tb = []
        tdif_tb = []
        weight_tb = []
        for startx in range(0, xfield.shape[0], boxsize):
            for starty in range(0, xfield.shape[1], boxsize):
                xdif_tb.append(
                    xfield[startx:startx+boxsize, starty:starty+boxsize])
                ydif_tb.append(
                    yfield[startx:startx+boxsize, starty:starty+boxsize])
                tdif_tb.append(
                    tfield[startx:startx+boxsize, starty:starty+boxsize])
                newweight = ((
                    (d_1[startx:startx+boxsize, starty:starty+boxsize]).sum() +
                    (d_2[startx:startx+boxsize, starty:starty+boxsize]).sum())
                    / boxsize/boxsize/2.0)
                # this is the weight used in steps
                newweight = 1.0-np.exp(-newweight/0.8)
                weight_tb.append(newweight)
        weight_tb = np.array(weight_tb)
        weight_tb[weight_tb < 0.01] = 0
        return xdif_tb, ydif_tb, tdif_tb, weight_tb

    @staticmethod
    def rebinvel(xfield, yfield, boxsize, myshape, origshape):
        """
        in: xfield, yfield  : u and v box velocity fields
            boxsize   : side length in pixels of each velocity box
            myshape   : shape of the field in velocity box units
            origshape : shape of the field in pixels
        out:  umat_t  : u velocity field
            vmat_t  : v velocity field
        Function to reshape the velocity vectors containing the box velocities,
        to velocity pixel maps
        """
        umat_t = np.zeros(origshape)
        vmat_t = np.zeros(origshape)
        for ii in range(myshape[0]):
            for jj in range(myshape[1]):  # size limited to origshape
                umat_t[ii*boxsize:(ii+1)*boxsize,
                       jj*boxsize:(jj+1)*boxsize] = xfield[ii, jj]
                vmat_t[ii*boxsize:(ii+1)*boxsize,
                       jj*boxsize:(jj+1)*boxsize] = yfield[ii, jj]
        return umat_t, vmat_t

    @staticmethod
    def makekernel(msize):
        """ make a kernel to smooth the input fields """
        temp = 1 - np.abs(np.linspace(-1, 1, msize*2+1))
        kernel = temp.reshape(msize*2+1, 1) * temp.reshape(1, msize*2+1)
        kernel /= kernel.sum()   # kernel should sum to 1!
        return kernel

    def smoothing(self, d_x, sidel, method=2):
        '''
        smoothing used to apply on the field to estimate partial derivatives
        '''
        if method == 1:
            kernel = self.makekernel(sidel)
            dxn = scipy.signal.convolve2d(d_x, kernel, mode='same',
                                          boundary="symm")
        elif method == 2:  # type of smoothing used in steps
            dxn = scipy.ndimage.filters.uniform_filter(d_x, size=sidel*2+1,
                                                       mode='nearest')
        return dxn

    @staticmethod
    def setupweightgrid(field):
        '''
        setup a weightfield:
        0 ---------------0
        |2.5 4 ---- 4 2.5|
        |4   6 -----6   4|
        ||   |      |   ||
        ||   |      |   ||
        |4   6 -----6   4|
        |2.5 4------4 2.5|
        0----------------0
        '''
        xdim, ydim = field.shape+np.array([2, 2])
        zz = np.ones([xdim, ydim])
        zz[0, :] = 0
        zz[:, 0] = 0
        zz[-1, :] = 0
        zz[:, -1] = 0  # halo points
        zz = zz * 6.  # field points
        zz[1, 1:-1] = 4.
        zz[1:-1, 1] = 4.
        zz[-2, 1:-1] = 4.
        zz[1:-1, -2] = 4  # edge points
        zz[1, 1] = 2.5
        zz[-2, -2] = 2.5
        zz[1, -2] = 2.5
        zz[-2, 1] = 2.5  # corner ps
        return zz, xdim, ydim

    @staticmethod
    def smallkernel():
        '''
        kernel representing the weighting implemented in steps. Used if
        the smartsmooth is used in the convolution mode (i.e. asinsteps=False)
        '''
        mkernel = np.array([[0.5, 1, 0.5], [1, 0, 1], [0.5, 1, 0.5]])/6.
        return mkernel

    def find_neighbour_image(self, field, ww=None):
        '''
        this can replace the cumbersome for loops in the smart smooth.
        However, it does handle edges not exactly the right way.
        From http://stackoverflow.com/questions/22669252/
        how-exactly-does-the-re%E2%80%8C%E2%80%8Bflect-mode-for-
        scipy%E2%80%8C%E2%80%8Bs-ndimage-filters-wo%E2%80%8C%E2%80%8Brk:

        mode       |   Ext   |         Input          |   Ext
        -----------+---------+------------------------+---------
        'mirror'   | 4  3  2 | 1  2  3  4  5  6  7  8 | 7  6  5
        'reflect'  | 3  2  1 | 1  2  3  4  5  6  7  8 | 8  7  6
        'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8
        'constant' | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0
        'wrap'     | 6  7  8 | 1  2  3  4  5  6  7  8 | 1  2  3
        Hence, the mode that corresponds most closely to the for loop solution
        is "mirror" (or reflect) reflect is the default, I leave it with that).
        '''
        mkernel = self.smallkernel()
        if ww is None:
            wfield = field
        else:
            wfield = field*ww
        ofield = scipy.ndimage.convolve(wfield, mkernel)
        return ofield

    def smartsmooth(self, xo, yo, x, y, w, pweight):
        """
        implements the smart smoothing (i.e. 0 that are 0 because there is no
        structure to calculate the advection from, are not included in the
        smoothing of the velocity field) as in steps
        """
        xnew = np.zeros(x.shape)
        ynew = np.zeros(y.shape)
        wnew = np.zeros(w.shape)
        xsm = np.zeros(x.shape)
        ysm = np.zeros(y.shape)
        nweight = 1.0 - pweight

        xnew = self.find_neighbour_image(x, w)
        ynew = self.find_neighbour_image(y, w)
        wnew = self.find_neighbour_image(w)
        xsm = self.find_neighbour_image(x)
        ysm = self.find_neighbour_image(y)

        # xsm will stay as the normal (unweighted average) for all points
        # where the neigbours have no weight and the point has no weight
        # below, everywhere where the neigbours have weight, I replace the
        # value with the weighted average of the neigbours
        xsm[abs(wnew) > 0] = xnew[abs(wnew) > 0]/wnew[abs(wnew) > 0]
        ysm[abs(wnew) > 0] = ynew[abs(wnew) > 0]/wnew[abs(wnew) > 0]
        # if the point itself has weight, I use a weighted sum of the neigbour
        # points and the point (however, it is not the point in iteration, it
        # is the original value!)
        xsm[abs(w) > 0] = (xnew[abs(w) > 0]*nweight + pweight * xo[abs(w) > 0]
                           * w[abs(w) > 0]) / (nweight*wnew[abs(w) > 0] +
                                               pweight*w[abs(w) > 0])
        ysm[abs(w) > 0] = (ynew[abs(w) > 0]*nweight + pweight * yo[abs(w) > 0]
                           * w[abs(w) > 0]) / (nweight*wnew[abs(w) > 0] +
                                               pweight*w[abs(w) > 0])
        return xsm, ysm

    def makeofc(self, xdif_t, ydif_t, tdif_t, subboxsize, d1, d2,
                iterations=50, pweight=0.1):
        """
        this implements the OFC algorithm, assuming all points in a box with
        subboxsize sidelength have the same velocity components.

        NOTE CS: the following code executes regardless of "asinsteps" value...
        (was default False - I removed this option)
        """
        # This part is for doing it the way it is done in STEPS----------------
        # (a) make subboxes
        # pweight = 0.1 # this is the weight given to the point as opposed to
        # its neigbours
        xdif_tb, ydif_tb, tdif_tb, weight_tb = self.makesubboxes(
            xdif_t, ydif_t, tdif_t, d1, d2, subboxsize)
        uvec = []
        vvec = []
        # (b) solve ofc on subboxes
        for xdif, ydif, tdif in zip(
                xdif_tb, ydif_tb, tdif_tb):
            II, dd = self.makeIandDmat(xdif, ydif, tdif)
            try:
                u = self.calcU(II, dd)
            except:
                u = [0, 0]
            uvec.append(u[0])
            vvec.append(u[1])
        uvec = np.array(uvec)
        vvec = np.array(vvec)
        # print uvec
        weight_tb = np.array(weight_tb)
        # (c) reorder block velocities in 2D
        myshape = [int((xdif_t.shape[0]-1)/subboxsize) + 1,
                   int((xdif_t.shape[1]-1)/subboxsize) + 1]
        umat = uvec.reshape(myshape)
        vmat = vvec.reshape(myshape)
        # pause()
        weights = weight_tb.reshape(myshape)
        # 03.11.2016 build in a check to detect insane velocities and put them
        # to an average of neigbouring
        flag = (np.abs(umat) + np.abs(vmat)) > vmat.shape[0]/3.
        umat[flag] = 0
        vmat[flag] = 0
        weights[flag] = 0
        # (d) do some smart smoothing
        conv_vec = []
        umatn = np.copy(umat)
        vmatn = np.copy(vmat)
        for _ in range(iterations):
            umatold = umatn
            umatn, vmatn = self.smartsmooth(umat, vmat, umatn, vmatn, weights,
                                            pweight)
            conv_vec.append((abs(umatold-umatn)).sum())
        # (e) rebin block velocities to 2D field velocities
        umat_f, vmat_f = self.rebinvel(umatn, vmatn, subboxsize, myshape,
                                       xdif_t.shape)
        smn = int(subboxsize/3)
        umat_f = self.smoothing(umat_f, smn, method=1)
        vmat_f = self.smoothing(vmat_f, smn, method=1)
        # ---------------------------------------------------------------------
        return umat_f, vmat_f
