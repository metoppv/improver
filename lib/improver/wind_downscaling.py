# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
"""Module containing wind downscaling plugins."""


import copy
import itertools

from cf_units import Unit
import iris
import numpy as np

from improver.constants import RMDI


# Scale parameter to determine reference height
ABSOLUTE_CORRECTION_TOL = 0.04

# Scaling parameter to determine reference height
HREF_SCALE = 2.0

# Von Karman's constant
VONKARMAN = 0.4

# Default roughness length for sea points
Z0M_SEA = 0.0001


class FrictionVelocity(object):

    """"Class to calculate the friction velocity.

    This holds the function to calculate the friction velocity u_star,
    given a reference height h_ref, the velocity at the reference
    height u_href and the surface roughness z_0.

    """

    def __init__(self, u_href, h_ref, z_0, mask):
        """Initialise the class.

        Parameters:
        -----------
        u_href: 2D np.array (float)
            wind speed at h_ref
        h_ref:  2D np.array (float)
            reference height
        z_0:    2D np.array (float)
            vegetative roughness length
        mask:   2D np.array (logical)
            where True, calculate u*

        comments:
            * z_0 and h_ref need to have identical units.
            * the calculated friction velocity will have the units of the
                supplied velocity u_href.

        """
        self.u_href = u_href
        self.h_ref = h_ref
        self.z_0 = z_0
        self.mask = mask

    def calc_ustar(self):
        """Function to calculate the friction velocity.

        Returns:
        --------
        ustar:  2D array (float)
            friction velocity

        """
        ustar = np.ones(self.u_href.shape) * RMDI
        ustar[self.mask] = VONKARMAN * (self.u_href[self.mask]/np.log
                                        (self.h_ref[self.mask] /
                                         self.z_0[self.mask]))
        return ustar


class RoughnessCorrectionUtilities(object):

    """Class to calculate the height and roughness wind corrections.

    This holds functions to calculate the roughness and height
    corrections given the ancil files:
    * standard deviation of hight in grid cell as sigma (model grid on pp grid)
    * Silhouette roughness as a_over_s (model grid on pp grid)
    * vegetative roughness length z_0 (model grid on pp grid)
    * post-processing grid orography pporo
    * model grid orography interpolated on post-processing grid modoro
    * height level 3D/ 1D grid
    and
    * windspeed 3D field on height level 3D grid (from above).

    """

    def __init__(self, a_over_s, sigma, z_0, pporo, modoro, ppres, modres):
        """Set up roughness and height correction.

        This sets up the parameters used for roughness and height
        correction given the ancillary file inputs:

        Parameters:
        ----------
        a_over_s: 2D array (float)
            Silhouette roughness field, dimensionless ancillary data,
            calculated according to Robinson (2008)
        sigma: 2D array (float)
            Standard deviation field of height in the grid cell, units
            of length
        z_0: 2D array (float)
            Vegetative roughness height field, units of length
        pporo: 2D array (float)
            Post processing grid orography field
        modoro: 2D array (float)
            Model orography field interpolated to post processing grid
        ppres: scalar (float)
            Grid cell length of post processing grid
        modres: scalar (float)
            Grid cell length of model grid

        """
        self.a_over_s = a_over_s
        self.z_0 = z_0
        if z_0 is None:
            self.l_no_winddownscale = True
        else:
            self.l_no_winddownscale = False
        self.pporo = pporo
        self.modoro = modoro
        self.h_over_2 = self.sigma2hover2(sigma)  # half peak to trough height
        self.hcmask, self.rcmask = self.setmask()  # HC mask, RC mask
        if not self.l_no_winddownscale:
            self.z_0[z_0 <= 0] = Z0M_SEA
        self.dx_min = ppres/2.  # scales smaller than this not resolved in pp
        # the original code had hardcoded 500
        self.dx_max = 3.*modres  # scales larger than this resolved in model
        # the original code had hardcoded 4000
        self.wavenum = self.calc_wav()  # k = 2*pi / L
        self.h_ref = self.calc_h_ref()
        self.refinemask()  # HC mask needs to be updated for missing orography
        self.h_at0 = self.delta_height()  # pp orography - model orography

    def refinemask(self):
        """Remask over RMDI and NaN orography.

        The mask for HC needs to be False where either of the
        orographies (model or pp) has an invalid number. This cannot be
        done before because the mask is used to calculate the
        wavenumber which can and should be calculated for all points
        where h_over_2 and a_over_s is a valid number.

        """
        self.hcmask[self.pporo == RMDI] = False
        self.hcmask[self.modoro == RMDI] = False
        self.hcmask[np.isnan(self.pporo)] = False
        self.hcmask[np.isnan(self.modoro)] = False

    def setmask(self):
        """Create a ~land-sea mask.

        Create a mask that is basically a land-sea mask:
        Both, the standard deviation and the silouette roughness, are 0
        over the sea. A standard deviation of 0 results in a RMDI for
        h_over_2.

        Returns:
        -------
        hcmask: 2D array (logical)
            True for land-points, false for Sea (HC)
        rcmask: 2D array (logical)
            additionally False for invalid z_0 (RC)

        """
        hcmask = np.full(self.h_over_2.shape, True, dtype=bool)
        hcmask[self.h_over_2 <= 0] = False
        hcmask[self.a_over_s <= 0] = False
        hcmask[np.isnan(self.h_over_2)] = False
        hcmask[np.isnan(self.a_over_s)] = False
        rcmask = np.copy(hcmask)
        if not self.l_no_winddownscale:
            rcmask[self.z_0 <= 0] = False
            rcmask[np.isnan(self.z_0)] = False
        return hcmask, rcmask

    @staticmethod
    def sigma2hover2(sigma):
        """Calculate the half peak-to-trough height.

        The ancillary data used to estimate the peak to trough height
        contains the standard deviation of height in a cell. For
        sine-waves, this relates to the amplitude of the wave as:

        Amplitude = sigma * sqrt(2)

        The amplitude would correspond to half the peak-to-trough
        height (h_o_2).

        Parameters:
        -----------
            sigma: 2D array
                standard deviation of height in grid cell.

        Returns:
        --------
            h_o_2: 2D array
                of half peak-to-trough height.

        Comments:
            Points that had sigma = 0 (i.e. sea points) are set to
            RMDI.

        """
        h_o_2 = np.ones(sigma.shape) * RMDI
        h_o_2[sigma > 0] = sigma[sigma > 0] * np.sqrt(2.0)
        return h_o_2

    def calc_wav(self):
        """Calculate wavenumber k of typical orographic lengthscale.

        Function to calculate wavenumber k of typical orographic
        lengthscale L:
            k = 2*pi / L (1)

        L is approximated from half the peak-to-trough height h_over_2
        and the silhoutte roughness a_over_s (average of up-slopes per
        unit length over several cross-sections through a grid cell)
        as:
            L = 2*h_over_2 / a_over_s (2)

        a_over_s is dimensionless since it is the sum of up-slopes
        measured in the same unit lengths as it is calculated over.

        h_over_2 is calculated from the standard deviation of height in
        a grid cell, sigma, as:
            h_over_2 = sqrt(2) * sigma

        which is based on the assumptions of sine waves, see
        sigma2hover2.

        From eq. (1) and (2) it follows that:
            k = 2*pi / (2*h_over_2 / a_over_s)
              = a_over_s * pi / h_over_2

        Returns:
        --------
        wavn: 2D np.array
            wavenumber in units of inverse units of supplied h_over_2.

        """
        wavn = np.ones(self.a_over_s.shape) * RMDI
        wavn[self.hcmask] = (self.a_over_s[self.hcmask] /
                             self.h_over_2[self.hcmask]*np.pi)
        wavn[wavn > np.pi/self.dx_min] = np.pi/self.dx_min
        wavn[self.h_over_2 == 0] = RMDI
        wavn[abs(wavn) < np.pi/self.dx_max] = np.pi/self.dx_max
        return wavn

    def calc_h_ref(self):
        """Calculate the reference height for roughness correction.

        The reference height below which the flow is in equilibrium
        with the vegetative roughness is proportional to 1/wavenum
        (Howard & Clark, 2007).

        Vosper (2009) and Clark (2009) argue that at the reference
        height, the perturbation should have decayed to a fraction
        epsilon (ABSOLUTE_CORRECTION_TOL). The factor alpha
        implements eq. 1.3 in Clark (2009): UK Climatology - Wind
        Screening Tool. See also Vosper (2009) for a motivation.

        alpha is the log of scale parameter to determine reference
        height which is currently set to 0.04 (this corresponds to
        epsilon in both Vosper and Clark)

        Returns:
        --------
        h_ref: 2D np.array (float)
            reference height for roughness correction

        """
        alpha = -np.log(ABSOLUTE_CORRECTION_TOL)
        aparam = np.ones(self.wavenum.shape) * RMDI
        h_ref = np.ones(self.wavenum.shape) * RMDI
        aparam[self.hcmask] = alpha + np.log(self.wavenum[self.hcmask] *
                                             self.h_over_2[self.hcmask])
        aparam[aparam > 1.0] = 1.0
        aparam[aparam < 0.0] = 0.0
        h_ref[self.hcmask] = aparam[self.hcmask] / self.wavenum[self.hcmask]
        h_ref[h_ref < 1.0] = 1.0
        h_ref = np.minimum(h_ref, HREF_SCALE*self.h_over_2)
        h_ref[h_ref < 1.0] = 1.0
        h_ref[~self.hcmask] = 0.0
        return h_ref

    def roughness_correction_sub(self, hgrid, uold, mask):
        """Function to perform the roughness correction.

        Parameters:
        ----------
        hgrid: 3D or 1D np.array (float)
            height above orography
        uold: 3D np.array (float)
            original velocities at hgrid.

        Returns:
        --------
        unew: 3D np.array (float)
            Corrected wind speed on hgrid. Above href, this is
            equal to uold.

        Comments:
            Replace the windspeed profile below the reference height with one
            that increases logarithmic with height, bound by the original
            velocity uhref at the reference height h_ref and by a 0 velocity at
            the vegetative roughness height z_0

        """
        uhref = self.calc_u_at_h(uold, hgrid, self.h_ref, mask)
        if hgrid.ndim == 1:
            hgrid = hgrid.reshape((1, 1, )+(hgrid.shape[0],))
        ustar = FrictionVelocity(uhref, self.h_ref, self.z_0,
                                 mask).calc_ustar()
        unew = np.copy(uold)
        mhref = self.h_ref
        mhref[~mask] = RMDI
        cond = hgrid < (self.h_ref).reshape(self.h_ref.shape+(1,))
        unew[cond] = (
            ustar.reshape(ustar.shape+(1,))*np.ones(unew.shape)
            )[cond] * (
                np.log(hgrid/(np.reshape(self.z_0, self.z_0.shape + (1,)) *
                              np.ones(unew.shape)))[cond])/VONKARMAN
        return unew

    def calc_u_at_h(self, u_in, h_in, hhere, mask, dolog=False):
        """Function to interpolate u_in on h_in at hhere.

        Parameters:
        ----------
        u_in: 3D array (float)
            velocity on h_in layer, last dim is height
        h_in: 3D or 1D array (float)
            height layer array
        hhere: 2D array (float)
            height grid to interpolate at
        (dolog: scalar (logial)
            if True, log interpolation, default False)

        Returns:
        -------
        uath: 2D array (float)
            velocity interpolated at h

        """
        u_in = np.ma.masked_less(u_in, 0.0)
        h_in = np.ma.masked_less(h_in, 0.0)
        # h_in.mask = u_in.mask
        # If I allow 1D height grids, I think I cannot do the hop over.

        # Ignores the height at the position where u_in is RMDI,"hops over"
        hhere = np.ma.masked_less(hhere, 0.0)
        upidx = np.argmax(h_in > hhere.reshape(hhere.shape+(1,)), axis=2)
        # loidx = np.maximum(upidx-1, 0) #if RMDI, need below
        loidx = np.argmin(np.ma.masked_less(hhere.reshape(hhere.shape+(1,)) -
                                            h_in, 0.0), axis=2)

        if h_in.ndim == 3:
            hup = h_in.take(upidx.flatten()+np.arange(0, upidx.size *
                                                      h_in.shape[2],
                                                      h_in.shape[2]))
            hlow = h_in.take(loidx.flatten()+np.arange(0, loidx.size *
                                                       h_in.shape[2],
                                                       h_in.shape[2]))
        elif h_in.ndim == 1:
            hup = h_in[upidx].flatten()
            hlow = h_in[loidx].flatten()
        uup = u_in.take(upidx.flatten()+np.arange(0, upidx.size*u_in.shape[2],
                                                  u_in.shape[2]))
        ulow = u_in.take(loidx.flatten()+np.arange(0, loidx.size*u_in.shape[2],
                                                   u_in.shape[2]))
        mask = mask.flatten()
        uath = np.full(mask.shape, RMDI, dtype=float)
        if dolog:
            uath[mask] = self.loginterpol(hup[mask], hlow[mask],
                                          hhere.flatten()[mask],
                                          uup[mask], ulow[mask])
        else:
            uath[mask] = self.interp1d(hup[mask], hlow[mask],
                                       hhere.flatten()[mask],
                                       uup[mask], ulow[mask])
        uath = np.reshape(uath, hhere.shape)
        return uath

    @staticmethod
    def interp1d(xup, xlow, at_x, yup, ylow):
        """Simple 1D linear interpolation for 2D grid inputs level.

        Parameters:
        ----------
        xlow: 2D np.array (float)
            lower x-bins
        xup: 2D np.array (float)
            upper x-bins
        at_x: 2D np.array (float)
            x values to interpolate y at
        yup: 2D np.array(float)
            y(xup)
        ylow: 2D np.array (float)
            y(xlow)

        Returns:
        -------
        interp: 2D np.array (float)
            y(at_x) assuming a lin function between xlow and xup

        """
        interp = np.full(xup.shape, RMDI, dtype=float)
        diffs = (xup - xlow)
        interp[diffs != 0] = (
            ylow[diffs != 0]+((at_x[diffs != 0]-xlow[diffs != 0]) /
                              diffs[diffs != 0]*(yup[diffs != 0] -
                                                 ylow[diffs != 0])))
        interp[diffs == 0] = at_x[diffs == 0]/xup[diffs == 0]*(yup[diffs == 0])
        return interp

    @staticmethod
    def loginterpol(x_u, x_l, at_x, y_u, y_l):
        """Simple 1D log interpolation y(x), except if lowest layer is
        ground level.

        Parameters:
        ----------
        x_l: 2D np.array (float)
            lower x-bins
        x_u: 2D np.array (float)
            upper x-bins
        at_x: 2D np.array (float)
            x values to interpolate y at
        y_u: 2D np.array (float)
            y(x_u)
        y_l: 2D np.array (float)
            y(x_l)

        Returns:
        -------
        loginterp: 2D np.array (float)
            y(at_x) assuming a log function between x_l and x_u

        """
        ain = np.full(x_u.shape, RMDI, dtype=float)
        loginterp = np.full(x_u.shape, RMDI, dtype=float)
        mfrac = x_u/x_l
        mtest = (x_u/x_l != 1) & (at_x != x_u)
        ain[mtest] = (y_u[mtest] - y_l[mtest])/np.log(mfrac[mtest])
        loginterp[mtest] = ain[mtest]*np.log(at_x[mtest]/x_u[mtest])+y_u[mtest]
        mtest = (x_u/x_l == 1)  # below lowest layer, make lin interp
        loginterp[mtest] = at_x[mtest]/x_u[mtest] * (y_u[mtest])
        mtest = (at_x == x_u)  # just use y_u
        loginterp[mtest] = y_u[mtest]
        return loginterp

    def height_corr_sub(self, u_a, heightg, mask, onemfrac):
        """Function to calculate the additive height correction.

        Parameters:
        ----------
        u_a: 2D array (float)
            outer velocity, e.g. velocity at h_ref_orig
        heightg: 1D or 3D array
            heights above orography
        onemfrac: currently, scalar = 1.
            In principle, it is a function of position and height, e.g.
            a 3D array (float)

        Returns:
        -------
        hc_add: 3D array (float)
            additive height correction to wind speed

        Comments:
            The height correction is a disturbance of the flow that
            decays exponentially with height. The larger the vertical
            offset h_at0 (the higher the unresolved hill), the larger
            is the disturbance.

            The more smooth the distrubance (the larger the horizontal
            scale of the disturbance), the smaller the height
            correction (hence, a larger wavenumber results in a larger
            disturbance).
            hc_add = exp(-height*wavenumber)*u(href)*h_at_0*k

        """
        (xdim, ydim) = u_a.shape
        if heightg.ndim == 1:
            zdim = heightg.shape[0]
            heightg = heightg.reshape((1, 1, zdim))
        elif heightg.ndim == 3:
            zdim = heightg.shape[2]
        ml2 = self.h_at0*self.wavenum
        expon = np.ones([xdim, ydim, zdim])
        mult = (self.wavenum).reshape(self.wavenum.shape+(1,))*heightg
        expon[mult > 0.0001] = np.exp(-mult[mult > 0.0001])
        hc_add = (
            expon*u_a.reshape(u_a.shape+(1,)) *
            ml2.reshape(ml2.shape+(1,))*onemfrac)
        hc_add[~mask, :] = 0
        return hc_add

    def delta_height(self):
        """Function to calculate pp-grid diff from model grid.

        Calculate the difference between pp-grid height and model
        grid height.

        Returns:
        -------
        deltZ: 2D np.array (float)
            height difference, ppgrid-model

        """
        delt_z = np.ones(self.pporo.shape) * RMDI
        delt_z[self.hcmask] = self.pporo[self.hcmask]-self.modoro[self.hcmask]
        return delt_z

    def do_rc_hc_all(self, hgrid, uorig):
        """Function to call HC and RC (height and roughness corrections).

        Parameters:
        ----------
        hgrid: 1D or 3D array (float)
            height grid of wind input
        uorig: 3D array (float)
            wind speed on these levels

        Returns:
        -------
            result: 3D array
                sum of  unew: 3D array (float) RC corrected windspeed
                on levels HC: 3D array (float) HC additional part

        """
        if hgrid.ndim == 3:
            condition1 = ((hgrid == RMDI).any(axis=2))
            self.hcmask[condition1] = False
            self.rcmask[condition1] = False
        mask_rc = np.copy(self.rcmask)
        mask_rc[(uorig == RMDI).any(axis=2)] = False
        mask_hc = np.copy(self.hcmask)
        mask_hc[(uorig == RMDI).any(axis=2)] = False
        if not self.l_no_winddownscale:
            unew = self.roughness_correction_sub(hgrid, uorig, mask_rc)
        else:
            unew = uorig
        uhref_orig = self.calc_u_at_h(uorig, hgrid, 1.0/self.wavenum, mask_hc)
        mask_hc[uhref_orig <= 0] = False
        onemfrac = 1.0
        # onemfrac = 1.0 - BfuncFrac(nx,ny,nz,heightvec,z_0,waveno, Ustar, UI)
        hc_add = self.height_corr_sub(uhref_orig, hgrid, mask_hc, onemfrac)
        result = unew + hc_add
        result[result < 0.] = 0  # HC can be negative if pporo<modeloro
        return result


class RoughnessCorrection(object):

    """Plugin to orographically-correct 3d wind speeds."""

    zcoordnames = ["height", "model_level_number"]
    xcoordnames = ["projection_x_coordinate", "grid_longitude", "longitude"]
    ycoordnames = ["projection_y_coordinate", "grid_latitude", "latitude"]
    tcoordnames = ["time", "forecast_time"]

    def __init__(self, a_over_s_cube, sigma_cube, pporo_cube,
                 modoro_cube, modres, z0_cube=None,
                 height_levels_cube=None):
        """Initialise the RoughnessCorrection instance.

        Parameters
        ----------
        a_over_s_cube: 2D cube
            model silhouette roughness on pp grid. dimensionless
        sigma_cube: 2D cube
            standard deviation of model orography height on pp grid.
            In m.
        pporo_cube: 2D cube
            pp orography. In m
        modoro_cube: 2D cube
            model orography interpolated on pp grid. In m
        modres: float
            original avearge model resolution in m
        (height_levels_cube: 3D or 1D cube)
            height of input velocity field. Can be position dependent
        (z0_cube: 2D cube)
            vegetative roughness length in m. If not given, do not do
            any RC

        """
        if hasattr(a_over_s_cube, "operation"):
            a_over_s_cube = a_over_s_cube.operation()
        if hasattr(sigma_cube, "operation"):
            sigma_cube = sigma_cube.operation()
        if hasattr(pporo_cube, "operation"):
            pporo_cube = pporo_cube.operation()
        if hasattr(modoro_cube, "operation"):
            modoro_cube = modoro_cube.operation()
        if hasattr(z0_cube, "operation"):
            z0_cube = z0_cube.operation()
        if hasattr(height_levels_cube, "operation"):
            height_levels_cube = height_levels_cube.operation()

        x_name, y_name, _, _ = self.find_coord_names(pporo_cube)
        # Some checking that all the grids match
        if not (self.check_ancils(a_over_s_cube, sigma_cube, z0_cube,
                                  pporo_cube, modoro_cube)):
            raise ValueError("ancillary grids are not consistent")
        # I want this ordering. Careful: iris.cube.Cube.slices is unreliable.
        self.a_over_s = next(a_over_s_cube.slices([y_name, x_name]))
        self.sigma = next(sigma_cube.slices([y_name, x_name]))
        try:
            self.z_0 = next(z0_cube.slices([y_name, x_name]))
        except AttributeError:
            self.z_0 = z0_cube
        except Exception as exc:
            emsg = "'{0}' while z0 setting. Arguments '{1}'."
            raise ValueError(emsg.format(exc.message, exc.args))
        self.pp_oro = next(pporo_cube.slices([y_name, x_name]))
        self.model_oro = next(modoro_cube.slices([y_name, x_name]))
        self.ppres = self.calc_av_ppgrid_res(pporo_cube)
        self.modres = modres
        self.height_levels = height_levels_cube
        if z0_cube is None:
            self.l_no_winddownscale = True
        else:
            self.l_no_winddownscale = False
        self.x_name = None
        self.y_name = None
        self.z_name = None
        self.t_name = None

    def find_coord_names(self, cube):
        """Extract x, y, z, and time coordinate names.

        Parameters:
        ----------
        cube: cube
            some iris cube to find coordinate names from

        Returns:
        -------
        xname: str
            name of the axis name in x-direction
        yname: str
            name of the axis name in y-direction
        zname: str
            name of the axis name in z-direction
        tname: str
            name of the axis name in t-direction

        """
        clist = set([cube.coords()[iii].name() for iii in
                     range(len(cube.coords()))])
        try:
            xname = list(clist.intersection(self.xcoordnames))[0]
        except IndexError:
            xname = None
        except Exception as exc:
            print("'{0}' while xname setting. Args: {1}.".format(exc.message,
                                                                 exc.args))
        try:
            yname = list(clist.intersection(self.ycoordnames))[0]
        except IndexError:
            yname = None
        except Exception as exc:
            print("'{0}' while yname setting. Args: {1}.".format(exc.message,
                                                                 exc.args))
        try:
            zname = list(clist.intersection(self.zcoordnames))[0]
        except IndexError:
            zname = None
        except Exception as exc:
            print("'{0}' while zname setting. Args: {1}.".format(exc.message,
                                                                 exc.args))
        try:
            tname = list(clist.intersection(self.tcoordnames))[0]
        except IndexError:
            tname = None
        except Exception as exc:
            print("'{0}' while tname setting. Args: {1}.".format(exc.message,
                                                                 exc.args))
        return xname, yname, zname, tname

    def calc_av_ppgrid_res(self, a_cube):
        """Calculate average grid resolution from a cube.

        Parameters:
        ----------
        a_cube: cube
            cube to calculate average resolution of

        Returns:
        -------
        float
            average grid resolution.

        """
        x_name, y_name, _, _ = self.find_coord_names(a_cube)
        [exp_xname, exp_yname] = ["projection_x_coordinate",
                                  "projection_y_coordinate"]
        exp_unit = Unit("m")
        if (x_name is not exp_xname) or (y_name is not exp_yname):
            raise ValueError("cannot currently calculate resolution")
        try:
            xres = (np.diff(a_cube.coord(x_name).bounds)).mean()
            yres = (np.diff(a_cube.coord(y_name).bounds)).mean()
        except Exception:
            xres = (np.diff(a_cube.coord(x_name).points)).mean()
            yres = (np.diff(a_cube.coord(y_name).points)).mean()
        if (
                (a_cube.coord(x_name).units != exp_unit) or
                (a_cube.coord(y_name).units != exp_unit)):
            raise ValueError("cube axis have units different from m.")
        return (xres + yres) / 2.0

    @staticmethod
    def check_ancils(a_over_s_cube, sigma_cube, z0_cube, pp_oro_cube,
                     model_oro_cube):
        """Check ancils grid and units.

        Check if ancil cubes are on the same grid and if they have the
        expected units. The testing for "same grid" might be replaced
        if there is a general utils function made for it or so.

        Parameters:
        ----------
        a_over_s_cube: iris cube
            holding the silhoutte roughness field
        sigma_cube: iris cube
            holding the standard deviation of height in a grid cell
        z0_cube: iris cube or None
            holding the vegetative roughness field
        pp_oro_cube: iris cube
            holding the post processing grid orography
        model_oro_cube: iris cube
            holding the model orography on post processing grid

        Returns:
        -------
        logical
            describing whether or not the tests passed

        """
        alist = [a_over_s_cube, sigma_cube, pp_oro_cube, model_oro_cube]
        unwanted_coord_list = [
            "time", "height", "model_level_number", "forecast_time",
            "forecast_reference_time", "forecast_period"]
        for field, exp_unit in zip(alist, [None, Unit("m"), Unit("m"),
                                           Unit("m")]):
            for unwanted_coord in unwanted_coord_list:
                try:
                    field.remove_coord(unwanted_coord)
                except Exception:
                    pass
            if field.units != exp_unit:
                msg = ('{} ancil field has unexpected unit:'
                       ' {} (expected) vs. {} (actual)')
                raise ValueError(
                    msg.format(field.name(), exp_unit, field.units))
        if z0_cube is not None:
            alist.append(z0_cube)
            for unwanted_coord in unwanted_coord_list:
                try:
                    z0_cube.remove_coord(unwanted_coord)
                except Exception:
                    pass
            if z0_cube.units != Unit('m'):
                msg = ("z0 ancil has unexpected unit: should be {} "
                       "is {}")
                raise ValueError(msg.format(Unit('m'), z0_cube.units))
        mlist = list(itertools.permutations(alist, 2))
        oklist = []
        for entr in mlist:
            oklist.append(entr[0].coords() == entr[1].coords())
            # HybridHeightToPhenomOnPressure._cube_compatibility_check(entr[0],
            # entr[1])
        return np.array(oklist).all()  # replace by a return value of True

    def find_coord_order(self, mcube):
        """Extract coordinate ordering within a cube.

        Figure out the order of the xyzt dimensions in a cube.
        iris.cube.Cube.slices seems not always to return the order one
        specifies, hence I need to find out the order and work with
        transpose instead.

        Parameters:
        ----------
        mcube: iris cube
            cube to check the order of coordinate axis

        Returns:
        -------
        xpos: integer
            position of x axis.
        ypos: integer
            position of y axis.
        zpos: integer
            position of z axis.
        tpos: integer
            position of t axis.

        """
        xpos = ypos = zpos = tpos = np.nan
        for iii, mcc in enumerate(mcube.coords()):
            try:
                if mcc == mcube.coord(self.x_name):
                    xpos = iii
                elif mcc == mcube.coord(self.y_name):
                    ypos = iii
                elif mcc == mcube.coord(self.z_name):
                    zpos = iii
                elif mcc == mcube.coord(self.t_name):
                    tpos = iii
            except Exception:
                pass
        return xpos, ypos, zpos, tpos

    def find_heightgrid(self, wind):
        """Setup the height grid.

        Setup the height grid either from the 1D or 3D height grid
        that was supplied to the plugin or from the z-axis information
        from the wind grid.

        Parameters:
        ----------
        wind: 3D or 4D iris cube
            representing the wind data.

        Returns:
        -------
        hld: 1D or 3D np.array
            representing the height grid.

        """
        if self.height_levels is None:
            hld = wind.coord(self.z_name).points
        else:
            hld = iris.util.squeeze(self.height_levels)
            if np.isnan(hld.data).any() or (hld.data == RMDI).any():
                raise ValueError("height grid contains invalid points")
            if hld.ndim == 3:
                try:
                    xap, yap, zap, _ = self.find_coord_order(hld)
                    hld.transpose([yap, xap, zap])
                except Exception:
                    raise ValueError("height grid different from wind grid")
            else:
                try:
                    hld = next(hld.slices([self.z_name]))
                except Exception:
                    raise ValueError("height z coordinate differs from wind z")
            hld = hld.data
        return hld

    def check_wind_ancil(self, xwp, ywp):
        """Check wind vs ancillary file grids.

        Check if wind and ancillary files are on the same grid and if
        they have the same ordering.

        Parameters:
        xwp: integer
            representing the position of the x-axis in the wind cube
        ywp: integer
            representing the position of the y-axis of the wind cube

        """
        xap, yap, _, _ = self.find_coord_order(self.pp_oro)
        if xwp - ywp != xap-yap:
            if np.isnan(xap) or np.isnan(yap):
                raise ValueError("ancillary grid different from wind grid")
            else:
                raise ValueError("xy-orientation: ancillary differ from wind")

    def process(self, cube):
        """Adjust the 4d wind field - cube - (x, y, z including times).

        Parameters
        ----------
        cube - iris.cube.Cube
            The wind cube to be operated upon. Should be wind speed on
            height_levels for all desired forecast times.

        Returns
        -------
        cube
            The 4d wind field with roughness and height correction
            applied in the same order as the input cube.

        """
        wind = cube
        if not isinstance(wind, iris.cube.Cube):
            msg = "wind input is not a cube, but {}"
            raise ValueError(msg.format(type(wind)))
        (self.x_name, self.y_name, self.z_name,
         self.t_name) = self.find_coord_names(wind)
        xwp, ywp, zwp, twp = self.find_coord_order(wind)
        try:
            wind.transpose([ywp, xwp, zwp, twp])  # problems with slices
        except ValueError:
            wind.transpose([ywp, xwp, zwp])
        rchc_list = iris.cube.CubeList()
        if self.l_no_winddownscale:
            z0_data = None
        else:
            z0_data = self.z_0.data
        roughness_correction = RoughnessCorrectionUtilities(
            self.a_over_s.data, self.sigma.data, z0_data, self.pp_oro.data,
            self.model_oro.data, self.ppres, self.modres)
        self.check_wind_ancil(xwp, ywp)
        hld = self.find_heightgrid(wind)
        for time_slice in wind.slices([self.y_name, self.x_name, self.z_name],
                                      ordered=False):
            if np.isnan(time_slice.data).any() or (time_slice.data < 0.).any():
                msg = ('{} has invalid wind data')
                raise ValueError(msg.format(time_slice.coord(self.t_name)))
            rc_hc = copy.deepcopy(time_slice)
            rc_hc.data = roughness_correction.do_rc_hc_all(hld,
                                                           time_slice.data)
            rchc_list.append(rc_hc)
        cube = rchc_list.merge()[0]
        # reorder wind and cube as original
        try:
            wind.transpose(np.argsort([ywp, xwp, zwp, twp]))
            cube.transpose(np.argsort([twp, ywp, xwp, zwp]))
        except ValueError:
            wind.transpose(np.argsort([ywp, xwp, zwp]))
            cube.transpose(np.argsort([ywp, xwp, zwp]))
        return cube
