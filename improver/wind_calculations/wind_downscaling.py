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
"""Module containing wind downscaling plugins."""

import copy
import itertools

import iris
import numpy as np
from cf_units import Unit
from iris.exceptions import CoordinateNotFoundError

from improver import BasePlugin, PostProcessingPlugin
from improver.constants import RMDI

# Scale parameter to determine reference height
ABSOLUTE_CORRECTION_TOL = 0.04

# Scaling parameter to determine reference height
HREF_SCALE = 2.0

# Von Karman's constant
VONKARMAN = 0.4

# Default roughness length for sea points
Z0M_SEA = 0.0001


class FrictionVelocity(BasePlugin):
    """"Class to calculate the friction velocity.

    This holds the function to calculate the friction velocity u_star,
    given a reference height h_ref, the velocity at the reference
    height u_href and the surface roughness z_0.

    """

    def __init__(self, u_href, h_ref, z_0, mask):
        """Initialise the class.

        Args:
            u_href (numpy.ndarray):
                A 2D array of float32 for the wind speed at h_ref
            h_ref (numpy.ndarray):
                A 2D array of float32 for the reference heights
            z_0 (numpy.ndarray):
                A 2D array of float32 for the vegetative roughness lengths
            mask (numpy.ndarray):
                A 2D array of booleans where True indicates calculate u*

        Notes:
            * z_0 and h_ref need to have identical units.
            * the calculated friction velocity will have the units of the
                supplied velocity u_href.

        """
        self.u_href = u_href
        self.h_ref = h_ref
        self.z_0 = z_0
        self.mask = mask

        # Check that input cubes are the same size
        array_sizes = [np.size(u_href), np.size(h_ref), np.size(z_0),
                       np.size(mask)]

        if not all(x == array_sizes[0] for x in array_sizes):
            raise ValueError('Different size input arrays u_href, h_ref, z_0, '
                             'mask')

    def process(self):
        """Function to calculate the friction velocity.

        ustar = K * (u_href / ln(h_ref / z_0))

        where ustar is the friction velocity, K is Von Karman's
        constant, u_ref is the wind speed at the reference height,
        h_ref is the reference height and z_0 is the vegetative
        roughness length.

        Returns:
            numpy.ndarray:
                A 2D array of float32 friction velocities

        """
        ustar = np.full(self.u_href.shape, RMDI, dtype=np.float32)
        numerator = self.u_href[self.mask]
        with np.errstate(invalid='ignore'):
            denominator = np.log(self.h_ref[self.mask] / self.z_0[self.mask])
        ustar[self.mask] = VONKARMAN * (numerator / denominator)
        return ustar


class RoughnessCorrectionUtilities:
    """Class to calculate the height and roughness wind corrections.

    This holds functions to calculate the roughness and height
    corrections given the ancil files:

     * standard deviation of height in grid cell as sigma (model grid on
       pp grid)
     * Silhouette roughness as a_over_s (model grid on pp grid)
     * vegetative roughness length z_0 (model grid on pp grid)
     * post-processing grid orography pporo
     * model grid orography interpolated on post-processing grid modoro
     * height level 3D/ 1D grid
     * windspeed 3D field on height level 3D grid (from above).

    """

    def __init__(self, a_over_s, sigma, z_0, pporo, modoro, ppres, modres):
        """Set up roughness and height correction.

        This sets up the parameters used for roughness and height
        correction given the ancillary file inputs:

        Args:
            a_over_s (numpy.ndarray):
                2D array float32 - Silhouette roughness field, dimensionless
                ancillary data, calculated according to Robinson, D. (2008)
                - Ancillary file creation for the UM, Unified Model
                Documentation Paper 73.
            sigma (numpy.ndarray):
                2D array float32 - Standard deviation field of height in the
                grid cell, units of length
            z_0 (numpy.ndarray):
                2D array float32 - Vegetative roughness height field,
                units of length
            pporo (numpy.ndarray):
                2D array float32 - Post processing grid orography field
            modoro (numpy.ndarray):
                2D array float32 - Model orography field interpolated to post
                processing grid
            ppres (float):
                Float - Grid cell length of post processing grid
            modres (float):
                Float - Grid cell length of model grid

        """
        self.a_over_s = a_over_s
        self.z_0 = z_0
        self.pporo = pporo
        self.modoro = modoro
        self.h_over_2 = self.sigma2hover2(sigma)  # half peak to trough height
        self.hcmask, self.rcmask = self._setmask()  # HC mask, RC mask
        if self.z_0 is not None:
            self.z_0[z_0 <= 0] = Z0M_SEA
        self.dx_min = ppres / 2.  # scales smaller than this not resolved in pp
        # the original code had hardcoded 500
        self.dx_max = 3. * modres  # scales larger than this resolved in model
        # the original code had hardcoded 4000
        self.wavenum = self._calc_wav()  # k = 2*pi / L
        self.h_ref = self._calc_h_ref()
        self._refinemask()  # HC mask needs to be updated for missing orography
        self.h_at0 = self._delta_height()  # pp orography - model orography

    def _refinemask(self):
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

    def _setmask(self):
        """Create a ~land-sea mask.

        Create a mask that is basically a land-sea mask:
        Both, the standard deviation and the silouette roughness, are 0
        over the sea. A standard deviation of 0 results in a RMDI for
        h_over_2.

        Returns:
            (tuple): tuple containing:
                **hcmask** (numpy.ndarray):
                    2D array of booleans- True for land-points,
                    false for Sea (HC)
                **rcmask** (numpy.ndarray):
                    2D array of booleans- additionally False for
                    invalid z_0 (RC)

        """
        hcmask = np.full(self.h_over_2.shape, True, dtype=bool)
        hcmask[self.h_over_2 <= 0] = False
        hcmask[self.a_over_s <= 0] = False
        hcmask[np.isnan(self.h_over_2)] = False
        hcmask[np.isnan(self.a_over_s)] = False
        rcmask = np.copy(hcmask)
        if self.z_0 is not None:
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

        Args:
            sigma (numpy.ndarray):
                2D array of float32 - standard deviation of height in
                grid cell.

        Returns:
            numpy.ndarray:
                2D array of float32 - half peak-to-trough height.

        Comments:
            Points that had sigma = 0 (i.e. sea points) are set to
            RMDI.

        """
        h_o_2 = np.full(sigma.shape, RMDI, dtype=np.float32)
        h_o_2[sigma > 0] = sigma[sigma > 0] * np.sqrt(2.0)
        return h_o_2

    def _calc_wav(self):
        """Calculate wavenumber k of typical orographic lengthscale.

        Function to calculate wavenumber k of typical orographic
        lengthscale L:

        .. math::
          :label:

            k = 2 * \\pi / L

        L is approximated from half the peak-to-trough height h_over_2
        and the silhoutte roughness a_over_s (average of up-slopes per
        unit length over several cross-sections through a grid cell)
        as:

        .. math::
          :label:

            L = 2 * \\rm{h\\_over\\_2} / \\rm{a\\_over\\_s}

        a_over_s is dimensionless since it is the sum of up-slopes
        measured in the same unit lengths as it is calculated over.

        h_over_2 is calculated from the standard deviation of height in
        a grid cell, sigma, as:

        .. math::
          :label:

            \\rm{h\\_over\\_2} = \\sqrt{2} * \\rm{sigma}

        which is based on the assumptions of sine waves, see
        sigma2hover2.

        From eq. (1) and (2) it follows that:

        .. math::
          :label:

            k = 2*\\pi / (2 * \\rm{h\\_over\\_2} / \\rm{a\\_over\\_s)}
              = \\rm{a\\_over\\_s} * \\pi / \\rm{h\\_over\\_2}

        Returns:
            numpy.ndarray:
                2D array float32 - wavenumber in units of inverse units of
                supplied h_over_2.

        """
        wavn = np.full(self.a_over_s.shape, RMDI, dtype=np.float32)
        wavn[self.hcmask] = (
            (self.a_over_s[self.hcmask] * np.pi) / self.h_over_2[self.hcmask]
        )
        wavn[wavn > np.pi / self.dx_min] = np.pi / self.dx_min
        wavn[self.h_over_2 == 0] = RMDI
        wavn[abs(wavn) < np.pi / self.dx_max] = np.pi / self.dx_max
        return wavn

    def _calc_h_ref(self):
        """Calculate the reference height for roughness correction.

        The reference height below which the flow is in equilibrium
        with the vegetative roughness is proportional to 1/wavenum
        (Howard & Clark, 2007).

        Vosper (2009) and Clark (2009) argue that at the reference
        height, the perturbation should have decayed to a fraction
        epsilon (ABSOLUTE_CORRECTION_TOL). The factor alpha
        implements eq. 1.3 in Clark (2009): UK Climatology - Wind
        Screening Tool. See also Vosper (2009) for a motivation.
        For a freely available external reference, see the Virtual Met
        Mast Version 1 Methodology and Verification paper under
        www.thecrownestate.co.uk.

        alpha is the log of scale parameter to determine reference
        height which is currently set to 0.04 (this corresponds to
        epsilon in both Vosper and Clark)

        Returns:
            numpy.ndarray:
                2D array float32 - reference height for roughness correction

        """
        alpha = -np.log(ABSOLUTE_CORRECTION_TOL)
        tunable_param = np.full(self.wavenum.shape, RMDI, dtype=np.float32)
        h_ref = np.full(self.wavenum.shape, RMDI, dtype=np.float32)
        tunable_param[self.hcmask] = (
            alpha + np.log(self.wavenum[self.hcmask] *
                           self.h_over_2[self.hcmask]))
        tunable_param[tunable_param > 1.0] = 1.0
        tunable_param[tunable_param < 0.0] = 0.0
        h_ref[self.hcmask] = (
            tunable_param[self.hcmask] / self.wavenum[self.hcmask])
        h_ref[h_ref < 1.0] = 1.0
        h_ref = np.minimum(h_ref, HREF_SCALE * self.h_over_2)
        h_ref[h_ref < 1.0] = 1.0
        h_ref[~self.hcmask] = 0.0
        return h_ref

    def calc_roughness_correction(self, hgrid, uold, mask):
        """Function to perform the roughness correction.

        Args:
            hgrid (numpy.ndarray):
                3D or 1D array float32 - height above orography
            uold (numpy.ndarray):
                3D array float32 - original velocities at hgrid.
            mask (numpy.ndarray):
                 2D array of bools that is True for land-points, False for Sea
                 and False for invalid z_0.

        Returns:
            numpy.ndarray:
                3D np.array float32 - Corrected wind speed on hgrid. Above
                href, this is equal to uold.

        Comments:
            Replace the windspeed profile below the reference height with one
            that increases logarithmically with height, bound by the original
            velocity uhref at the reference height h_ref and by a 0 velocity at
            the vegetative roughness height z_0

        """
        uhref = self._calc_u_at_h(uold, hgrid, self.h_ref, mask)
        if hgrid.ndim == 1:
            hgrid = hgrid[np.newaxis, np.newaxis, :]
        ustar = FrictionVelocity(uhref, self.h_ref, self.z_0,
                                 mask).process()
        unew = np.copy(uold)
        mhref = self.h_ref
        mhref[~mask] = RMDI
        cond = hgrid < self.h_ref[:, :, np.newaxis]

        # Create array of ones.
        arr_ones = np.ones(unew.shape, dtype=np.float32)

        first_arg = (ustar[:, :, np.newaxis] * arr_ones)[cond]
        sec_arg = np.log(hgrid /
                         (np.reshape(self.z_0, self.z_0.shape + (1,)) *
                          arr_ones))[cond]

        unew[cond] = (first_arg * sec_arg) / VONKARMAN

        return unew

    def _calc_u_at_h(self, u_in, h_in, hhere, mask, dolog=False):
        """Function to interpolate u_in on h_in at hhere.

        Args:
            u_in (numpy.ndarray):
                3D array float32 - velocity on h_in layer, last dim is height
            h_in(numpy.ndarray):
                3D or 1D array float32 - height layer array
            hhere (numpy.ndarray):
                2D array float32 - height grid to interpolate at
            mask (numpy.ndarray):
                2D array of bools - mask the final result for uath
            dolog (bool):
                if True, log interpolation, default False

        Returns:
            numpy.ndarray:
                2D array float32 - velocity interpolated at h

        """
        u_in = np.ma.masked_less(u_in, 0.0)
        h_in = np.ma.masked_less(h_in, 0.0)
        # h_in.mask = u_in.mask
        # If I allow 1D height grids, I think I cannot do the hop over.

        # Ignores the height at the position where u_in is RMDI,"hops over"
        hhere = np.ma.masked_less(hhere, 0.0)
        upidx = np.argmax(h_in > hhere[:, :, np.newaxis], axis=2)
        # loidx = np.maximum(upidx-1, 0) #if RMDI, need below
        loidx = np.argmin(np.ma.masked_less(hhere[:, :, np.newaxis] -
                                            h_in, 0.0), axis=2)

        if h_in.ndim == 3:
            hup = h_in.take(upidx.flatten() + np.arange(0, upidx.size *
                                                        h_in.shape[2],
                                                        h_in.shape[2]))
            hlow = h_in.take(loidx.flatten() + np.arange(0, loidx.size *
                                                         h_in.shape[2],
                                                         h_in.shape[2]))
        elif h_in.ndim == 1:
            hup = h_in[upidx].flatten()
            hlow = h_in[loidx].flatten()
        uup = u_in.take(
            upidx.flatten() +
            np.arange(0, upidx.size * u_in.shape[2], u_in.shape[2])) \
            # pylint: disable=unsubscriptable-object
        ulow = u_in.take(
            loidx.flatten() +
            np.arange(0, loidx.size * u_in.shape[2], u_in.shape[2])) \
            # pylint: disable=unsubscriptable-object
        mask = mask.flatten()
        uath = np.full(mask.shape, RMDI, dtype=np.float32)
        if dolog:
            uath[mask] = self._interpolate_log(hup[mask], hlow[mask],
                                               hhere.flatten()[mask],
                                               uup[mask], ulow[mask])
        else:
            uath[mask] = self._interpolate_1d(hup[mask], hlow[mask],
                                              hhere.flatten()[mask],
                                              uup[mask], ulow[mask])
        uath = np.reshape(uath, hhere.shape)
        return uath

    @staticmethod
    def _interpolate_1d(xup, xlow, at_x, yup, ylow):
        """Simple 1D linear interpolation for 2D grid inputs level.

        Args:
            xup (numpy.ndarray):
                2D array float32 - upper x-bins
            xlow (numpy.ndarray):
                2D array float32 - lower x-bins
            at_x (numpy.ndarray):
                2D array float32 - x values to interpolate y at
            yup (numpy.ndarray):
                2D array float32 - y(xup)
            ylow (numpy.ndarray):
                2D array float32 - y(xlow)

        Returns:
            numpy.ndarray:
                2D array float32 - y(at_x) assuming a lin function
                between xlow and xup

        """
        interp = np.full(xup.shape, RMDI, dtype=np.float32)
        diffs = (xup - xlow)
        interp[diffs != 0] = (
            ylow[diffs != 0] + ((at_x[diffs != 0] - xlow[diffs != 0]) /
                                diffs[diffs != 0] * (yup[diffs != 0] -
                                                     ylow[diffs != 0])))
        interp[diffs == 0] = at_x[diffs == 0] / xup[diffs == 0] * (
            yup[diffs == 0])
        return interp

    @staticmethod
    def _interpolate_log(xup, xlow, at_x, yup, ylow):
        """Simple 1D log interpolation y(x), except if lowest layer is
        ground level.

        Args:
            xup (numpy.ndarray):
                2D array float32 - upper x-bins
            xlow (numpy.ndarray):
                2D array float32 - lower x-bins
            at_x (numpy.ndarray):
                2D array float32 - x values to interpolate y at
            yup (numpy.ndarray):
                2D array float32 - y(xup)
            ylow (numpy.ndarray):
                2D array float32 -y(xlow)

        Returns:
            numpy.ndarray:
                2D array float32 - y(at_x) assuming a log function
                between xlow and xup

        """
        ain = np.full(xup.shape, RMDI, dtype=np.float32)
        loginterp = np.full(xup.shape, RMDI, dtype=np.float32)
        mfrac = xup / xlow
        mtest = (xup / xlow != 1) & (at_x != xup)
        ain[mtest] = (yup[mtest] - ylow[mtest]) / np.log(mfrac[mtest])
        loginterp[mtest] = ain[mtest] * np.log(at_x[mtest] / xup[mtest]) + yup[
            mtest]
        mtest = (xup / xlow == 1)  # below lowest layer, make lin interp
        loginterp[mtest] = at_x[mtest] / xup[mtest] * (yup[mtest])
        mtest = (at_x == xup)  # just use yup
        loginterp[mtest] = yup[mtest]
        return loginterp

    def _calc_height_corr(self, u_a, heightg, mask, onemfrac):
        """Function to calculate the additive height correction.

        Args:
            u_a (numpy.ndarray):
                2D array float32 - outer velocity, e.g. velocity at h_ref_orig
            heightg (numpy.ndarray):
                1D or 3D array float32 - heights above orography
            mask (numpy.ndarray):
                3D array of bools - Masks the hc_add result
            onemfrac (float or numpy.ndarray):
                Currently, scalar = 1. But can be a function of position and
                height, e.g. a 3D array (float32)

        Returns:
            numpy.ndarray:
                3D array float32 - additive height correction to wind speed

        Comments:
            The height correction is a disturbance of the flow that
            decays exponentially with height. The larger the vertical
            offset h_at0 (the higher the unresolved hill), the larger
            is the disturbance.

            The more smooth the disturbance (the larger the horizontal
            scale of the disturbance), the smaller the height
            correction (hence, a larger wavenumber results in a larger
            disturbance).
            hc_add = exp(-height*wavenumber)*u(href)*h_at_0*wavenumber

            A final factor of 1 is assumed and omitted for the Bessel
            function term.

        """
        (xdim, ydim) = u_a.shape
        if heightg.ndim == 1:
            zdim = heightg.shape[0]
            heightg = heightg[np.newaxis, np.newaxis, :]
        elif heightg.ndim == 3:
            zdim = heightg.shape[2]
        ml2 = self.h_at0 * self.wavenum
        expon = np.ones([xdim, ydim, zdim], dtype=np.float32)
        mult = self.wavenum[:, :, np.newaxis] * heightg
        expon[mult > 0.0001] = np.exp(-mult[mult > 0.0001])
        hc_add = (expon * u_a[:, :, np.newaxis] *
                  ml2[:, :, np.newaxis] * onemfrac)
        hc_add[~mask, :] = 0
        return hc_add

    def _delta_height(self):
        """Function to calculate pp-grid diff from model grid.

        Calculate the difference between pp-grid height and model
        grid height.

        Returns:
            numpy.ndarray:
                2D array float32 - height difference, ppgrid-model

        """
        delt_z = np.full(self.pporo.shape, RMDI, dtype=np.float32)
        delt_z[self.hcmask] = self.pporo[self.hcmask] - self.modoro[
            self.hcmask]
        return delt_z

    def do_rc_hc_all(self, hgrid, uorig):
        """Function to call HC and RC (height and roughness corrections).

        Args:
            hgrid (numpy.ndarray):
                1D or 3D array float32 - height grid of wind input
            uorig (numpy.ndarray):
                3D array float32 - wind speed on these levels

        Returns:
            numpy.ndarray:
                sum of  unew: 3D array float32 - RC corrected windspeed
                on levels HC: 3D array float32 - HC additional part

        Friedrich, M. M., 2016
        Wind Downscaling Program (Internal Met Office Report)

        """
        if hgrid.ndim == 3:
            condition1 = ((hgrid == RMDI).any(axis=2))
            self.hcmask[condition1] = False
            self.rcmask[condition1] = False
        mask_rc = np.copy(self.rcmask)
        mask_rc[(uorig == RMDI).any(axis=2)] = False
        mask_hc = np.copy(self.hcmask)
        mask_hc[(uorig == RMDI).any(axis=2)] = False
        if self.z_0 is not None:
            unew = self.calc_roughness_correction(hgrid, uorig, mask_rc)
        else:
            unew = uorig
        uhref_orig = self._calc_u_at_h(uorig, hgrid, 1.0 / self.wavenum,
                                       mask_hc)
        mask_hc[uhref_orig <= 0] = False
        # Setting this value to 1, is equivalent to setting the
        # Bessel function to 1. (Friedrich, 2016)
        # Example usage if the Bessel function was not set to 1 is:
        # onemfrac = 1.0 - BfuncFrac(nx,ny,nz,heightvec,z_0,waveno, Ustar, UI)
        onemfrac = 1.0
        hc_add = self._calc_height_corr(uhref_orig, hgrid, mask_hc, onemfrac)
        result = unew + hc_add
        result[result < 0.] = 0  # HC can be negative if pporo<modeloro
        return result.astype(np.float32)


class RoughnessCorrection(PostProcessingPlugin):
    """Plugin to orographically-correct 3d wind speeds."""

    zcoordnames = ["height", "model_level_number"]
    tcoordnames = ["time", "forecast_time"]

    def __init__(self, a_over_s_cube, sigma_cube, pporo_cube,
                 modoro_cube, modres, z0_cube=None,
                 height_levels_cube=None):
        """Initialise the RoughnessCorrection instance.

        Args:
            a_over_s_cube (iris.cube.Cube):
                2D - model silhouette roughness on pp grid. dimensionless
            sigma_cube (iris.cube.Cube):
                2D - standard deviation of model orography height on pp grid.
                In m.
            pporo_cube (iris.cube.Cube):
                2D - pp orography. In m
            modoro_cube (iris.cube.Cube):
                2D - model orography interpolated on pp grid. In m
            modres (float):
                original average model resolution in m
            height_levels_cube (iris.cube.Cube):
                3D or 1D - height of input velocity field.
                Can be position dependent
            z0_cube (iris.cube.Cube):
                2D - vegetative roughness length in m. If not given, do not do
                any RC
        """
        # Standard Python 'float' type is either single or double depending on
        # system and there is no reliable method of finding which from the
        # variable. So force to numpy.float32 by default.
        modres = np.float32(modres)

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
        self.pp_oro = next(pporo_cube.slices([y_name, x_name]))
        self.model_oro = next(modoro_cube.slices([y_name, x_name]))
        self.ppres = self.calc_av_ppgrid_res(pporo_cube)
        self.modres = modres
        self.height_levels = height_levels_cube
        self.x_name = None
        self.y_name = None
        self.z_name = None
        self.t_name = None

    def find_coord_names(self, cube):
        """Extract x, y, z, and time coordinate names.

        Args:
            cube (iris.cube.Cube):
                some iris cube to find coordinate names from

        Returns:
            (tuple): tuple containing:
                **xname** (str):
                    name of the axis name in x-direction
                **yname** (str):
                    name of the axis name in y-direction
                **zname** (str):
                    name of the axis name in z-direction
                **tname** (str):
                    name of the axis name in t-direction

        """
        clist = {cube.coords()[i].name() for i in range(len(cube.coords()))}
        try:
            xname = cube.coord(axis="x").name()
        except CoordinateNotFoundError as exc:
            print("'{0}' while xname setting. Args: {1}.".format(exc,
                                                                 exc.args))
        try:
            yname = cube.coord(axis="y").name()
        except CoordinateNotFoundError as exc:
            print("'{0}' while yname setting. Args: {1}.".format(exc,
                                                                 exc.args))
        if clist.intersection(self.zcoordnames):
            zname = list(clist.intersection(self.zcoordnames))[0]
        else:
            zname = None

        if clist.intersection(self.tcoordnames):
            tname = list(clist.intersection(self.tcoordnames))[0]
        else:
            tname = None
        return xname, yname, zname, tname

    def calc_av_ppgrid_res(self, a_cube):
        """Calculate average grid resolution from a cube.

        Args:
            a_cube (iris.cube.Cube):
                Cube to calculate average resolution of

        Returns:
            np.float32:
                Average grid resolution.

        """
        x_name, y_name, _, _ = self.find_coord_names(a_cube)
        [exp_xname, exp_yname] = ["projection_x_coordinate",
                                  "projection_y_coordinate"]
        exp_unit = Unit("m")
        if (x_name != exp_xname) or (y_name != exp_yname):
            raise ValueError("cannot currently calculate resolution")

        if (a_cube.coord(x_name).bounds is None and
                a_cube.coord(y_name).bounds is None):
            xres = (np.diff(a_cube.coord(x_name).points)).mean()
            yres = (np.diff(a_cube.coord(y_name).points)).mean()
        else:
            xres = (np.diff(a_cube.coord(x_name).bounds)).mean()
            yres = (np.diff(a_cube.coord(y_name).bounds)).mean()
        if (
                (a_cube.coord(x_name).units != exp_unit) or
                (a_cube.coord(y_name).units != exp_unit)):
            raise ValueError("cube axis have units different from m.")
        return (abs(xres) + abs(yres)) / 2.0

    @staticmethod
    def check_ancils(a_over_s_cube, sigma_cube, z0_cube, pp_oro_cube,
                     model_oro_cube):
        """Check ancils grid and units.

        Check if ancil cubes are on the same grid and if they have the
        expected units. The testing for "same grid" might be replaced
        if there is a general utils function made for it or so.

        Args:
            a_over_s_cube (iris.cube.Cube):
                holding the silhouette roughness field
            sigma_cube (iris.cube.Cube):
                holding the standard deviation of height in a grid cell
            z0_cube (iris.cube.Cube or None):
                holding the vegetative roughness field
            pp_oro_cube (iris.cube.Cube):
                holding the post processing grid orography
            model_oro_cube (iris.cube.Cube):
                holding the model orography on post processing grid

        Returns:
            numpy.ndarray:
                Containing bools describing whether or not the tests passed

        """
        ancil_list = [a_over_s_cube, sigma_cube, pp_oro_cube, model_oro_cube]
        unwanted_coord_list = [
            "time", "height", "model_level_number", "forecast_time",
            "forecast_reference_time", "forecast_period"]
        for field, exp_unit in zip(ancil_list, [1, Unit("m"),
                                                Unit("m"), Unit("m")]):
            for unwanted_coord in unwanted_coord_list:
                try:
                    field.remove_coord(unwanted_coord)
                except CoordinateNotFoundError:
                    pass
            if field.units != exp_unit:
                msg = ('{} ancil field has unexpected unit:'
                       ' {} (expected) vs. {} (actual)')
                raise ValueError(
                    msg.format(field.name(), exp_unit, field.units))
        if z0_cube is not None:
            ancil_list.append(z0_cube)
            for unwanted_coord in unwanted_coord_list:
                try:
                    z0_cube.remove_coord(unwanted_coord)
                except CoordinateNotFoundError:
                    pass
            if z0_cube.units != Unit('m'):
                msg = ("z0 ancil has unexpected unit: should be {} "
                       "is {}")
                raise ValueError(msg.format(Unit('m'), z0_cube.units))
        permutated_ancil_list = list(itertools.permutations(ancil_list, 2))
        oklist = []
        for entry in permutated_ancil_list:
            x_axis_flag = (
                entry[0].coord(axis="y") == entry[1].coord(axis="y"))
            y_axis_flag = (
                entry[0].coord(axis="x") == entry[1].coord(axis="x"))
            oklist.append(x_axis_flag & y_axis_flag)
            # HybridHeightToPhenomOnPressure._cube_compatibility_check(entr[0],
            # entr[1])
        return np.array(oklist).all()  # replace by a return value of True

    def find_coord_order(self, mcube):
        """Extract coordinate ordering within a cube.

        Use coord_dims to assess the dimension associated with a particular
        dimension coordinate. If a coordinate is not a dimension coordinate,
        then a NaN value will be returned for that coordinate.

        Args:
            mcube (iris.cube.Cube):
                cube to check the order of coordinate axis

        Returns:
            (tuple): tuple containing:
                **xpos** (int):
                    position of x axis.
                **ypos** (int):
                    position of y axis.
                **zpos** (int):
                    position of z axis.
                **tpos** (int):
                    position of t axis.

        """
        coord_names = [self.x_name, self.y_name, self.z_name, self.t_name]
        positions = len(coord_names) * [np.nan]
        for coord_index, coord_name in enumerate(coord_names):
            if mcube.coords(coord_name, dim_coords=True):
                positions[coord_index], = mcube.coord_dims(coord_name)
        return positions

    def find_heightgrid(self, wind):
        """Setup the height grid.

        Setup the height grid either from the 1D or 3D height grid
        that was supplied to the plugin or from the z-axis information
        from the wind grid.

        Args:
            wind (iris.cube.Cube):
                3D or 4D - representing the wind data.

        Returns:
            numpy.ndarray:
                1D or 3D array - representing the height grid.

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
                except KeyError:
                    raise ValueError("height grid different from wind grid")
            elif hld.ndim == 1:
                try:
                    hld = next(hld.slices([self.z_name]))
                except CoordinateNotFoundError:
                    raise ValueError("height z coordinate differs from wind z")
            else:
                raise ValueError("hld must have a dimension length of "
                                 "either 3 or 1"
                                 "hld.ndim is {}".format(hld.ndim))
            hld = hld.data
        return hld

    def check_wind_ancil(self, xwp, ywp):
        """Check wind vs ancillary file grids.

        Check if wind and ancillary files are on the same grid and if
        they have the same ordering.

        Args:
            xwp (int):
                representing the position of the x-axis in the wind cube
            ywp (int):
                representing the position of the y-axis of the wind cube

        """
        xap, yap, _, _ = self.find_coord_order(self.pp_oro)
        if xwp - ywp != xap - yap:
            if np.isnan(xap) or np.isnan(yap):
                raise ValueError("ancillary grid different from wind grid")
            raise ValueError("xy-orientation: ancillary differ from wind")

    def process(self, input_cube):
        """Adjust the 4d wind field - cube - (x, y, z including times).

        Args:
            input_cube (iris.cube.Cube):
                The wind cube to be operated upon. Should be wind speed on
                height_levels for all desired forecast times.

        Returns:
            iris.cube.Cube:
                The 4d wind field with roughness and height correction
                applied in the same order as the input cube.

        Raises
        ------
        TypeError: If input_cube is not a cube.

        """
        if not isinstance(input_cube, iris.cube.Cube):
            msg = "wind input is not a cube, but {}"
            raise TypeError(msg.format(type(input_cube)))
        (self.x_name, self.y_name, self.z_name,
         self.t_name) = self.find_coord_names(input_cube)
        xwp, ywp, zwp, twp = self.find_coord_order(input_cube)
        if np.isnan(twp):
            input_cube.transpose([ywp, xwp, zwp])
        else:
            input_cube.transpose([ywp, xwp, zwp, twp])  # problems with slices
        rchc_list = iris.cube.CubeList()
        if self.z_0 is None:
            z0_data = None
        else:
            z0_data = self.z_0.data
        roughness_correction = RoughnessCorrectionUtilities(
            self.a_over_s.data, self.sigma.data, z0_data, self.pp_oro.data,
            self.model_oro.data, self.ppres, self.modres)
        self.check_wind_ancil(xwp, ywp)
        hld = self.find_heightgrid(input_cube)
        for time_slice in input_cube.slices_over("time"):
            if np.isnan(time_slice.data).any() or (time_slice.data < 0.).any():
                msg = ('{} has invalid wind data')
                raise ValueError(msg.format(time_slice.coord(self.t_name)))
            rc_hc = copy.deepcopy(time_slice)
            rc_hc.data = roughness_correction.do_rc_hc_all(
                hld, time_slice.data)
            rchc_list.append(rc_hc)
        output_cube = rchc_list.merge_cube()
        # reorder input_cube and output_cube as original
        if np.isnan(twp):
            input_cube.transpose(np.argsort([ywp, xwp, zwp]))
            output_cube.transpose(np.argsort([ywp, xwp, zwp]))
        else:
            input_cube.transpose(np.argsort([ywp, xwp, zwp, twp]))
            output_cube.transpose(np.argsort([twp, ywp, xwp, zwp]))

        return output_cube
