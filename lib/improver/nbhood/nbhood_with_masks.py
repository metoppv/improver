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
"""Utilities for using neighbourhood processing."""


import iris


class ApplyNeighbourhoodProcessingWithAMask(object):

    def __init__(
            self, coord_for_masking, neighbourhood_method, radii, lead_times,
            unweighted_mode, ens_factor):
        """
        Initialise the class.

        Args:
            coord_for_masking : string
                String matching the name of the coordinate that will be used
                for masking.
            neighbourhood_method : str
                Name of the neighbourhood method to use. Options: 'circular',
                'square'.
            radii : float or List (if defining lead times)
                The radii in metres of the neighbourhood to apply.
                Rounded up to convert into integer number of grid
                points east and north, based on the characteristic spacing
                at the zero indices of the cube projection-x and y coords.
            lead_times : None or List
                List of lead times or forecast periods, at which the radii
                within 'radii' are defined. The lead times are expected
                in hours.
            unweighted_mode : boolean
                If True, use a circle with constant weighting.
                If False, use a circle for neighbourhood kernel with
                weighting decreasing with radius.
            ens_factor : float
                The factor with which to adjust the neighbourhood size
                for more than one ensemble member.
                If ens_factor = 1.0 this essentially conserves ensemble
                members if every grid square is considered to be the
                equivalent of an ensemble member.
                Optional, defaults to 1.0
        """
        self.coord_for_masking = coord_for_masking
        self.neighbourhood_method = neighbourhood_method
        self.radii = radii
        self.lead_times = lead_times
        self.unweighted_mode = unweighted_mode
        self.ens_factor = ens_factor

    def __repr__(self):
        pass

    def process(self, cube, mask_cube):

        cube_slices = iris.cube.CubeList([])
        for cube_slice in mask_cube.slices_over(coord_for_masking):
            cube_slice = NeighbourhoodProcessing(
                self.neighbourhood_method, self.radii, self.lead_times,
                self.unweighted_mode, self.ens_factor).process(
                    cube_slice, mask_cube=mask_cube)
            cube_slices.append(cube_slice)
        concatenated_cube = cube_slices.concatenate_cube()

        single_cube = concatenated_cube.collapsed(coord_for_masking, iris.analysis.MAX)
        return single_cube
