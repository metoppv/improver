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
"""Module containing to apply neighbourhood processing to a vicinity."""


class ProbabilityOfOccurrence(object):

    """
    Calculate the probability of a phenomenon occurring within specified
    distance.
    """

    def __init__(self, distance, neighbourhood_method, radii, lead_times=None,
                 unweighted_mode=False, ens_factor=1.0):
        """
        Initialise the class.

        Args:
            distance : float
                Distance in metres used to define whether there is an
                occurrence within the vicinity.
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
        self.distance = distance
        self.neighbourhood_method = neighbourhood_method
        self.radii = radii
        self.lead_times = lead_times
        self.unweighted_mode = unweighted_mode
        self.ens_factor = ens_factor

    def process(self, cube):
        """
        Identify the probability of having a phenomenon occur within a
        vicinity.
        The steps for this are as follows:
        1. Calculate the occurrence of a phenomenon within a defined vicinity.
        2. If the cube contains a realization dimension coordinate, find the
           mean.
        3. Compute neighbourhood processing.

        Args:
            cube : Iris.cube.Cube
                A cube that has been thresholded.

        Returns:
            cube : Iris.cube.Cube
                A cube containing neighbourhood probabilities to represent the
                probability of an occurrence within the vicinity given a
                pre-defined spatial uncertainty.

        """
        cube = OccurrenceWithinVicinity(distance).process(cube)
        if cube.dim_coords('realization'):
            cube = cube.collapsed('realization', iris.analysis.MEAN)
        cube = NeighbourhoodProcessing(
            self.neighbourhood_method, self.radii, self.lead_times,
            self.unweighted_mode, self.ens_factor).process(cube)
        return cube
