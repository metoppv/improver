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
"""Provide support utilities for time lagging ensembles"""

from improver.utilities.temporal import (
    unify_forecast_reference_time, cycletime_to_datetime,
    find_latest_cycletime)
from improver.utilities.cube_manipulation import merge_cubes


class GenerateTimeLaggedEnsemble(object):

    """
    A plugin to combine realizations from different forecast cycles into one
    cube.
    """

    def __init__(self, cycletime=None):
        """
        Initialise class.

        Keyword Args:
            cycletime (str):
                A string of form YYYYMMDDTHHMMZ describing the
                forecast_reference_time we want the resulting cube to be
                relative to. Default None in which case the latest
                forecast_reference_time from the input cubes is used.
        """
        self.cycletime = cycletime

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<GenerateTimeLaggedEnsemble: cycletime: {}>')
        return result.format(self.cycletime)

    def process(self, cubelist):
        """
        Take an input cubelist containing forecasts from different cycles and
        merges them into a single cube.

        The steps taken are:
            1. If no cycletime is given then find the latest cycle time from
               the input cubes.
            2. Update the forecast periods in each input cube to be relative
               to the new cycletime.
            3. Checks if there are duplicate realization numbers. If a
               duplicate is found, renumbers all of the realizations to remove
               any duplicates.
            4. Merge cubes into one cube, removing any metadata that
               doesn't match.
        """
        if self.cycletime is None:
            cycletime = find_latest_cycletime(cubelist)
        else:
            cycletime = cycletime_to_datetime(self.cycletime)
        cubelist = unify_forecast_reference_time(cubelist, cycletime)

        duplicates=[]

        for cube in cubelist[1:]:
            for item in cube.coord("realization").points:
                if item in cubelist[0].coord("realization").points:
                    duplicates.append(item)

        if len(duplicates) > 0:
            duplicates_exist = True
        else:
            duplicates_exist = False

        if duplicates_exist:
            first_realization = 0
            for cube in cubelist:
                n_realization = len(cube.coord("realization").points)
                new_realization = []

                for i in range(first_realization,first_realization+n_realization):
                    new_realization.append(i)
                cube.coord("realization").points=new_realization
                first_realization = len(new_realization)

        lagged_ensemble = merge_cubes(cubelist)
        return lagged_ensemble
