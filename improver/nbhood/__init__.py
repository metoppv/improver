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
"""init for nbhood"""


def radius_by_lead_time(radii, lead_times):
    """
    Parse radii and lead_times provided to CLIs that use neighbourhooding.
    If no lead times are provided, return the first radius for use at all
    lead times. If lead times are provided, ensure there are sufficient
    radii to assign one to each lead time. If so return two lists, else raise
    an exception.

    Args:
        radii (list of str):
            Radii as a list provided by clize.
        lead_times (list of str or None):
            Lead times as a list provided by clize, or None if not set.
    Returns:
        (tuple): tuple containing:
            **radius_or_radii** (float or list of floats):
                Radii as a float or list of floats.
            **lead_times** (None or list of ints):
                Lead times in hours as a list of ints or None.
    Raises:
        ValueError: If multiple radii are provided without any lead times.
        ValueError: If radii and lead_times lists are on unequal lengths.
    """
    if lead_times is None:
        if not len(radii) == 1:
            raise ValueError("Multiple radii have been supplied but no "
                             "associated lead times.")
        radius_or_radii = float(radii[0])
    else:
        if not len(radii) == len(lead_times):
            raise ValueError("If leadtimes are supplied, it must be a list"
                             " of equal length to a list of radii.")
        radius_or_radii = [float(x) for x in radii]
        lead_times = [int(x) for x in lead_times]

    return radius_or_radii, lead_times
