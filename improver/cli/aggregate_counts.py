#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
"""Script to calculate hits, misses, false alarms and no detections and append
counts to a log file."""

import os
import sys
import numpy as np

from iris import Constraint

from improver.utilities.load import load_cube
from improver.utilities.temporal import datetime_to_cycletime


def get_counts(obs, fcst, threshold):
    """Compute counts of hits, misses, false alarms and correct no detections
    at this threshold"""
    count = np.ones_like(obs)
    hits = np.sum(count[np.where((obs >= threshold) & (fcst >= threshold))])
    misses = np.sum(count[np.where((obs >= threshold) & (fcst < threshold))])
    false = np.sum(count[np.where((obs < threshold) & (fcst >= threshold))])
    no_det = np.sum(count[np.where((obs < threshold) & (fcst < threshold))])
    return hits, misses, false, no_det


def main(obs_path, fcst_path, log_path, lead_time_minutes, threshold, threshold_unit='mm h-1'):
    """
    Args:
        obs_path (str):
            Full path to observation netcdf file
        fcst_path (str):
            Full path to forecast (nowcast or UKV) netcdf file
        log_path (str):
            Directory to write / append log file
        lead_time_minutes (int)
        threshold (float)
        threshold_unit (str)

    """
    # Read radar observation and forecast at appropriate lead time
    lead_time_seconds = 60 * int(lead_time_minutes)
    obs = load_cube(obs_path)
    fcst = load_cube(fcst_path, Constraint(forecast_period=lead_time_seconds))

    # Check validity times
    obs_time = obs.coord("time").cell(0).point
    fcst_time = fcst.coord("time").cell(0).point
    if fcst_time != obs_time:
        msg = 'Forecast for {} does not match observation at {}'
        raise ValueError(msg.format(fcst_time, obs_time))

    # For this threshold calculate hits, misses, false alarms and missed detections
    obs.convert_units(threshold_unit)
    fcst.convert_units(threshold_unit)
    hits, misses, false_alarms, no_det = get_counts(obs.data, fcst.data, threshold)

    # Format line
    cycletime = datetime_to_cycletime(obs_time)
    line = f'{cycletime} {lead_time_minutes} {hits} {misses} {false_alarms} {no_det}'

    # Append line to log file (one per month)
    fname = os.path.join(log_path, f'{cycletime[:6]}_{threshold}_counts.log')
    with open(fname, "a") as dtf:
        dtf.write(line+'\n')


if __name__ == "__main__":
    """Execute with arguments"""
    args = sys.argv[1:4]
    args.append(int(sys.argv[4]))
    args.append(float(sys.argv[5]))
    if len(sys.argv) > 6:
        args.append(sys.argv[6])
    main(*args)

