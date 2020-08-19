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
from improver.utilities.temporal import (
    datetime_to_cycletime, extract_nearest_time_point
)


def get_counts(obs, fcst, threshold):
    """Compute counts of hits, misses, false alarms and correct no detections
    at this threshold"""
    count = np.ones(obs.shape, dtype=np.int32)
    mask = obs.mask + fcst.mask

    hits = np.sum(count[np.where((obs >= threshold) & (fcst >= threshold) & (~mask))])
    misses = np.sum(count[np.where((obs >= threshold) & (fcst < threshold) & (~mask))])
    false = np.sum(count[np.where((obs < threshold) & (fcst >= threshold) & (~mask))])
    no_det = np.sum(count[np.where((obs < threshold) & (fcst < threshold) & (~mask))])

    return hits, misses, false, no_det


def main(obs_path, fcst_path, log_path, model, thresholds_mmh):
    """
    Args:
        obs_path (str):
            Full path to observation netcdf file
        fcst_path (str):
            Full path to forecast (nowcast or UKV) netcdf file
        log_path (str):
            Directory to write / append log file
        model (str):
            Source model identifier for output file name
        thresholds_mmh (list of float)
    """
    # Read inputs
    obs = load_cube(obs_path)
    all_fcsts = load_cube(fcst_path)

    # Extract required forecast validity time
    obs_time = obs.coord("time").cell(0).point
    try:
        fcst = extract_nearest_time_point(all_fcsts, obs_time, allowed_dt_difference=0)
    except ValueError:
        # we expect the UKV not to have matching 15 minute forcasts; so if a
        # matching forecast is not available, exit without error
        return

    # Convert to threshold units
    obs.convert_units('mm h-1')
    fcst.convert_units('mm h-1')

    # Calculate hits, misses, false alarms and missed detections for these thresholds
    cycletime = datetime_to_cycletime(obs_time)
    lead_time_minutes = int(fcst.coord("forecast_period").points[0] / 60)
    lines = []
    for threshold in thresholds_mmh:
        hits, misses, false_alarms, no_det = get_counts(obs.data, fcst.data, threshold)
        line = f'{cycletime} {lead_time_minutes:3} {threshold:5.3} {hits:6} {misses:6} {false_alarms:6} {no_det:6}'
        lines.append(line)

    # Append line to log file (one per month)
    fname = os.path.join(log_path, f'{cycletime[:6]}_{model}_counts.log')
    with open(fname, "a") as dtf:
        for line in lines:
            dtf.write(line+'\n')


if __name__ == "__main__":
    """Execute with arguments"""
    args = sys.argv[1:5]
    args.append([float(t) for t in sys.argv[5:]])
    main(*args)

