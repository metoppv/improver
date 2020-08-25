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

import argparse
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


def main(obs_path, fcst_paths, log_path, model, thresholds_mmh):
    """
    Args:
        obs_path (str):
            Full path to observation netcdf file
        fcst_paths (list of str):
            List of full paths to forecast (nowcast or UKV) netcdf files
        log_path (str):
            Directory to write / append log file
        model (str):
            Source model identifier for output file name
        thresholds_mmh (list of float)
    """
    # Load "truth" and set up constants
    obs = load_cube(obs_path)
    obs.convert_units('mm h-1')
    obs_time = obs.coord("time").cell(0).point
    cycletime = datetime_to_cycletime(obs_time)

    lines = []
    for fcst_path in fcst_paths:    

        all_fcsts = load_cube(fcst_path)

        try:
            fcst = extract_nearest_time_point(all_fcsts, obs_time, allowed_dt_difference=0)
        except ValueError:
            # we expect the UKV not to have matching 15 minute forcasts; so if a
            # matching forecast is not available, exit without error
            continue

        fcst.convert_units('mm h-1')
        lead_time_minutes = int(fcst.coord("forecast_period").points[0] / 60)

        for threshold in thresholds_mmh:
            hits, misses, false_alarms, no_det = get_counts(obs.data, fcst.data, threshold)
            line = (f'{cycletime} {lead_time_minutes:3} {threshold:5.3} {hits:6} '
                    f'{misses:6} {false_alarms:6} {no_det:6}')
            lines.append(line)

    # Append lines to log file
    fname = os.path.join(log_path, f'{cycletime[:6]}_{model}_counts.log')
    with open(fname, "a") as dtf:
        for line in lines:
            dtf.write(line+'\n')


if __name__ == "__main__":
    """Execute with arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('obs_path', type=str)
    parser.add_argument('fcst_paths', type=str, nargs='+')
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--thresholds', type=float, nargs='+')
    args = parser.parse_args()
    main(args.obs_path, args.fcst_paths, args.log_path, args.model, args.thresholds)
