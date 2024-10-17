# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Expected datatypes and units for time-type coordinates"""

from collections import namedtuple

import numpy as np

TimeSpec = namedtuple("TimeSpec", ("calendar", "dtype", "units"))

DT_FORMAT = "%Y%m%dT%H%MZ"

_TIME_REFERENCE_SPEC = TimeSpec(
    calendar="gregorian", dtype=np.int64, units="seconds since 1970-01-01 00:00:00"
)

_TIME_INTERVAL_SPEC = TimeSpec(calendar=None, dtype=np.int32, units="seconds")

TIME_COORDS = {
    "time": _TIME_REFERENCE_SPEC,
    "forecast_reference_time": _TIME_REFERENCE_SPEC,
    "blend_time": _TIME_REFERENCE_SPEC,
    "forecast_period": _TIME_INTERVAL_SPEC,
    "UTC_offset": _TIME_INTERVAL_SPEC,
}
