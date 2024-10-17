# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing profiling utilities."""

import atexit
import cProfile
import pstats
import sys
from cProfile import Profile
from typing import Optional


def profile_start() -> Profile:
    """Start and return a new profiler instance.

    Returns:
        Active cProfile.Profile instance.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    return profiler


def profile_hook_enable(dump_filename: Optional[str] = None) -> None:
    """Register a hook to gather and dump profile information at exit.

    Args:
        dump_filename:
            File path to dump profiling info into at exit.
    """
    profiler = profile_start()
    atexit.register(profile_stop, profiler, dump_filename=dump_filename)


def profile_stop(
    profiler: Profile,
    sort_field: str = "cumulative",
    dump_filename: str = None,
    dump_line_count: int = 100,
) -> None:
    """Stop a given profiler and print or dump stats.

    Args:
        profiler:
            Active profiling instance.
        sort_field:
            pstats.Stats sort field for ordering profiling results.
        dump_filename:
            File path to dump profiling stats into.
        dump_line_count:
            Maximum lines to print out.
    """
    profiler.disable()
    stats = pstats.Stats(profiler, stream=sys.stderr).sort_stats(sort_field)
    if dump_filename is None:
        stats.print_stats(dump_line_count)
    else:
        stats.dump_stats(dump_filename)
