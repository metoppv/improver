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
"""Module containing profiling utilities."""

import atexit
import cProfile
import pstats
import sys


def profile_start():
    """Start and return a new profiler instance.

    Returns:
        Active cProfile.Profile instance.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    return profiler


def profile_hook_enable(dump_filename=None):
    """Register a hook to gather and dump profile information at exit.

    Args:
        dump_filename (str):
            File path to dump profiling info into at exit.
    """
    profiler = profile_start()
    atexit.register(profile_stop, profiler, dump_filename=dump_filename)


def profile_stop(profiler, sort_field='cumulative', dump_filename=None,
                 dump_line_count=100):
    """Stop a given profiler and print or dump stats.

    Args:
        profiler (cProfile.Profile):
            Active profiling instance.
        sort_field (str):
            pstats.Stats sort field for ordering profiling results.
        dump_filename (str):
            File path to dump profiling stats into.
        dump_line_count (int):
            Maximum lines to print out.
    """
    profiler.disable()
    stats = pstats.Stats(profiler, stream=sys.stderr).sort_stats(sort_field)
    if dump_filename is None:
        stats.print_stats(dump_line_count)
    else:
        stats.dump_stats(dump_filename)
