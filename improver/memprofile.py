# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2022 Met Office.
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
"""Module containing maximum memory profiling utilities."""

import sys
import time
import tracemalloc
from datetime import datetime
from queue import Queue
from resource import RUSAGE_SELF, getrusage
from threading import Thread
from typing import Callable, Tuple


def memory_profile_start(outfile_prefix: str) -> Tuple[Thread, Queue]:
    """Starts the memory tracking profiler.

    Args:
        outfile_prefix:
            Prefix for the generated output. 2 files will
            be generated: \\*_SNAPSHOT and \\*_MAX_TRACKER.

    Returns:
        - Active Thread tracking the memory.
        - Active Queue for communication to the thread.
    """
    queue = Queue()
    thread = Thread(target=memory_monitor, args=(queue, outfile_prefix))
    thread.start()
    return thread, queue


def memory_profile_end(queue: Queue, thread: Thread) -> None:
    """Ends the memory tracking profiler.

    Args:
        queue:
            Active queue instance to communicate with active thread.
        thread:
            Active thread instance running memory tracking.
    """
    queue.put("Stop")
    thread.join()


def memory_monitor(queue: Queue, outfile_prefix: str) -> None:
    """Function to track memory usage, should be run in a separate
    thread to the main program.

    Samples max_rss every 0.1s, if the max_rss is higher than the
    previous max_rss, then creates a tracemalloc snapshot. There is a
    performance overhead when using this.

    Args:
        queue:
            Active queue instance to communicate with the thread.
        outfile_prefix:
            Prefix for the generated output. 2 files will
            be generated: \\*_SNAPSHOT and \\*_MAX_TRACKER.
    """
    tracemalloc.start()
    old_max = 0
    snapshot = None
    wait_time = 0.1

    fout = open("{}_MAX_TRACKER".format(outfile_prefix), "w")
    b2mb = 1 / 1048576
    if sys.platform == "linux":
        # linux outputs max_rss in KB not B
        b2mb = 1 / 1024

    while True:
        if queue.empty():
            time.sleep(wait_time)
            max_rss = getrusage(RUSAGE_SELF).ru_maxrss
            if max_rss > old_max:
                snapshot = tracemalloc.take_snapshot()
                line = "{} max RSS {:.2f} MiB".format(datetime.now(), max_rss * b2mb)
                print(line, file=fout)
                old_max = max_rss
        else:
            snapshot.dump("{}_SNAPSHOT".format(outfile_prefix))
            fout.close()
            tracemalloc.stop()
            return


def memory_profile_decorator(func: Callable, outfile_prefix: str) -> Callable:
    """A decorator for convenience of running.

    Args:
        func:
            function to track the maximum memory of.
        outfile_prefix:
            Prefix for the generated output. 2 files will
            be generated: \\*_SNAPSHOT and \\*_MAX_TRACKER.

    Returns:
        The wrapper
    """

    def wrapper(*args, **kwargs):
        thread, queue = memory_profile_start(outfile_prefix)
        results = func(*args, **kwargs)
        memory_profile_end(queue, thread)
        return results

    return wrapper
