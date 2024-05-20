# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Test wide setup and configuration"""

import pytest


@pytest.fixture(autouse=True)
def thread_control(monkeypatch):
    """
    Wrap all tests with a limit to one thread via threadpoolctl.

    The threadpoolctl library handles a variety of numerical libraries including
    OpenBLAS, MKL and OpenMP, using their library specific interfaces during runtime.
    Environment variable settings for these need to be applied before starting the
    python interpreter, which is not possible from inside pytest.
    This limitation to one thread avoids thread contention when parallel processing
    is handled at larger scale such as Dask or pytest-xdist. See dask documentation:
    https://docs.dask.org/en/stable/array-best-practices.html#avoid-oversubscribing-threads
    """
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            yield
    except ModuleNotFoundError:
        yield
    return
