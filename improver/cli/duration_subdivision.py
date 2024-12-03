#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to divide period diagnostics into shorter periods."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    *,
    target_period: int,
    fidelity: int,
    day_mask: bool = False,
    night_mask: bool = False,
):
    """Subdivide a duration diagnostic, e.g. sunshine duration, into
    shorter periods, optionally applying a night mask to ensure that
    quantities defined only in the day or night are not spread into
    night or day periods respectively.

    This is a very simple approach. In the case of sunshine duration
    the duration is divided up evenly across the short periods defined
    by the fidelity argument. These are then optionally masked to zero
    for chosen periods (day or night). Values in the non-zeroed periods
    are then renormalised relative to the original period total, such
    that the total across the whole period ought to equal the original. This
    is not always possible as the night mask applied is simpler than e.g. the
    radiation scheme impact on a 3D orography. As such the renormalisation
    could yield durations longer than the fidelity period in each
    non-zeroed period as it tries to allocate e.g. 5 hours of sunlight
    across 4 non-zeroed hours. This is not physical, so the renormalisation
    is partnered with a clip that limits the duration allocated to the
    renormalised periods to not exceed their length. The result of this
    is that the original sunshine durations cannot be recovered for points
    that are affected. Instead the calculated night mask is limiting the
    accuracy to allow the subdivision to occur. This is the cost of this
    method.

    Note that this method cannot account for any e.g. cloud that is
    affecting the sunshine duration in a period. If a 6-hour period is
    split into three 2-hour periods the split will be even regardless of
    when thick cloud might occur.

    Args:
        cube (iris.cube.Cube):
            The original duration diagnostic cube.
        target_period (int):
            The time period described by the output cubes in seconds.
            The data will be reconstructed into non-overlapping periods.
            The target_period must be a factor of the original period.
        fidelity (int):
            The shortest increment in seconds into which the input periods are
            divided and to which the night mask is applied. The
            target periods are reconstructed from these shorter periods.
            Shorter fidelity periods better capture where the day / night
            dicriminator falls.
        night_mask (bool):
            If true, points that fall at night are zeroed and duration
            reallocated to day time periods as much as possible.
        day_mask (bool):
            If true, points that fall in the day time are zeroed and
            duration reallocated to night time periods as much as possible.
    Returns:
        iris.cube.Cube:
            A cube containing the target period data with a time dimension
            with an entry for each period. These periods combined span the
            original cube's period.
    """
    from improver.utilities.temporal_interpolation import DurationSubdivision

    plugin = DurationSubdivision(
        target_period, fidelity, night_mask=night_mask, day_mask=day_mask
    )
    result = plugin.process(cube)
    return result
