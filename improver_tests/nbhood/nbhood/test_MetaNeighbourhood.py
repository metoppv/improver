# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from unittest.mock import patch, sentinel

import pytest

from improver.nbhood.nbhood import MetaNeighbourhood


class HaltExecution(Exception):
    pass


@patch("improver.nbhood.nbhood.as_cube")
def test_as_cube_called(mock_as_cube):
    mock_as_cube.side_effect = [None, HaltExecution]
    try:
        MetaNeighbourhood(
            lead_times=[1, 2, 3],
            radii=[1, 2, 3],
            neighbourhood_output=sentinel.ngoutput,
        )(sentinel.cube, mask=sentinel.mask)
    except HaltExecution:
        pass
    mock_as_cube.assert_any_call(sentinel.cube)
    mock_as_cube.assert_any_call(sentinel.mask)


@pytest.mark.parametrize(
    "neighbourhood_output, weighted_mode, neighbourhood_shape, degrees_as_complex, exception_msg",
    [
        (
            "percentiles",
            True,
            "square",
            False,
            "weighted_mode cannot be used with" 'neighbourhood_output="percentiles"',
        ),
        (
            "percentiles",
            False,
            "square",
            True,
            "Cannot generate percentiles from complex numbers",
        ),
        (
            "probabilities",
            False,
            "circular",
            True,
            "Cannot process complex numbers with circular neighbourhoods",
        ),
    ],
)
def test___init___exceptions(
    neighbourhood_output,
    weighted_mode,
    neighbourhood_shape,
    degrees_as_complex,
    exception_msg,
):
    """Exception when passing """
    args = [neighbourhood_output]
    kwargs = {
        "weighted_mode": weighted_mode,
        "neighbourhood_shape": neighbourhood_shape,
        "degrees_as_complex": degrees_as_complex,
    }
    kwargs.update(dict(lead_times=[1, 2, 3], radii=[1, 2, 3]))
    with pytest.raises(RuntimeError, match=exception_msg):
        MetaNeighbourhood(*args, **kwargs)
