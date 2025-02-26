# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the threshold.Threshold plugin."""

import numpy as np
import pytest
from iris.coords import CellMethod, DimCoord
from iris.cube import Cube

from improver.threshold import Threshold


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        # a threshold of zero is used with a multiplicative fuzzy factor
        (
            {"threshold_values": 0.0, "fuzzy_factor": 0.6},
            "Invalid threshold with fuzzy factor",
        ),
        # a fuzzy factor of minus 1 is given
        (
            {"threshold_values": 0.6, "fuzzy_factor": -1.0},
            "Invalid fuzzy_factor: must be >0 and <1: -1.0",
        ),
        # a fuzzy factor of zero is given
        (
            {"threshold_values": 0.6, "fuzzy_factor": 0.0},
            "Invalid fuzzy_factor: must be >0 and <1: 0.0",
        ),
        # a fuzzy factor of unity is given
        (
            {"threshold_values": 0.6, "fuzzy_factor": 1.0},
            "Invalid fuzzy_factor: must be >0 and <1: 1.0",
        ),
        # a fuzzy factor of 2 is given
        (
            {"threshold_values": 0.6, "fuzzy_factor": 2.0},
            "Invalid fuzzy_factor: must be >0 and <1: 2.0",
        ),
        # fuzzy_factor and fuzzy_bounds both set
        (
            {"threshold_config": {"0.6": [0.4, 0.8]}, "fuzzy_factor": 2.0},
            "Invalid combination of keywords",
        ),
        # fuzzy_bounds contains one value
        (
            {"threshold_config": {"0.6": [0.4]}},
            "Invalid bounds for one threshold: \\(0.4,\\).",
        ),
        # fuzzy_bounds contains three values
        (
            {"threshold_config": {"0.6": [0.4, 0.8, 1.2]}},
            "Invalid bounds for one threshold: \\(0.4, 0.8, 1.2\\).",
        ),
        # fuzzy_bounds do not bound threshold - upper bound too low
        (
            {"threshold_config": {"0.6": [0.4, 0.5]}},
            "Threshold must be within bounds: !\\( 0.4 <= 0.6 <= 0.5 \\)",
        ),
        # fuzzy_bounds do not bound threshold - lower bound too high
        (
            {"threshold_config": {"0.6": [0.7, 0.8]}},
            "Threshold must be within bounds: !\\( 0.7 <= 0.6 <= 0.8 \\)",
        ),
        # comparison_operator is invalid
        (
            {"threshold_values": 0.6, "comparison_operator": "invalid"},
            'String "invalid" does not match any known comparison_operator',
        ),
        # collapse coordinate is not percentile or realization
        (
            {"threshold_values": 0.6, "collapse_coord": "kittens"},
            "Can only collapse over one or a combination of realization",
        ),
        # collapse coordinate is a list with invalid options
        (
            {"threshold_values": 0.6, "collapse_coord": ["kittens", "puppies"]},
            "Can only collapse over one or a combination of realization",
        ),
        # collapse coordinate is a list containing both percentile and realization
        (
            {"threshold_values": 0.6, "collapse_coord": ["percentile", "realization"]},
            "Cannot collapse over both percentile and realization coordinates.",
        ),
        # threshold values provided as argument and via config
        (
            {"threshold_values": 0.6, "threshold_config": {"0.6"}},
            "threshold_config and threshold_values are mutually exclusive arguments",
        ),
        # at least one set means of defining the thresholds must be used.
        ({}, "One of threshold_config or threshold_values must be provided."),
    ],
)
def test_init(kwargs, exception):
    """Test the exceptions raised by the __init__ method."""
    with pytest.raises(ValueError, match=exception):
        Threshold(**kwargs)


def test_init_fill_mask():
    """Test the that a fill_mask argument is made into a float type within
    the __init__ method."""

    fill_masked = int(5)
    plugin = Threshold(threshold_values=[0], fill_masked=fill_masked)
    assert isinstance(plugin.fill_masked, float)


@pytest.mark.parametrize(
    "diagnostic,units", [("precipitation_rate", "mm/hr"), ("air_temperature", "K")]
)
def test__add_threshold_coord(default_cube, diagnostic, units):
    """Test the _add_threshold_coord method for diagnostics with
    different units."""

    ref_cube = default_cube.copy()
    default_cube.rename(diagnostic)
    default_cube.units = units
    plugin = Threshold(threshold_values=1)
    plugin.threshold_coord_name = default_cube.name()
    plugin._add_threshold_coord(default_cube, 1)

    assert default_cube.ndim == ref_cube.ndim
    if diagnostic == "air_temperature":
        assert diagnostic in [
            coord.standard_name for coord in default_cube.coords(dim_coords=False)
        ]
    else:
        assert diagnostic in [
            coord.long_name for coord in default_cube.coords(dim_coords=False)
        ]

    threshold_coord = default_cube.coord(diagnostic)
    assert threshold_coord.var_name == "threshold"
    assert threshold_coord.points[0] == 1
    assert threshold_coord.units == default_cube.units
    assert threshold_coord.dtype == np.float64


@pytest.mark.parametrize(
    "kwargs,n_realizations,n_times,data",
    [
        # A typical case with float inputs
        ({}, 1, 1, np.zeros(25, dtype=np.float32).reshape(5, 5)),
        # A case with integer inputs, where the data is converted to
        # float32 type, allowing for non-integer thresholded values,
        # i.e. due to the application of fuzzy thresholds.
        ({}, 1, 1, np.zeros(25, dtype=np.int8).reshape(5, 5)),
        # Test removal of realization coordinate if it is collapsed
        (
            {"collapse_coord": "realization"},
            2,
            1,
            np.zeros(18, dtype=np.int8).reshape(2, 3, 3),
        ),
        # Test time coordinate remains even if it is collapsed
        (
            {"collapse_coord": ["realization", "time"]},
            2,
            2,
            np.zeros(36, dtype=np.int8).reshape(2, 2, 3, 3),
        ),
    ],
)
def test_attributes_and_types(kwargs, custom_cube):
    """Test that the returned cube has the expected type and attributes."""

    expected_attributes = {
        "source": "Unit test",
        "institution": "Met Office",
        "title": "Post-Processed IMPROVER unit test",
    }
    default_kwargs = {"threshold_values": 12, "fuzzy_factor": (5 / 6)}
    default_kwargs.update(kwargs)
    plugin = Threshold(**default_kwargs)
    result = plugin(custom_cube)

    assert isinstance(result, Cube)
    assert result.dtype == np.float32
    assert not result.coords("realization")
    assert result.coords("time")
    for key, attribute in expected_attributes.items():
        assert result.attributes[key] == attribute


@pytest.mark.parametrize(
    "comparison_operator",
    ["gt", "lt", "ge", "le", ">", "<", ">=", "<=", "GT", "LT", "GE", "LE"],
)
@pytest.mark.parametrize("vicinity", [None, [4000], [3000, 6000]])
@pytest.mark.parametrize("threshold_values", [-0.5, 0, [0.2, 0.4]])
@pytest.mark.parametrize("threshold_units", ["mm/hr", "mm/day"])
def test_threshold_metadata(
    default_cube,
    threshold_coord,
    expected_cube_name,
    comparison_operator,
    vicinity,
    threshold_values,
    threshold_units,
):
    """ "Test that the metadata relating to the thresholding options, on both
    the cube and threshold coordinate is as expected. Many combinations of
    options are tested, including:

      - All ways of specifying the various comparators
      - With and without a single or multiple vicinity radii
      - Negative, zero, and multi-valued threshold values
      - Threshold units that match and differ from the input data
    """

    kwargs = {
        "threshold_values": threshold_values,
        "comparison_operator": comparison_operator,
        "vicinity": vicinity,
        "threshold_units": threshold_units,
    }
    ref_cube_name = default_cube.name()
    plugin = Threshold(**kwargs)
    result = plugin(default_cube)

    assert result.name() == expected_cube_name.format(cube_name=ref_cube_name)
    assert result.coord(var_name="threshold") == threshold_coord

    if vicinity:
        expected_vicinity = DimCoord(
            vicinity, long_name="radius_of_vicinity", units="m"
        )
        assert result.coord("radius_of_vicinity") == expected_vicinity


def test_vicinity_as_empty_list(default_cube):
    """Test that a vicinity set as an empty list does not result in
    a vicinity diagnostic or any exceptions. Tested for a deterministic,
    single realization, and multi-realization cube."""

    expected_cube_name = "probability_of_precipitation_rate_above_threshold"
    plugin = Threshold(threshold_values=0.5, vicinity=list())
    result = plugin(default_cube)

    assert result.name() == expected_cube_name


@pytest.mark.parametrize("collapse", (None, ["realization"], ["realization", "time"]))
@pytest.mark.parametrize("comparator", ("gt", "lt", "le", "ge"))
@pytest.mark.parametrize(
    "kwargs,expected_single_value,expected_multi_value",
    [
        # Note that the expected values given here are for
        # thresholding with a ">" or ">=" comparator. If the comparator
        # is "<" or "<=", the values are inverted as (1 - expected_value).
        # diagnostic value(s) above threshold value
        ({"threshold_values": 0.1}, 1.0, [1.0, 1.0]),
        # diagnostic value(s) below threshold value
        ({"threshold_values": 1.0}, 0.0, [0.0, 0.0]),
        # diagnostic value at threshold value, multi-realization values either side,
        # fuzziness applied
        ({"threshold_values": 0.5, "fuzzy_factor": 0.5}, 0.5, [0.4, 0.6]),
        # diagnostic value(s) above threshold value, fuzziness applied
        ({"threshold_values": 0.4, "fuzzy_factor": 0.5}, 0.75, [0.625, 0.875]),
        # diagnostic value(s) below threshold value, fuzziness applied
        ({"threshold_values": 0.8, "fuzzy_factor": 0.5}, 0.125, [0.0625, 0.1875]),
        # diagnostic value(s) below the fuzzy bounds
        ({"threshold_values": 2.0, "fuzzy_factor": 0.5}, 0.0, [0.0, 0.0]),
        # diagnostic value(s) above the fuzzy bounds
        ({"threshold_values": 0.2, "fuzzy_factor": 0.8}, 1.0, [1.0, 1.0]),
        # asymmetric fuzzy bounds applied, diagnostic value(s) below threshold value
        ({"threshold_config": {"0.6": [0.4, 0.7]}}, 0.25, [0.125, 0.375]),
        # asymmetric fuzzy bounds applied, diagnostic value(s) above threshold value
        ({"threshold_config": {"0.4": [0.0, 0.6]}}, 0.75, [0.625, 0.875]),
        # asymmetric fuzzy bounds applied, diagnostic value(s) at threshold value
        ({"threshold_config": {"0.5": [0.4, 0.7]}}, 0.5, [0.25, 0.625]),
        # fuzzy bounds set to "None" in the threshold config
        ({"threshold_config": {"1.0": "None"}}, 0.0, [0.0, 0.0]),
    ],
)
def test_expected_values(default_cube, kwargs, collapse, comparator, expected_result):
    """Test that thresholding yields the expected data values for
    different configurations. Variations tried here are:

      - Different threshold values relative to the diagnostic value(s)
      - Use of fuzzy thresholds, specified in different ways.
      - Deterministic, single realization, multi-realization, and multi-
        realization/multi-time inputs
      - Different threshold comparators. Note that the tests have been
        engineered such that there are no cases where the difference
        between "ge" and "gt", or "le" and "lt" are signficant, these
        are tested elsewhere.
      - Collapsing and not collapsing the realization coordinate when
        present.
    """

    local_kwargs = kwargs.copy()

    if collapse and set(collapse).issubset(
        [crd.name() for crd in default_cube.coords(dim_coords=True)]
    ):
        local_kwargs.update({"collapse_coord": collapse})
    elif collapse:
        # Skip tests in which the coordinates to be collapsed are not
        # dimension coordinates of the input cube.
        pytest.skip()

    # Check the time coordinate returned is as expected even if the time
    # coordinate is being collapsed.
    expected_time_coord = default_cube.coord("time")
    if collapse and "time" in collapse:
        expected_time_coord = expected_time_coord.collapsed()
        expected_time_coord.points = expected_time_coord.bounds[0][-1]

    local_kwargs.update({"comparison_operator": comparator})
    plugin = Threshold(**local_kwargs)
    result = plugin(default_cube)

    assert result.data.shape == expected_result.shape
    np.testing.assert_array_almost_equal(result.data, expected_result)
    assert result.coord("time") == expected_time_coord


@pytest.mark.parametrize(
    "kwargs,n_realizations,n_times,data,expected_result",
    [
        # diagnostic value at threshold value, no fuzziness, various comparators
        (
            {"threshold_values": 0.0, "comparison_operator": "gt"},
            1,
            1,
            np.zeros((3, 3)),
            np.zeros((3, 3), dtype=np.float32),
        ),
        (
            {"threshold_values": 0.0, "comparison_operator": "lt"},
            1,
            1,
            np.zeros((3, 3)),
            np.zeros((3, 3), dtype=np.float32),
        ),
        (
            {"threshold_values": 0.0, "comparison_operator": "ge"},
            1,
            1,
            np.zeros((3, 3)),
            np.ones((3, 3), dtype=np.float32),
        ),
        (
            {"threshold_values": 0.0, "comparison_operator": "le"},
            1,
            1,
            np.zeros((3, 3)),
            np.ones((3, 3), dtype=np.float32),
        ),
        # diagnostic value at threshold value of 0, with fuzziness bounding.
        # All comparators will give the same result for this symmetric case.
        (
            {"threshold_config": {"0.": [-1, 1]}, "comparison_operator": "gt"},
            1,
            1,
            np.zeros((3, 3)),
            np.full((3, 3), 0.5, dtype=np.float32),
        ),
        # negative diagnostic value above negative threshold with fuzziness.
        (
            {"threshold_config": {"-1.": [-2, 0]}, "comparison_operator": "gt"},
            1,
            1,
            np.ones((3, 3)) * -0.5,
            np.ones((3, 3), dtype=np.float32) * 0.75,
        ),
        # diagnostic value above a negative threshold value
        (
            {"threshold_values": -1.0},
            1,
            1,
            np.zeros((3, 3)),
            np.ones((3, 3), dtype=np.float32),
        ),
        # masked inputs, with outputs retaining masking
        (
            {"threshold_values": 1.0},
            1,
            1,
            np.ma.masked_array([[2.0, 0], [3.0, 3.0]], mask=[[0, 0], [1, 1]]),
            np.ma.masked_array(
                [[1.0, 0], [0, 0]], mask=[[0, 0], [1, 1]], dtype=np.float32
            ),
        ),
        # masked inputs, with masked values filled and return type an unmasked array
        (
            {"threshold_values": 1.0, "fill_masked": np.inf},
            1,
            1,
            np.ma.masked_array([[2.0, 0], [3, 3]], mask=[[0, 0], [1, 1]]),
            np.array([[1.0, 0], [1.0, 1.0]], dtype=np.float32),
        ),
        # masked inputs with fuzzy bounds
        (
            {"threshold_values": 0.5, "fuzzy_factor": 0.5},
            1,
            1,
            np.ma.masked_array([[0.5, 0], [3.0, 3.0]], mask=[[0, 0], [1, 1]]),
            np.ma.masked_array(
                [[0.5, 0], [0, 0]], mask=[[0, 0], [1, 1]], dtype=np.float32
            ),
        ),
        # multi-realization, masked in one realization, returns expected value when the
        # realization coordinate is collapsed. This value is 1 as the second
        # realization is excluded from the averaging due to masking.
        (
            {"threshold_values": 0.5, "collapse_coord": "realization"},
            2,
            1,
            np.ma.masked_array(
                [
                    [[1.0, 0], [0, 0]],  # 2x2 realization 0
                    [[0, 0], [0, 0]],
                ],  # 2x2 realization 1
                mask=[[[0, 0], [0, 0]], [[1, 0], [0, 0]]],  # 2x2 realization 0
            ),  # 2x2 realization 1
            np.array([[1.0, 0], [0, 0]], dtype=np.float32),
        ),
        # multi-realization, multi-time, masked in one realization, returns
        # expected value when the realization and time coordinates are
        # collapsed. This value is 1 as the second realization is excluded
        # from the averaging due to masking.
        (
            {"threshold_values": 0.5, "collapse_coord": ["realization", "time"]},
            2,
            2,
            np.ma.masked_array(
                [
                    [[1.0, 0], [0, 0]],  # 2x2 realization 0
                    [[0, 0], [0, 0]],
                ],  # 2x2 realization 1
                mask=[[[0, 0], [0, 0]], [[1, 0], [0, 0]]],  # 2x2 realization 0
            ),  # 2x2 realization 1
            np.array([[1.0, 0], [0, 0]], dtype=np.float32),
        ),
        # Vicinity processing, with one central value exceeding the threshold
        # resulting in the whole domain returning probabilities of 1
        (
            {"threshold_values": 0.5, "vicinity": 3000},
            1,
            1,
            np.array([[0, 0, 0], [0, 1.0, 0.0], [0, 0, 0]]),
            np.ones((3, 3), dtype=np.float32),
        ),
        # Vicinity processing, with one corner value exceeding the threshold
        # resulting in neighbouring cells probabilities of 1 within the limit
        # of the defined vicinity radius (2x2km grid cells)
        (
            {"threshold_values": 0.5, "vicinity": 3000},
            1,
            1,
            np.array([[1.0, 0, 0], [0, 0, 0.0], [0, 0, 0]]),
            np.array([[1.0, 1.0, 0], [1.0, 1.0, 0.0], [0, 0, 0]], dtype=np.float32),
        ),
        # Vicinity processing, with one masked value exceeding the threshold.
        # This masked point is not considered, and so zeros are returned
        # at neighbouring points within the vicinity radius. The masking is
        # preserved in the resulting thresholded data.
        (
            {"threshold_values": 0.5, "vicinity": 3000},
            1,
            1,
            np.ma.masked_array(
                [[0, 2.0, 0], [0, 0, 0.0], [0, 0, 0]],
                mask=[[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            ),
            np.ma.masked_array(
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                mask=[[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                dtype=np.float32,
            ),
        ),
        # multiple thresholds and multiple vicinities; input a 5x5 grid
        # with value 1 at the top-left point. For the first threshold
        # probabilities of 1 are returned in the top left corner over
        # 2x2 cells for the first vicinity, and over 3x3 cells for the
        # second vicinity. The second threshold returns only 0 probabilities.
        (
            {"threshold_values": [0.5, 1.5], "vicinity": [3000, 5000]},
            1,
            1,
            np.r_[[1], [0] * 24].reshape((5, 5)).astype(np.float32),
            np.stack(
                [
                    np.stack(
                        [
                            np.r_[[1] * 2, [0] * 3, [1] * 2, [0] * 18].reshape((5, 5)),
                            np.r_[
                                [1] * 3, [0] * 2, [1] * 3, [0] * 2, [1] * 3, [0] * 12
                            ].reshape((5, 5)),
                        ]
                    ),
                    np.zeros((2, 5, 5)),
                ]
            ).astype(np.float32),
        ),
        # Multiple thresholds applied.
        (
            {"threshold_values": [0.5, 1.5]},
            1,
            1,
            np.ones((3, 3)),
            np.stack(
                [np.ones((3, 3), dtype=np.float32), np.zeros((3, 3), dtype=np.float32)]
            ),
        ),
        # threshold value units are specified and differ from the data; if units matches
        # the returned probabilities would be 1, but taking into account the units
        # the returned probabilities are 0.
        (
            {"threshold_values": 0.5, "threshold_units": "m/s"},
            1,
            1,
            np.ones((3, 3)),
            np.zeros((3, 3), dtype=np.float32),
        ),
        # value of np.inf tested with an above and below comparator, as well as an
        # equivalence test with a threshold set to np.inf.
        (
            {"threshold_values": 1.0e6, "comparison_operator": "gt"},
            1,
            1,
            np.array([[np.inf, 0], [0, 0]]),
            np.array([[1.0, 0], [0, 0]], dtype=np.float32),
        ),
        (
            {"threshold_values": 1.0e6, "comparison_operator": "lt"},
            1,
            1,
            np.array([[np.inf, 0], [0, 0]]),
            np.array([[0, 1.0], [1.0, 1.0]], dtype=np.float32),
        ),
        (
            {"threshold_values": np.inf, "comparison_operator": "ge"},
            1,
            1,
            np.array([[np.inf, 0], [0, 0]]),
            np.array([[1.0, 0], [0, 0]], dtype=np.float32),
        ),
        # Data varying across time and realization dimensions. For the first
        # time, realization 0 (R0) contains all 2s, R1 contains all 1s. For the
        # second time R0 contains all 4s, R1 contains all 5s. The three tests
        # below demonstrate that result varies depending upon the dimension
        # that is collapsed. Collapsing both leads to a 3x3 array of 0.5.
        # Collapsing just the realizations leads to a 2x3x3 array with values
        # 0 for the first time, 1 for the second. Collapsing just the time
        # dimension leads to a 2x3x3 array with values 0.5 for both realizations.
        (
            {"threshold_values": [3], "collapse_coord": ["realization", "time"]},
            2,
            2,
            np.r_[[2] * 9, [1] * 9, [4] * 9, [5] * 9].reshape((2, 2, 3, 3)),
            np.full((3, 3), 0.5).astype(np.float32),
        ),
        # As above but only collapsing realizations
        (
            {"threshold_values": [3], "collapse_coord": ["realization"]},
            2,
            2,
            np.r_[[2] * 9, [1] * 9, [4] * 9, [5] * 9].reshape((2, 2, 3, 3)),
            np.stack([np.full((3, 3), 0), np.full((3, 3), 1)]).astype(np.float32),
        ),
        # As above but only collapsing time
        (
            {"threshold_values": [3], "collapse_coord": ["time"]},
            2,
            2,
            np.r_[[2] * 9, [1] * 9, [4] * 9, [5] * 9].reshape((2, 2, 3, 3)),
            np.full((2, 3, 3), 0.5).astype(np.float32),
        ),
    ],
)
def test_bespoke_expected_values(custom_cube, kwargs, expected_result):
    """Test that thresholding yields the expected data values for
    different configurations. Variations tried here are:

      - Different threshold values relative to the diagnostic value(s)
      - Use of fuzzy thresholds, specified in different ways.
      - Deterministic, single realization, and multi-realization inputs
      - Different threshold comparators. Note that the tests have been
        engineered such that there are no cases where the difference
        between "ge" and "gt", or "le" and "lt" are signficant, these
        are tested elsewhere.
      - Collapsing and not collapsing the realization coordinate when
        present.
    """
    plugin = Threshold(**kwargs)
    result = plugin(custom_cube)

    assert result.data.shape == expected_result.shape
    assert np.allclose(result.data, expected_result)
    assert type(result.data) == type(expected_result)
    assert result.data.dtype == expected_result.dtype
    if np.ma.is_masked(result.data):
        assert (result.data.mask == expected_result.mask).all()


@pytest.mark.parametrize(
    "kwargs,n_realizations,n_times,data,mask,expected_result",
    [
        # the land corner has a value of 1, this is not spread across the
        # other points within the vicinity as these are sea points.
        (
            {"threshold_values": 0.5, "vicinity": 3000},
            1,
            1,
            np.array([[0, 0, 1.0], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 1.0], [0, 0, 0], [0, 0, 0]], dtype=np.float32),
        ),
        # a sea corner has a value of 1, this is spread across the
        # other sea points within the vicinity, but not the central
        # point as this is land.
        (
            {"threshold_values": 0.5, "vicinity": 3000},
            1,
            1,
            np.array([[0, 0, 0], [0, 0, 0], [1.0, 0, 0]]),
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            np.array([[0, 0, 0], [1.0, 0, 0], [1.0, 1.0, 0]], dtype=np.float32),
        ),
        # a vicinity that is large enough to affect all points and a
        # complex mask to check land points are unaffected by the
        # spread of values from a sea point.
        (
            {"threshold_values": 0.5, "vicinity": 5000},
            1,
            1,
            np.array([[0, 0, 0], [0, 0, 0], [1.0, 0, 0]]),
            np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]]),
            np.array([[0, 1.0, 0], [1.0, 0, 1.0], [1.0, 0, 1.0]], dtype=np.float32),
        ),
    ],
)
def test_vicinity_with_landmask(custom_cube, landmask, kwargs, expected_result):
    """Test that the application of a maximum in neighbourhood method
    (vicinity) returns the expected values when a mask is provided
    to limit the spread of values to matching types, e.g. a high
    probability on land is only spread to other land points.
    """
    plugin = Threshold(**kwargs)
    result = plugin(custom_cube, landmask)

    assert result.data.shape == expected_result.shape
    assert np.allclose(result.data, expected_result)
    assert type(result.data) == type(expected_result)
    assert result.data.dtype == expected_result.dtype


@pytest.mark.parametrize(
    "n_realizations,n_times,data", [(1, 1, np.array([[0, np.nan], [1, 1]]))]
)
def test_nan_handling(custom_cube):
    """Test that an exception is raised if the input data contains an
    unmasked NaN."""

    plugin = Threshold(threshold_values=0.5)
    with pytest.raises(ValueError, match="NaN detected in input cube data"):
        plugin(custom_cube)


@pytest.mark.parametrize(
    "n_realizations,n_times,data,mask", [(1, 1, np.zeros((2, 2)), np.zeros((2, 2)))]
)
def test_landmask_no_vicinity(custom_cube, landmask):
    """Test that an exception is raised if a landmask is provided but
    vicinity processing is not being applied."""

    plugin = Threshold(threshold_values=0.5)
    msg = "Cannot apply land-mask cube without in-vicinity processing"
    with pytest.raises(ValueError, match=msg):
        plugin(custom_cube, landmask)


@pytest.mark.parametrize("n_realizations,n_times,data", [(1, 1, np.zeros((2, 2)))])
def test_cell_methods(custom_cube):
    """Test that cell methods are modified as expected when present
    on the input cube."""

    custom_cube.add_cell_method(CellMethod("max", coords="time"))
    plugin = Threshold(threshold_values=0.5)
    result = plugin(custom_cube)

    (cell_method,) = result.cell_methods
    assert cell_method.method == "max"
    assert cell_method.coord_names[0] == "time"
    assert cell_method.comments[0] == "of precipitation_rate"


@pytest.mark.parametrize("n_realizations,n_times,data", [(4, 1, np.zeros((4, 2, 2)))])
def test_percentile_collapse(custom_cube):
    """Test that a percentile coordinate can be collapsed to calculate
    an average. These percentiles must be equally spaced and centred on
    the 50th percentile ensure they may be considered as equivalent to
    realizations."""

    expected_result = np.zeros((2, 2, 2), dtype=np.float32)
    expected_result[0, 0, 0] = 0.25

    percentile_coord = DimCoord([20, 40, 60, 80], long_name="percentile", units="%")
    custom_cube.remove_coord("realization")
    custom_cube.add_dim_coord(percentile_coord, 0)
    custom_cube.data[0, 0, 0] = 1

    plugin = Threshold(threshold_values=[0.5, 1.5], collapse_coord="percentile")
    result = plugin(custom_cube)

    assert not result.coords("percentile")
    assert np.allclose(result.data, expected_result)
    assert (result.coord(var_name="threshold").points == [0.5, 1.5]).all()


def test_threshold_unit_conversion(default_cube):
    """Test threshold coordinate points after undergoing unit conversion.
    Specifically ensuring that small floating point values have no floating
    point precision errors after the conversion (float equality check with no
    tolerance). The data units are mm/hr, and the threshold units are
    micro-metres/hr."""

    plugin = Threshold(threshold_values=[0.03, 0.09, 0.1], threshold_units="um hr-1")
    result = plugin(default_cube)

    assert (
        result.coord(var_name="threshold").points
        == np.array([3e-5, 9.0e-05, 1e-4], dtype="float32")
    ).all()
    assert result.coord(var_name="threshold").units == "mm hr-1"
