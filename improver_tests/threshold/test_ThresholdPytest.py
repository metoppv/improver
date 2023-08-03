# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
"""Unit tests for the threshold.Threshold plugin."""


import numpy as np
import pytest
from iris.coords import CellMethod, DimCoord
from iris.cube import Cube

from improver.threshold import Threshold as Threshold


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
            "Can only collapse over a realization coordinate or a percentile",
        ),
    ],
)
def test_init(kwargs, exception):
    with pytest.raises(ValueError, match=exception):
        Threshold(**kwargs)


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
    "n_realizations,data",
    [
        # A typical case with float inputs
        (1, np.zeros((25), dtype=np.float32).reshape(5, 5)),
        # A case with integer inputs, where the data is converted to
        # float32 type, allowing for non-integer thresholded values,
        # i.e. due to the application of fuzzy thresholds.
        (1, np.zeros((25), dtype=np.int8).reshape(5, 5)),
    ],
)
def test_attributes_and_types(custom_cube, n_realizations, data):
    """Test that the returned cube has the expected type and attributes."""

    expected_attributes = {
        "source": "Unit test",
        "institution": "Met Office",
        "title": "Post-Processed IMPROVER unit test",
    }
    plugin = Threshold(threshold_values=12, fuzzy_factor=(5 / 6))
    result = plugin(custom_cube)

    assert isinstance(result, Cube)
    assert result.dtype == np.float32
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
    """"Test that the metadata relating to the thresholding options, on both
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

    if vicinity is not None:
        expected_vicinity = DimCoord(
            vicinity, long_name="radius_of_vicinity", units="m"
        )
        assert result.coord("radius_of_vicinity") == expected_vicinity


@pytest.mark.parametrize("collapse", (False, True))
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
    ],
)
def test_expected_values(default_cube, kwargs, collapse, comparator, expected_result):
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

    local_kwargs = kwargs.copy()

    if collapse and default_cube.coords("realization", dim_coords=True):
        local_kwargs.update({"collapse_coord": "realization"})
    elif collapse:
        # No need to repeat tests in which collapse_coord is not used.
        pytest.skip()

    local_kwargs.update({"comparison_operator": comparator})
    plugin = Threshold(**local_kwargs)
    result = plugin(default_cube)

    assert result.data.shape == expected_result.shape
    assert np.allclose(result.data, expected_result)


@pytest.mark.parametrize(
    "kwargs,n_realizations,data,expected_result",
    [
        # diagnostic value at threshold value, no fuzziness, various comparators
        (
            {"threshold_values": 0.0, "comparison_operator": "gt"},
            1,
            np.zeros((3, 3)),
            np.zeros((3, 3), dtype=np.float32),
        ),
        (
            {"threshold_values": 0.0, "comparison_operator": "lt"},
            1,
            np.zeros((3, 3)),
            np.zeros((3, 3), dtype=np.float32),
        ),
        (
            {"threshold_values": 0.0, "comparison_operator": "ge"},
            1,
            np.zeros((3, 3)),
            np.ones((3, 3), dtype=np.float32),
        ),
        (
            {"threshold_values": 0.0, "comparison_operator": "le"},
            1,
            np.zeros((3, 3)),
            np.ones((3, 3), dtype=np.float32),
        ),
        # diagnostic value at threshold value of 0, with fuzziness bounding.
        # All comparators will give the same result for this symmetric case.
        (
            {"threshold_config": {"0.": [-1, 1]}, "comparison_operator": "gt"},
            1,
            np.zeros((3, 3)),
            np.ones((3, 3), dtype=np.float32) * 0.5,
        ),
        # negative diagnostic value above negative threshold with fuzziness.
        (
            {"threshold_config": {"-1.": [-2, 0]}, "comparison_operator": "gt"},
            1,
            np.ones((3, 3)) * -0.5,
            np.ones((3, 3), dtype=np.float32) * 0.75,
        ),
        # diagnostic value above a negative threshold value
        (
            {"threshold_values": -1.0},
            1,
            np.zeros((3, 3)),
            np.ones((3, 3), dtype=np.float32),
        ),
        # masked inputs, with outputs retaining masking
        (
            {"threshold_values": 1.0},
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
            np.ma.masked_array([[2.0, 0], [3, 3]], mask=[[0, 0], [1, 1]]),
            np.array([[1.0, 0], [1.0, 1.0]], dtype=np.float32),
        ),
        # masked inputs with fuzzy bounds
        (
            {"threshold_values": 0.5, "fuzzy_factor": 0.5},
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
            np.array([[0, 0, 0], [0, 1.0, 0.0], [0, 0, 0]]),
            np.ones((3, 3), dtype=np.float32),
        ),
        # Vicinity processing, with one corner value exceeding the threshold
        # resulting in neighbourhing cells probabilities of 1 within the limit
        # of the defined vicinity radius (2x2km grid cells)
        (
            {"threshold_values": 0.5, "vicinity": 3000},
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
            np.r_[[1], [0] * 24].reshape((5, 5)),
            np.stack(
                [
                    np.stack(
                        [
                            np.r_[[1] * 2, [0] * 3, [1] * 2, [0] * 18]
                            .reshape((5, 5))
                            .astype(np.float32),
                            np.r_[[1] * 3, [0] * 2, [1] * 3, [0] * 2, [1] * 3, [0] * 12]
                            .reshape((5, 5))
                            .astype(np.float32),
                        ]
                    ),
                    np.stack(
                        [
                            np.zeros((5, 5), dtype=np.float32),
                            np.zeros((5, 5), dtype=np.float32),
                        ]
                    ),
                ]
            ),
        ),
        # Multiple thresholds applied.
        (
            {"threshold_values": [0.5, 1.5]},
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
            np.ones((3, 3)),
            np.zeros((3, 3), dtype=np.float32),
        ),
    ],
)
def test_bespoke_expected_values(
    custom_cube, kwargs, n_realizations, data, expected_result
):
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


@pytest.mark.parametrize("n_realizations,data", [(1, np.array([[0, np.nan], [1, 1]]))])
def test_nan_handling(custom_cube, n_realizations, data):
    """Test that an exception is raised if the input data contains an
    unmasked NaN."""

    plugin = Threshold(threshold_values=0.5)
    with pytest.raises(ValueError, match="NaN detected in input cube data"):
        plugin(custom_cube)


@pytest.mark.parametrize("n_realizations,data", [(1, np.zeros((2, 2)))])
def test_cell_methods(custom_cube, n_realizations, data):
    """Test that cell methods are modified as expected when present
    on the input cube."""

    custom_cube.add_cell_method(CellMethod("max", coords="time"))
    plugin = Threshold(threshold_values=0.5)
    result = plugin(custom_cube)

    (cell_method,) = result.cell_methods
    assert cell_method.method == "max"
    assert cell_method.coord_names[0] == "time"
    assert cell_method.comments[0] == "of precipitation_rate"


@pytest.mark.parametrize("n_realizations,data", [(4, np.zeros((4, 2, 2)))])
def test_percentile_collapse(custom_cube, n_realizations, data):
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
