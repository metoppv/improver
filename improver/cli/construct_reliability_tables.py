#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to construct reliability tables for use in reliability calibration."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube,
    truth_attribute,
    n_probability_bins: int = 5,
    single_value_lower_limit: bool = False,
    single_value_upper_limit: bool = False,
    aggregate_coordinates: cli.comma_separated_list = None,
):
    """Populate reliability tables for use in reliability calibration.

    Loads historical forecasts and gridded truths that are compared to build
    reliability tables. Reliability tables are returned as a cube with a
    leading threshold dimension that matches that of the forecast probability
    cubes and the thresholded truth.

    Args:
        cubes (list of iris.cube.Cube):
            A list of cubes containing the historical probability forecasts and
            corresponding truths used for calibration. These cubes must include
            the same diagnostic name in their names, and must both have
            equivalent threshold coordinates. The cubes will be distinguished
            using the user provided truth attribute.
        truth_attribute (str):
            An attribute and its value in the format of "attribute=value",
            which must be present on truth cubes.
        n_probability_bins (int):
            The total number of probability bins required in the reliability
            tables. If single value limits are turned on, these are included in
            this total. If using single_value_limits this value must be at
            least 3.
        single_value_lower_limit (bool):
            Mandates that the lowest bin should be single valued, with a small
            precision tolerance, defined as 1.0E-6. The bin is thus 0 to 1.0E-6.
        single_value_upper_limit (bool):
            Mandates that the highest bin should be single valued, with a small
            precision tolerance, defined as 1.0E-6. The bin is thus (1 - 1.0E-6) to 1.
        aggregate_coordinates (List[str]):
            An optional list of coordinates over which to aggregate the reliability
            calibration table using summation. This is equivalent to constructing
            then using aggregate-reliability-tables but with reduced memory
            usage due to avoiding large intermediate data.

    Returns:
        iris.cube.Cube:
            Reliability tables for the forecast diagnostic with a leading
            threshold coordinate.
    """
    from improver.calibration import split_forecasts_and_truth
    from improver.calibration.reliability_calibration import (
        ConstructReliabilityCalibrationTables,
    )

    forecast, truth, _ = split_forecasts_and_truth(cubes, truth_attribute)

    return ConstructReliabilityCalibrationTables(
        n_probability_bins=n_probability_bins,
        single_value_lower_limit=single_value_lower_limit,
        single_value_upper_limit=single_value_upper_limit,
    )(forecast, truth, aggregate_coordinates)
