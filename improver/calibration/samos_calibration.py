# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
This module defines all the "plugins" specific to Standardised Anomaly Model Output
Statistics (SAMOS).
"""
import iris

from improver import BasePlugin, PostProcessingPlugin
from improver.utilities.statistical import GAMFit, GAMPredict
from improver.utilities.mathematical_operations import (
    CalculateClimateAnomalies,
    CalculateForecastValueFromClimateAnomaly
)
from iris.analysis import MEAN, STD_DEV
from iris.cube import Cube, CubeList
from iris.util import new_axis
from numpy.ma import masked_all_like
from typing import Dict, List, Optional
from xarray import DataArray
from pandas import merge


class TrainGAMsForSAMOS(BasePlugin):
    def __init__(
        self,
        model_specification: Dict,
        max_iter: int = 100,
        tol: float = 0.0001,
        distribution: str = "normal",
        link: str = "identity",
        fit_intercept: bool = True,
    ):
        """
        Class for fitting Generalised Additive Models (GAMs) to training data for use in
        a Standardised Anomaly Model Output Statistics (SAMOS) calibration scheme.

        Two GAMs are trained: one modelling the mean of the training data and one
        modelling the standard deviation.

        Args:
            model_specification:
                a list containing lists of three items (in order):
                    1. a string containing a single pyGAM term; one of 'l' (linear),
                    's' (spline), 'te' (tensor), or 'f' (factor)
                    2. a list of integers which correspond to the features to be
                    included in that term
                    3. a dictionary of kwargs to be included when defining the term
            max_iter:
                a pyGAM argument which determines the maximum iterations allowed when
                fitting the GAM
            tol:
                a pyGAM argument determining the tolerance used to define the stopping
                criteria
            distribution:
                a pyGAM argument determining the distribution to be used in the model
            link:
                a pyGAM argument determining the link function to be used in the model
            fit_intercept:
                a pyGAM argument determining whether to include an intercept term in
                the model
        """
        self.model_specification = model_specification
        self.max_iter = max_iter
        self.tol = tol
        self.distribution = distribution
        self.link = link
        self.fit_intercept = fit_intercept

    @staticmethod
    def calculate_cube_statistics(input_cube: Cube) -> CubeList:
        """Function to calculate mean and standard deviation of the input cube. If the
        cube has a realization dimension then statistics will be calculated by
        collapsing over this dimension. Otherwise, a rolling window calculation over
        the time dimension will be used.

        Returns:
            CubeList containing a mean cube and standard deviation cube.
        """
        removed_coords = []
        if input_cube.coords("realization"):
            # Calculate forecast mean and standard deviation over the realization
            # coordinate.
            input_mean = input_cube.collapsed("realization", MEAN)
            input_mean.remove_coord("realization")
            input_sd = input_cube.collapsed("realization", STD_DEV)
            input_sd.remove_coord("realization")
        else:
            # Pad the time coordinate of the input cube, then calculate the mean and
            # standard deviation using a rolling window over the time coordinate.
            time_coord = input_cube.coord("time")
            increments = time_coord.points[1:] - time_coord.points[:-1]
            if len(set(increments)) > 1:
                msg = "AAAAHHHH"
                raise ValueError(msg)
            else:
                # Remove forecast reference time and forecast period coordinates if they
                # exist on the input cube in order to allow extension of the time
                # coordinate. These coords are saved and added back to the output cubes.
                for coord in ["forecast_reference_time", "forecast_period"]:
                    if input_cube.coords(coord):
                        removed_coords.append(input_cube.coord(coord).copy())
                        input_cube.remove_coord(coord)

                increment = increments[0]
                pad_width = 2  # No. of points to add to each end of time coordinate

                # Get first and last time slices in the input cube, to be used to create
                # cube slices which are before/after the first/last slice.
                first_slice = input_cube.extract(
                    iris.Constraint(time=time_coord.cell(0))
                )
                last_slice = input_cube.extract(
                    iris.Constraint(time=time_coord.cell(-1))
                )

                padded_cube = iris.cube.CubeList([input_cube])
                for i in range(pad_width):
                    # Create cubes with an earlier/later time to use to pad the input.
                    # All data in these cubes is masked to prevent them contributing to
                    # later calculations.
                    early_cube = first_slice.copy()
                    early_cube.coord("time").points = (early_cube.coord("time").points
                                                       - (i + 1) * increment)
                    early_cube = new_axis(early_cube, "time")
                    early_cube.data = masked_all_like(early_cube.data)

                    late_cube = last_slice.copy()
                    late_cube.coord("time").points = (late_cube.coord("time").points
                                                      + (i + 1) * increment)
                    late_cube = new_axis(late_cube, "time")
                    late_cube.data = masked_all_like(late_cube.data)

                    padded_cube.extend([early_cube, late_cube])

                padded_cube = padded_cube.concatenate_cube()

                # Calculate mean and standard deviation using rolling window over padded
                # time coordinate. Remove bounds from this coordinate in resulting cubes
                # as they aren't needed for later calculations.
                input_mean = padded_cube.rolling_window(
                    coord="time", aggregator=MEAN, window=(2 * pad_width) + 1
                )
                input_mean.coord("time").bounds = None
                input_sd = padded_cube.rolling_window(
                    coord="time", aggregator=STD_DEV, window=(2 * pad_width) + 1
                )
                input_sd.coord("time").bounds = None
                if removed_coords:
                    for coord in removed_coords:
                        kwargs = {"data_dims": 0} if len(coord.points) > 1 else {}
                        input_mean.add_aux_coord(coord, **kwargs)
                        input_sd.add_aux_coord(coord, **kwargs)

        return CubeList([input_mean, input_sd])

    @staticmethod
    def prepare_data_for_gam(
        input_cube: Cube,
        additional_fields: Optional[CubeList] = None,
    ):
        """
        Convert input cubes in to a single, combined dataframe.
        """
        # Convert to Pandas dataframe via Xarray as version 3.0.3 of Iris does not
        # handle converting cubes with more than 2 dimensions to dataframes.
        df = DataArray().from_iris(input_cube).to_dataframe()
        df.reset_index(inplace=True)
        if additional_fields:
            for cube in additional_fields:
                new_df = DataArray().from_iris(cube).to_dataframe()
                new_df.reset_index(inplace=True)
                df = merge(
                    left=df,
                    right=new_df[["latitude", "longitude", cube.name()]],
                    how='left'
                )

        return df

    def process(
        self,
        input_cube: Cube,
        features: List,
        additional_fields: Optional[CubeList] = None,
    ):
        """
        Args:
            input_cube:
                Historic forecasts or observations from the training dataset. Must
                contain at least one of:
                - a realization coordinate
                - a time coordinate with more than one point and equally spaced points
            features:
                The list of features. These must be either coordinates on input_cube or
                share a name with a cube in additional_predictors. The index of each
                feature should match the indices used in model_specification.
            additional_fields:
                Additional fields to use as supplementary predictors.
        Returns:
            Fitted GAM models for the input_cube mean and standard deviation.
        """
        if not input_cube.coords('realization'):
            if not input_cube.coords('time'):
                msg = (
                    "The input cube must contain at least one of a realization or time "
                    "coordinate in order to allow the calculation of means and "
                    "standard deviations. The following coordinates were found: "
                    f"{input_cube.coords()}.")
                raise ValueError(msg)
            elif len(input_cube.coord('time').points) == 1:
                msg = (
                    "The input cube does not contain a realization coordinate. In "
                    "order to calculate means and standard deviations the time "
                    "coordinate must contain more than one point. The following time "
                    f"coordinate was found: {input_cube.coord('time')}.")
                raise ValueError(msg)

        # Calculate mean and standard deviation from input cube.
        stat_cubes = self.calculate_cube_statistics(input_cube)

        # Create list to put fitted GAM models in.
        output = []

        # Initialize plugin used to fit GAMs.
        plugin = GAMFit(
            model_specification=self.model_specification,
            max_iter=self.max_iter,
            tol=self.tol,
            distribution=self.distribution,
            link=self.link,
            fit_intercept=self.fit_intercept,
        )

        for stat_cube in stat_cubes:
            df = self.prepare_data_for_gam(stat_cube, additional_fields)

            X_input = df[features].values
            y_input = df[input_cube.name()].values

            output.append(plugin.process(X_input, y_input))

        return output
