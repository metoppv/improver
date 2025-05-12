# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
This module defines all the "plugins" specific to Standardised Anomaly Model Output
Statistics (SAMOS).
"""
from improver import BasePlugin, PostProcessingPlugin
from improver.utilities.statistical import GAMFit, GAMPredict
from improver.utilities.mathematical_operations import (
    CalculateClimateAnomalies,
    CalculateForecastValueFromClimateAnomaly
)
from improver.utilities.cube_manipulation import collapsed
import iris
from iris.cube import Cube, CubeList
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

        Raises:
            ValueError if input_cube does not contain a realization or time dimension.
        """

    @staticmethod
    def prepare_data_for_gam(
        input_cube: Cube,
        additional_fields: Optional[CubeList] = None,
    ):
        """
        Convert input cubes in to a single, combined dataframe.
        """
        # Convert to Pandas dataframe via Xarray as version of Iris does not handle
        # converting cubes with more than 2 dimensions.
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
                Historic forecasts from the training dataset. This cube must contain
                a 'realization' dimension.
            features:
                The list of features. These must be either coordinates on the forecasts
                cube or share a name with a cube in additional_predictors. The index of
                each feature should match the indices used in model_specification.
            additional_fields:
                Additional fields to use as supplementary predictors.
        Returns:
            Fitted GAM models for the forecast mean and standard deviation.
        """
        if input_cube.coords('realization') is None:
            msg = ("The input forecast cube must contain a realization coordinate in "
                   "order to allow the calculation of means and standard deviations. "
                   f"The following coordinates were found: {input_cube.coords()}")
            raise ValueError(msg)

        # Calculate forecast mean and standard deviation over the realization coordinate
        input_mean = collapsed(input_cube, "realization", iris.analysis.MEAN)
        input_sd = collapsed(input_cube, "realization", iris.analysis.STD_DEV)

        # Create list to put fitted GAM models in
        output = []

        # Initialize plugin used to fit GAMs
        plugin = GAMFit(
            model_specification=self.model_specification,
            max_iter=self.max_iter,
            tol=self.tol,
            distribution=self.distribution,
            link=self.link,
            fit_intercept=self.fit_intercept,
        )

        for stat_cube in [input_mean, input_sd]:
            df = self.prepare_data_for_gam(stat_cube, additional_fields)

            X_input = df[features].values
            y_input = df[forecasts.name()].values

            output.append(plugin.process(X_input, y_input))

        return output
