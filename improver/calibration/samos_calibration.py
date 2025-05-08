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
    def prepare_data_for_gam(
        forecasts: Cube,
        additional_fields: Optional[CubeList] = None,
    ):
        """
        Convert input cubes in to a single, combined dataframe.
        """
        df = DataArray().from_iris(forecasts).to_dataframe()
        if additional_fields:
            for cube in additional_fields:
                new_df = DataArray().from_iris(cube).to_dataframe()
                df = merge(left=df, right=new_df, how='left')
        df.drop_duplicates(
            subset=['latitude', 'longitude', 'altitude'], keep='first', inplace=True
        )

        return df

    def process(
        self,
        forecasts: Cube,
        features: List,
        additional_fields: Optional[CubeList] = None,
    ):
        """
        Args:
            forecasts:
                Historic forecasts from the training dataset. This cube should contain
                a 'realization' dimension.
            features:
                The list of features. These must be either coordinates on the forecasts
                cube or share a name with a cube in additional_predictors. The index of
                each feature should match the indices use in model_specification.
            additional_fields:
                Additional fields to use as supplementary predictors.
        Returns:
            Fitted GAM models for the forecast mean and standard deviation.
        """
        if forecasts.coords('realization') is None:
            msg = ("The input forecast cube must contain a realization coordinate in "
                   "order to allow the calculation of means and standard deviations. "
                   f"The following coordinates were found: {forecasts.coords()}")
            raise ValueError(msg)

        # calculate forecast mean and standard deviation over the realization coordinate
        forecast_mean = collapsed(forecasts, "realization", iris.analysis.MEAN)
        forecast_sd = collapsed(forecasts, "realization", iris.analysis.STD_DEV)

        # create list to put fitted GAM models in
        output = []

        for forecast_stat in [forecast_mean, forecast_sd]:
            df = self.prepare_data_for_gam(forecast_stat, additional_fields)

            X_input = df[features].values
            y_input = df[forecasts.name()].values

            plugin = GAMFit(
                model_specification=self.model_specification,
                max_iter=self.max_iter,
                tol=self.tol,
                distribution=self.distribution,
                link=self.link,
                fit_intercept=self.fit_intercept,
            )

            output.append(plugin.process(X_input, y_input))

        return output
