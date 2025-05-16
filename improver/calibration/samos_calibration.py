# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
This module defines all the "plugins" specific to Standardised Anomaly Model Output
Statistics (SAMOS).
"""
import iris
import iris.pandas
from cf_units import Unit

from improver.calibration.utilities import check_predictor

iris.FUTURE.pandas_ndim = True
import pandas as pd

from improver import BasePlugin, PostProcessingPlugin
from improver.utilities.generalized_additive_models import GAMFit, GAMPredict
from improver.utilities.mathematical_operations import (
    CalculateClimateAnomalies,
    CalculateForecastValueFromClimateAnomaly
)
from improver.calibration.emos_calibration import (
    EstimateCoefficientsForEnsembleCalibration,
)
from iris.analysis import MEAN, STD_DEV
from iris.cube import Cube, CubeList
from iris.util import new_axis
from numpy.ma import masked_all_like
from typing import Dict, List, Optional, Union
from pandas import merge
import datetime


def prepare_data_for_gam(
    input_cube: Cube,
    additional_fields: Optional[CubeList] = None,
) -> pd.DataFrame:
    """
    Convert input cubes in to a single, combined dataframe.

    Args:
        input_cube: A cube of forecast or observation data.
        additional_fields: Additional cubes with points which can be matched with points
        in input_cube by matching spatial coordinates values.

    Returns:
        A pandas dataframe containing the following columns:
        1. A column with the same name as input_cube containing the original cube data
        2. A series of columns derived from the input_cube dimension coordinates
        3. A series of columns associated with any auxiliary coordinates (scalar or
        otherwise) of input_cube
        4. One column associated with each of the cubes in additional cubes, with column
        names matching the associated cube
    """
    # Convert to Pandas dataframe via Xarray as version 3.0.3 of Iris does not
    # handle converting cubes with more than 2 dimensions to dataframes.
    spatial_coords = [
        input_cube.coord(axis="x").name(),
        input_cube.coord(axis="y").name()
    ]
    df = iris.pandas.as_data_frame(
        input_cube,
        add_aux_coords=True,
        add_cell_measures =True,
        add_ancillary_variables=True
    )
    df.reset_index(inplace=True)
    if additional_fields:
        for cube in additional_fields:
            new_df = iris.pandas.as_data_frame(
                cube,
                add_aux_coords=True,
                add_cell_measures =True,
                add_ancillary_variables=True
            )
            new_df.reset_index(inplace=True)
            match_coords = spatial_coords.copy()
            match_coords.append(cube.name())
            df = merge(
                left=df,
                right=new_df[match_coords],
                how='left'
            )

    return df


def convert_dataframe_to_cube(
    df: pd.DataFrame,
    template_cube: Cube,
):
    """Function to convert a Pandas dataframe to Iris cube format. The result is a copy
    of template_cube with data from df. The diagnostic name and all of the dimension
    coordinates on template_cube must be columns of df.
    """
    dim_coords = [c.name() for c in template_cube.coords(dim_coords=True)]
    diagnostic = template_cube.name()

    df.set_index(dim_coords, inplace=True)
    df.sort_index(inplace=True)

    converted_cube = iris.pandas.as_cubes(df[[diagnostic]])[0]  # as_cubes() returns a cubelist
    result = template_cube.copy(data=converted_cube.data)

    return result


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
        modelling the standard deviation. These can then be used to convert forecasts or
        observations to climatological anomalies. This plugin should be run separately
        for forecast and observation data.

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
                for coord in [
                    "forecast_reference_time", "forecast_period", "blend_time"
                ]:
                    if input_cube.coords(coord):
                        removed_coords.append(input_cube.coord(coord).copy())
                        input_cube.remove_coord(coord)

                increment = increments[0]
                pad_width = 2  # No. of points to add to each end of time coordinate.

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

    def process(
        self,
        input_cube: Cube,
        features: List[str],
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
            df = prepare_data_for_gam(stat_cube, additional_fields)

            X_input = df[features].values
            y_input = df[input_cube.name()].values

            output.append(plugin.process(X_input, y_input))

        return output


class TrainEMOSForSAMOS(BasePlugin):
    """Class to calculate Ensemble Model Output Statistics (EMOS) coefficients to
    calibrate climate anomaly forecasts given training data including forecasts and
    verifying observations and four Generalized Additive Models (GAMs) which model:
    - forecast mean,
    - forecast standard deviation,
    - observation mean,
    - observation standard deviation.

    This class first calculates climatological means and standard deviations by
    predicting them from the input GAMs. Following this, the input forecasts and
    observations are converted to climatological anomalies using the predicted means
    and standard deviations. Finally, EMOS coefficients are calculated from the
    climatological anomaly training data.
    """
    def __init__(
        self,
        distribution: str,
        emos_kwargs: Optional[Dict] = None,
    ) -> None:
        """Information.

        Args:
            distribution:
                Name of distribution. Assume that a calibrated version of the
                climate anomaly forecast could be represented using this distribution.
            emos_kwargs: Keyword arguments accepted by the
                EstimateCoefficientsForEnsembleCalibration plugin. Should not contain
                a distribution argument.
        """
        self.distribution = distribution
        self.emos_kwargs = emos_kwargs if emos_kwargs else {}

    def process(
        self,
        historic_forecasts: Cube,
        truths: Cube,
        forecast_gams: List,
        truth_gams: List,
        gam_features: List[str],
        gam_additional_fields: Optional[CubeList] = None,
        emos_additional_fields: Optional[CubeList] = None,
        land_sea_mask: Optional[Cube] = None,
    ):
        diagnostic = historic_forecasts.name()

        forecast_df = prepare_data_for_gam(historic_forecasts, gam_additional_fields)
        truth_df = prepare_data_for_gam(truths, gam_additional_fields)

        # Calculate climatological means and standard deviations of forecasts and
        # observations using previously fitted GAMs.
        forecast_mean_pred = GAMPredict().process(
            forecast_gams[0], forecast_df[gam_features]
        )
        forecast_sd_pred = GAMPredict().process(
            forecast_gams[1], forecast_df[gam_features]
        )
        truth_mean_pred = GAMPredict().process(truth_gams[0], truth_df[gam_features])
        truth_sd_pred = GAMPredict().process(truth_gams[1], truth_df[gam_features])

        # Convert forecast and truth means and standard deviations into cubes
        forecast_df[diagnostic] = forecast_mean_pred
        forecast_mean = convert_dataframe_to_cube(forecast_df, historic_forecasts)

        forecast_df[diagnostic] = forecast_sd_pred
        forecast_sd = convert_dataframe_to_cube(forecast_df, historic_forecasts)

        truth_df[diagnostic] = truth_mean_pred
        truth_mean = convert_dataframe_to_cube(truth_df, truths)

        truth_df[diagnostic] = truth_sd_pred
        truth_sd = convert_dataframe_to_cube(truth_df, truths)

        # Convert forecasts and truths to climatological anomalies.
        forecast_ca = CalculateClimateAnomalies().process(
            historic_forecasts, forecast_mean, forecast_sd
        )
        truth_ca = CalculateClimateAnomalies().process(truths, truth_mean, truth_sd)

        plugin = EstimateCoefficientsForEnsembleCalibration(
            distribution=self.distribution, **self.emos_kwargs,
        )
        return plugin(forecast_ca, truth_ca, landsea_mask=land_sea_mask)
