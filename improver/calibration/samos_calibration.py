# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
This module defines all the "plugins" specific to Standardised Anomaly Model Output
Statistics (SAMOS).
"""

from typing import Dict, List, Optional, Sequence, Tuple

import iris
import iris.pandas
import pandas as pd
from iris.analysis import MEAN, STD_DEV
from iris.cube import Cube, CubeList
from iris.util import new_axis
from numpy import float32
from numpy.ma import masked_all_like
from pandas import merge

from improver import BasePlugin, PostProcessingPlugin
from improver.calibration.emos_calibration import (
    ApplyEMOS,
    EstimateCoefficientsForEnsembleCalibration,
    convert_to_realizations,
    generate_forecast_from_distribution,
    get_attribute_from_coefficients,
    get_forecast_type,
)
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.utilities.generalized_additive_models import GAMFit, GAMPredict
from improver.utilities.mathematical_operations import CalculateClimateAnomalies

# Setting to allow cubes with more than 2 dimensions to be converted to/from dataframes.
iris.FUTURE.pandas_ndim = True


def prepare_data_for_gam(
    input_cube: Cube,
    additional_fields: Optional[CubeList] = None,
) -> pd.DataFrame:
    """
    Convert input cubes in to a single, combined dataframe.

    Args:
        input_cube: A cube of forecast or observation data.
        additional_fields: Additional cubes with points which can be matched with points
        in input_cube by matching spatial coordinate values.

    Returns:
        A pandas dataframe containing the following columns:
        1. A column with the same name as input_cube containing the original cube data
        2. A series of columns derived from the input_cube dimension coordinates
        3. A series of columns associated with any auxiliary coordinates (scalar or otherwise) of input_cube
        4. One column associated with each of the cubes in additional cubes, with column names matching the associated cube

    """
    # Check if we are dealing with spot data.
    wmo_id_present = "wmo_id" in [c.name() for c in input_cube.coords()]
    if not wmo_id_present:
        spatial_coords = [
            input_cube.coord(axis="x").name(),
            input_cube.coord(axis="y").name(),
        ]

    df = iris.pandas.as_data_frame(
        input_cube,
        add_aux_coords=True,
        add_cell_measures=True,
        add_ancillary_variables=True,
    )
    df.reset_index(inplace=True)
    if additional_fields:
        for cube in additional_fields:
            new_df = iris.pandas.as_data_frame(
                cube,
                add_aux_coords=True,
                add_cell_measures=True,
                add_ancillary_variables=True,
            )
            new_df.reset_index(inplace=True)
            match_coords = ["wmo_id"] if wmo_id_present else spatial_coords.copy()
            match_coords.append(cube.name())
            df = merge(left=df, right=new_df[match_coords], how="left")

    return df


def convert_dataframe_to_cube(
    df: pd.DataFrame,
    template_cube: Cube,
) -> Cube:
    """Function to convert a Pandas dataframe to Iris cube format by using a template
    cube.

    Args:
        df: a Pandas dataframe which must contain at least the following columns:
            1. a column matching the name of template_cube
            2. a series of columns with names which match the dimension coordinates on
            template_cube
        template_cube: A cube which will provide the metadata for the output cube

    Returns:
        A copy of template_cube containing data from df.
    """
    dim_coords = [c.name() for c in template_cube.coords(dim_coords=True)]
    diagnostic = template_cube.name()

    indexed_df = df.set_index(dim_coords, inplace=False)
    indexed_df.sort_index(inplace=True)

    # The as_cubes() function returns a cubelist. In this case, the cubelist contains
    # only one element.
    converted_cube = iris.pandas.as_cubes(indexed_df[[diagnostic]])[0]
    result = template_cube.copy(data=converted_cube.data)

    return result


def get_climatological_stats(
    input_cube: Cube,
    gams: List,
    gam_features: List[str],
    additional_cubes: Optional[CubeList],
) -> Tuple[Cube, Cube]:
    """Function to predict climatological means and standard deviations given fitted
    GAMs for each statistic and cubes which can be used to construct a dataframe
    containing all required features for those GAMs.

    Args:
        input_cube
        gams: A list containing two fitted GAMs, the first for predicting the
            climatological mean of the locations in input_cube and the second
            predicting the climatoloigcal standard deviation
        gam_features:
            The list of features. These must be either coordinates on input_cube or
            share a name with a cube in additional_cubes. The index of each
            feature should match the indices used in model_specification.
        additional_cubes:
            Additional fields to use as supplementary predictors.

    Returns:
        A pair of cubes containing climatological mean and climatological standard
        deviation predictions respectively.
    """
    diagnostic = input_cube.name()

    df = prepare_data_for_gam(input_cube, additional_cubes)

    # Calculate climatological means and standard deviations using previously
    # fitted GAMs.
    mean_pred = GAMPredict().process(gams[0], df[gam_features])
    sd_pred = GAMPredict().process(gams[1], df[gam_features])

    # Convert means and standard deviations into cubes
    df[diagnostic] = mean_pred
    mean_cube = convert_dataframe_to_cube(df, input_cube)

    df[diagnostic] = sd_pred
    sd_cube = convert_dataframe_to_cube(df, input_cube)

    return mean_cube, sd_cube


class TrainGAMsForSAMOS(BasePlugin):
    """
    Class for fitting Generalised Additive Models (GAMs) to training data for use in
    a Standardised Anomaly Model Output Statistics (SAMOS) calibration scheme.

    Two GAMs are trained: one modelling the mean of the training data and one
    modelling the standard deviation. These can then be used to convert forecasts or
    observations to climatological anomalies. This plugin should be run separately
    for forecast and observation data.
    """

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
        Initialize the class.

        Args:
            model_specification:
                a list containing three items (in order):
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
    def calculate_statistic_by_rolling_window(input_cube: Cube, pad_width=2):
        """Function to calculate mean and standard deviation of input_cube using a
        rolling window calculation over the time coordinate.

        The input_cube time coordinate is padded by the pad_width at the beginning and
        end of the time coordinate, to ensure that the result of the rolling window
        calculation has the same shape as input_cube. Additionally, any missing time
        points in the input cube are filled with masked data, so that the rolling window
        is always taken over a fixed length time period.

        If there are any time periods for a given site where the rolling window would
        contain one or zero valid data points, then that site is removed from the cube
        prior to the calculation.
        """
        removed_coords = []
        # Pad the time coordinate of the input cube, then calculate the mean and
        # standard deviation using a rolling window over the time coordinate.
        time_coord = input_cube.coord("time")
        increments = time_coord.points[1:] - time_coord.points[:-1]
        min_increment = increments.min()
        if all(x % min_increment == 0 for x in increments):
            padded_cube = input_cube.copy()
            # Remove time related coordinates other than the coordinate called
            # "time" on the input cube in order to allow extension of the "time"
            # coordinate. These coordinates are saved and added back to the output
            # cubes.
            for coord in [
                "forecast_reference_time",
                "forecast_period",
                "blend_time",
            ]:
                if padded_cube.coords(coord):
                    removed_coords.append(padded_cube.coord(coord).copy())
                    padded_cube.remove_coord(coord)

            # Create slices of artificial cube data to pad the existing cube time
            # coordinate and fill any gaps. This ensures that all of the time points
            # are equally spaced and the padding ensures that the output of the
            # rolling window calculation is the same shape as the input cube.
            existing_points = time_coord.points
            desired_points = [
                x
                for x in range(
                    min(existing_points) - (min_increment * pad_width),
                    max(existing_points) + (min_increment * pad_width),
                    min_increment,
                )
            ]

            # Slice input_cube over time dimension so that the artificial cubes can be
            # concatenated correctly.
            padded_cubelist = iris.cube.CubeList([])
            for cslice in padded_cube.slices_over("time"):
                padded_cubelist.append(new_axis(cslice.copy(), "time"))

            # For each desired point which doesn't already correspond to a time point
            # on the cube, create a new cube slice with that time point with all data
            # masked.
            cslice = padded_cube.extract(iris.Constraint(time=time_coord.cell(0)))
            for point in desired_points:
                if point not in existing_points:
                    # Create a new cube slice with time point equal to point.
                    new_slice = cslice.copy()
                    new_slice.coord("time").points = point
                    new_slice = new_axis(new_slice, "time")
                    new_slice.data = masked_all_like(new_slice.data)
                    padded_cubelist.append(new_slice)
            padded_cube = padded_cubelist.concatenate_cube()

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

            # Add any removed time coordinates back on to the mean and standard
            # deviation cubes.
            for coord in removed_coords:
                time_dim = padded_cube.coord_dims("time")
                kwargs = {"data_dims": time_dim} if len(coord.points) > 1 else {}
                input_mean.add_aux_coord(coord, **kwargs)
                input_sd.add_aux_coord(coord, **kwargs)

            return input_mean, input_sd

    def calculate_cube_statistics(self, input_cube: Cube) -> CubeList:
        """Function to calculate mean and standard deviation of the input cube. If the
        cube has a realization dimension then statistics will be calculated by
        collapsing over this dimension. Otherwise, a rolling window calculation over
        the time dimension will be used.

        Args:
            input_cube:

        Returns:
            CubeList containing a mean cube and standard deviation cube.
        """
        # if input_cube.coords("realization"):
        #     # Calculate forecast mean and standard deviation over the realization
        #     # coordinate.
        #     input_mean = input_cube.collapsed("realization", MEAN)
        #     input_mean.remove_coord("realization")
        #     input_sd = input_cube.collapsed("realization", STD_DEV)
        #     input_sd.remove_coord("realization")
        # else:
        #     input_mean, input_sd = self.calculate_statistic_by_rolling_window(
        #         input_cube
        #     )

        collapse_coords = ["time"]
        for coord in ["realization", "percentile"]:
            if input_cube.coords(coord):
                collapse_coords.append(coord)
        input_mean = input_cube.collapsed(collapse_coords, MEAN)
        input_sd = input_cube.collapsed(collapse_coords, STD_DEV)
        # Remove the realization and percentile coordinates from the mean and standard
        # deviation cubes.
        for coord in collapse_coords:
            for cube in [input_mean, input_sd]:
                if cube.coords(coord):
                    cube.remove_coord(coord)

        return CubeList([input_mean, input_sd])

    def process(
        self,
        input_cube: Cube,
        features: List[str],
        additional_fields: Optional[CubeList] = None,
    ) -> List:
        """
        Function to fit GAMs to model the mean and standard deviation of the input_cube
        for use in SAMOS.

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
            A list containing fitted GAMs which model the input_cube mean and
            standard deviation.
        """
        if not input_cube.coords("realization"):
            if not input_cube.coords("time"):
                msg = (
                    "The input cube must contain at least one of a realization or time "
                    "coordinate in order to allow the calculation of means and "
                    "standard deviations. The following coordinates were found: "
                    f"{input_cube.coords()}."
                )
                raise ValueError(msg)
            elif len(input_cube.coord("time").points) == 1:
                msg = (
                    "The input cube does not contain a realization coordinate. In "
                    "order to calculate means and standard deviations the time "
                    "coordinate must contain more than one point. The following time "
                    f"coordinate was found: {input_cube.coord('time')}."
                )
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
        """Initialize the class.

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

    def climate_anomaly_emos(
        self,
        forecast_cubes: List[Cube],
        truth_cubes: List[Cube],
        additional_fields: Optional[CubeList] = None,
        landsea_mask: Optional[Cube] = None,
    ) -> CubeList:
        """Function to convert forecasts and truths to climate anomalies then calculate
        EMOS coefficients for the climate anomalies.

        Args:
            forecast_cubes:
                A list of three cubes: a cube containing historic forecasts, a cube
                containing climatological mean predictions and a cube containing
                climatoloigcal standard deviation predictions.
            truth_cubes:
                A list of three cubes: a cube containing historic truths, a cube
                containing climatological mean predictions and a cube containing
                climatoloigcal standard deviation predictions.
            additional_fields:
                Additional fields to use as supplementary predictors.
            landsea_mask:
                The optional cube containing a land-sea mask. If provided, only
                land points are used to calculate the coefficients. Within the
                land-sea mask cube land points should be specified as ones,
                and sea points as zeros.

        Returns:
            CubeList constructed using the coefficients provided and using
            metadata from the historic_forecasts cube. Each cube within the
            cubelist is for a separate EMOS coefficient e.g. alpha, beta,
            gamma, delta.
        """
        # Convert forecasts and truths to climatological anomalies.
        forecast_ca = CalculateClimateAnomalies(ignore_temporal_mismatch=True).process(
            *forecast_cubes
        )
        truth_ca = CalculateClimateAnomalies(ignore_temporal_mismatch=True).process(
            *truth_cubes
        )

        plugin = EstimateCoefficientsForEnsembleCalibration(
            distribution=self.distribution,
            **self.emos_kwargs,
        )
        return plugin.process(
            historic_forecasts=forecast_ca,
            truths=truth_ca,
            additional_fields=additional_fields,
            landsea_mask=landsea_mask,
        )

    def process(
        self,
        historic_forecasts: Cube,
        truths: Cube,
        forecast_gams: List,
        truth_gams: List,
        gam_features: List[str],
        gam_additional_fields: Optional[CubeList] = None,
        emos_additional_fields: Optional[CubeList] = None,
        landsea_mask: Optional[Cube] = None,
    ) -> CubeList:
        """Function to convert historic forecasts and truths to climatoligcal anomalies,
        then fit EMOS coefficients to these anomalies.

        Args:
            historic_forecasts:
                Historic forecasts from the training dataset.
            truths:
                Truths from the training dataset.
            forecast_gams:
                A list containing two fitted GAMs, the first for predicting the
                climatological mean of the locations in historic_forecasts and the
                second predicting the climatoloigcal standard deviation.
            truth_gams:
                A list containing two fitted GAMs, the first for predicting the
                climatological mean of the locations in truths and the second
                predicting the climatoloigcal standard deviation.
            gam_features:
                The list of features. These must be either coordinates on input_cube or
                share a name with a cube in additional_cubes. The index of each
                feature should match the indices used in model_specification.
            gam_additional_fields:
                Additional fields to use as supplementary predictors in the GAMs.
            emos_additional_fields:
                Additional fields to use as supplementary predictors in EMOS.
            landsea_mask:
                The optional cube containing a land-sea mask. If provided, only
                land points are used to calculate the EMOS coefficients. Within the
                land-sea mask cube land points should be specified as ones,
                and sea points as zeros.

        Returns:
            CubeList constructed using the coefficients provided and using
            metadata from the historic_forecasts cube. Each cube within the
            cubelist is for a separate EMOS coefficient e.g. alpha, beta,
            gamma, delta.
        """
        forecast_mean, forecast_sd = get_climatological_stats(
            historic_forecasts, forecast_gams, gam_features, gam_additional_fields
        )
        truth_mean, truth_sd = get_climatological_stats(
            truths, truth_gams, gam_features, gam_additional_fields
        )

        emos_coefficients = self.climate_anomaly_emos(
            forecast_cubes=[historic_forecasts, forecast_mean, forecast_sd],
            truth_cubes=[truths, truth_mean, truth_sd],
            additional_fields=emos_additional_fields,
            landsea_mask=landsea_mask,
        )

        return emos_coefficients


class ApplySAMOS(PostProcessingPlugin):
    """
    Class to calibrate an input forecast using SAMOS given the following inputs:
    - Two GAMs which model, respectively, the climatological mean and standard
    deviation of the forecast. This allows the forecast to be converted to
    climatological anomalies.
    - A set of EMOS coefficients which can be applied to correct the climatological
    anomalies.
    """

    def __init__(self, percentiles: Optional[Sequence] = None):
        """Initialize class.

        Args:
            percentiles:
                The set of percentiles used to create the calibrated forecast.
        """
        self.percentiles = [float32(p) for p in percentiles] if percentiles else None

    def process(
        self,
        forecast: Cube,
        forecast_gams: List,
        gam_features: List[str],
        emos_coefficients: CubeList,
        gam_additional_fields: Optional[CubeList] = None,
        emos_additional_fields: Optional[CubeList] = None,
        prob_template: Optional[Cube] = None,
        realizations_count: Optional[int] = None,
        ignore_ecc_bounds: bool = True,
        tolerate_time_mismatch: bool = False,
        predictor: str = "mean",
        randomise: bool = False,
        random_seed: Optional[int] = None,
    ):
        """Calibrate input forecast using GAMs to convert the forecast to climatological
         anomalies and pre-calculated EMOS coefficients to apply to those anomalies.

        Args:
            forecast:
                Uncalibrated forecast as probabilities, percentiles or
                realizations.
            forecast_gams:
                A list containing two fitted GAMs, the first for predicting the
                climatological mean of the locations in historic_forecasts and the
                second predicting the climatological standard deviation.
            gam_features:
                The list of features. These must be either coordinates on input_cube or
                share a name with a cube in additional_cubes. The index of each
                feature should match the indices used in model_specification.
            emos_coefficients:
                EMOS coefficients.
            gam_additional_fields:
                Additional fields to use as supplementary predictors in the GAMs.
            emos_additional_fields:
                Additional fields to use as supplementary predictors in EMOS.
            prob_template:
                A cube containing a probability forecast that will be used as
                a template when generating probability output when the input
                format of the forecast cube is not probabilities i.e. realizations
                or percentiles.
            realizations_count:
                Number of realizations to use when generating the intermediate
                calibrated forecast from probability or percentile inputs
            ignore_ecc_bounds:
                If True, allow percentiles from probabilities to exceed the ECC
                bounds range.  If input is not probabilities, this is ignored.
            tolerate_time_mismatch:
                If True, tolerate a mismatch in validity time and forecast
                period for coefficients vs forecasts. Use with caution!
            predictor:
                Predictor to be used to calculate the location parameter of the
                calibrated distribution.  Value is "mean" or "realizations".
            randomise:
                Used in generating calibrated realizations.  If input forecast
                is probabilities or percentiles, this is ignored.
            random_seed:
                Used in generating calibrated realizations.  If input forecast
                is probabilities or percentiles, this is ignored.

        Returns:
            Calibrated forecast in the form of the input (ie probabilities
            percentiles or realizations).
        """
        input_forecast_type = get_forecast_type(forecast)

        forecast_as_realizations = forecast.copy()
        if input_forecast_type != "realizations":
            forecast_as_realizations = convert_to_realizations(
                forecast.copy(), realizations_count, ignore_ecc_bounds
            )

        forecast_mean, forecast_sd = get_climatological_stats(
            forecast_as_realizations, forecast_gams, gam_features, gam_additional_fields
        )
        forecast_ca = CalculateClimateAnomalies(ignore_temporal_mismatch=True).process(
            diagnostic_cube=forecast_as_realizations,
            mean_cube=forecast_mean,
            std_cube=forecast_sd,
        )

        # Returns parameters which describe a climate anomaly distribution.
        location_parameter, scale_parameter = ApplyEMOS(
            percentiles=self.percentiles
        ).process(
            forecast=forecast_ca,
            coefficients=emos_coefficients,
            additional_fields=emos_additional_fields,
            prob_template=prob_template,
            realizations_count=realizations_count,
            ignore_ecc_bounds=ignore_ecc_bounds,
            tolerate_time_mismatch=tolerate_time_mismatch,
            predictor=predictor,
            randomise=randomise,
            random_seed=random_seed,
            return_parameters=True,
        )

        # The data in these cubes are identical along the realization dimensions.
        forecast_mean = next(forecast_mean.slices_over("realization"))
        forecast_sd = next(forecast_sd.slices_over("realization"))

        # Transform location and scale parameters so that they represent a distribution
        # in the units of the original forecast, rather than climatological anomalies.
        forecast_units = (
            forecast.units
            if input_forecast_type != "probabilities"
            else find_threshold_coordinate(forecast).units
        )
        location_parameter.data = (
            location_parameter.data * forecast_sd.data
        ) + forecast_mean.data
        location_parameter.units = forecast_units

        # The scale parameter returned by ApplyEMOS is the standard deviation for a
        # normal distribution. To get the desired standard deviation in
        # realization/percentile space we must multiply by the estimated forecast
        # standard deviation.
        scale_parameter.data = scale_parameter.data * forecast_sd.data
        scale_parameter.units = forecast_units

        # Generate output in desired format from distribution.
        self.distribution = {
            "name": get_attribute_from_coefficients(emos_coefficients, "distribution"),
            "location": location_parameter,
            "scale": scale_parameter,
            "shape": get_attribute_from_coefficients(
                emos_coefficients, "shape_parameters", optional=True
            ),
        }

        template = prob_template if prob_template else forecast
        result = generate_forecast_from_distribution(
            self.distribution, template, self.percentiles, randomise, random_seed
        )

        return result
