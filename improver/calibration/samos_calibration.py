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

try:
    import pygam
except ModuleNotFoundError:
    # Define dummy class to avoid type hint errors.
    class pygam:
        def GAM(self):
            pass


from iris.analysis import MEAN, STD_DEV
from iris.cube import Cube, CubeList
from iris.util import new_axis
from numpy import array, clip, float32, int64, nan
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
from improver.ensemble_copula_coupling.utilities import get_bounds_of_distribution
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.utilities.cube_manipulation import collapse_realizations
from improver.utilities.generalized_additive_models import GAMFit, GAMPredict
from improver.utilities.mathematical_operations import CalculateClimateAnomalies

# Setting to allow cubes with more than 2 dimensions to be converted to/from dataframes.
iris.FUTURE.pandas_ndim = True


def prepare_data_for_gam(
    input_cube: Cube,
    additional_fields: Optional[CubeList] = None,
    unique_site_id_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert input cubes in to a single, combined dataframe.

    Each of the input cubes is converted to a pandas dataframe. The dataframe derived
    from input_cube then forms the left in a series of left dataframe joins with those
    derived from each cube in additional_fields. The x and y coordinates are used to
    perform this join. This means that the resulting combined dataframe will contain all
    of the sites/grid points in input_cube, but not any other sites/grid points in the
    additional_fields cubes.

    Args:
        input_cube:
            A cube of forecast or observation data.
        additional_fields:
            Additional cubes with points which can be matched with points
            in input_cube by matching spatial coordinate values.
        unique_site_id_key:
            If working with spot data and available, the name of the coordinate
            in the input cubes that contains unique site IDs, e.g. "wmo_id" if
            all sites have a valid wmo_id.

    Returns:
        A pandas dataframe with rows equal to the number of sites/grid points in
        input_cube and containing the following columns:
        1. A column with the same name as input_cube containing the original cube data
        2. A series of columns derived from the input_cube dimension coordinates
        3. A series of columns associated with any auxiliary coordinates (scalar or otherwise) of input_cube
        4. One column associated with each of the cubes in additional cubes, with column names matching the associated cube


    """
    df = iris.pandas.as_data_frame(
        input_cube,
        add_aux_coords=True,
        add_cell_measures=True,
        add_ancillary_variables=True,
    )
    df.reset_index(inplace=True)

    if additional_fields:
        # Check if we are dealing with spot data.
        site_data = "spot_index" in [c.name() for c in input_cube.coords()]

        # For site data we should use unique IDs wherever possible. As a
        # fallback we can match on latitude, longitude and altitude. We
        # need altitude to accommodate e.g. sites with a lower and upper
        # forecast altitude. If we are working with gridded data we use the
        # spatial coordinates of the input cube.
        if site_data and unique_site_id_key is not None:
            match_coords = [unique_site_id_key]
        elif site_data:
            match_coords = ["latitude", "longitude", "altitude"]
        else:
            match_coords = [
                input_cube.coord(axis="X").name(),
                input_cube.coord(axis="Y").name(),
            ]
        for cube in additional_fields:
            new_df = iris.pandas.as_data_frame(
                cube,
                add_aux_coords=True,
                add_cell_measures=True,
                add_ancillary_variables=True,
            )
            new_df.reset_index(inplace=True)
            df = merge(left=df, right=new_df[match_coords + [cube.name()]], how="left")

    return df


def convert_dataframe_to_cube(
    df: pd.DataFrame,
    template_cube: Cube,
) -> Cube:
    """Function to convert a Pandas dataframe to Iris cube format by using a template
    cube. The input template_cube provides all metadata for the output.

    Args:
        df: A Pandas dataframe which must contain at least the following columns:
            1. A column matching the name of template_cube
            2. A series of columns with names which match the dimension coordinates on
            template_cube. The data in these columns should match the points on the
            corresponding dimension of template_cube.
        template_cube: A cube which will provide all metadata for the output cube

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
    sd_clip: float = 0.25,
    unique_site_id_key: Optional[str] = None,
) -> Tuple[Cube, Cube]:
    """Function to predict climatological means and standard deviations given fitted
    GAMs for each statistic and cubes which can be used to construct a dataframe
    containing all required features for those GAMs.

    Args:
        input_cube
        gams: A list containing two fitted GAMs, the first for predicting the
            climatological mean of the locations in input_cube and the second
            predicting the climatological standard deviation.
        gam_features:
            The list of features. These must be either coordinates on input_cube or
            share a name with a cube in additional_cubes. The index of each
            feature should match the indices used in model_specification.
        additional_cubes:
            Additional fields to use as supplementary predictors.
        sd_clip:
            The minimum standard deviation value to allow when predicting from the GAM.
            Any predictions below this value will be set to this value.
        unique_site_id_key:
            If working with spot data and available, the name of the coordinate
            in the input cubes that contains unique site IDs, e.g. "wmo_id" if
            all sites have a valid wmo_id.

    Returns:
        A pair of cubes containing climatological mean and climatological standard
        deviation predictions respectively.
    """
    diagnostic = input_cube.name()

    df = prepare_data_for_gam(
        input_cube, additional_cubes, unique_site_id_key=unique_site_id_key
    )

    # Calculate climatological means and standard deviations using previously
    # fitted GAMs.
    mean_pred = GAMPredict().process(gams[0], df[gam_features])
    sd_pred = GAMPredict().process(gams[1], df[gam_features])

    # Convert means and standard deviations into cubes
    df[diagnostic] = mean_pred
    mean_cube = convert_dataframe_to_cube(df, input_cube)

    df[diagnostic] = sd_pred
    sd_cube = convert_dataframe_to_cube(df, input_cube)
    sd_cube.data = clip(sd_cube.data, a_min=sd_clip, a_max=None)

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
        model_specification: list[list[str], list[int], dict],
        max_iter: int = 100,
        tol: float = 0.0001,
        distribution: str = "normal",
        link: str = "identity",
        fit_intercept: bool = True,
        window_length: int = 11,
        unique_site_id_key: Optional[str] = None,
    ):
        """
        Initialize the class.
        Args:
            model_specification:
                A list of lists which each contain three items (in order):
                    1. a string containing a single pyGAM term; one of 'linear',
                    'spline', 'tensor', or 'factor'
                    2. a list of integers which correspond to the features to be
                    included in that term
                    3. a dictionary of kwargs to be included when defining the term
            max_iter:
                A pyGAM argument which determines the maximum iterations allowed when
                fitting the GAM
            tol:
                A pyGAM argument determining the tolerance used to define the stopping
                criteria
            distribution:
                A pyGAM argument determining the distribution to be used in the model
            link:
                A pyGAM argument determining the link function to be used in the model
            fit_intercept:
                A pyGAM argument determining whether to include an intercept term in
                the model
            window_length:
                The length of the rolling window used to calculate the mean and standard
                deviation of the input cube when the input cube does not have a
                realization dimension coordinate. This must be an odd integer greater
                than 1.
            unique_site_id_key:
                An optional key to use for uniquely identifying each site in the
                training data. If not provided, the default behavior is to use the
                spatial coordinates (latitude, longitude) of each site.
        """
        self.model_specification = model_specification
        self.max_iter = max_iter
        self.tol = tol
        self.distribution = distribution
        self.link = link
        self.fit_intercept = fit_intercept
        self.unique_site_id_key = unique_site_id_key

        if window_length < 3 or window_length % 2 == 0 or window_length % 1 != 0:
            raise ValueError(
                "The window_length input must be an odd integer greater than 1. "
                f"Received: {window_length}."
            )
        else:
            self.window_length = window_length

    def apply_aggregator(
        self, padded_cube: Cube, aggregator: iris.analysis.WeightedAggregator
    ) -> Cube:
        """
        Internal function to apply rolling window aggregator to padded cube.
        Args:
            padded_cube:
                The cube to have rolling window calculation applied to.
            aggregator:
                The aggregator to use in the rolling window calculation.

        Returns:
            A cube containing the result of the rolling window calculation. Any
            cell methods and time bounds are removed from the cube as they are not
            necessary for later calculations.
        """
        summary_cube = padded_cube.rolling_window(
            coord="time", aggregator=aggregator, window=self.window_length
        )
        summary_cube.cell_methods = ()
        summary_cube.coord("time").bounds = None
        summary_cube.coord("time").points = summary_cube.coord("time").points.astype(
            int64
        )
        summary_cube.data = summary_cube.data.filled(nan)
        return summary_cube

    def calculate_statistic_by_rolling_window(self, input_cube: Cube):
        """Function to calculate mean and standard deviation of input_cube using a
        rolling window calculation over the time coordinate.

        The input_cube time coordinate is padded at the beginning and end of the time
        coordinate, to ensure that the result of the rolling window calculation has the
        same shape as input_cube. Additionally, any missing time points in the input
        cube are filled with masked data, so that the rolling window is always taken
        over a period containing an equal number of time points.
        """
        removed_coords = []
        pad_width = int((self.window_length - 1) / 2)

        # Pad the time coordinate of the input cube, then calculate the mean and
        # standard deviation using a rolling window over the time coordinate.
        time_coord = input_cube.coord("time")
        increments = time_coord.points[1:] - time_coord.points[:-1]
        min_increment = increments.min()

        if not all(x % min_increment == 0 for x in increments):
            raise ValueError(
                "The increments between points in the time coordinate of the input "
                "cube must be divisible by the smallest increment between points to "
                "allow for rolling window calculations to be performed over the time "
                "coordinate. The increments between points in the time coordinate "
                f"were: {increments}. The smallest increment was: {min_increment}."
            )

        padded_cube = input_cube.copy()

        # Check if we are dealing with a period diagnostic.
        if padded_cube.coord("time").bounds is not None:
            bounds_width = (
                padded_cube.coord("time").bounds[0][1]
                - padded_cube.coord("time").bounds[0][0]
            )
        else:
            bounds_width = None
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
        # coordinate and fill any gaps. This ensures that all the time points
        # are equally spaced and the padding ensures that the output of the
        # rolling window calculation is the same shape as the input cube.
        existing_points = time_coord.points
        desired_points = array(
            [
                x
                for x in range(
                    min(existing_points) - (min_increment * pad_width),
                    max(existing_points) + (min_increment * pad_width) + 1,
                    min_increment,
                )
            ],
            dtype=int64,
        )

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
                if bounds_width:
                    # Add correct bounds to point if required.
                    new_slice.coord("time").bounds = [point - bounds_width, point]
                new_slice = new_axis(new_slice, "time")
                new_slice.data = masked_all_like(new_slice.data)
                padded_cubelist.append(new_slice)
        padded_cube = padded_cubelist.concatenate_cube()

        aggregated_cubes = {}
        for aggregator in [MEAN, STD_DEV]:
            aggregated_cubes[aggregator.name()] = self.apply_aggregator(
                padded_cube, aggregator
            )

        # Create constraint to extract only those time points which were present in
        # the original input cube.
        constr = iris.Constraint(
            time=lambda cell: cell.point in input_cube.coord("time").cells()
        )
        input_mean = aggregated_cubes["mean"].extract(constr)
        input_sd = aggregated_cubes["standard_deviation"].extract(constr)

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

        The rolling window method calculates a statistic over data in a fixed time
        window and assigns the value of the statistic to the central time in the window.
        For example, for data points [0.0, 1.0, 2.0, 1.0, 0.0] each valid in
        consecutive hours T+0, T+1, T+2, T+3, T+4, the mean calculated by a rolling
        window of width 5 would be 0.8. This value would be associated with T+2 in the
        resulting cube.

        To enable this calculation to produce a cube of the same dimensions as
        input_cube, the data in input_cube is first padded with additional data. For a
        rolling window of width 5, 2 data slices are added to the start and end of the
        input_cube time coordinate. The data in these slices are masked so that they
        don't affect the calculated statistics.

        Args:
            input_cube:
                A cube with at least one of the following coordinates:
                1. A realization dimension coordinate
                2. A time coordinate with more than one point and evenly spaced points.

        Returns:
            CubeList containing a mean cube and standard deviation cube.

        Raises:
            ValueError: If input_cube does not contain a realization coordinate and
            does contain a time coordinate with unevenly spaced points.
        """
        if input_cube.coords("realization"):
            input_mean = collapse_realizations(input_cube, method="mean")
            input_sd = collapse_realizations(input_cube, method="std_dev")
        else:
            input_mean, input_sd = self.calculate_statistic_by_rolling_window(
                input_cube
            )

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
                share a name with a cube in additional_fields. The index of each
                feature should match the indices used in model_specification.
            additional_fields:
                Additional fields to use as supplementary predictors.

        Returns:
            A list containing fitted GAMs which model the input_cube mean and
            standard deviation.

        Raises:
            ValueError: If input_cube does not contain at least one of a realization or
            time coordinate.
            ValueError: If the input cube does not have a realization coordinate and the
            time coordinate that it does have contains only one point.
        """
        if not input_cube.coords("realization", dim_coords=True):
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
            df = prepare_data_for_gam(
                stat_cube, additional_fields, unique_site_id_key=self.unique_site_id_key
            )
            feature_values = df[features].values
            targets = df[input_cube.name()].values
            output.append(plugin.process(feature_values, targets))

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
        unique_site_id_key: Optional[str] = None,
    ) -> None:
        """Initialize the class.
        Args:
            distribution:
                Name of distribution. Assume that a calibrated version of the
                climate anomaly forecast could be represented using this distribution.
            emos_kwargs: Keyword arguments accepted by the
                EstimateCoefficientsForEnsembleCalibration plugin. Should not contain
                a distribution argument.
            unique_site_id_key:
                If working with spot data and available, the name of the coordinate
                in the input cubes that contains unique site IDs, e.g. "wmo_id" if
                all sites have a valid wmo_id.
        """
        self.distribution = distribution
        self.emos_kwargs = emos_kwargs if emos_kwargs else {}
        self.unique_site_id_key = unique_site_id_key

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
                containing climatological mean predictions of the forecasts and a cube
                containing climatological standard deviation predictions of the
                forecasts.
            truth_cubes:
                A list of three cubes: a cube containing historic truths, a cube
                containing climatological mean predictions of the truths and a cube
                containing climatological standard deviation predictions of the truths.
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
        """Function to convert historic forecasts and truths to climatological
        anomalies, then fit EMOS coefficients to these anomalies.

        Args:
            historic_forecasts:
                Historic forecasts from the training dataset.
            truths:
                Truths from the training dataset.
            forecast_gams:
                A list containing two fitted GAMs, the first for predicting the
                climatological mean of the locations in historic_forecasts and the
                second predicting the climatological standard deviation. Appropriate
                GAMs are produced by the TrainGAMsForSAMOS plugin.
            truth_gams:
                A list containing two fitted GAMs, the first for predicting the
                climatological mean of the locations in truths and the second
                predicting the climatological standard deviation. Appropriate
                GAMs are produced by the TrainGAMsForSAMOS plugin.
            gam_features:
                The list of features. These must be either coordinates on input_cube or
                share a name with a cube in gam_additional_fields. The index of each
                feature must match the indices used in model_specification.
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
            historic_forecasts,
            forecast_gams,
            gam_features,
            gam_additional_fields,
            unique_site_id_key=self.unique_site_id_key,
        )
        truth_mean, truth_sd = get_climatological_stats(
            truths,
            truth_gams,
            gam_features,
            gam_additional_fields,
            unique_site_id_key=self.unique_site_id_key,
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

    def __init__(
        self,
        percentiles: Optional[Sequence] = None,
        unique_site_id_key: Optional[str] = None,
    ):
        """Initialize class.

        Args:
            percentiles:
                The set of percentiles used to create the calibrated forecast.
            unique_site_id_key:
                If working with spot data and available, the name of the coordinate
                in the input cubes that contains unique site IDs, e.g. "wmo_id" if
                all sites have a valid wmo_id.
        """
        self.percentiles = [float32(p) for p in percentiles] if percentiles else None
        self.unique_site_id_key = unique_site_id_key

    def process(
        self,
        forecast: Cube,
        forecast_gams: List,
        truth_gams: List,
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
            truth_gams:
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
            forecast_as_realizations,
            forecast_gams,
            gam_features,
            gam_additional_fields,
            unique_site_id_key=self.unique_site_id_key,
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
        truth_mean, truth_sd = get_climatological_stats(
            forecast_as_realizations,
            truth_gams,
            gam_features,
            gam_additional_fields,
            unique_site_id_key=self.unique_site_id_key,
        )

        # The data in these cubes are identical along the realization dimensions.
        truth_mean = next(truth_mean.slices_over("realization"))
        truth_sd = next(truth_sd.slices_over("realization"))

        # Transform location and scale parameters so that they represent a distribution
        # in the units of the original forecast, rather than climatological anomalies.
        forecast_units = (
            forecast.units
            if input_forecast_type != "probabilities"
            else find_threshold_coordinate(forecast).units
        )
        location_parameter.data = (
            location_parameter.data * truth_sd.data
        ) + truth_mean.data
        location_parameter.units = forecast_units

        # The scale parameter returned by ApplyEMOS is the standard deviation for a
        # normal distribution. To get the desired standard deviation in
        # realization/percentile space we must multiply by the estimated forecast
        # standard deviation.
        scale_parameter.data = scale_parameter.data * truth_sd.data
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

        if input_forecast_type != "probabilities" and not prob_template:
            # Enforce that the result is within sensible bounds.
            bounds_pairing = get_bounds_of_distribution(
                bounds_pairing_key=result.name(), desired_units=result.units
            )
            result.data = clip(
                result.data, a_min=bounds_pairing[0], a_max=bounds_pairing[1]
            )

        # Enforce correct dtype.
        result.data = result.data.astype(dtype=float32)

        return result
