# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugins to perform quantile regression using random forests."""

from typing import Optional

import iris
import joblib
import numpy as np
import pandas as pd
from iris.cube import Cube, CubeList
from quantile_forest import RandomForestQuantileRegressor

from improver import BasePlugin, PostProcessingPlugin
from improver.constants import DAYS_IN_YEAR, HOURS_IN_DAY
from improver.utilities.cube_manipulation import enforce_coordinate_ordering


def _expand_dims(
    template_cube: Cube, expansion_array: np.ndarray, features: list[str]
) -> np.ndarray:
    """Expand dimensions of the expansion_array to match the dimensions of the template
    cube. For example, the dimensions of the expansion_array may only span a portion
    of the dimensions present in the template cube. If the expansion_array has
    singleton dimensions that are not present in the template cube, these singleton
    dimension will be squeezed away.

    Args:
        template_cube: Cube that acts as a template for the output. The dimension
            coordinates and coordinates associated with the dimension coordinates on
            this cube will be used to expand the expansion_array.
        expansion_array: Array that will be expanded to match the dimensions of the
            template_cube.
        features: List of feature names that are present in the template_cube. These
            may correspond to dimension coordinates or coordinates associated with
            dimension coordinates or scalar coordinates.

    Returns:
        Array with dimensions expanded to match the template_cube.
    """
    # The aim here is to find the names of the dimension coordinates but check
    # whether the name of any coordinate associated with a dimension coordinate matches
    # the features provided. If there is a match, then use the feature name, rather than
    # the dimension coordinate name.
    dim_coord_names = [c.name() for c in template_cube.coords(dim_coords=True)]
    refined_dim_coord_names = []
    for coord in template_cube.coords():
        # Ignore scalar coordinates.
        if len(template_cube.coord_dims(coord)) > 0:
            associated_coords = [
                c.name()
                for c in template_cube.coords(
                    dimensions=template_cube.coord_dims(coord)
                )
            ]
            feature_associated_coords = list(
                set(associated_coords).intersection(features)
            )
            if feature_associated_coords:
                refined_dim_coord_names.append(feature_associated_coords[0])
            elif coord.name() in dim_coord_names:
                refined_dim_coord_names.append(coord.name())
    refined_dim_coord_names = list(set(refined_dim_coord_names))

    expansion_dim_names = list(set(refined_dim_coord_names) - set(features))

    # If the expansion_dims and refined_dim_coord_names match, this implies that
    # the features provided are not dimension coordinates nor associated with any
    # dimension coordinate. In this case, any singleton dimensions on the
    # expansion_array should be removed, as it isn't relevant for defining the shape
    # of the template cube.
    if expansion_dim_names == refined_dim_coord_names:
        expansion_array = np.squeeze(expansion_array)

    dims = []
    for dim_name in expansion_dim_names:
        dims.extend(template_cube.coord_dims(dim_name))

    return np.expand_dims(expansion_array, dims)


def prep_feature(
    template_cube: Cube,
    feature_cube: Cube,
    feature: str,
) -> np.ndarray:
    """Prepare the feature values for the quantile regression random forest model.
    Each expected feature is handled separately. The key steps are:
        - Collapse the template cube over the realization dimension to get a shape
        template.
        - Extract the feature values from the feature cube provided.
        - Handle different types of features separately (mean, std, spatial coordinates,
        forecast period, model weights, day of year, hour of day, and static features)
        to ensure they are broadcasted correctly to match the template cube shape.
        Time coordinates are the most complex to handle as they can be scalar, span a
        single dimension or span multiple dimensions.
        - Flatten the feature values to create a 1D array.
        - Set the output dtype based on the feature type.
        - Return the flattened array of feature values.

    Args:
        template_cube (cube):
            The forecast cube that acts only as a template.
        feature_cube (cube):
            Feature cube that is either static or dynamic.
        feature:
            The feature to be extracted from the associated feature_cube.
            If "static" then the cube itself acts as the feature.
    Returns:
        feature_values (numpy.ndarray):
            Flattened array of feature values.
    """
    # Collapse the template cube to get a shape template.
    collapsed_cube = template_cube.collapsed(["realization"], iris.analysis.MEAN)

    # Handle each expected feature type separately.
    if "mean" == feature:
        feature_values = feature_cube.collapsed(
            ["realization"], iris.analysis.MEAN
        ).data.flatten()
    elif "std" == feature:
        feature_values = feature_cube.collapsed(
            ["realization"], iris.analysis.STD_DEV
        ).data.flatten()
    elif feature in ["latitude", "longitude", "altitude"]:
        coord_multidim = _expand_dims(
            collapsed_cube, feature_cube.coord(feature).points, ["spot_index"]
        )
        feature_values = np.broadcast_to(coord_multidim, collapsed_cube.shape).flatten()
    elif feature == "forecast_period":
        if len(feature_cube.coord_dims("forecast_period")) == 1:
            coord_multidim = _expand_dims(
                collapsed_cube,
                feature_cube.coord("forecast_period").points,
                ["forecast_period"],
            )
        else:
            coord_multidim = feature_cube.coord("forecast_period").points

        feature_values = np.broadcast_to(coord_multidim, collapsed_cube.shape).flatten()
    elif feature == "model_weights":
        if len(feature_cube.coord_dims("forecast_period")) == 1:
            coord_multidim = _expand_dims(
                collapsed_cube,
                feature_cube.coord("model_weights").points,
                ["forecast_period"],
            )
        else:
            coord_multidim = feature_cube.coord("model_weights").points

        feature_values = np.broadcast_to(coord_multidim, collapsed_cube.shape).flatten()
    elif feature in ["day_of_year", "day_of_year_sin", "day_of_year_cos"]:
        frt_dims = feature_cube.coord_dims("forecast_reference_time")
        fp_dims = feature_cube.coord_dims("forecast_period")
        frt_coord = feature_cube.coord("forecast_reference_time")
        fp_coord = feature_cube.coord("forecast_period")

        if len(frt_dims) == 0 and len(fp_dims) == 0:
            time_point = frt_coord.cell(0).point._to_real_datetime() + pd.to_timedelta(
                fp_coord.points[0], unit=str(fp_coord.units)
            )
            day_of_year = np.array([np.int32(time_point.strftime("%j"))])
        elif frt_dims == fp_dims:
            # Forecast reference time and forecast period share a dimension coordinate.
            # Forecast reference time and forecast period must be mixed together
            # along one dimension.
            day_of_year = []
            for frt_cell, fp_point in zip(frt_coord.cells(), fp_coord.points):
                time_point = frt_cell.point._to_real_datetime() + pd.to_timedelta(
                    fp_point, unit=str(fp_coord.units)
                )
                day_of_year.append(np.int32(time_point.strftime("%j")))
            day_of_year = np.array(day_of_year)
            # frt_dims and fp_dims are the same, so we can choose one of them to pop.
            day_of_year = _expand_dims(
                collapsed_cube, day_of_year, ["forecast_reference_time"]
            )
        else:
            # Forecast reference time and forecast period are different dimensions.
            day_of_year = np.zeros((len(frt_coord.points), len(fp_coord.points)))
            for i, frt_cell in enumerate(frt_coord.cells()):
                for j, fp_point in enumerate(fp_coord.points):
                    time_point = frt_cell.point._to_real_datetime() + pd.to_timedelta(
                        fp_point, unit=str(fp_coord.units)
                    )
                    day_of_year[i, j] = time_point.strftime("%j")

            day_of_year = _expand_dims(
                collapsed_cube,
                day_of_year,
                ["forecast_reference_time", "forecast_period"],
            )

        feature_values = np.broadcast_to(day_of_year, collapsed_cube.shape).flatten()
        if feature == "day_of_year_sin":
            feature_values = np.sin(2 * np.pi * feature_values / (DAYS_IN_YEAR + 1))
        elif feature == "day_of_year_cos":
            feature_values = np.cos(2 * np.pi * feature_values / (DAYS_IN_YEAR + 1))
    elif feature in ["hour_of_day", "hour_of_day_sin", "hour_of_day_cos"]:
        frt_dims = feature_cube.coord_dims("forecast_reference_time")
        fp_dims = feature_cube.coord_dims("forecast_period")
        frt_coord = feature_cube.coord("forecast_reference_time")
        fp_coord = feature_cube.coord("forecast_period")

        if len(frt_dims) == 0 and len(fp_dims) == 0:
            time_point = frt_coord.cell(0).point._to_real_datetime() + pd.to_timedelta(
                fp_coord.points[0], unit=str(fp_coord.units)
            )
            hour_of_day = np.array([np.int32(time_point.hour)])
        elif frt_dims == fp_dims:
            # Forecast reference time and forecast period share a dimension coordinate.
            # Forecast reference time and forecast period must be mixed together
            # along one dimension.
            hour_of_day = []
            for frt_cell, fp_point in zip(frt_coord.cells(), fp_coord.points):
                time_point = frt_cell.point._to_real_datetime() + pd.to_timedelta(
                    fp_point, unit=str(fp_coord.units)
                )
                hour_of_day.append(np.int32(time_point.hour))
            hour_of_day = np.array(hour_of_day)

            # frt_dims and fp_dims are the same, so we can choose one of them to pop.
            hour_of_day = _expand_dims(
                collapsed_cube, hour_of_day, ["forecast_reference_time"]
            )
        else:
            # Forecast reference time and forecast period are different dimensions.
            hour_of_day = np.zeros((len(frt_coord.points), len(fp_coord.points)))
            for i, frt_cell in enumerate(frt_coord.cells()):
                for j, fp_point in enumerate(fp_coord.points):
                    time_point = frt_cell.point._to_real_datetime() + pd.to_timedelta(
                        fp_point, unit=str(fp_coord.units)
                    )
                    hour_of_day[i, j] = time_point.hour

            hour_of_day = _expand_dims(
                collapsed_cube,
                hour_of_day,
                ["forecast_reference_time", "forecast_period"],
            )

        feature_values = np.broadcast_to(hour_of_day, collapsed_cube.shape).flatten()
        if feature == "hour_of_day_sin":
            feature_values = np.sin(2 * np.pi * feature_values / HOURS_IN_DAY)
        elif feature == "hour_of_day_cos":
            feature_values = np.cos(2 * np.pi * feature_values / HOURS_IN_DAY)
    elif feature == "day_of_training_period":
        if len(feature_cube.coord_dims("day_of_training_period")) == 1:
            coord_multidim = _expand_dims(
                collapsed_cube,
                feature_cube.coord("day_of_training_period").points,
                ["day_of_training_period"],
            )
        else:
            coord_multidim = feature_cube.coord("day_of_training_period").points

        feature_values = np.broadcast_to(coord_multidim, collapsed_cube.shape).flatten()
    elif feature == "static":
        coord_multidim = _expand_dims(collapsed_cube, feature_cube.data, ["spot_index"])
        feature_values = np.broadcast_to(coord_multidim, collapsed_cube.shape).flatten()
    else:
        # If the feature is not one of the expected types, raise an error.
        msg = f"Feature {feature} is not supported."
        raise ValueError(msg)

    # Ensure the feature values are returned with the correct dtype.
    if feature in ["mean", "std", "static"]:
        feature_values = feature_values.astype(feature_cube.dtype)
    elif feature in [
        "day_of_year",
        "day_of_year_sin",
        "day_of_year_cos",
        "hour_of_day",
        "hour_of_day_sin",
        "hour_of_day_cos",
    ]:
        feature_values = feature_values.astype(np.float32)
    else:
        feature_values = feature_values.astype(feature_cube.coord(feature).dtype)

    return feature_values


def _check_valid_transformation(transformation: str):
    """Check if the transformation is one of the supported types.
    Args:
        transformation: Transformation to be checked.
    Raises:
        ValueError: If the transformation is not one of the supported types.
    """
    if transformation not in ["log", "log10", "sqrt", "cbrt", None]:
        msg = (
            "Currently the only supported transformations are log, log10, sqrt "
            f"and cbrt. The transformation supplied was {transformation}."
        )
        raise ValueError(msg)


class TrainQuantileRegressionRandomForests(BasePlugin):
    """Plugin to train a model using quantile regression random forests."""

    def __init__(
        self,
        feature_config: dict[str, list[str]],
        n_estimators: int,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
        transformation: Optional[str] = None,
        pre_transform_addition: np.float32 = 0,
        compression: int = 5,
        model_output: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialise the plugin.
        Args:
            feature_config (dict):
                Feature configuration defining the features to be used for quantile
                regression. The configuration is a dictionary of strings, where the
                keys are the names of the input cube(s) supplied, and the values are
                a list. This list can contain both computed features, such as the
                mean or standard deviation (std), or static features, such as the
                altitude. The computed features will be computed using the cube defined
                in the dictionary key. If the key is the feature itself e.g. a distance
                to water cube, then the value should state "static". This will ensure
                the cube's data is used as the feature. The config will have the
                structure:
                    "DYNAMIC_VARIABLE_NAME": ["FEATURE1", "FEATURE2"] e.g:
                    {
                    "air_temperature": ["mean", "std", "altitude"],
                    "visibility_at_screen_level": ["mean", "std"]
                    "distance_to_water": ["static"],
                    }
            n_estimators (int):
                Number of trees in the forest.
            max_depth (int):
                Maximum depth of the tree.
            random_state (int):
                Random seed for reproducibility.
            transformation (str):
                Transformation to be applied to the data before fitting.
            pre_transform_addition (float):
                Value to be added before transformation.
            compression (int):
                Compression level for saving the model.
            model_output (str):
                Full path including model file name that will store the pickled model.
            kwargs:
                Additional keyword arguments for the quantile regression model.
        """

        self.feature_config = feature_config
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.transformation = transformation
        _check_valid_transformation(self.transformation)
        self.pre_transform_addition = pre_transform_addition
        self.compression = compression
        self.output = model_output
        self.kwargs = kwargs
        self.expected_coordinate_order = ["forecast_reference_time", "forecast_period"]

    def fit_qrf(
        self, forecast_features: np.ndarray, target: np.ndarray
    ) -> RandomForestQuantileRegressor:
        """Fit the quantile regression random forest model.
        Args:
            forecast_features (numpy.ndarray):
                Array of forecast features.
            target (numpy.ndarray):
                Array of target values.
        Returns:
            qrf_model (RandomForestQuantileRegressor):
                Fitted quantile regression model.
        """

        qrf_model = RandomForestQuantileRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            **self.kwargs,
        )
        qrf_model.fit(forecast_features, target)
        return qrf_model

    def _organise_truth_data(
        self, forecast_cube: Cube, truth_cube: Cube
    ) -> list[np.ndarray]:
        """Organise the truth data, so that the validity time matches the validity
        time of the forecast. This might mean that the truth data is repeated, if, for
        example, the forecast has multiple forecast reference times and multiple
        forecast periods that have the same validity time."""
        frt_dims = forecast_cube.coord_dims("forecast_reference_time")
        fp_dims = forecast_cube.coord_dims("forecast_period")

        frt_coord = forecast_cube.coord("forecast_reference_time")
        fp_coord = forecast_cube.coord("forecast_period")

        time_datetimes = []

        # Handle the case where both forecast reference time and forecast period are
        # non-dimensional coordinates (i.e., the cube has no time dimensions).
        if frt_dims is None and fp_dims is None:
            time_datetimes.append(
                frt_coord.cell(0).point._to_real_datetime()
                + pd.to_timedelta(fp_coord.points, unit=str(fp_coord.units))
            )
        elif frt_dims == fp_dims:
            # Handle the case where forecast reference time and forecast period share
            # the same dimension. This means both are mixed together along one
            # dimension.
            for frt, fp in zip(
                list(frt_coord.cells()),
                fp_coord.points,
            ):
                time_datetimes.append(
                    frt.point._to_real_datetime()
                    + pd.to_timedelta(
                        fp, unit=str(forecast_cube.coord("forecast_period").units)
                    )
                )
        else:
            # Handle the case where forecast reference time and forecast period are on
            # different dimensions. This requires iterating over all combinations of
            # forecast period and forecast reference time.
            enforce_coordinate_ordering(
                forecast_cube,
                self.expected_coordinate_order[::-1],
                anchor_start=True,
            )
            for fp in fp_coord.points:
                for frt in list(frt_coord.cells()):
                    time_datetimes.append(
                        frt.point._to_real_datetime()
                        + pd.to_timedelta(
                            fp, unit=str(forecast_cube.coord("forecast_period").units)
                        )
                    )

        truth_data_list = []
        for time_datetime in time_datetimes:
            constr = iris.Constraint(time=lambda cell: cell.point == time_datetime)
            truth_data_list.append(truth_cube.extract(constr).data)
        return truth_data_list

    def process(
        self,
        forecast_cube: Cube,
        truth_cube: Cube,
        feature_cubes: Optional[CubeList] = None,
    ) -> None:
        """Train a quantile regression random forests model.
        Args:
            forecast_cube:
                Cube containing the realization forecasts. If the cube provided
                contains multiple forecast periods, then the cube is expected to have
                forecast period, forecast reference time, realization
                and spot_index as the dimensions. This cube is only used as a template.
                If the forecast cube is a feature cube, then it should also be provided
                within the feature_cubes list.
            truth_cube:
                Cube containing the truths. The truths should have the same validity
                times as the forecast. If the same validity time occurs multiple times
                within the forecast cube (i.e. due to e.g. a forecast reference time
                of 20170102T0000Z and a lead time of T+36 having the same validity time
                as a forecast reference time of 20170103T0000Z and a lead time of T+6),
                then the truth data will be repeated.
            feature_cubes:
                List of additional feature cubes. The name of the cube should match a
                key in the feature_config dictionary.

        References:
            Johnson. (2024). quantile-forest: A Python Package for Quantile
            Regression Forests. Journal of Open Source Software, 9(93), 5976.
            https://doi.org/10.21105/joss.05976.
            Meinshausen, N. (2006). Quantile regression forests.
            Journal of Machine Learning Research,
            7(35), 983–999. http://jmlr.org/papers/v7/meinshausen06a.html
            Taillardat, M., O. Mestre, M. Zamo, and P. Naveau, 2016: Calibrated
            Ensemble Forecasts Using Quantile Regression Forests and Ensemble Model
            Output Statistics. Mon. Wea. Rev., 144, 2375–2393,
            https://doi.org/10.1175/MWR-D-15-0260.1.
            Taillardat, M. and Mestre, O.: From research to applications – examples of
            operational ensemble post-processing in France using machine learning,
            Nonlin. Processes Geophys., 27, 329–347,
            https://doi.org/10.5194/npg-27-329-2020, 2020.
        """

        if self.transformation:
            forecast_cube.data = getattr(np, self.transformation)(
                forecast_cube.data + self.pre_transform_addition
            )
            truth_cube.data = getattr(np, self.transformation)(
                truth_cube.data + self.pre_transform_addition
            )

        # Ensure the forecast cube has the correct dimension ordering for prep_feature.

        coord_dims = [
            forecast_cube.coord_dims(c) for c in self.expected_coordinate_order
        ]
        coord_names = []
        for dim in coord_dims:
            if len(dim) == 0:
                continue
            if len(forecast_cube.coords(dimensions=dim, dim_coords=True)) == 0:
                continue
            coord_names.append(
                forecast_cube.coord(dimensions=dim, dim_coords=True).name()
            )
        enforce_coordinate_ordering(
            forecast_cube,
            [coord_names],
            anchor_start=True,
        )

        feature_values = []

        for feature_name in self.feature_config.keys():
            feature_cube = feature_cubes.extract(iris.Constraint(feature_name))
            if not feature_cube:
                msg = (
                    f"Feature cube for {feature_name} not found in the provided "
                    "feature cubes."
                )
                raise ValueError(msg)
            for feature in self.feature_config[feature_name]:
                print("feature = ", feature)
                feature_values.append(
                    prep_feature(forecast_cube, feature_cube[0], feature)
                )
        feature_values = np.array(feature_values).T
        truth_data_list = self._organise_truth_data(forecast_cube, truth_cube)
        target_values = np.array(truth_data_list).flatten()

        # Fit the quantile regression model
        qrf_model = self.fit_qrf(feature_values, target_values)

        joblib.dump(qrf_model, self.output, compress=self.compression)


class ApplyQuantileRegressionRandomForests(PostProcessingPlugin):
    """Plugin to apply a trained model using quantile regression random forests."""

    def __init__(
        self,
        feature_config: dict[str, list[str]],
        quantiles: list[np.float32],
        transformation: str = None,
        pre_transform_addition: np.float32 = 0,
    ) -> None:
        """Initialise the plugin.
        Args:
            feature_config (dict):
                Feature configuration defining the features to be used for quantile regression.
                The configuration is a dictionary of strings, where the keys are the names of
                the input cube(s) supplied, and the values are a list. This list can contain both
                computed features, such as the mean or standard deviation (std), or static
                features, such as the altitude. The computed features will be computed using
                the cube defined in the dictionary key. If the key is the feature itself e.g.
                a distance to water cube, then the value should state "static". This will ensure
                the cube's data is used as the feature.
                The config will have the structure:
                "DYNAMIC_VARIABLE_NAME": ["FEATURE1", "FEATURE2"] e.g:
                {
                "air_temperature": ["mean", "std", "altitude"],
                "visibility_at_screen_level": ["mean", "std"]
                "distance_to_water": ["static"],
                }
            quantiles (float):
                Quantiles used for prediction (values ranging from 0 to 1)
            transformation (str):
                Transformation to be applied to the data before fitting.
            pre_transform_addition (float):
                Value to be added before transformation.

        Raises:
            ValueError: If the transformation is not one of the supported types.
        """

        self.feature_config = feature_config
        self.quantiles = quantiles
        self.transformation = transformation
        _check_valid_transformation(self.transformation)
        self.pre_transform_addition = pre_transform_addition

    def _reverse_transformation(self, forecast_cube: Cube):
        """Reverse the transformation applied to the data prior to fitting the QRF.
        The forecast cube provided is modified in place.

        Args:
            forecast_cube: Forecast to be calibrated.
        """
        if self.transformation:
            if self.transformation == "log":
                forecast_cube.data = (
                    np.exp(forecast_cube.data) - self.pre_transform_addition
                )
            elif self.transformation == "log10":
                forecast_cube.data = (
                    10 ** (forecast_cube.data) - self.pre_transform_addition
                )
            elif self.transformation == "sqrt":
                forecast_cube.data = forecast_cube.data**2 - self.pre_transform_addition
            elif self.transformation == "cbrt":
                forecast_cube.data = forecast_cube.data**3 - self.pre_transform_addition

    def process(
        self,
        qrf_model: RandomForestQuantileRegressor,
        feature_cubes: CubeList,
        template_forecast_cube: Cube,
    ) -> Cube:
        """Apply a quantile regression random forests model.

        Args:
            qrf_model: A trained QRF model.
            feature_cubes: CubeList of features. This should include the forecast to be
                calibrated, if that has been used as a feature in the training, and
                any features that can be provided as cubes.
            template_forecast_cube: Template forecast cube that provides the required
                metadata and shape for the output cube. The data from this cube will not
                be used.

        Returns:
            Calibrated forecast cube with the same metadata as the template
            forecast cube.

        """
        feature_values = []

        for feature_name in self.feature_config.keys():
            feature_cube = feature_cubes.extract_cube(iris.Constraint(feature_name))

            # Transform the feature cube data if a transformation is specified.
            if (
                self.transformation
                and set(["mean", "std"]).intersection(self.feature_config[feature_name])
                and feature_cube.name() == template_forecast_cube.name()
            ):
                feature_cube.data = getattr(np, self.transformation)(
                    feature_cube.data + self.pre_transform_addition
                )

            for feature in self.feature_config[feature_name]:
                feature_values.append(
                    prep_feature(template_forecast_cube, feature_cube, feature)
                )

        feature_values = np.array(feature_values).T
        calibrated_forecast = qrf_model.predict(
            feature_values, quantiles=self.quantiles
        )
        calibrated_forecast = np.float32(calibrated_forecast)

        calibrated_forecast_cube = template_forecast_cube.copy(
            data=np.broadcast_to(calibrated_forecast.T, template_forecast_cube.shape)
        )
        self._reverse_transformation(calibrated_forecast_cube)

        return calibrated_forecast_cube
