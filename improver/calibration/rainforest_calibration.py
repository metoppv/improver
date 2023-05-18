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
"""RainForests calibration Plugins.

.. Further information is available in:
.. include:: extended_documentation/calibration/rainforests_calibration/
   rainforests_calibration.rst

"""

from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import cf_units as unit
import numpy as np
from iris.analysis import MEAN
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from numpy import ndarray

from improver import PostProcessingPlugin
from improver.ensemble_copula_coupling.utilities import (
    get_bounds_of_distribution,
    interpolate_pointwise,
)
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.cube_manipulation import add_coordinate_to_cube, compare_coords


class ApplyRainForestsCalibration(PostProcessingPlugin):
    """Generic class to calibrate input forecast via RainForests.

    The choice of tree-model library is determined from package availability, and whether
    all required models files are available. Treelite is the preferred option, defaulting
    to lightGBM if requirements are missing.
    """

    def __new__(cls, model_config_dict: dict, threads: int = 1):
        """Initialise class object based on package and model file availability.

        Args:
            model_config_dict:
                Dictionary containing Rainforests model configuration variables.
            threads:
                Number of threads to use during prediction with tree-model objects.

        Dictionary is of format::

            {
                "-50.0" : {
                    "lightgbm_model" : "<path_to_lightgbm_model_object>",
                    "treelite_model" : "<path_to_treelite_model_object>"
                },
                "-25.0" : {
                    "lightgbm_model" : "<path_to_lightgbm_model_object>",
                    "treelite_model" : "<path_to_treelite_model_object>"
                },
                ...,
                "50.0" : {
                    "lightgbm_model" : "<path_to_lightgbm_model_object>",
                    "treelite_model" : "<path_to_treelite_model_object>"
                }
            }

        The keys specify the error threshold value, while the associated values
        are the path to the corresponding tree-model objects for that threshold.

        Treelite predictors are used if treelite_runitme is an installed dependency
        and an associated path has been provided for all thresholds, otherwise lightgbm
        Boosters are used as the default tree model type.
        """
        try:
            # Use treelite class, unless subsequent conditions fail.
            cls = ApplyRainForestsCalibrationTreelite
            # Try and initialise the treelite_runtime library to test if the package
            # is available.
            import treelite_runtime  # noqa: F401

            # Check that all required files have been specified.
            treelite_model_filenames = [
                threshold_dict.get("treelite_model")
                for threshold_dict in model_config_dict.values()
            ]
            if None in treelite_model_filenames:
                raise ValueError(
                    "Path to treelite model missing for one or more error thresholds "
                    "in model_config_dict, defaulting to using lightGBM models."
                )
        except (ModuleNotFoundError, ValueError):
            # Default to lightGBM.
            cls = ApplyRainForestsCalibrationLightGBM
            # Ensure all required files have been specified.
            lightgbm_model_filenames = [
                threshold_dict.get("lightgbm_model")
                for threshold_dict in model_config_dict.values()
            ]
            if None in lightgbm_model_filenames:
                raise ValueError(
                    "Path to lightgbm model missing for one or more error thresholds "
                    "in model_config_dict."
                )
        return super(ApplyRainForestsCalibration, cls).__new__(cls)

    def process(self) -> None:
        """Subclasses should override this function."""
        raise NotImplementedError(
            "Process function must be called via subclass method."
        )


class ApplyRainForestsCalibrationLightGBM(ApplyRainForestsCalibration):
    """Class to calibrate input forecast given via RainForests approach using light-GBM
    tree models"""

    def __new__(cls, model_config_dict: dict, threads: int = 1):
        """Check all model files are available before initialising."""
        lightgbm_model_filenames = [
            threshold_dict.get("lightgbm_model")
            for threshold_dict in model_config_dict.values()
        ]
        if None in lightgbm_model_filenames:
            raise ValueError(
                "Path to lightgbm model missing for one or more error thresholds "
                "in model_config_dict."
            )
        return super(ApplyRainForestsCalibration, cls).__new__(cls)

    def __init__(self, model_config_dict: dict, threads: int = 1):
        """Initialise the tree model variables used in the application of RainForests
        Calibration. LightGBM Boosters are used for tree model predictors.

        Args:
            model_config_dict:
                Dictionary containing Rainforests model configuration variables.
            threads:
                Number of threads to use during prediction with tree-model objects.

        Dictionary is of format::

            {
                "-50.0" : {
                    "lightgbm_model" : "<path_to_lightgbm_model_object>"
                },
                "-25.0" : {
                    "lightgbm_model" : "<path_to_lightgbm_model_object>"
                },
                ...,
                "50.0" : {
                    "lightgbm_model" : "<path_to_lightgbm_model_object>"
                }
            }

        The keys specify the error threshold value, while the associated values
        are the path to the corresponding tree-model objects for that threshold.
        """
        from lightgbm import Booster

        # Dictionary keys represent error thresholds, however may be strings as they
        # are sourced from json files. In order use these in processing, and to sort
        # them in a sensible fashion, we shall cast the key values as float32.
        sorted_model_config_dict = OrderedDict(
            sorted({np.float32(k): v for k, v in model_config_dict.items()}.items())
        )

        self.error_thresholds = np.array([*sorted_model_config_dict.keys()])
        self.model_input_converter = np.array
        lightgbm_model_filenames = [
            Path(threshold_dict.get("lightgbm_model")).expanduser()
            for threshold_dict in sorted_model_config_dict.values()
        ]
        self.tree_models = [
            Booster(model_file=str(file)).reset_parameter({"num_threads": threads})
            for file in lightgbm_model_filenames
        ]

    def _check_num_features(self, features: CubeList) -> None:
        """Check that the correct number of features has been passed into the model.

        Args:
            features:
                Cubelist containing feature variables.
        """
        expected_num_features = self.tree_models[0].num_feature()
        if expected_num_features != len(features):
            raise ValueError(
                "Number of expected features does not match number of feature cubes."
            )

    def _align_feature_variables(
        self, feature_cubes: CubeList, forecast_cube: Cube
    ) -> Tuple[CubeList, Cube]:
        """Ensure that feature cubes have consistent dimension coordinates. If realization
        dimension present in any cube, all cubes lacking this dimension will have realization
        dimension added and broadcast along this new dimension.

        This situation occurs when derived fields (such as accumulated solar radiation)
        are used as predictors. As these fields do not contain a realization dimension,
        they must be broadcast to match the NWP fields that do contain realization, so that
        all features have consistent shape.

        In the case of deterministic models (those without a realization dimension), a
        realization dimension is added to allow consistent behaviour between ensemble and
        deterministic models.

        Args:
            feature_cubes:
                Cubelist containing feature variables to align.
            forecast_cube:
                Cube containing the forecast variable to align.

        Returns:
            - feature_cubes with realization coordinate added to each cube if absent
            - forecast_cube with realization coordinate added if absent

        Raises:
            ValueError:
                if feature/forecast variables have inconsistent dimension coordinates
                (excluding realization dimension), or if feature/forecast variables have
                different length realization coordinate over cubes containing a realization
                coordinate.
        """
        combined_cubes = CubeList(list([*feature_cubes, forecast_cube]))

        # Compare feature cube coordinates, raise error if dim-coords don't match
        compare_feature_coords = compare_coords(
            combined_cubes, ignored_coords=["realization"]
        )
        for misaligned_coords in compare_feature_coords:
            for coord_info in misaligned_coords.values():
                if coord_info["data_dims"] is not None:
                    raise ValueError(
                        "Mismatch between non-realization dimension coords."
                    )

        # Compare realization coordinates across cubes where present;
        # raise error if realization coordinates don't match, otherwise set
        # common_realization_coord to broadcast over.
        realization_coords = {
            variable.name(): variable.coord("realization")
            for variable in combined_cubes
            if variable.coords("realization")
        }
        if not realization_coords:
            # Case I: realization_coords is empty. Add single realization dim to all cubes.
            common_realization_coord = DimCoord(
                [0], standard_name="realization", units=1
            )
        else:
            # Case II: realization_coords is not empty.
            # Note: In future, another option here could be to filter to common realization
            # values using filter_realizations() in utilities.cube_manipulation.
            variables_with_realization = list(realization_coords.keys())
            sample_realization = realization_coords[variables_with_realization[0]]
            for feature in variables_with_realization[1:]:
                if realization_coords[feature] != sample_realization:
                    raise ValueError("Mismatch between realization dimension coords.")
            common_realization_coord = sample_realization

        # Add realization coord to cubes where absent by broadcasting along this dimension
        aligned_cubes = CubeList()
        for cube in combined_cubes:
            if not cube.coords("realization"):
                expanded_cube = add_coordinate_to_cube(
                    cube, new_coord=common_realization_coord
                )
                aligned_cubes.append(expanded_cube)
            else:
                aligned_cubes.append(cube)

        # Make data contiguous (required for numba interpolation)
        for cube in aligned_cubes:
            if not cube.data.flags["C_CONTIGUOUS"]:
                cube.data = np.ascontiguousarray(cube.data, dtype=cube.data.dtype)

        return aligned_cubes[:-1], aligned_cubes[-1]

    def _prepare_error_probability_cube(self, forecast_cube):
        """Initialise a cube with the same dimensions as the input forecast_cube,
        with an additional threshold dimension added as the leading dimension.

        Args:
            forecast_cube:
                Cube containing the forecast to be calibrated.

        Returns:
            An empty probability cube.
        """
        # Create a template for error CDF, with threshold the leading dimension.
        forecast_error_variable = f"forecast_error_of_{forecast_cube.name()}"

        error_probability_cube = create_new_diagnostic_cube(
            name=f"probability_of_{forecast_error_variable}_above_threshold",
            units="1",
            template_cube=forecast_cube,
            mandatory_attributes=generate_mandatory_attributes([forecast_cube]),
        )
        error_threshold_coord = DimCoord(
            self.error_thresholds,
            long_name=forecast_error_variable,
            var_name="threshold",
            units=forecast_cube.units,
            attributes={"spp__relative_to_threshold": "above"},
        )
        error_probability_cube = add_coordinate_to_cube(
            error_probability_cube, new_coord=error_threshold_coord,
        )

        return error_probability_cube

    def _prepare_features_array(self, feature_cubes: CubeList) -> ndarray:
        """Convert gridded feature cubes into a numpy array, with feature variables
        sorted alphabetically.

        Note: It is expected that feature_cubes has been aligned using
        _align_feature_variables prior to calling this function.

        Args:
            feature_cubes:
                Cubelist containing the independent feature variables for prediction.

        Returns:
            Array containing flattened feature variables,

        Raises:
            ValueError:
                If flattened cubes have differing length.
        """
        # Get the names of features and sort alphabetically
        feature_variables = [cube.name() for cube in feature_cubes]
        feature_variables.sort()

        # Unpack the cube-data into an array to feed into the tree-models.
        features_list = []
        for feature in feature_variables:
            cube = feature_cubes.extract_cube(feature)
            features_list.append(cube.data.ravel()[:, np.newaxis])
        features_arr = np.concatenate(features_list, axis=1)

        return features_arr

    def _make_decreasing(self, probability_data: ndarray) -> ndarray:
        """Enforce monotonicity on the error CDF data, where threshold dimension
        is assumed to be the leading dimension.

        This is achieved by identifying the minimum value progressively along
        the leading dimension by comparing to all preceding probability values along
        this dimension. The same is done for maximum values, comparing to all
        succeeding values along the leading dimension. Averaging these resulting
        arrays results in an array decreasing monotonically in the threshold dimension.

        Args:
            probability_data:
                The error probability data as exceedence probabilities.

        Returns:
            The error probability data, enforced to be monotonically decreasing along
            the leading dimension.
        """
        lower = np.minimum.accumulate(probability_data, axis=0)
        upper = np.flip(
            np.maximum.accumulate(np.flip(probability_data, axis=0), axis=0), axis=0
        )
        return 0.5 * (upper + lower)

    def _evaluate_probabilities(
        self,
        forecast_data: ndarray,
        input_data: ndarray,
        forecast_variable: str,
        forecast_variable_unit: str,
        output_data: ndarray,
    ) -> None:
        """Evaluate probability that error in forecast exceeds thresholds, setting
        the result to 1 when `forecast + threshold` is less than or equal to
        the lower bound of forecast_variable, as defined in constants.BOUNDS_FOR_ECDF`.

        Args:
            forecast_data:
                1-d containing data for the variable to be calibrated.
            input_data:
                2-d array of data for the feature variables of the model
            forecast_variable:
                name of forecast variable
            forecast_variable_unit:
                unit of forecast variable
            output_data:
                array to populate with output; will be modified in place
        """

        input_dataset = self.model_input_converter(input_data)

        bounds_data = get_bounds_of_distribution(
            forecast_variable, forecast_variable_unit
        )
        lower_bound_in_fcst_units = bounds_data[0]

        for threshold_index, model in enumerate(self.tree_models):
            # it is assumed that models have been trained using the >= operator
            # (i.e. they predict the probabilty that error >= threshold)
            threshold = self.error_thresholds[threshold_index]
            if threshold >= 0:
                # In this case, for all values of forecast we have
                # forecast + threshold >= forecast >= lower_bound_in_fcst_units
                prediction = model.predict(input_dataset)
            else:
                # In this case, we have error > threshold if and only if
                # observations > forecast + threshold, which has probability 1
                # if forecast + threshold < lower_bound_in_fcst_units
                prediction = np.ones(input_data.shape[0], dtype=np.float32)
                forecast_bool = forecast_data + threshold >= lower_bound_in_fcst_units
                if np.any(forecast_bool):
                    input_subset = self.model_input_converter(input_data[forecast_bool])
                    prediction[forecast_bool] = model.predict(input_subset)
            output_data[threshold_index, :] = np.reshape(
                prediction, output_data.shape[1:]
            )
        return

    def _calculate_error_probabilities(
        self, forecast_cube: Cube, feature_cubes: CubeList,
    ) -> Cube:
        """Evaluate the error exceedence probabilities for forecast_cube using the tree_models,
        with the associated feature_cubes taken as inputs to the tree_model predictors.

        Args:
            forecast_cube:
                Cube containing the variable to be calibrated.
            feature_cubes:
                Cubelist containing the independent feature variables for prediction.

        Returns:
            A cube containing error exceedence probabilities.

        Raises:
            ValueError:
                If an unsupported model object is passed. Expects lightgbm Booster, or
                treelite_runtime Predictor (if treelite dependency is available).
        """

        error_probability_cube = self._prepare_error_probability_cube(forecast_cube)

        input_dataset = self._prepare_features_array(feature_cubes)

        forecast_data = forecast_cube.data.ravel()

        self._evaluate_probabilities(
            forecast_data,
            input_dataset,
            forecast_cube.name(),
            forecast_cube.units,
            error_probability_cube.data,
        )

        # Enforcing monotonicity
        error_probability_cube.data = self._make_decreasing(error_probability_cube.data)

        return error_probability_cube

    def _get_ensemble_distributions(
        self, error_CDF: Cube, forecast: Cube, output_thresholds: List[float]
    ) -> Cube:
        """
        Add the error CDF to the NWP forecast and extract probabilities at output thresholds
        for all realizations.

        Args:
            error_CDF:
                Cube containing the CDF probabilities for each realization across the error
                thresholds.
            forecast:
                Cube containing NWP ensemble forecast.
            output_thresholds:
                Ordered list of thresholds at which to calculate the output probabilities.

        Returns:
            Cube containing probabilities at output thresholds for all realizations. Dimensions
            are same as forecast cube with additional threshold dimension first.
        """

        # Add error to forecast. The thresholds for each grid point are obtained
        # by adding the forecast at that point to error_values; thus
        # we get a different set of thresholds for each grid point.
        error_values = error_CDF.coord(var_name="threshold").points
        error_values = np.expand_dims(
            error_values, axis=tuple(range(1, forecast.ndim + 1))
        )
        thresholds = error_values + forecast.data[np.newaxis, :]
        probabilities = error_CDF.data

        # Add an extra threshold below/above with one/zero probability to
        # ensure that the PDF covers the full probability range.
        # Thresholds are usually float32 with epsilon of ~= 1.1e-7
        eps = np.finfo(thresholds.dtype).eps
        # For small values (especially exactly zero), at least epsilon;
        # for larger values, the next representable float number.
        thresholds = np.concatenate(
            [
                thresholds[[0]] - np.maximum(eps, np.abs(thresholds[[0]] * eps)),
                thresholds,
                thresholds[[-1]] + np.maximum(eps, np.abs(thresholds[[-1]] * eps)),
            ],
            axis=0,
        )
        probabilities = np.concatenate(
            [
                np.ones((1,) + probabilities.shape[1:], np.float32),
                probabilities,
                np.zeros((1,) + probabilities.shape[1:], np.float32),
            ],
            axis=0,
            dtype=np.float32,
        )

        # get bounds
        bounds_data = get_bounds_of_distribution(forecast.name(), forecast.units)
        lower_bound_in_fcst_units = bounds_data[0]

        # Set probability to 1 for thresholds <= lower bound
        threshold_below_bound = thresholds <= lower_bound_in_fcst_units
        thresholds = np.where(
            threshold_below_bound, lower_bound_in_fcst_units, thresholds
        )
        probabilities = np.where(threshold_below_bound, 1, probabilities)

        # Interpolate to a common set of output thresholds.
        output_thresholds = np.sort(output_thresholds).astype(np.float32)
        output_array = interpolate_pointwise(
            output_thresholds, thresholds, probabilities
        )

        # Make output cube
        probability_cube = create_new_diagnostic_cube(
            name=f"probability_of_{forecast.name()}_above_threshold",
            units="1",
            template_cube=forecast,
            mandatory_attributes=generate_mandatory_attributes([forecast]),
            optional_attributes=forecast.attributes,
        )
        threshold_coord = DimCoord(
            output_thresholds,
            standard_name=forecast.name(),
            var_name="threshold",
            units=forecast.units,
            attributes={"spp__relative_to_threshold": "greater_than_or_equal_to"},
        )
        probability_cube = add_coordinate_to_cube(
            probability_cube, new_coord=threshold_coord,
        )
        probability_cube.data = output_array
        return probability_cube

    def process(
        self,
        forecast_cube: Cube,
        feature_cubes: CubeList,
        output_thresholds: List,
        threshold_units: str = None,
    ) -> Cube:
        """Apply rainforests calibration to forecast cube.

        Ensemble forecasts must be in realization representation. Deterministic forecasts
        can be processed to produce a pseudo-ensemble; a realization dimension will be added
        to deterministic forecast cubes if one is not already present.

        The calibration is done in a situation dependent fashion using a series of
        decision-tree models to construct representative error distributions which are
        then used to map each input ensemble member onto a series of realisable values.

        These error distributions are formed in a two-step process:

        1. Evaluate error CDF defined over the specified error_thresholds. Each exceedence
        probability is evaluated using the corresponding decision-tree model.

        2. Interpolate the CDF to extract a series of percentiles for the error distribution.
        The error percentiles are then applied to each associated ensemble realization to
        produce an error distribution. Distributions for different ensemble members are
        then interpolated to the same set of output thresholds and averaged in probability space.

        Args:
            forecast_cube:
                Cube containing the forecast to be calibrated; must be as realizations.
            feature_cubes:
                Cubelist containing the feature variables (physical parameters) used as inputs
                to the tree-models for the generation of the associated error distributions.
                Feature cubes are expected to have the same dimensions as forecast_cube, with
                the exception of the realization dimension. Where the feature_cube contains a
                realization dimension this is expected to be consistent, otherwise the cube will
                be broadcast along the realization dimension.
            output_thresholds (list):
                List of thresholds at which to evaluate output probabilties.
            threshold_units (str):
                Units in which output_thresholds are specified. If None, assumed to be the same as
                forecast_cube.
        Returns:
            The calibrated forecast cube.

        Raises:
            RuntimeError:
                If the number of tree-models is inconsistent with the number of error
                thresholds.
        """
        # Check that the correct number of feature variables has been supplied.
        self._check_num_features(feature_cubes)

        # Align forecast and feature datasets
        aligned_features, aligned_forecast = self._align_feature_variables(
            feature_cubes, forecast_cube
        )

        # Evaluate the error CDF using tree-models.
        error_CDF = self._calculate_error_probabilities(
            aligned_forecast, aligned_features
        )

        if threshold_units:
            original_threshold_unit = unit.Unit(threshold_units)
            forecast_unit = forecast_cube.units
            output_thresholds_in_forecast_units = [
                original_threshold_unit.convert(x, forecast_unit)
                for x in output_thresholds
            ]
        else:
            output_thresholds_in_forecast_units = output_thresholds

        # Calculate probabilities at output thresholds
        probabilities_by_realization = self._get_ensemble_distributions(
            error_CDF, aligned_forecast, output_thresholds_in_forecast_units
        )

        # Average over realizations
        output_cube = probabilities_by_realization.collapsed("realization", MEAN)
        output_cube.remove_coord("realization")

        return output_cube


class ApplyRainForestsCalibrationTreelite(ApplyRainForestsCalibrationLightGBM):
    """Class to calibrate input forecast given via RainForests approach using treelite
    compiled tree models"""

    def __new__(cls, model_config_dict: dict, threads: int = 1):
        """Check required dependency and all model files are available before initialising."""
        # Try and initialise the treelite_runtime library to test if the package
        # is available.
        import treelite_runtime  # noqa: F401

        treelite_model_filenames = [
            threshold_dict.get("treelite_model")
            for threshold_dict in model_config_dict.values()
        ]
        if None in treelite_model_filenames:
            raise ValueError(
                "Path to treelite model missing for one or more error thresholds "
                "in model_config_dict."
            )
        return super(ApplyRainForestsCalibration, cls).__new__(cls)

    def __init__(self, model_config_dict: dict, threads: int = 1):
        """Initialise the tree model variables used in the application of RainForests
        Calibration. Treelite Predictors are used for tree model predictors.

        Args:
            model_config_dict:
                Dictionary containing Rainforests model configuration variables.
            threads:
                Number of threads to use during prediction with tree-model objects.

        Dictionary is of format::

            {
                "-50.0" : {
                    "treelite_model" : "<path_to_treelite_model_object>"
                },
                "-25.0" : {
                    "treelite_model" : "<path_to_treelite_model_object>"
                },
                ...,
                "50.0" : {
                    "treelite_model" : "<path_to_treelite_model_object>"
                }
            }

        The keys specify the error threshold value, while the associated values
        are the path to the corresponding tree-model objects for that threshold.
        """
        from treelite_runtime import DMatrix, Predictor

        # Dictionary keys represent error thresholds, however may be strings as they
        # are sourced from json files. In order use these in processing, and to sort
        # them in a sensible fashion, we shall cast the key values as float32.
        sorted_model_config_dict = OrderedDict(
            sorted({np.float32(k): v for k, v in model_config_dict.items()}.items())
        )

        self.error_thresholds = np.array([*sorted_model_config_dict.keys()])
        self.model_input_converter = DMatrix
        treelite_model_filenames = [
            Path(threshold_dict.get("treelite_model")).expanduser()
            for threshold_dict in sorted_model_config_dict.values()
        ]
        self.tree_models = [
            Predictor(libpath=str(file), verbose=False, nthread=threads)
            for file in treelite_model_filenames
        ]

    def _check_num_features(self, features: CubeList) -> None:
        """Check that the correct number of features has been passed into the model.
        Args:
            features:
                Cubelist containing feature variables.
        """
        expected_num_features = self.tree_models[0].num_feature
        if expected_num_features != len(features):
            raise ValueError(
                "Number of expected features does not match number of feature cubes."
            )
