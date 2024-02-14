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


import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from cf_units import Unit
from iris.analysis import MEAN
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from numpy import ndarray

from improver import PostProcessingPlugin
from improver.constants import MINUTES_IN_HOUR, SECONDS_IN_MINUTE
from improver.ensemble_copula_coupling.utilities import (
    get_bounds_of_distribution,
    interpolate_multiple_rows_same_x,
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

    def __new__(
        cls,
        model_config_dict: Dict[str, Dict[str, Dict[str, str]]],
        threads: int = 1,
        bin_data: bool = False,
    ):

        """Initialise class object based on package and model file availability.

        Args:
            model_config_dict:
                Dictionary containing Rainforests model configuration variables.
            threads:
                Number of threads to use during prediction with tree-model objects.
            bin_data:
                Bin data according to splits used in models. This speeds up prediction
                if there are many data points which fall into the same bins for all threshold
                models. Limits the calculation of common feature values by only calculating
                them once.

        Dictionary is of format::

        {
        "24": {
            "0.000010": {
                "lightgbm_model": "<path_to_lightgbm_model_object>",
                "treelite_model": "<path_to_treelite_model_object>"
            },
            "0.000050": {
                "lightgbm_model": "<path_to_lightgbm_model_object>",
                "treelite_model": "<path_to_treelite_model_object>"
            },
            "0.000100": {
                "lightgbm_model": "<path_to_lightgbm_model_object>",
                "treelite_model": "<path_to_treelite_model_object>"
            },
        }

        The keys specify the lead times and model threshold values, while the
        associated values are the path to the corresponding tree-model objects
        for that lead time and threshold.

        Treelite predictors are used if treelite_runtime is an installed dependency
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
            ApplyRainForestsCalibration.check_filenames(
                "treelite_model", model_config_dict
            )
        except (ModuleNotFoundError, ValueError):
            # Default to lightGBM.
            cls = ApplyRainForestsCalibrationLightGBM
            # Ensure all required files have been specified.
            ApplyRainForestsCalibration.check_filenames(
                "lightgbm_model", model_config_dict
            )
        return super(ApplyRainForestsCalibration, cls).__new__(cls)

    def _check_num_features(self, features: CubeList) -> None:
        """Check that the correct number of features has been passed into the model.
        Args:
            features:
                Cubelist containing feature variables.
        """
        expected_num_features = self._get_num_features()
        if expected_num_features != len(features):
            raise ValueError(
                "Number of expected features does not match number of feature cubes."
            )

    def _get_feature_splits(self, model_config_dict) -> Dict[int, List[ndarray]]:
        """Get the combined feature splits (over all thresholds) for each lead time.

        Args:
            model_config_dict: dictionary of the same format expected by __init__

        Returns:
            dict where keys are the lead times and the values are lists of lists.
            The outer list has length equal to the number of model features, and it contains
            the lists of feature splits for each feature. Each feature's list of splits is ordered.
        """
# These string patterns are defined by light-gbm and are used for finding the feature and
# threshold information in the model .txt files.
        split_feature_string = "split_feature="
        feature_threshold_string = "threshold="
        combined_feature_splits = {}
        for lead_time in model_config_dict.keys():
            all_splits = [set() for i in range(self._get_num_features())]
            for threshold_str in model_config_dict[lead_time].keys():
                lgb_model_filename = Path(
                    model_config_dict[lead_time][threshold_str].get("lightgbm_model")
                ).expanduser()
                with open(lgb_model_filename, "r") as f:
                    for line in f:
                        if line.startswith(split_feature_string):
                            line = line[len(split_feature_string) : -1]
# This deals with the common situation where there is no splits on this line.
                                continue
                            features = [int(x) for x in line.split(" ")]
                        elif line.startswith(feature_threshold_string):
                            line = line[len(feature_threshold_string) : -1]
                            if len(line) == 0:
                                continue
                            splits = [float(x) for x in line.split(" ")]
                            for feature_ind, threshold in zip(features, splits):
                                all_splits[feature_ind].add(threshold)
            combined_feature_splits[np.float32(lead_time)] = [
                np.sort(list(x)) for x in all_splits
            ]
        return combined_feature_splits

    @staticmethod
    def check_filenames(
        key_name: str, model_config_dict: Dict[str, Dict[str, Dict[str, str]]]
    ):
        """Check whether files specified by model_config_dict exist,
        and raise an error if any are missing.

        Args:
            key_name: 'treelite_model' or 'lightgbm_model' are the expected names.
            model_config_dict: Dictionary containing Rainforests model configuration variables.
        """
        if key_name not in ["lightgbm_model", "treelite_model"]:
            raise ValueError("key_name must be 'lightgbm_model' or 'treelite_model'")
        model_filenames = []
        for lead_time in model_config_dict.keys():
            for threshold in model_config_dict[lead_time].keys():
                model_filenames.append(
                    model_config_dict[lead_time][threshold].get(key_name)
                )
        if None in model_filenames:
            if key_name == "lightgbm_model":
                raise ValueError(
                    "Path to lightgbm model missing for one or more model thresholds "
                    "in model_config_dict."
                )
            elif key_name == "treelite_model":
                raise ValueError(
                    "Path to treelite model missing for one or more model thresholds "
                    "in model_config_dict, defaulting to using lightGBM models."
                )

    def _parse_model_config(
        self, model_config_dict: Dict[str, Dict[str, Dict[str, str]]]
    ) -> Dict[np.float32, Dict[np.float32, Dict[str, str]]]:
        """Parse the model config dictionary, set self.lead_times and self.model_thresholds,
        and return a sorted version of the config dictionary.

        Args:
            model_config_dict: Nested dictionary with string keys. Keys of outer level are
            lead times, and keys of inner level are thresholds.

        Returns:
            Dictionary with the same nested structure as model_config_dict, but
            the lead time and threshold keys now have type np.float.
        """

        sorted_model_config_dict = OrderedDict()
        for key, lead_time_dict in model_config_dict.items():
            sorted_model_config_dict[np.float32(key)] = OrderedDict(
                sorted({np.float32(k): v for k, v in lead_time_dict.items()}.items())
            )

        self.lead_times = np.sort(np.array([*sorted_model_config_dict.keys()]))
        if len(self.lead_times) > 0:
            self.model_thresholds = np.sort(
                np.array([*sorted_model_config_dict[self.lead_times[0]].keys()])
            )
        else:
            warnings.warn(
                "Model config does not match the expected specification; calibration will not work"
            )
            self.model_thresholds = np.array([])
        return sorted_model_config_dict


class ApplyRainForestsCalibrationLightGBM(ApplyRainForestsCalibration):
    """Class to calibrate input forecast given via RainForests approach using light-GBM
    tree models"""

    def __new__(
        cls,
        model_config_dict: Dict[str, Dict[str, Dict[str, str]]],
        threads: int = 1,
        bin_data: bool = False,
    ):
        """Check all model files are available before initialising."""
        ApplyRainForestsCalibration.check_filenames("lightgbm_model", model_config_dict)
        return super(ApplyRainForestsCalibration, cls).__new__(cls)

    def __init__(
        self,
        model_config_dict: Dict[str, Dict[str, Dict[str, str]]],
        threads: int = 1,
        bin_data: bool = False,
    ):
        """Initialise the tree model variables used in the application of RainForests
        Calibration. LightGBM Boosters are used for tree model predictors.

        Args:
            model_config_dict:
                Dictionary containing Rainforests model configuration variables.
            threads:
                Number of threads to use during prediction with tree-model objects.
            bin_data:
                Bin data according to splits used in models. This speeds up prediction
                if there are many data points which fall into the same bins for all threshold
                models. Limits the calculation of common feature values by only calculating
                them once.

        Dictionary is of format::

            {
                "24": {
                    "0.000010": {
                        "lightgbm_model": "<path_to_lightgbm_model_object>",
                        "treelite_model": "<path_to_treelite_model_object>"
                    },
                    "0.000050": {
                        "lightgbm_model": "<path_to_lightgbm_model_object>",
                        "treelite_model": "<path_to_treelite_model_object>"
                    },
                    "0.000100": {
                        "lightgbm_model": "<path_to_lightgbm_model_object>",
                        "treelite_model": "<path_to_treelite_model_object>"
                    },
                }
            }

        The keys specify the lead times and model threshold values, while the
        associated values are the path to the corresponding tree-model objects
        for that lead time and threshold.
        """
        from lightgbm import Booster

        sorted_model_config_dict = self._parse_model_config(model_config_dict)
        self.model_input_converter = np.array
        self.tree_models = {}
        for lead_time in self.lead_times:
            # check all lead times have the same thresholds
            curr_thresholds = np.array([*sorted_model_config_dict[lead_time].keys()])
            if np.any(curr_thresholds != self.model_thresholds):
                raise ValueError(
                    "The same thresholds must be used for all lead times. "
                    f"Lead time {self.lead_times[0]} has thresholds: {self.model_thresholds},"
                    f"lead time {lead_time} has thresholds: {curr_thresholds}"
                )
            for threshold in self.model_thresholds:
                model_filename = Path(
                    sorted_model_config_dict[lead_time][threshold].get("lightgbm_model")
                ).expanduser()
                self.tree_models[lead_time, threshold] = Booster(
                    model_file=str(model_filename)
                ).reset_parameter({"num_threads": threads})

        self.bin_data = bin_data
        if self.bin_data:
            self.combined_feature_splits = self._get_feature_splits(model_config_dict)

    def _get_num_features(self) -> int:
        return next(iter(self.tree_models.values())).num_feature()

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

    def _prepare_threshold_probability_cube(self, forecast_cube, thresholds):
        """Initialise a cube with the same dimensions as the input forecast_cube,
        with an additional threshold dimension added as the leading dimension.

        Args:
            forecast_cube:
                Cube containing the forecast to be calibrated.
            thresholds:
                Points of the the threshold dimension.

        Returns:
            An empty probability cube.
        """
        # Create a template for CDF, with threshold the leading dimension.
        forecast_variable = forecast_cube.name()

        probability_cube = create_new_diagnostic_cube(
            name=f"probability_of_{forecast_variable}_above_threshold",
            units="1",
            template_cube=forecast_cube,
            mandatory_attributes=generate_mandatory_attributes([forecast_cube]),
            optional_attributes=forecast_cube.attributes,
        )
        threshold_coord = DimCoord(
            thresholds,
            standard_name=forecast_variable,
            var_name="threshold",
            units=forecast_cube.units,
            attributes={"spp__relative_to_threshold": "greater_than_or_equal_to"},
        )
        probability_cube = add_coordinate_to_cube(
            probability_cube, new_coord=threshold_coord,
        )

        return probability_cube

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
        """Enforce monotonicity on the CDF data, where threshold dimension
        is assumed to be the leading dimension.

        This is achieved by identifying the minimum value progressively along
        the leading dimension by comparing to all preceding probability values along
        this dimension. The same is done for maximum values, comparing to all
        succeeding values along the leading dimension. Averaging these resulting
        arrays results in an array decreasing monotonically in the threshold dimension.

        Args:
            probability_data:
                The probability data as exceedence probabilities.

        Returns:
            The probability data, enforced to be monotonically decreasing along
            the leading dimension.
        """
        lower = np.minimum.accumulate(probability_data, axis=0)
        upper = np.flip(
            np.maximum.accumulate(np.flip(probability_data, axis=0), axis=0), axis=0
        )
        return 0.5 * (upper + lower)

    def _evaluate_probabilities(
        self, input_data: ndarray, lead_time_hours: int, output_data: ndarray,
    ) -> None:
        """Evaluate probability that forecast exceeds thresholds.

        Args:
            input_data:
                2-d array of data for the feature variables of the model
            lead_time_hours:
                lead time in hours
            output_data:
                array to populate with output; will be modified in place
        """

        if np.float32(lead_time_hours) in self.lead_times:
            model_lead_time = np.float32(lead_time_hours)
        else:
            # find closest model lead time
            best_ind = np.argmin(np.abs(self.lead_times - lead_time_hours))
            model_lead_time = self.lead_times[best_ind]

        if self.bin_data:
            # bin by feature splits
            feature_splits = self.combined_feature_splits[model_lead_time]
            binned_data = np.empty(input_data.shape, dtype=np.int32)
            n_features = len(feature_splits)
            for i in range(n_features):
                binned_data[:, i] = np.digitize(
                    input_data[:, i], bins=feature_splits[i]
                )
            # sort so rows in the same bins are grouped
            sort_ind = np.lexsort(
                tuple([binned_data[:, i] for i in range(n_features)])
            )
            sorted_data = binned_data[sort_ind]
            reverse_sort_ind = np.argsort(sort_ind)
            # we only need to predict for rows which are different from the previous row
            diff = np.any(np.diff(sorted_data, axis=0) != 0, axis=1)
            predict_rows = np.concatenate([[0], np.nonzero(diff)[0] + 1])
            data_for_prediction = input_data[sort_ind][predict_rows]
            full_prediction = np.empty((input_data.shape[0],))
            # forward fill code inspired by this: https://stackoverflow.com/a/41191127
            fill_inds = np.zeros(len(full_prediction), dtype=np.int32)
            fill_inds[predict_rows] = predict_rows
            fill_inds = np.maximum.accumulate(fill_inds)
            dataset_for_prediction = self.model_input_converter(data_for_prediction)

            for threshold_index, threshold in enumerate(self.model_thresholds):
                model = self.tree_models[model_lead_time, threshold]
                prediction = model.predict(dataset_for_prediction)
                prediction = np.clip(prediction, 0, 1)
                full_prediction[predict_rows] = prediction
                full_prediction = full_prediction[fill_inds]
                # restore original order
                full_prediction = full_prediction[reverse_sort_ind]
                output_data[threshold_index, :] = np.reshape(
                    full_prediction, output_data.shape[1:]
                )
        else:
            dataset_for_prediction = self.model_input_converter(input_data)
            for threshold_index, threshold in enumerate(self.model_thresholds):
                model = self.tree_models[model_lead_time, threshold]
                prediction = model.predict(dataset_for_prediction)
                prediction = np.clip(prediction, 0, 1)
                output_data[threshold_index, :] = np.reshape(
                    prediction, output_data.shape[1:]
                )

    def _calculate_threshold_probabilities(
        self, forecast_cube: Cube, feature_cubes: CubeList,
    ) -> Cube:
        """Evaluate the threshold exceedence probabilities for each ensemble member in
        forecast_cube using the tree_models, with the associated feature_cubes taken as
        inputs to the tree_model predictors.

        Args:
            forecast_cube:
                Cube containing the variable to be calibrated.
            feature_cubes:
                Cubelist containing the independent feature variables for prediction.

        Returns:
            A cube containing threshold exceedence probabilities.

        Raises:
            ValueError:
                If an unsupported model object is passed. Expects lightgbm Booster, or
                treelite_runtime Predictor (if treelite dependency is available).
        """

        threshold_probability_cube = self._prepare_threshold_probability_cube(
            forecast_cube, self.model_thresholds
        )

        input_dataset = self._prepare_features_array(feature_cubes)

        lead_time_hours = forecast_cube.coord("forecast_period").points[0] / (
            SECONDS_IN_MINUTE * MINUTES_IN_HOUR
        )

        self._evaluate_probabilities(
            input_dataset, lead_time_hours, threshold_probability_cube.data,
        )

        # Enforcing monotonicity
        threshold_probability_cube.data = self._make_decreasing(
            threshold_probability_cube.data
        )

        return threshold_probability_cube

    def _get_ensemble_distributions(
        self, per_realization_CDF: Cube, forecast: Cube, output_thresholds: ndarray
    ) -> Cube:
        """
        Interpolate probabilities calculated at model thresholds to extract probabilities
        at output thresholds for all realizations.

        Args:
            per_realization_CDF:
                Cube containing the CDF probabilities for each ensemble member at model
                thresholds, with threshold as the first dimension.
            forecast:
                Cube containing NWP ensemble forecast.
            output_thresholds:
                Sorted array of thresholds at which to calculate the output probabilities.

        Returns:
            Cube containing probabilities at output thresholds for all realizations. Dimensions
            are same as forecast cube with additional threshold dimension first.
        """

        input_probabilities = per_realization_CDF.data
        output_thresholds = np.array(output_thresholds, dtype=np.float32)
        bounds_data = get_bounds_of_distribution(forecast.name(), forecast.units)
        lower_bound = bounds_data[0].astype(np.float32)
        if (len(self.model_thresholds) == len(output_thresholds)) and np.allclose(
            self.model_thresholds, output_thresholds
        ):
            output_probabilities = np.copy(input_probabilities.data)
        else:
            # add lower bound with probability 1
            input_probabilities = np.concatenate(
                [
                    np.ones((1,) + input_probabilities.shape[1:], dtype=np.float32),
                    input_probabilities,
                ],
                axis=0,
            )
            input_thresholds = np.concatenate([[lower_bound], self.model_thresholds])
            # reshape to 2 dimensions
            input_probabilities_2d = np.reshape(
                input_probabilities, (input_probabilities.shape[0], -1)
            )
            output_probabilities_2d = interpolate_multiple_rows_same_x(
                output_thresholds, input_thresholds, input_probabilities_2d.transpose()
            )
            output_probabilities = np.reshape(
                output_probabilities_2d.transpose(),
                (len(output_thresholds),) + input_probabilities.shape[1:],
            )
            # force interpolated probabilities to be monotone (sometimes they
            # are not due to small floating-point errors)
            output_probabilities = self._make_decreasing(output_probabilities)

        # set probability for lower bound to 1
        if np.isclose(output_thresholds[0], lower_bound):
            output_probabilities[0, :] = 1

        # Make output cube
        probability_cube = self._prepare_threshold_probability_cube(
            forecast, output_thresholds
        )
        probability_cube.data = output_probabilities.astype(np.float32)
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
        decision-tree models to construct representative probability distributions for
        each input ensemble member which are then blended to give the calibrated
        distribution for the full ensemble.

        These distributions are formed in a two-step process:

        1. Evaluate CDF defined over the specified model thresholds for each ensemble member.
        Each threshold exceedance probability is evaluated using the corresponding
        decision-tree model.

        2. Interpolate each ensemble member distribution to the output thresholds.

        Args:
            forecast_cube:
                Cube containing the forecast to be calibrated; must be as realizations.
            feature_cubes:
                Cubelist containing the feature variables (physical parameters) used as inputs
                to the tree-models for the generation of the associated probability distributions.
                Feature cubes are expected to have the same dimensions as forecast_cube, with
                the exception of the realization dimension. Where the feature_cube contains a
                realization dimension this is expected to be consistent, otherwise the cube will
                be broadcast along the realization dimension.
            output_thresholds:
                Set of output thresholds.
            threshold_units:
                Units in which output_thresholds are specified. If None, assumed to be the same as
                forecast_cube.

        Returns:
            The calibrated forecast cube.

        Raises:
            RuntimeError:
                If the number of tree models is inconsistent with the number of model
                thresholds.
        """
        # Check that the correct number of feature variables has been supplied.
        self._check_num_features(feature_cubes)

        # Align forecast and feature datasets
        aligned_features, aligned_forecast = self._align_feature_variables(
            feature_cubes, forecast_cube
        )

        # Evaluate the CDF using tree models.
        per_realization_CDF = self._calculate_threshold_probabilities(
            aligned_forecast, aligned_features
        )

        # convert units of output thresholds
        if threshold_units:
            original_threshold_unit = Unit(threshold_units)
            forecast_unit = forecast_cube.units
            output_thresholds_in_forecast_units = np.array(
                [
                    original_threshold_unit.convert(x, forecast_unit)
                    for x in output_thresholds
                ]
            )
        else:
            output_thresholds_in_forecast_units = np.array(output_thresholds)

        # Calculate probabilities at output thresholds
        interpolated_per_realization_CDF = self._get_ensemble_distributions(
            per_realization_CDF, aligned_forecast, output_thresholds_in_forecast_units
        )

        # Average over realizations
        calibrated_probability_cube = interpolated_per_realization_CDF.collapsed(
            "realization", MEAN
        )
        calibrated_probability_cube.remove_coord("realization")

        return calibrated_probability_cube


class ApplyRainForestsCalibrationTreelite(ApplyRainForestsCalibrationLightGBM):
    """Class to calibrate input forecast given via RainForests approach using treelite
    compiled tree models"""

    def __new__(
        cls,
        model_config_dict: Dict[str, Dict[str, Dict[str, str]]],
        threads: int = 1,
        bin_data: bool = False,
    ):

        """Check required dependency and all model files are available before initialising."""
        # Try and initialise the treelite_runtime library to test if the package
        # is available.
        import treelite_runtime  # noqa: F401

        # Check that all required files have been specified.
        ApplyRainForestsCalibration.check_filenames("treelite_model", model_config_dict)
        return super(ApplyRainForestsCalibration, cls).__new__(cls)

    def __init__(
        self,
        model_config_dict: Dict[str, Dict[str, Dict[str, str]]],
        threads: int = 1,
        bin_data: bool = False,
    ):
        """Initialise the tree model variables used in the application of RainForests
        Calibration. Treelite Predictors are used for tree model predictors.

        Args:
            model_config_dict:
                Dictionary containing Rainforests model configuration variables.
            threads:
                Number of threads to use during prediction with tree-model objects.
            bin_data:
                Bin data according to splits used in models. This speeds up prediction
                if there are many data points which fall into the same bins for all threshold
                models. Limits the calculation of common feature values by only calculating
                them once.

        Dictionary is of format::

            {
                "24": {
                    "0.000010": {
                        "lightgbm_model": "<path_to_lightgbm_model_object>",
                        "treelite_model": "<path_to_treelite_model_object>"
                    },
                    "0.000050": {
                        "lightgbm_model": "<path_to_lightgbm_model_object>",
                        "treelite_model": "<path_to_treelite_model_object>"
                    },
                    "0.000100": {
                        "lightgbm_model": "<path_to_lightgbm_model_object>",
                        "treelite_model": "<path_to_treelite_model_object>"
                    },
                }
            }

        The keys specify the model threshold value, while the associated values
        are the path to the corresponding tree-model objects for that threshold.
        """
        from treelite_runtime import DMatrix, Predictor

        sorted_model_config_dict = self._parse_model_config(model_config_dict)
        self.model_input_converter = DMatrix
        self.tree_models = {}
        for lead_time in self.lead_times:
            # check all lead times have the same thresholds
            curr_thresholds = np.array([*sorted_model_config_dict[lead_time].keys()])
            if np.any(curr_thresholds != self.model_thresholds):
                raise ValueError("The same thresholds must be used for all lead times.")
            for threshold in self.model_thresholds:
                model_filename = Path(
                    sorted_model_config_dict[lead_time][threshold].get("treelite_model")
                ).expanduser()
                self.tree_models[lead_time, threshold] = Predictor(
                    libpath=str(model_filename), verbose=False, nthread=threads
                )

        self.bin_data = bin_data
        if self.bin_data:
            self.combined_feature_splits = self._get_feature_splits(model_config_dict)

    def _get_num_features(self) -> int:
        return next(iter(self.tree_models.values())).num_feature
