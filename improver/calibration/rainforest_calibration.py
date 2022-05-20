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
"""RainForests calibration Plugins."""

import warnings
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from iris.util import new_axis
from numpy import ndarray
from pandas import DataFrame

from improver import PostProcessingPlugin
from improver.cli import generate_percentiles
from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    ConvertProbabilitiesToPercentiles,
    RebadgePercentilesAsRealizations,
)
from improver.ensemble_copula_coupling.utilities import choose_set_of_percentiles
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.synthetic_data.set_up_test_cubes import add_coordinate
from improver.utilities.cube_manipulation import compare_coords

# Passed to choose_set_of_percentiles to set of evenly spaced percentiles
DEFAULT_ERROR_PERCENTILES_COUNT = 19
DEFAULT_OUTPUT_REALIZATIONS_COUNT = 100


class ApplyRainForestsCalibration(PostProcessingPlugin):
    """Class to calibrate input forecast given a series of RainForests tree models."""

    def __init__(self, model_config_dict: dict, threads: int):
        """Initialise the tree model variables used in the application of RainForests
        Calibration.

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
        from lightgbm import Booster

        try:
            from treelite_runtime import Predictor
        except ModuleNotFoundError:
            warnings.warn(
                "Module treelite_runtime unavailable. Defaulting to using lightgbm Boosters."
            )
            self.treelite_enabled = False
        else:
            self.treelite_enabled = True

        # Dictionary keys represent error thresholds, however may be strings as they
        # are sourced from json files. In order use these in processing, and to sort
        # them in a sensible fashion, we shall cast the key values as float32.
        sorted_model_config_dict = OrderedDict(
            sorted({np.float32(k): v for k, v in model_config_dict.items()}.items())
        )

        self.error_thresholds = np.array([*sorted_model_config_dict.keys()])

        lightgbm_model_filenames = [
            threshold_dict.get("lightgbm_model")
            for threshold_dict in sorted_model_config_dict.values()
        ]
        treelite_model_filenames = [
            threshold_dict.get("treelite_model")
            for threshold_dict in sorted_model_config_dict.values()
        ]
        if (None not in treelite_model_filenames) and self.treelite_enabled:
            self.tree_models = [
                Predictor(libpath=file, verbose=False, nthread=threads)
                for file in treelite_model_filenames
            ]
        else:
            if None in lightgbm_model_filenames:
                raise ValueError(
                    "Path to lightgbm model missing for one or more error thresholds "
                    "in model_config_dict."
                )
            self.tree_models = [
                Booster(model_file=file).reset_parameter({"num_threads": threads})
                for file in lightgbm_model_filenames
            ]

    def _check_num_features(self, features: CubeList) -> None:
        """Check that the correct number of features has been passed into the model."""
        sample_tree_model = self.tree_models[0]
        if self.treelite_enabled:
            from treelite_runtime import Predictor

            if isinstance(sample_tree_model, Predictor):
                expected_num_features = sample_tree_model.num_feature
            else:
                expected_num_features = sample_tree_model.num_feature()
        else:
            expected_num_features = sample_tree_model.num_feature()

        if expected_num_features != len(features):
            raise ValueError(
                "Number of expected features does not match number of feature cubes."
            )

    def _align_feature_variables(
        self, feature_cubes: CubeList, forecast_cube: Cube
    ) -> Tuple[CubeList, Cube]:
        """Ensure that feature cubes have consistent dimension coordinates. If
        realization dimension present in any cube, all cubes lacking this dimension
        will have realization dimension added and broadcast along this new dimension.

        Args:
            feature_cubes:
                cube list containing feature variables to align.
            forecast_cube:
                cube containing the forecast variable to align.

        Returns:
            - feature_cubes with realization coordinate added to each cube if absent
            - forecast_cube with realization coordinate added if absent

        Raises:
            ValueError:
                if feature/forecast variables have inconsistent dimension coordinates
                (excluding realization dimension), or if feature/forecast variables have
                different length realization coordinate over cubes containing this coordinate.
        """
        combined_cubes = CubeList(list([*feature_cubes, forecast_cube]))

        # Compare feature cube coordinates, raise error if dim-coords don't match
        compare_feature_coords = compare_coords(
            combined_cubes, ignored_coords=["realization"]
        )
        misaligned_dim_coords = [
            coord_info["coord"]
            for misaligned_coords in compare_feature_coords
            for coord, coord_info in misaligned_coords.items()
            if coord_info["data_dims"] is not None
        ]
        if misaligned_dim_coords:
            raise ValueError(
                f"Dimension coords do not match between: {misaligned_dim_coords}"
            )

        # Compare realization coordinates across cubes where present;
        # raise error if realization coordinates don't match, otherwise set
        # common_realization_coord to broadcast over.
        realization_coords = {
            variable.name(): variable.coords("realization")
            for variable in combined_cubes
            if variable.coords("realization")
        }
        if not realization_coords:
            # Case I: realization_coords is empty. Add single realization dim to all cubes.
            common_realization_coord = DimCoord(
                [0], standard_name="realization", units=1, var_name="realization"
            )
        else:
            # Case II: realization_coords is not empty.
            variables_with_realization = list(realization_coords.keys())
            sample_variable = variables_with_realization[0]
            sample_realization = realization_coords[sample_variable][0]
            misaligned_realizations = [
                feature
                for feature in variables_with_realization[1:]
                if realization_coords[feature][0] != sample_realization
            ]
            if misaligned_realizations:
                misaligned_realizations.append(sample_variable)
                raise ValueError(
                    f"Realization coords  do not match between: {misaligned_realizations}"
                )
            common_realization_coord = sample_realization

        # Add realization coord to cubes where absent by broadcasting along this dimension
        aligned_cubes = CubeList()
        for i_cube, cube in enumerate(combined_cubes):
            if not cube.coords("realization"):
                cube = combined_cubes[i_cube]
                expanded_cube = add_coordinate(
                    cube,
                    common_realization_coord.points,
                    common_realization_coord.name(),
                    common_realization_coord.units,
                    dtype=common_realization_coord.dtype,
                )
                # The above method will add realization as an aux_coord for deterministic
                # data (single realization). We need this as a dim-coord, so we will add
                # an axis for realization.
                if expanded_cube.coord_dims("realization") != (0,):
                    expanded_cube = new_axis(expanded_cube, scalar_coord="realization")
                aligned_cubes.append(expanded_cube)
            else:
                aligned_cubes.append(combined_cubes[i_cube])

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
        error_probability_cube = add_coordinate(
            error_probability_cube,
            self.error_thresholds,
            forecast_error_variable,
            forecast_cube.units,
            attributes={"spp__relative_to_threshold": "above"},
        )
        error_probability_cube.coord(forecast_error_variable).var_name = "threshold"

        return error_probability_cube

    def _prepare_features_dataframe(self, feature_cubes: Cube) -> DataFrame:
        """Convert gridded feature cubes into a dataframe, with feature variables
        sorted alphabetically.

        Args:
            feature_cubes:
                Cubelist containing the independent feature variables for prediction.

        Returns:
            Dataframe containing flattened feature variables.

        Raises:
            ValueError:
                If flattened cubes have differing length.
        """
        # Get the names of features and sort alphabetically
        feature_variables = [cube.name() for cube in feature_cubes]
        feature_variables.sort()

        # Unpack the cube-data into dataframe to feed into the tree-models.
        features_df = pd.DataFrame()
        for feature in feature_variables:
            cube = feature_cubes.extract_cube(feature)
            data = cube.data.flatten()
            if (len(features_df) > 0) and (len(data) != len(features_df)):
                raise RuntimeError("Input cubes have differing sizes.")
            features_df[feature] = data

        return features_df

    def _make_decreasing(self, input_array: ndarray) -> ndarray:
        """Make np.array monotone decreasing in the first dimension.

        Args:
            input_array: the array to make monotone

        Returns:
            array: an array of same shape as the input, where np.diff(axis=0)
            is non-negative
        """
        lower = np.minimum.accumulate(input_array, axis=0)
        upper = np.flip(
            np.maximum.accumulate(np.flip(input_array, axis=0), axis=0), axis=0
        )
        return 0.5 * (upper + lower)

    def _calculate_error_probabilities(
        self, forecast_cube: Cube, feature_cubes: CubeList,
    ) -> Cube:
        """Evaluate the error exceedence probabilities for forecast_cube from tree_models and
        the associated feature_cubes.

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
        from lightgbm import Booster

        if self.treelite_enabled:
            from treelite_runtime import DMatrix, Predictor

        error_probability_cube = self._prepare_error_probability_cube(forecast_cube)

        features_df = self._prepare_features_dataframe(feature_cubes)

        if isinstance(self.tree_models[0], Booster):
            # Use GBDT models for calculation.
            input_dataset = features_df
        elif self.treelite_enabled:
            if isinstance(self.tree_models[0], Predictor):
                # Use treelite models for calculation.
                input_dataset = DMatrix(features_df.values)
            else:
                raise ValueError("Unsupported model object passed.")
        else:
            raise ValueError("Unsupported model object passed.")

        for threshold_index, model in enumerate(self.tree_models):
            prediction = model.predict(input_dataset)
            error_probability_cube.data[threshold_index, ...] = np.reshape(
                prediction, forecast_cube.data.shape
            )

        # Enforcing monotonicity
        error_probability_cube.data = self._make_decreasing(error_probability_cube.data)

        return error_probability_cube

    def _extract_error_percentiles(
        self, error_probability_cube, error_percentiles_count
    ):
        """Extract error percentile values from the error exceedence probabilities.

        Args:
            error_probability_cube:
                A cube containing error exceedence probabilities.
            error_percentiles_count:
                The number of error percentiles to extract. The resulting percentiles
                will be evenly spaced on the interval (0, 100).

        Returns:
            Cube containing percentile values for the error distributions.
        """
        error_percentiles = choose_set_of_percentiles(
            error_percentiles_count, sampling="quantile",
        )
        error_percentiles_cube = ConvertProbabilitiesToPercentiles().process(
            error_probability_cube, percentiles=error_percentiles
        )

        return error_percentiles_cube

    def _apply_error_to_forecast(
        self, forecast_cube: Cube, error_percentiles_cube: Cube
    ) -> Cube:
        """Apply the error distributions (as error percentiles) to the forecast cube. The result
        is a series (sub-ensemble) of values for each forecast realization.

        Args:
            forecast_cube:
                Cube containing the forecast to be calibrated.
            error_percentiles_cube:
                Cube containing percentile values for the error distributions.
        Returns:
            Cube containing the forecast sub-ensembles.
        """
        # Apply the error_percentiles to the forecast_cube
        forecast_subensembles_data = (
            forecast_cube.data[:, np.newaxis] + error_percentiles_cube.data
        )
        # Negative can values arise when the forecast is between error thresholds.
        # When there is a lower bound on the observable value (0.0 in the case of rainfall),
        # error thresholds below the forecast value should have probability of exceedence
        # of 1, however it is possible that when forecast value is between thresholds that
        # the linear interpolation in mapping from probabilities to percentiles, can give
        # percentile values that lie below the forecast value. Consequently, when applied
        # to the forecast, these result in negative values in the sub-ensemble.
        forecast_subensembles_data = np.maximum(0.0, forecast_subensembles_data)
        # Return cube containing forecast subensembles
        return create_new_diagnostic_cube(
            name=forecast_cube.name(),
            units=forecast_cube.units,
            template_cube=error_percentiles_cube,
            mandatory_attributes=generate_mandatory_attributes([forecast_cube]),
            optional_attributes=forecast_cube.attributes,
            data=forecast_subensembles_data,
        )

    def _stack_subensembles(self, forecast_subensembles: Cube) -> Cube:
        """Stacking the realization and percentile dimensions in forecast_subensemble
        into a single realization dimension. Realization and percentile are assumed to
        be the first and second dimensions respectively.

        Args:
            input_cube:
                Cube containing the forecast_subensembles.

        Returns:
            Cube containing single realization dimension in place of the realization
            and percentile dimensions in forecast_subensemble.

        Raises:
            ValueError:
                if realization and percentile are not the first and second
                dimensions.
        """
        realization_percentile_dims = (
            *forecast_subensembles.coord_dims("realization"),
            *forecast_subensembles.coord_dims("percentile"),
        )
        if realization_percentile_dims != (0, 1):
            raise ValueError("Invalid dimension coordinate ordering.")
        realization_size = len(forecast_subensembles.coord("realization").points)
        percentile_size = len(forecast_subensembles.coord("percentile").points)
        new_realization_coord = DimCoord(
            points=np.arange(realization_size * percentile_size, dtype=np.int32),
            standard_name="realization",
            units="1",
        )
        # As we are stacking the first two dimensions, we need to subtract 1 from all
        # dimension position values.
        dim_coords_and_dims = [(new_realization_coord, 0)]
        dim_coords = forecast_subensembles.coords(dim_coords=True)
        for coord in dim_coords:
            if coord.name() not in ["realization", "percentile"]:
                dims = tuple(
                    d - 1 for d in forecast_subensembles.coord_dims(coord.name())
                )
                dim_coords_and_dims.append((coord, dims))
        aux_coords_and_dims = []
        aux_coords = forecast_subensembles.coords(dim_coords=False)
        for coord in aux_coords:
            dims = tuple(d - 1 for d in forecast_subensembles.coord_dims(coord.name()))
            aux_coords_and_dims.append((coord, dims))
        # Stack the first two dimensions.
        superensemble_data = np.reshape(
            forecast_subensembles.data, (-1,) + forecast_subensembles.data.shape[2:]
        )
        superensemble_cube = Cube(
            superensemble_data,
            standard_name=forecast_subensembles.standard_name,
            long_name=forecast_subensembles.long_name,
            var_name=forecast_subensembles.var_name,
            units=forecast_subensembles.units,
            dim_coords_and_dims=dim_coords_and_dims,
            aux_coords_and_dims=aux_coords_and_dims,
            attributes=forecast_subensembles.attributes,
        )
        return superensemble_cube

    def _combine_subensembles(
        self, forecast_subensembles: Cube, output_realizations_count: Optional[int]
    ) -> Cube:
        """Combine the forecast sub-ensembles into a single ensemble. This is done by
        first stacking the sub-ensembles into a single super-ensemble and then resampling
        the super-ensemble to produce a subset of output realizations.

        Args:
            forecast_subensembles:
                Cube containing a series of forecast sub-ensembles.
            output_realizations_count:
                The number of ensemble realizations that will be extracted from the
                super-ensemble. If realizations_count is None, all realizations will
                be returned.

        Returns:
            Cube containing single realization dimension.
        """
        superensemble_cube = self._stack_subensembles(forecast_subensembles)

        if output_realizations_count is None:
            warnings.warn(
                Warning(
                    "output_realizations_count not specified. Returning all realizations from the "
                    "full super-ensemble."
                )
            )
            return superensemble_cube

        output_percentiles = choose_set_of_percentiles(
            output_realizations_count, sampling="quantile",
        )
        percentile_cube = generate_percentiles.process(
            superensemble_cube,
            coordinates="realization",
            percentiles=output_percentiles,
        )
        reduced_ensemble = RebadgePercentilesAsRealizations()(percentile_cube)

        return reduced_ensemble

    def process(
        self,
        forecast_cube: Cube,
        feature_cubes: CubeList,
        error_percentiles_count: int = DEFAULT_ERROR_PERCENTILES_COUNT,
        output_realizations_count: int = DEFAULT_OUTPUT_REALIZATIONS_COUNT,
    ) -> Cube:
        """Apply rainforests calibration to forecast cube.

        Ensemble forecasts must be in realization representation. Deterministic forecasts
        can be processed to produce a pseudo-ensemble; a realization dimension will be added
        to deterministic forecast cubes if one is not already present.

        The calibration is done in a situation dependent fashion using a series of
        decision-tree models to construct representative error distributions which are
        then used to map each input ensemble member onto a series of realisable values.

        These error distributions are formed in a two-step process:

        1. Evalute error CDF defined over the specified error_thresholds. Each exceedence
        probability is evaluated using the corresponding decision-tree model.

        2. Interpolate the CDF to extract a series of percentiles for the error distribution.
        The error percentiles are then applied to each associated ensemble realization to
        produce a series of realisable values; collectively these series form a calibrated
        super-ensemble, which is then sub-sampled to provide the calibrated forecast.

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
            error_percentiles_count:
                The number of error percentiles to extract from the associated error CDFs
                evaluated via the tree-models. These error percentiles are applied to each
                ensemble realization to produce a series of values, which collectively form
                the super-ensemble. The resulting super-ensemble will be of
                size = forecast.realization.size * error_percentiles_count.
            output_realizations_count:
                The number of ensemble realizations that will be extracted from the
                super-ensemble. If realizations_count is None, all realizations will
                be returned.

        Returns:
            The calibrated forecast cube.

        Raises:
            ValueError:
                If the number of tree-models is inconsistent with the number of error
                thresholds.
        """
        # Check that tree-model object available for each error threshold.
        if len(self.error_thresholds) != len(self.tree_models):
            raise ValueError(
                "tree_models must be of the same size as error_thresholds."
            )

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

        # Extract error percentiles from error CDF.
        error_percentiles = self._extract_error_percentiles(
            error_CDF, error_percentiles_count
        )

        # Apply error to forecast cube.
        forecast_subensembles = self._apply_error_to_forecast(
            aligned_forecast, error_percentiles
        )

        # Combine sub-ensembles into a single consolidated ensemble.
        calibrated_output = self._combine_subensembles(
            forecast_subensembles, output_realizations_count
        )

        return calibrated_output
