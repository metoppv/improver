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
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from improver import PostProcessingPlugin
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

    # Does this belong somewhere else?
    def _add_coordinate_to_cube(
        self,
        input_cube: Cube,
        new_coord: DimCoord,
        new_dim_location: Optional[int] = None,
        copy_metadata: Optional[bool] = True,
    ) -> Cube:
        """Create a copy of input cube with an additional dimension coordinate
        added to the cube at the specified axis. The data from input cube is broadcast
        over this new dimension.
        Args:
            input cube:
                cube to add realization dimension to.
            new_coord:
                new coordinate to add to input cube.
            new_dim_location:
                position in cube.data to position the new dimension coord.
            copy_metadata:
                flag as to whether to carry metadata over to output cube.

        Returns:
            output_cube
        """
        input_dim_count = len(input_cube.dim_coords)

        new_dim_coords = list(input_cube.dim_coords) + [new_coord]
        new_dims = list(range(input_dim_count + 1))
        new_dim_coords_and_dims = list(zip(new_dim_coords, new_dims))

        aux_coords = input_cube.aux_coords
        aux_coord_dims = [input_cube.coord_dims(coord.name()) for coord in aux_coords]
        new_aux_coords_and_dims = list(zip(aux_coords, aux_coord_dims))

        new_coord_size = len(new_coord.points)
        new_data = np.broadcast_to(
            input_cube.data, shape=(new_coord_size,) + input_cube.shape
        )
        new_data = input_cube.data[..., np.newaxis] * np.ones(
            shape=new_coord_size, dtype=input_cube.dtype
        )

        output_cube = Cube(
            new_data,
            dim_coords_and_dims=new_dim_coords_and_dims,
            aux_coords_and_dims=new_aux_coords_and_dims,
        )
        if copy_metadata:
            output_cube.metadata = input_cube.metadata

        if new_dim_location is not None:
            final_dim_order = np.insert(
                np.arange(input_dim_count), new_dim_location, values=input_dim_count
            )
            output_cube.transpose(final_dim_order)

        return output_cube

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
                expanded_cube = self._add_coordinate_to_cube(
                    cube, common_realization_coord, new_dim_location=0
                )
                aligned_cubes.append(expanded_cube)
            else:
                aligned_cubes.append(combined_cubes[i_cube])

        return aligned_cubes[:-1], aligned_cubes[-1]

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

        # Extract error percentiles from error CDF.

        # Apply error to forecast cube.

        # Combine sub-ensembles into a single consolidated ensemble.

        # Until calibration processing steps are added, the behaviour here will be to return
        # the forecast cube without calibration. This will enable other integration work to
        # proceed concurrently with development of this Plugin.
        return forecast_cube
