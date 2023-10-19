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
"""Plugin to calculate blend weights and blend data across a dimension"""

import warnings
from copy import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import iris
import numpy as np
from iris.cube import Cube, CubeList

from improver import PostProcessingPlugin
from improver.blending import MODEL_BLEND_COORD, MODEL_NAME_COORD
from improver.blending.spatial_weights import SpatiallyVaryingWeightsFromMask
from improver.blending.utilities import (
    get_coords_to_remove,
    match_site_forecasts,
    record_run_coord_to_attr,
    update_blended_metadata,
    update_record_run_weights,
)
from improver.blending.weighted_blend import (
    MergeCubesForWeightedBlending,
    WeightedBlendAcrossWholeDimension,
)
from improver.blending.weights import (
    ChooseDefaultWeightsLinear,
    ChooseDefaultWeightsNonLinear,
    ChooseWeightsLinear,
)
from improver.utilities.spatial import (
    check_if_grid_is_equal_area,
    distance_to_number_of_grid_cells,
)


class WeightAndBlend(PostProcessingPlugin):
    """
    Wrapper class to calculate weights and blend data across cycles or models
    """

    def __init__(
        self,
        blend_coord: str,
        wts_calc_method: str,
        weighting_coord: Optional[str] = None,
        wts_dict: Optional[Dict[str, Dict[str, Any]]] = None,
        y0val: Optional[float] = None,
        ynval: Optional[float] = None,
        cval: Optional[float] = None,
        inverse_ordering: bool = False,
    ) -> None:
        """
        Initialise central parameters

        Args:
            blend_coord:
                Coordinate over which blending will be performed (eg "model"
                for grid blending)
            wts_calc_method:
                Method to use to calculate weights used in blending.
                "linear" (default): calculate linearly varying blending weights.
                "nonlinear": calculate blending weights that decrease
                exponentially with increasing blending coordinates.
                "dict": calculate weights using a dictionary passed in.
            weighting_coord:
                Name of coordinate over which linear weights should be scaled.
                This coordinate must be available in the weights dictionary.
            wts_dict:
                Dictionary from which to calculate blending weights. Dictionary
                format is as specified in
                improver.blending.weights.ChooseWeightsLinear
            y0val:
                The relative value of the weighting start point (lowest value of
                blend coord) for choosing default linear weights.
                If used this must be a positive float or 0.
            ynval:
                The relative value of the weighting end point (highest value of
                blend coord) for choosing default linear weights. This must be a
                positive float or 0.
                Note that if blending over forecast reference time, ynval >= y0val
                would normally be expected (to give greater weight to the more
                recent forecast).
            cval:
                Factor used to determine how skewed the non-linear weights will be.
                A value of 1 implies equal weighting.
            inverse_ordering:
                Option to invert weighting order for non-linear weights plugin
                so that higher blend coordinate values get higher weights (eg
                if cycle blending over forecast reference time).
        """
        self.blend_coord = blend_coord
        self.wts_calc_method = wts_calc_method
        self.weighting_coord = None

        if self.wts_calc_method == "dict":
            self.weighting_coord = weighting_coord
            self.wts_dict = wts_dict
        elif self.wts_calc_method == "linear":
            self.y0val = y0val
            self.ynval = ynval
        elif self.wts_calc_method == "nonlinear":
            self.cval = cval
            self.inverse_ordering = inverse_ordering
        else:
            raise ValueError(
                "Weights calculation method '{}' unrecognised".format(
                    self.wts_calc_method
                )
            )

    def _calculate_blending_weights(self, cube: Cube) -> Cube:
        """
        Wrapper for plugins to calculate blending weights by the appropriate
        method.

        Args:
            cube:
                Cube of input data to be blended

        Returns:
            Cube containing 1D array of weights for blending
        """
        if self.wts_calc_method == "dict":
            if "model" in self.blend_coord:
                config_coord = MODEL_NAME_COORD
            else:
                config_coord = self.blend_coord

            weights = ChooseWeightsLinear(
                self.weighting_coord, self.wts_dict, config_coord_name=config_coord
            )(cube)

        elif self.wts_calc_method == "linear":
            weights = ChooseDefaultWeightsLinear(y0val=self.y0val, ynval=self.ynval)(
                cube, self.blend_coord
            )

        elif self.wts_calc_method == "nonlinear":
            weights = ChooseDefaultWeightsNonLinear(self.cval)(
                cube, self.blend_coord, inverse_ordering=self.inverse_ordering
            )

        return weights

    def _update_spatial_weights(
        self, cube: Cube, weights: Cube, fuzzy_length: float
    ) -> Cube:
        """
        Update weights using spatial information

        Args:
            cube:
                Cube of input data to be blended
            weights:
                Initial 1D cube of weights scaled by self.weighting_coord
            fuzzy_length:
                Distance (in metres) over which to smooth weights at domain
                boundaries

        Returns:
            Updated 3D cube of spatially-varying weights
        """
        check_if_grid_is_equal_area(cube)
        grid_cells = distance_to_number_of_grid_cells(
            cube, fuzzy_length, return_int=False
        )
        plugin = SpatiallyVaryingWeightsFromMask(
            self.blend_coord, fuzzy_length=grid_cells
        )
        weights = plugin(cube, weights)
        return weights

    def _remove_zero_weighted_slices(
        self, cube: Cube, weights: Cube
    ) -> Tuple[Cube, Cube]:
        """Removes any cube and weights slices where the 1D weighting factor
        is zero

        Args:
            cube:
                The data cube to be blended
            weights:
                1D cube of weights varying along self.blend_coord

        Returns:
            - Data cube without zero-weighted slices
            - Weights without zeroes
        """
        slice_out_vals = []
        for wslice in weights.slices_over(self.blend_coord):
            if np.sum(wslice.data) == 0:
                slice_out_vals.append(wslice.coord(self.blend_coord).points[0])

        if not slice_out_vals:
            return cube, weights

        constraint = iris.Constraint(
            coord_values={self.blend_coord: lambda x: x not in slice_out_vals}
        )
        cube = cube.extract(constraint)
        weights = weights.extract(constraint)
        return cube, weights

    def process(
        self,
        cubelist: Union[List[Cube], CubeList],
        cycletime: Optional[str] = None,
        model_id_attr: Optional[str] = None,
        record_run_attr: Optional[str] = None,
        spatial_weights: bool = False,
        fuzzy_length: float = 20000,
        attributes_dict: Optional[Dict[str, str]] = None,
    ) -> Cube:
        """
        Merge a cubelist, calculate appropriate blend weights and compute the
        weighted mean. Returns a single cube collapsed over the dimension
        given by self.blend_coord.

        Args:
            cubelist:
                List of cubes to be merged and blended
                If blending site forecasts, this list can optionally include a
                neighbour cube ancillary as a reference sitelist. This is used
                to ensure the reference sitelist is produced. At least one of the
                input cubes must contain sites that match the reference set.
            cycletime:
                The forecast reference time to be used after blending has been
                applied, in the format YYYYMMDDTHHMMZ. If not provided, the
                blended file takes the latest available forecast reference time
                from the input datasets supplied.
            model_id_attr:
                The name of the dataset attribute to be used to identify the source
                model when blending data from different models.
            record_run_attr:
                The name of the dataset attribute to be used to store model and
                cycle sources in metadata, e.g. when blending data from different
                models. Requires model_id_attr.
            spatial_weights:
                If True, this option will result in the generation of spatially
                varying weights based on the masks of the data we are blending.
                The one dimensional weights are first calculated using the chosen
                weights calculation method, but the weights will then be adjusted
                spatially based on where there is masked data in the data we are
                blending. The spatial weights are calculated using the
                SpatiallyVaryingWeightsFromMask plugin.
            fuzzy_length:
                When calculating spatially varying weights we can smooth the
                weights so that areas close to areas that are masked have lower
                weights than those further away. This fuzzy length controls the
                scale over which the weights are smoothed. The fuzzy length is in
                terms of m, the default is 20km. This distance is then converted
                into a number of grid squares, which does not have to be an
                integer. Assumes the grid spacing is the same in the x and y
                directions and raises an error if this is not true. See
                SpatiallyVaryingWeightsFromMask for more details.
            attributes_dict:
                Dictionary describing required changes to attributes after blending

        Returns:
            Cube of blended data.

        Raises:
            ValueError:
                If attempting to use record_run_attr without providing model_id_attr.

        Warns:
            UserWarning: If blending masked data without spatial weights.
                         This has not been fully tested.
        """
        if record_run_attr is not None and model_id_attr is None:
            raise ValueError(
                "record_run_attr can only be used with model_id_attr, which "
                "has not been provided."
            )

        if not isinstance(cubelist, CubeList):
            try:
                cubelist = CubeList(cubelist)
            except TypeError:
                cubelist = CubeList([cubelist])

        # If the cubes for blending are site forecasts and a reference cube
        # (neighbour cube) has been provided, check that the sites the forecasts
        # contain match one another. If not, attempt to construct matching cubes
        # for blending using the reference cube to set the expected sites.
        try:
            (reference_site_cube,) = [
                cube for cube in cubelist if cube.name() == "grid_neighbours"
            ]
        except ValueError:
            # ValueError for attempting to unpack a list with 0 elements.
            pass
        else:
            cubelist.remove(reference_site_cube)
            if cubelist[0].coords("spot_index"):
                cubelist = match_site_forecasts(cubelist, reference_site_cube)

        # Prepare cubes for weighted blending, including creating custom metadata
        # for multi-model blending. The merged cube has a monotonically ascending
        # blend coordinate. Plugin raises an error if blend_coord is not present on
        # all input cubes.
        merger = MergeCubesForWeightedBlending(
            self.blend_coord,
            weighting_coord=self.weighting_coord,
            model_id_attr=model_id_attr,
            record_run_attr=record_run_attr,
        )
        cube = merger(cubelist, cycletime=cycletime)

        if "model" in self.blend_coord:
            self.blend_coord = copy(MODEL_BLEND_COORD)

        # Record coordinates associated with the blend coord that will be removed
        # later once the blend coord has been collapsed.
        coords_to_remove = get_coords_to_remove(cube, self.blend_coord)

        weights = None
        if len(cube.coord(self.blend_coord).points) > 1:
            weights = self._calculate_blending_weights(cube)
            cube, weights = self._remove_zero_weighted_slices(cube, weights)

        if record_run_attr is not None and weights is not None:
            cube = update_record_run_weights(cube, weights, self.blend_coord)

        # Deal with case of only one input cube or non-zero-weighted slice
        if len(cube.coord(self.blend_coord).points) == 1:
            result = cube
        else:
            if spatial_weights:
                weights = self._update_spatial_weights(cube, weights, fuzzy_length)
            elif np.ma.is_masked(cube.data):
                # Raise warning if blending masked arrays using non-spatial weights.
                warnings.warn(
                    "Blending masked data without spatial weights has not been"
                    " fully tested."
                )

            # Blend across specified dimension
            BlendingPlugin = WeightedBlendAcrossWholeDimension(self.blend_coord)
            result = BlendingPlugin(cube, weights=weights)

        if record_run_attr is not None:
            record_run_coord_to_attr(result, cube, record_run_attr)

        # Remove custom metadata and and update time-type coordinates.  Remove
        # non-time-type coordinate that were previously associated with the blend
        # dimension (coords_to_remove).  Add user-specified and standard blend
        # attributes.
        update_blended_metadata(
            result,
            self.blend_coord,
            coords_to_remove=coords_to_remove,
            cycletime=cycletime,
            attributes_dict=attributes_dict,
            model_id_attr=model_id_attr,
        )

        return result
