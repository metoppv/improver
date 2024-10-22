# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""Spot data extraction from diagnostic fields using neighbour cubes."""

import warnings
from typing import List, Optional, Union

import iris
import numpy as np
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError

from improver import BasePlugin
from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    ConvertProbabilitiesToPercentiles,
    ResamplePercentiles,
)
from improver.metadata.probabilistic import find_percentile_coordinate
from improver.percentile import PercentileConverter
from improver.utilities.cube_extraction import extract_subcube
from improver.utilities.cube_manipulation import collapse_realizations

from .apply_lapse_rate import SpotLapseRateAdjust
from .spot_extraction import SpotExtraction
from .utilities import get_neighbour_finding_method_name


class SpotManipulation(BasePlugin):
    """
    A wrapper to the spot-extraction plugin that handles additional options
    for manipulating the spot-data. This allows for the extraction of
    percentiles, including the conversion to percentiles of probabilities.
    Lapse rate adjustment of spot temperatures is also possible.
    """

    def __init__(
        self,
        apply_lapse_rate_correction: bool = False,
        fixed_lapse_rate: Optional[float] = None,
        land_constraint: bool = False,
        similar_altitude: bool = False,
        extract_percentiles: Optional[Union[float, List[float]]] = None,
        ignore_ecc_bounds_exceedance: bool = False,
        skip_ecc_bounds: bool = False,
        new_title: Optional[str] = None,
        suppress_warnings: bool = False,
        realization_collapse: bool = False,
        subset_coord: str = None,
    ) -> None:
        """
        Initialise the wrapper plugin using the selected options.

        Args:
            apply_lapse_rate_correction (bool):
                Use to apply a lapse-rate correction to screen temperature
                forecasts so that they better represent the altitude of the
                spot site for which they have been extracted. This lapse rate
                will be applied for a fixed orographic difference between the
                site and grid point altitude. Differences in orography in
                excess of this fixed limit will use the Environmental Lapse
                Rate (also known as the Standard Atmosphere Lapse Rate).
                Lapse rate adjustment cannot be applied to existing spot
                forecasts that are passed in for subsetting.
            fixed_lapse_rate (float):
                If provided, use this fixed value as a lapse-rate for adjusting
                the forecast values if apply_lapse_rate_correction is True. This
                can be used instead of providing a lapse rate cube. Value is
                given in Kelvin / metre of temperature change with ascent. For
                example a dry adiabatic lapse rate would be given as -0.0098.
                This lapse rate will be applied for a fixed orographic difference
                between the site and gridpoint altitude. Differences in orography
                in excess of this fixed limit will use the Environmental Lapse
                Rate (also known as the Standard Atmosphere Lapse Rate).
            land_constraint (bool):
                Use to select the nearest-with-land-constraint neighbour-selection
                method from the neighbour_cube. This means that the grid points
                should be land points except for sites where none were found within
                the search radius when the neighbour cube was created. May be used
                with similar_altitude.
            similar_altitude (bool):
                Use to select the nearest-with-height-constraint
                neighbour-selection method from the neighbour_cube. These are grid
                points that were found to be the closest in altitude to the spot
                site within the search radius defined when the neighbour cube was
                created. May be used with land_constraint.
            extract_percentiles (list or int):
                If set to a percentile value or a list of percentile values,
                data corresponding to those percentiles will be returned. For
                example "25, 50, 75" will result in the 25th, 50th and 75th
                percentiles being returned from a cube of probabilities,
                percentiles or realizations. Deterministic input data will raise
                a warning message.
                Note that for percentile inputs, if the desired percentile(s) do
                not exist in the input cube the available percentiles will be
                resampled to produce those requested.
            ignore_ecc_bounds_exceedance (bool):
                Demotes exceptions where calculated percentiles are outside the ECC
                bounds range to warnings.
            skip_ecc_bounds (bool):
                If True, ECC bounds are not included when probabilities
                are converted to percentiles. This has the effect that percentiles
                outside of the range given by the input percentiles will be computed
                by nearest neighbour interpolation from the nearest available percentile,
                rather than using linear interpolation between the nearest available
                percentile and the ECC bound.
            new_title (str):
                New title for the spot-extracted data.  If None, this attribute is
                removed from the output cube since it has no prescribed standard
                and may therefore contain grid information that is no longer
                correct after spot-extraction.
            suppress_warnings (bool):
                Suppress warning output. This option should only be used if it
                is known that warnings will be generated but they are not required.
            realization_collapse (bool):
                Triggers equal-weighting blending of the realization coord if required.
                Use this if a threshold coord is also present on the input cube.
            subset_coord (str):
                If a spot cube is provided as input this plugin can return a subset of
                the sites based on the sites specified in the neighbour cube. To
                achieve this the plugin needs the name of the site ID coordinate to be
                used for matching, e.g. wmo_id. If subset_coord is not provided, and a
                spot forecast is passed in, the entire spot cube will be processed and
                returned. The neighbour selection method options have no impact if a
                spot cube is passed in.
        """
        self.neighbour_selection_method = get_neighbour_finding_method_name(
            land_constraint, similar_altitude
        )
        self.apply_lapse_rate_correction = apply_lapse_rate_correction
        self.fixed_lapse_rate = fixed_lapse_rate
        self.extract_percentiles = extract_percentiles
        self.ignore_ecc_bounds_exceedance = ignore_ecc_bounds_exceedance
        self.skip_ecc_bounds = skip_ecc_bounds
        self.new_title = new_title
        self.suppress_warnings = suppress_warnings
        self.realization_collapse = realization_collapse
        self.subset_coord = subset_coord

    def process(self, cubes: CubeList) -> Cube:
        """
        Call spot-extraction and other plugins to manipulate the resulting
        spot forecasts.

        Args:
            cubes:
                A list of cubes containing the diagnostic data to be extracted,
                a temperature lapse rate (optional) and the neighbour cube.

        Returns:
            Spot-extracted forecast data following any optional manipulations.

        Warns:
            If diagnostic cube is not a known probabilistic type.
            If a lapse rate cube was not provided, but the option to apply
            the lapse rate correction was enabled.
        """
        neighbour_cube = cubes[-1]
        cube = cubes[0]

        # If a spot forecast cube is passed in, constrain the sites to
        # those that are found in the neighbour cube if an ID coordinate on
        # which to constrain is provided, e.g. wmo_id. Otherwise pass the
        # spot forecast cube forwards unchanged.
        if cube.coords("spot_index"):
            if (
                self.apply_lapse_rate_correction is not False
                or self.fixed_lapse_rate is not None
            ):
                raise NotImplementedError(
                    "Lapse rate adjustment when subsetting an existing spot "
                    "forecast cube has not been implemented."
                )
            if self.subset_coord is None:
                result = cube
            else:
                try:
                    sites = neighbour_cube.coord(self.subset_coord).points
                except CoordinateNotFoundError as err:
                    raise ValueError(
                        "Subset_coord not found in neighbour cube."
                    ) from err
                # Exclude unset site IDs as this value is non-unique.
                sites = [item for item in sites if item != "None"]
                site_constraint = iris.Constraint(
                    coord_values={self.subset_coord: sites}
                )
                result = cube.extract(site_constraint)
                if not result:
                    raise ValueError("No spot sites retained after subsetting.")
        else:
            result = SpotExtraction(
                neighbour_selection_method=self.neighbour_selection_method
            )(neighbour_cube, cube, new_title=self.new_title)

        if self.realization_collapse:
            result = collapse_realizations(result)

        # If a probability or percentile diagnostic cube is provided, extract
        # the given percentile if available. This is done after the spot-extraction
        # to minimise processing time; usually there are far fewer spot sites than
        # grid points.
        if self.extract_percentiles:
            extract_percentiles = [np.float32(x) for x in self.extract_percentiles]
            try:
                perc_coordinate = find_percentile_coordinate(result)
            except CoordinateNotFoundError:
                if "probability_of_" in result.name():
                    result = ConvertProbabilitiesToPercentiles(
                        ecc_bounds_warning=self.ignore_ecc_bounds_exceedance,
                        skip_ecc_bounds=self.skip_ecc_bounds,
                    )(result, percentiles=extract_percentiles)
                    result = iris.util.squeeze(result)
                elif result.coords("realization", dim_coords=True):
                    fast_percentile_method = not np.ma.is_masked(result.data)
                    result = PercentileConverter(
                        "realization",
                        percentiles=extract_percentiles,
                        fast_percentile_method=fast_percentile_method,
                    )(result)
                else:
                    if not self.suppress_warnings:
                        msg = (
                            "Diagnostic cube is not a known probabilistic type. "
                            "The {} percentile(s) could not be extracted. The "
                            "spot-extracted outputs will be returned without "
                            "percentile extraction.".format(extract_percentiles)
                        )
                        warnings.warn(msg)
            else:
                if set(extract_percentiles).issubset(perc_coordinate.points):
                    constraint = [
                        "{}={}".format(perc_coordinate.name(), extract_percentiles)
                    ]
                    result = extract_subcube(result, constraint)
                else:
                    result = ResamplePercentiles()(
                        result, percentiles=extract_percentiles
                    )

        # Check whether a lapse rate cube has been provided
        if self.apply_lapse_rate_correction:
            if len(cubes) == 3:
                plugin = SpotLapseRateAdjust(
                    neighbour_selection_method=self.neighbour_selection_method
                )
                result = plugin(result, neighbour_cube, cubes[-2])
            elif self.fixed_lapse_rate is not None:
                plugin = SpotLapseRateAdjust(
                    neighbour_selection_method=self.neighbour_selection_method,
                    fixed_lapse_rate=self.fixed_lapse_rate,
                )
                result = plugin(result, neighbour_cube)
            elif not self.suppress_warnings:
                warnings.warn(
                    "A lapse rate cube or fixed lapse rate was not provided, but the "
                    "option to apply the lapse rate correction was enabled. No lapse rate "
                    "correction could be applied."
                )

        # Remove the internal model_grid_hash attribute if present.
        result.attributes.pop("model_grid_hash", None)

        return result
