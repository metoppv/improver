# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""init for calibration"""

from typing import List, Optional, Tuple

from iris.cube import Cube, CubeList
import numpy as np
import scipy

from improver.metadata.probabilistic import (
    get_diagnostic_cube_name_from_probability_name,
)
from improver.utilities.cube_manipulation import MergeCubes


def split_forecasts_and_truth(
    cubes: List[Cube], truth_attribute: str, land_sea_mask_name: bool
) -> Tuple[Cube, Cube, Optional[CubeList], Optional[Cube]]:
    """
    A common utility for splitting the various inputs cubes required for
    calibration CLIs. These are generally the forecast cubes, historic truths,
    and in some instances a land-sea mask is also required.

    Args:
        cubes:
            A list of input cubes which will be split into relevant groups.
            These include the historical forecasts, in the format supported by
            the calibration CLIs, and the truth cubes.
        truth_attribute:
            An attribute and its value in the format of "attribute=value",
            which must be present on truth cubes.

    Returns:
        - A cube containing all the historic forecasts.
        - A cube containing all the truth data.
        - If found within the input cubes list a land-sea mask will be
          returned, else None is returned.

    Raises:
        ValueError:
            An unexpected number of distinct cube names were passed in.
        IOError:
            More than one cube was identified as a land-sea mask.
        IOError:
            Missing truth or historical forecast in input cubes.
    """
    cubes_dict = {"truth": {}, "land_sea_mask": {}, "other": {}, "historic_forecasts": {}, "additional_fields": {}}
    # split non-land_sea_mask cubes on forecast vs truth
    truth_key, truth_value = truth_attribute.split("=")
    for cube in cubes:
        try:
            cube_name = get_diagnostic_cube_name_from_probability_name(cube.name())
        except ValueError:
            cube_name = cube.name()
        if cube.attributes.get(truth_key) == truth_value:
            cubes_dict["truth"].setdefault(cube_name, []).append(cube)
        elif cube_name == land_sea_mask_name:
            cubes_dict["land_sea_mask"].setdefault(cube_name, []).append(cube)
        else:
            blend_time_list = [c for c in cube.coords() if c.name() == "blend_time"]
            if len(blend_time_list):
                cube.remove_coord("blend_time")
            if cube.coords("forecast_period"):
                cube.coord("forecast_period").attributes = {}
            if cube.coords("forecast_reference_time"):
                cube.coord("forecast_reference_time").attributes = {}
            cubes_dict["other"].setdefault(cube_name, []).append(cube)

    if len(cubes_dict["truth"]) > 1:
        msg = (f"Truth supplied for multiple diagnostics {list(cubes_dict['truth'].keys())}. "
               "The truth should only exist for one diagnostic.")
        raise ValueError(msg)

    if land_sea_mask_name and not cubes_dict["land_sea_mask"]:
        raise IOError("Expected one cube for land-sea mask with "
                      f"the name {land_sea_mask_name}.")

    diag_name = list(cubes_dict["truth"].keys())[0]
    cubes_dict["historic_forecasts"] = cubes_dict["other"][diag_name]
    for k, v in cubes_dict["other"].items():
        if k != diag_name:
            cubes_dict["additional_fields"].setdefault(k, []).extend(v)

    missing_inputs = " and ".join(k for k, v in cubes_dict.items() if k in ["truth", "historic_forecasts"] and not v)
    if missing_inputs:
        raise IOError(f"Missing {missing_inputs} input.")

    truth = MergeCubes()(filter_obs(cubes_dict["truth"][diag_name]))
    forecast = MergeCubes()(cubes_dict["historic_forecasts"])
    additional_fields = CubeList([MergeCubes()(cubes_dict["additional_fields"][k]) for k in cubes_dict["additional_fields"]])
    return forecast, truth, additional_fields, cubes_dict["land_sea_mask"]


def filter_obs(spot_truths_cubelist: CubeList) -> CubeList:
    """Deal with observation sites that failed to provide an altitude, latitude
    and longitude. As we've ensured that each observation site has an entry
    for each timestep, even if no observation was available at that timestep,
    some observations will not have an altitude, latitudes or longitudes.
    There is also the potential for some sites to report different altitudes,
    latitudes and longitudes at different times. For simplicity, the altitude,
    latitude and longitude from the first timestep is used as the altitude,
    latitude and longitude throughout the input CubeList.

    Args:
        spot_truths_cubelist:
            Cubelist of spot truths

    Returns:
        CubeList of spot truths with consistent values for the altitude,
        latitude and longitude at each timestep.
    """
    if not all([c if c.coords("spot_index") else False for c in spot_truths_cubelist]):
        return spot_truths_cubelist
    altitudes = np.squeeze(
        scipy.stats.mode(
            np.stack([c.coord("altitude").points for c in spot_truths_cubelist]), axis=0
        )[0]
    )
    latitudes = np.squeeze(
        scipy.stats.mode(
            np.stack([c.coord(axis="y").points for c in spot_truths_cubelist]), axis=0
        )[0]
    )
    longitudes = np.squeeze(
        scipy.stats.mode(
            np.stack([c.coord(axis="x").points for c in spot_truths_cubelist]), axis=0
        )[0]
    )

    altitudes = np.nan_to_num(altitudes)
    latitudes = np.nan_to_num(latitudes)
    longitudes = np.nan_to_num(longitudes)

    for index, _ in enumerate(spot_truths_cubelist):

        spot_truths_cubelist[index].coord("altitude").points = altitudes
        spot_truths_cubelist[index].coord(axis="y").points = latitudes
        spot_truths_cubelist[index].coord(axis="x").points = longitudes

    return spot_truths_cubelist


def split_forecasts_and_coeffs(
    cubes: List[Cube], land_sea_mask_name: bool
) -> Tuple[Cube, Optional[Cube], CubeList, Optional[Cube]]:
    """
    A common utility for splitting the various inputs cubes required for
    calibration CLIs. These are generally the forecast cubes, historic truths,
    and in some instances a land-sea mask is also required.

    Args:
        cubes:
            A list of input cubes which will be split into relevant groups.
            These include the historical forecasts, in the format supported by
            the calibration CLIs, and the truth cubes.
        truth_attribute:
            An attribute and its value in the format of "attribute=value",
            which must be present on truth cubes.

    Returns:
        - A cube containing all the historic forecasts.
        - A cube containing all the truth data.
        - If found within the input cubes list a land-sea mask will be
          returned, else None is returned.

    Raises:
        ValueError:
            An unexpected number of distinct cube names were passed in.
        IOError:
            More than one cube was identified as a land-sea mask.
        IOError:
            Missing truth or historical forecast in input cubes.
    """

    cubes_dict = {"current_forecast": {}, "coefficients": {}, "land_sea_mask": {}, "additional_fields": {}, "other": {}}
    # split non-land_sea_mask cubes on forecast vs truth
    for cubelist in cubes:
        for cube in cubelist:
            try:
                cube_name = get_diagnostic_cube_name_from_probability_name(cube.name())
            except ValueError:
                cube_name = cube.name()
            if "emos_coefficient" in cube_name:
                cubes_dict["coefficients"].setdefault(cube_name, []).append(cube)
            elif cube_name == land_sea_mask_name:
                cubes_dict["land_sea_mask"] = cube
            else:
                cubes_dict["other"].setdefault(cube_name, []).append(cube)

    if land_sea_mask_name and not cubes_dict["land_sea_mask"]:
        raise IOError("Expected one cube for land-sea mask with "
                      f"the name {land_sea_mask_name}.")

    diagnostic_standard_name = list(set([v[0].attributes["diagnostic_standard_name"] for v in cubes_dict["coefficients"].values() if v[0].attributes.get("diagnostic_standard_name")]))
    if len(diagnostic_standard_name) == 1:
        diagnostic_standard_name, = diagnostic_standard_name
    else:
        msg = ("The coefficients cubes are expected to have one consistent "
               f"diagnostic_standard_name attribute, rather than {diagnostic_standard_name}")
        raise AttributeError(msg)

    cubes_dict["current_forecast"], = cubes_dict["other"][diagnostic_standard_name]
    for k, v in cubes_dict["other"].items():
        if k != diagnostic_standard_name:
            cubes_dict["additional_fields"].setdefault(k, []).extend(v)

    additional_fields = CubeList([cubes_dict["additional_fields"][k][0] for k in cubes_dict["additional_fields"]])
    coefficients = CubeList([cubes_dict["coefficients"][k][0] for k in cubes_dict["coefficients"]])

    return cubes_dict["current_forecast"], additional_fields, coefficients, cubes_dict["land_sea_mask"]