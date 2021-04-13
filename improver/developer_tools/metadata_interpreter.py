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
"""Module containing classes for metadata interpretation"""

from iris.coords import CellMethod
from iris.exceptions import CoordinateNotFoundError

from improver.metadata.check_datatypes import check_mandatory_standards
from improver.metadata.constants import PERC_COORD
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTES
from improver.metadata.probabilistic import (
    find_percentile_coordinate,
    find_threshold_coordinate,
    get_diagnostic_cube_name_from_probability_name,
    get_threshold_coord_name_from_probability_name,
)
from improver.utilities.cube_manipulation import get_coord_names, get_dim_coord_names

### Constants relating to metadata encoding

# Model name-to-attribute maps
MODEL_CODES = {
    "Nowcast": "nc_det",
    "Global": "gl_det",
    "MOGREPS-G": "gl_ens",
    "MOGREPS-UK": "uk_ens",
    "UKV": "uk_det",
}
MODEL_NAMES = dict((v, k) for k, v in MODEL_CODES.items())

# Diagnostics that differ from the PROB / PERC / DIAG pattern (not all are handled)
ANCILLARIES = [
    "surface_altitude",
    "land_fraction",
    "land_binary_mask",
    "grid_with_halo",
    "topographic_zone_weights",
    "topography_mask",
    "silhouette_roughness",
    "standard_deviation_of_height_in_grid_cell",
    "timezone_mask",
    "smoothing_coefficient_x",
    "smoothing_coefficient_y",
]
EMOS_COEFF_NAMES = [
    f"emos_coefficient_{coeff}" for coeff in ["alpha", "beta", "gamma", "delta"]
]
INTERMEDIATES = [
    "grid_neighbours",
    "grid_eastward_wind",
    "grid_northward_wind",
    "precipitation_advection_x_velocity",
    "precipitation_advection_y_velocity",
    "reliability_calibration_table",
] + EMOS_COEFF_NAMES

SPECIAL_CASES = ["weather_code", "wind_from_direction"] + INTERMEDIATES + ANCILLARIES

# Expected coordinates for different field types
SPOT_COORDS = ["spot_index", "latitude", "longitude", "altitude", "wmo_id"]
UNBLENDED_TIME_COORDS = ["time", "forecast_period", "forecast_reference_time"]
BLENDED_TIME_COORDS = ["time", "blend_time"]
LOCAL_TIME_COORDS = ["time", "time_in_local_timezone"]

# Compliant, required and forbidden cell methods
NONCOMP_CMS = [
    CellMethod(method="mean", coords="forecast_reference_time"),
    CellMethod(method="mean", coords="model_id"),
    CellMethod(method="mean", coords="model_configuration"),
    CellMethod(method="mean", coords="realization"),
]
NONCOMP_CM_METHODS = ["point", "weighted_mean"]
COMPLIANT_CM_METHODS = ["min", "max", "minimum", "maximum", "sum"]
PRECIP_ACCUM_CM = CellMethod(method="sum", coords="time")
PRECIP_ACCUM_NAMES = [
    "lwe_thickness_of_precipitation_amount",
    "lwe_thickness_of_sleetfall_amount",
    "lwe_thickness_of_snowfall_amount",
    "thickness_of_rainfall_amount",
]

# Compliant, required and forbidden attributes
NONCOMP_ATTRS = [
    "mosg__grid_type",
    "mosg__grid_domain",
    "mosg__grid_version",
    "mosg__forecast_run_duration",
    "grid_id",
    "source_realizations",
    "um_version",
]
DIAG_ATTRS = {
    "weather_code": ["weather_code", "weather_code_meaning"],
    "wind_gust": ["wind_gust_diagnostic"],
}
COMPLIANT_ATTRS = MANDATORY_ATTRIBUTES + [
    "Conventions",
    "least_significant_digit",
    "mosg__model_configuration",
]

# Expected substrings to be found in certain title attributes
BLEND_TITLE_SUBSTR = "IMPROVER Post-Processed Multi-Model Blend"
PP_TITLE_SUBSTR = "Post-Processed"
SPOT_TITLE_SUBSTR = "Spot Values"


class MOMetadataInterpreter:
    """Class to interpret an iris cube according to the Met Office specific
    IMPROVER standard.  This is intended as a debugging tool to aid developers
    in adding and modifying metadata within the code base."""

    PROB = "probabilities"
    PERC = "percentiles"
    DIAG = "realizations"
    ANCIL = "ancillary"

    def __init__(self):
        """Initialise class parameters, which store information about a cube to be
        parsed into a human-readable string by the
        :func:`~improver.developer_tools.metadata_interpreter.display_interpretation`
        function.
        """
        self.model_id_attr = "mosg__model_configuration"
        self.unhandled = False

        # set up empty strings to record any non-compliance (returned as one error
        # after all checks have been made) or warnings
        self.errors = []
        self.warnings = []
        # initialise information to be derived from input cube
        self.prod_type = "gridded"  # gridded or spot
        self.field_type = (
            None  # probabilities, percentiles, realizations, ancillary or name
        )
        self.diagnostic = None  # name
        self.relative_to_threshold = None  # for probability data only
        self.methods = ""  # human-readable interpretation of cell method(s)
        self.post_processed = (
            None  # True / False on whether significant processing applied
        )
        self.model = None  # human-readable model name
        self.blended = None  # has it been model blended (True / False)

    def check_probability_cube_metadata(self, cube):
        """Checks probability-specific metadata"""
        if cube.units != "1":
            self.errors.append(
                f"Expected units of 1 on probability data, got {cube.units}"
            )

        try:
            self.diagnostic = get_diagnostic_cube_name_from_probability_name(
                cube.name()
            )
        except ValueError as cause:
            # if the probability name is not valid
            self.errors.append(str(cause))

        expected_threshold_name = get_threshold_coord_name_from_probability_name(
            cube.name()
        )

        if not cube.coords(expected_threshold_name):
            msg = f"Cube does not have expected threshold coord '{expected_threshold_name}'; "
            try:
                threshold_name = find_threshold_coordinate(cube).name()
            except CoordinateNotFoundError:
                coords = [coord.name() for coord in cube.coords()]
                msg += (
                    f"no coord with var_name='threshold' found in all coords: {coords}"
                )
                self.errors.append(msg)
            else:
                msg += f"threshold coord has incorrect name '{threshold_name}'"
                self.errors.append(msg)
                self.check_threshold_coordinate_properties(
                    cube.name(), cube.coord(threshold_name)
                )
        else:
            threshold_coord = cube.coord(expected_threshold_name)
            self.check_threshold_coordinate_properties(cube.name(), threshold_coord)

    def check_threshold_coordinate_properties(self, cube_name, threshold_coord):
        """Checks threshold coordinate properties are correct and consistent with
        cube name"""
        if threshold_coord.var_name != "threshold":
            self.errors.append(
                f"Threshold coord {threshold_coord.name()} does not have "
                "var_name='threshold'"
            )

        try:
            self.relative_to_threshold = threshold_coord.attributes[
                "spp__relative_to_threshold"
            ]
        except KeyError:
            self.errors.append(
                f"{cube_name} threshold coordinate has no "
                "spp__relative_to_threshold attribute"
            )
            return

        if self.relative_to_threshold in ("greater_than", "greater_than_or_equal_to"):
            threshold_attribute = "above"
        elif self.relative_to_threshold in ("less_than", "less_than_or_equal_to"):
            threshold_attribute = "below"
        elif self.relative_to_threshold == "between_thresholds":
            # TODO remove this once we get rid of the "between thresholds" plugin and CLI
            threshold_attribute = "between"
            self.warnings.append("Between thresholds data are not fully supported")
        else:
            threshold_attribute = None
            self.errors.append(
                f"spp__relative_to_threshold attribute '{self.relative_to_threshold}' "
                "is not in permitted value set"
            )

        if threshold_attribute and threshold_attribute not in cube_name:
            self.errors.append(
                f"Cube name '{cube_name}' is not consistent with "
                f"spp__relative_to_threshold attribute '{self.relative_to_threshold}'"
            )

    def check_cell_methods(self, cube):
        """Checks cell methods are permitted and correct"""
        if any([substr in cube.name() for substr in PRECIP_ACCUM_NAMES]):
            msg = f"Expected sum over time cell method for {cube.name()}"
            if not cube.cell_methods:
                self.errors.append(msg)
            else:
                found_cm = False
                for cm in cube.cell_methods:
                    if (
                        cm.method == PRECIP_ACCUM_CM.method
                        and cm.coord_names == PRECIP_ACCUM_CM.coord_names
                    ):
                        found_cm = True
                if not found_cm:
                    self.errors.append(msg)

        if cube.cell_methods:
            for cm in cube.cell_methods:
                if cm.method in COMPLIANT_CM_METHODS:
                    self.methods += f" {cm.method} over {cm.coord_names[0]}"
                    if self.field_type == self.PROB:
                        if not cm.comments or cm.comments[0] != f"of {self.diagnostic}":
                            self.errors.append(
                                f"Cell method {cm} on probability data should have comment "
                                f"'of {self.diagnostic}'"
                            )
                    # check point and bounds on method coordinate
                    if "time" in cm.coord_names:
                        if cube.coord("time").bounds is None:
                            self.errors.append(
                                f"Cube of{self.methods} has no time bounds"
                            )

                elif cm in NONCOMP_CMS or cm.method in NONCOMP_CM_METHODS:
                    self.errors.append(f"Non-standard cell method {cm}")
                else:
                    # flag method which might be invalid, but we can't be sure
                    self.warnings.append(
                        f"Unexpected cell method {cm}. Please check the standard to "
                        "ensure this is valid"
                    )

    def _check_blend_and_model_attributes(self, attrs):
        """Interprets attributes for model and blending information
        and checks for self-consistency"""
        self.blended = True if BLEND_TITLE_SUBSTR in attrs["title"] else False

        if self.blended:
            if self.model_id_attr not in attrs:
                self.errors.append(f"No {self.model_id_attr} on blended file")
            else:
                codes = attrs[self.model_id_attr].split(" ")
                names = []
                for code in codes:
                    try:
                        names.append(MODEL_NAMES[code])
                    except KeyError:
                        self.errors.append(
                            f"Model ID attribute contains unrecognised model code {code}"
                        )
                self.model = ", ".join(names)
            return

        if self.model_id_attr in attrs:
            for key in MODEL_CODES:
                if (
                    f"{key} Model" in attrs["title"]
                    and attrs[self.model_id_attr] != MODEL_CODES[key]
                ):
                    self.errors.append(
                        f"Title {attrs['title']} is inconsistent with model ID "
                        f"attribute {attrs[self.model_id_attr]}"
                    )

            try:
                self.model = MODEL_NAMES[attrs[self.model_id_attr]]
            except KeyError:
                self.errors.append(
                    f"Attribute {attrs[self.model_id_attr]} is not a valid single "
                    "model.  If valid for blend, then title attribute is missing "
                    f"expected substring {BLEND_TITLE_SUBSTR}."
                )

    def check_attributes(self, attrs):
        """Checks for unexpected attributes, then interprets values for model
        information and checks for self-consistency"""
        if self.diagnostic in DIAG_ATTRS:
            permitted_attributes = COMPLIANT_ATTRS + DIAG_ATTRS[self.diagnostic]
        else:
            permitted_attributes = COMPLIANT_ATTRS.copy()

        if any([attr in NONCOMP_ATTRS for attr in attrs]):
            self.errors.append(
                f"Attributes {attrs.keys()} include one or more forbidden "
                f"values {[attr for attr in attrs if attr in NONCOMP_ATTRS]}"
            )
        elif any([attr not in permitted_attributes for attr in attrs]):
            self.warnings.append(
                f"{attrs.keys()} include unexpected attributes "
                f"{[attr for attr in attrs if attr not in permitted_attributes]}. "
                "Please check the standard to ensure this is valid."
            )

        if self.diagnostic in DIAG_ATTRS:
            required = DIAG_ATTRS[self.diagnostic]
            if any([req not in attrs for req in required]):
                self.errors.append(
                    f"Attributes {attrs.keys()} missing one or more required "
                    f"values {[req for req in required if req not in attrs]}"
                )

        if self.field_type != self.ANCIL:
            if not all([attr in attrs for attr in MANDATORY_ATTRIBUTES]):
                self.errors.append(
                    f"Attributes {attrs.keys()} missing one or more mandatory values "
                    f"{[req for req in MANDATORY_ATTRIBUTES if req not in attrs]}"
                )

            if "title" in attrs:
                self.post_processed = (
                    True
                    if PP_TITLE_SUBSTR in attrs["title"]
                    or BLEND_TITLE_SUBSTR in attrs["title"]
                    else False
                )
                # determination of whether file is blended depends on title
                self._check_blend_and_model_attributes(attrs)

    def _check_coords_present(self, coords, expected_coords):
        """Check whether all expected coordinates are present"""
        found_coords = [coord for coord in coords if coord in expected_coords]
        if not set(found_coords) == set(expected_coords):
            self.errors.append(
                f"Missing one or more coordinates: found {found_coords}, "
                f"expected {expected_coords}"
            )

    def _check_coord_bounds(self, cube, coord):
        """If coordinate has bounds, check points are equal to upper bound"""
        if cube.coord(coord).bounds is not None:
            upper_bounds = cube.coord(coord).bounds[..., 1]
            if not (cube.coord(coord).points == upper_bounds).all():
                self.errors.append(f"{coord} points should be equal to upper bounds")

    def check_spot_data(self, cube, coords):
        """Check spot coordinates"""
        self.prod_type = "spot"
        if "title" in cube.attributes:
            if SPOT_TITLE_SUBSTR not in cube.attributes["title"]:
                self.errors.append(
                    f"Title attribute {cube.attributes['title']} is not "
                    "consistent with spot data"
                )

        self._check_coords_present(coords, SPOT_COORDS)

    def run(self, cube):
        """Populates self-consistent interpreted parameters, or raises collated errors
        describing (as far as posible) how the metadata are a) not self-consistent,
        and / or b) not consistent with the Met Office IMPROVER standard.

        Although every effort has been made to return as much information as possible,
        collated errors may not be complete if the issue is fundamental. The developer
        is advised to rerun this tool after each fix, until no further problems are
        raised.
        """

        # 1) Interpret diagnostic and type-specific metadata, including cell methods
        if cube.name() in ANCILLARIES:
            self.field_type = self.ANCIL
            self.diagnostic = cube.name()
            if cube.cell_methods:
                self.errors.append(f"Unexpected cell methods {cube.cell_methods}")

        elif cube.name() in SPECIAL_CASES:
            self.field_type = self.diagnostic = cube.name()
            if cube.name() == "weather_code":
                if cube.cell_methods:
                    self.errors.append(f"Unexpected cell methods {cube.cell_methods}")
            elif cube.name() == "wind_from_direction":
                if cube.cell_methods:
                    expected = CellMethod(method="mean", coords="realization")
                    if len(cube.cell_methods) > 1 or cube.cell_methods[0] != expected:
                        self.errors.append(
                            f"Unexpected cell methods {cube.cell_methods}"
                        )
            else:
                self.unhandled = True
                return

        else:
            if "probability" in cube.name() and "threshold" in cube.name():
                self.field_type = self.PROB
                self.check_probability_cube_metadata(cube)
            else:
                self.diagnostic = cube.name()
                try:
                    perc_coord = find_percentile_coordinate(cube)
                except CoordinateNotFoundError:
                    coords = get_coord_names(cube)
                    if any(
                        [cube.coord(coord).var_name == "threshold" for coord in coords]
                    ):
                        self.field_type = self.PROB
                        self.check_probability_cube_metadata(cube)
                    else:
                        self.field_type = self.DIAG
                else:
                    self.field_type = self.PERC
                    if perc_coord.name() != PERC_COORD:
                        self.errors.append(
                            f"Percentile coordinate should have name {PERC_COORD}, "
                            f"has {perc_coord.name()}"
                        )

                    if perc_coord.units != "%":
                        self.errors.append(
                            "Percentile coordinate should have units of %, "
                            f"has {perc_coord.units}"
                        )

            self.check_cell_methods(cube)

        # 2) Interpret model and blend information from cube attributes
        self.check_attributes(cube.attributes)

        # 3) Check whether expected coordinates are present
        coords = get_coord_names(cube)
        if "spot_index" in coords:
            self.check_spot_data(cube, coords)

        if self.field_type == self.ANCIL:
            # there is no definitive standard for time coordinates on static ancillaries
            pass
        elif (
            cube.coords("time")
            and len(cube.coord_dims("time")) == 2
            and not self.blended
        ):
            # 2D time coordinates are only present on global day-max diagnostics that
            # use a local time zone coordinate. These do not have a 2D forecast period.
            expected_coords = set(LOCAL_TIME_COORDS + UNBLENDED_TIME_COORDS)
            expected_coords.discard("forecast_period")
            self._check_coords_present(coords, expected_coords)
        elif self.blended:
            self._check_coords_present(coords, BLENDED_TIME_COORDS)
        else:
            self._check_coords_present(coords, UNBLENDED_TIME_COORDS)

        # 4) Check points are equal to upper bounds for bounded time coordinates
        for coord in ["time", "forecast_period"]:
            if coord in get_coord_names(cube):
                self._check_coord_bounds(cube, coord)

        # 5) Check datatypes on data and coordinates
        try:
            check_mandatory_standards(cube)
        except ValueError as cause:
            self.errors.append(str(cause))

        # 6) Raise collated errors if present
        if self.errors:
            raise ValueError("\n".join(self.errors))


def _format_standard_cases(interpreter, verbose, vstring):
    """Format prob / perc / diagnostic information from a
    MOMetadataInterpreter instance"""
    field_type = interpreter.field_type.replace("_", " ")
    diagnostic = interpreter.diagnostic.replace("_", " ")
    if interpreter.relative_to_threshold:
        relative_to_threshold = interpreter.relative_to_threshold.replace("_", " ")

    rval = []
    rtt = (
        f" {relative_to_threshold} thresholds"
        if interpreter.field_type == interpreter.PROB
        else ""
    )
    rval.append(f"It contains {field_type} of {diagnostic}{rtt}")
    if verbose:
        rval.append(vstring("name, threshold coordinate (probabilities only)"))

    if interpreter.methods:
        rval.append(f"These {field_type} are of {diagnostic}{interpreter.methods}")
        if verbose:
            rval.append(vstring("cell methods"))

    ppstring = "some" if interpreter.post_processed else "no"
    rval.append(f"It has undergone {ppstring} significant post-processing")
    if verbose:
        rval.append(vstring("title attribute"))
    return rval


def display_interpretation(interpreter, verbose=False):
    """Prints metadata interpretation in human-readable form.  This should
    not be run on a MOMetadataInterpreter instance that has raised errors.

    Args:
        interpreter (MOMetadataInterpreter):
            Populated instance of MOMetadataInterpreter
        verbose (bool):
            Optional flag to include information about the source of the
            metadata interpretation (eg name, coordinates, attributes, etc)        

    Returns:
        str:
            Formatted string describing metadata in human-readable form
    """
    if interpreter.unhandled:
        return f"{interpreter.diagnostic} is not handled by this interpreter\n"

    def vstring(source_metadata):
        """Format additional message for verbose output"""
        return f"    Source: {source_metadata}"

    field_type = interpreter.field_type.replace("_", " ")
    output = []
    if field_type == "realizations":
        field_type_clause = f"file containing one or more {field_type}"
    else:
        field_type_clause = f"{field_type} file"
    output.append(f"This is a {interpreter.prod_type} {field_type_clause}")
    if verbose:
        output.append(vstring("name, coordinates"))

    if interpreter.diagnostic not in SPECIAL_CASES:
        output.extend(_format_standard_cases(interpreter, verbose, vstring))

    if interpreter.diagnostic in ANCILLARIES:
        output.append(f"This is a static ancillary with no time information")
    elif interpreter.blended:
        output.append(f"It contains blended data from models: {interpreter.model}")
        if verbose:
            output.append(vstring("title attribute, model ID attribute"))
    else:
        if interpreter.model:
            output.append(f"It contains data from {interpreter.model}")
            if verbose:
                output.append(vstring("model ID attribute"))
        else:
            output.append("It has no source model information and cannot be blended")
            if verbose:
                output.append(vstring("model ID attribute (missing)"))

    if interpreter.warnings:
        warning_string = "\n".join(interpreter.warnings)
        output.append(f"WARNINGS:\n{warning_string}")

    return "\n".join(output) + "\n"
