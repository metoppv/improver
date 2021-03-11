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

from improver.metadata.constants import PERC_COORD
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTES
from improver.metadata.probabilistic import (
    get_diagnostic_cube_name_from_probability_name,
    get_threshold_coord_name_from_probability_name,
)
from improver.utilities.cube_manipulation import get_coord_names


class MOMetadataInterpreter:
    """Class to interpret an iris cube according to the Met Office specific
    IMPROVER standard.  This is intended as a debugging tool to aid developers
    in adding and modifying metadata within the code base."""

    PROB = "probability"
    PERC = "percentile"
    DIAG = "realization"

    MODEL_CODES = {
        "Nowcast": "nc_det",
        "MOGREPS-G": "gl_ens",
        "MOGREPS-UK": "uk_ens",
        "UKV": "uk_det",
    }
    MODEL_NAMES = dict((v, k) for k, v in MODEL_CODES.iteritems())

    def __init__(self, verbose=False):
        """Initialise class parameters"""
        self.verbose = verbose
        self.model_id_attr = "mosg__model_configuration"

        # set up empty strings to record any non-compliance (returned as one error
        # after all checks have been made) or warnings
        self.error_string = ""
        self.warning_string = ""

        # type-specific metadata
        self.field_type = None
        self.diagnostic = None
        self.relative_to_threshold = None

        # cell method interpretation
        self.methods = ""

        # attribute interpretation
        self.post_processed = None
        self.model = None
        self.blendable = None
        self.blended = None

    def add_error(msg):
        """Appends new error message to string"""
        self.error_string += msg + "\n"

    def check_probability_cube_metadata(cube):
        """Checks probability-specific metadata"""
        try:
            self.diagnostic = get_diagnostic_cube_name_from_probability_name(
                cube.name()
            ).replace("_", " ")
        except ValueError as cause:
            # if the probability name is not valid
            self.add_error(cause.msg)

        expected_threshold_name = get_threshold_coord_name_from_probability_name(
            cube.name()
        )
        if not cube.coords(expected_threshold_name):
            msg = f"Cube does not have expected threshold coord '{expected_threshold_name}'"
            try:
                threshold_name = find_threshold_coordinate(cube)
            except CoordinateNotFoundError:
                coords = [coord.name() for coord in cube.coords()]
                msg = (
                    f"no coord with var_name='threshold' found in all coords: {coords}"
                )
                self.add_error(msg)
            else:
                msg += f"threshold coord has incorrect name '{threshold_name}'"
                self.add_error(msg)
                self.check_threshold_coordinate_properties(
                    cube.name(), cube.coord(threshold_name)
                )
        else:
            threshold_coord = cube.coord(expected_threshold_name)
            self.check_threshold_coordinate_properties(cube.name(), threshold_coord)

    def check_threshold_coordinate_properties(cube_name, threshold_coord):
        """Checks threshold coordinate properties are correct and consistent with
        cube name"""
        threshold_var_name = threshold_coord.var_name
        if threshold_var_name != "threshold":
            self.add_error(
                f"Threshold coord {threshold_coord.name()} does not have "
                'var_name="threshold"'
            )

        self.relative_to_threshold = threshold_coord.attributes(
            "spp__relative_to_threshold"
        ).replace("_", " ")

        if self.relative_to_threshold in ("greater than", "greater than or equal to"):
            threshold_attribute = "above"
        elif self.relative_to_threshold in ("less than", "less than or equal to"):
            threshold_attribute = "below"
        else:
            threshold_attribute = None
            self.add_error(
                f'spp__relative_to_threshold attribute "{self.relative_to_threshold}" '
                "is not in permitted value set"
            )

        if threshold_attribute not in cube_name:
            self.add_error(
                f'Cube name "{cube_name}" is not consistent with '
                f'spp__relative_to_threshold attribute "{self.relative_to_threshold}"'
            )

    def check_cell_methods(cell_methods):
        """Checks cell methods are permitted and correctly formatted"""
        for cm in cell_methods:
            if cm.method == "mean":
                if "forecast_reference_time" in cm.coords:
                    self.add_error(f"Non-standard cell method {cm}")
                elif "realization" in cm.coords:
                    # we expect wind direction to have a "mean over realizations" cell method
                    if "direction" != diagnostic:
                        self.add_error(f"Non-standard cell method {cm}")
                else:
                    # flag method which might be invalid, but we can't be sure
                    self.warning_string += (
                        f"Unexpected cell method {cm}. Please check the standard to "
                        "ensure this is valid\n"
                    )

            elif cm.method == "point":
                self.add_error(f"Non-standard cell method {cm}")

            elif cm.method in ["max", "min", "sum"]:
                self.methods += f" {cm.method} over {cm.coords}"
                if self.field_type == PROB:
                    if cm.comment[0] != f"of {diagnostic}":
                        self.add_error(
                            f"Cell method {cm} on probability data should have comment "
                            f'"of {self.diagnostic}"'
                        )

            else:
                # flag method which might be invalid, but we can't be sure
                self.warning_string += (
                    f"Unexpected cell method {cm}. Please check the standard to "
                    "ensure this is valid\n"
                )

    def check_attributes(attrs):
        """Checks for model information and consistency"""
        self.post_processed = "some" if "Post-Processed" in attrs[title] else "no"

        self.blended = True if "Blend" in title else False

        if self.blended:
            if self.model_id_attr not in attrs:
                self.add_error(f"No {self.model_id_attr} on blended file")
            else:
                codes = attrs[self.model_id_attr].split(" ")
                names = [MODEL_NAMES[code] for code in codes]
                self.model = names.join(", ")

        else:
            if self.model_id_attr in attrs:
                for key in MODEL_CODES:
                    if (
                        key in attrs[title]
                        and attrs[self.model_id_attr] != MODEL_CODES[key]
                    ):
                        self.add_error(
                            f"Title {attrs[title]} is inconsistent with model ID attribute "
                            f"{attrs[self.model_id_attr]}"
                        )
                self.model = MODEL_NAMES[attrs[self.model_id_attr]]
                self.blendable = True
            else:
                self.blendable = False

    def gen_output_string():
        """Generates file description in readable form"""

        def vstring(source_metadata):
            """Format additional message for verbose output"""
            return f"    Source: {source_metadata}\n"

        output_string = ""
        output_string += f"This is a {self.field_type} file\n"
        if self.verbose:
            output_string += vstring("name, coordinates")

        rtt = (
            " {self.relative_to_threshold} thresholds"
            if self.field_type == PROB
            else ""
        )
        output_string += f"It contains {self.field_type}s of {self.diagnostic}{rtt}\n"
        if self.verbose:
            output_string += vstring("name, threshold coordinate (if probability)")

        if self.methods:
            output_string += (
                f"These {self.field_type}s are of {self.diagnostic}{self.methods}\n"
            )
            if self.verbose:
                output_string += vstring("cell methods")

        output_string += (
            "It has undergone {self.post_processed} significant post-processing\n"
        )
        if self.verbose:
            output_string += vstring("title attribute")

        if self.blended:
            output_string += f"It contains blended data from models: {self.model}\n"
            if self.verbose:
                output_string += vstring("title attribute, model ID attribute")
        else:
            if blendable:
                output_string += f"It contains data from {self.model}\n"
                if self.verbose:
                    output_string += vstring("model ID attribute")
            else:
                output_string += (
                    "It has no source model information and cannot be blended\n"
                )
                if self.verbose:
                    output_string += vstring("model ID attribute (missing)")

        if self.warning_string:
            output_string += (
                f"WARNING: please check the following metadata: \n{self.warning_string}"
            )

        return output_string

    def process(cube):
        """Return string interpretation of cube metadata or raise errors"""

        if "probability" in cube.name():
            self.field_type = PROB
            self.check_probability_cube_metadata(cube)
        else:
            diagnostic = cube.name()
            coords = get_coord_names(cube)
            if PERC_COORD in coords:
                self.field_type = PERC
                perc_units = cube.coord(PERC_COORD).units
                if perc_units != "%":
                    self.add_error(
                        "Percentile coordinate should have units of %, "
                        f"has {perc_units}"
                    )
            else:
                self.field_type = DIAG

        if cube.cell_methods:
            self.check_cell_methods(cube.cell_methods)

        try:
            self.check_attributes(cube.attributes)
        except KeyError:
            self.add_error(
                "Cube is missing one or more of the mandatory attributes: "
                f"{MANDATORY_ATTRIBUTES}"
            )

        # raise errors if present; otherwise return output
        if self.error_string:
            raise ValueError(self.error_string)

        return self.gen_output_string()
