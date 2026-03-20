# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing a deterministic realization selector.

It interprets the cube's attribute: primary_input_realizations_to_clusters.
It can then select and extract a realization from the cube.
This can then be used as a deterministic realization.
"""

# Load in Packages
import iris
from iris.cube import Cube
import json
from improver import PostProcessingPlugin


# Define Plugin
class DeterministicRealizationSelector(PostProcessingPlugin):
    """Plugin which takes an Iris Cube with the attribute:
    primary_input_realizations_to_clusters.
    Then, extracts only the realization which contains the control member.
    Then returns the subsetted Iris Cube. 
    """

    def __init__(self, control_member=0) -> None:
        """Init Method, to set up processing for the Plugin.

        Args:
            control_member:
                The number of the ensemble member acting as the control member.
                Default value = 0.
        """
        self.control_member = control_member

    def find_control_key(self, cube: Cube, attribute: str) -> int | None:
        """Method takes the cube and finds the
        key (realization) that contains the control member.
        This is done by testing the attribute and the control member exists.
        Then finds the ensemble's respective key.

        Args:
            cube:
                Cube with realizations and
                the attribute to be searched through for the control member.
            attribute:
                The attribute which contains the cluster metadata,
                as a dictionary, to be searched through.
        Returns:
            control_key:
                Key (realization number) that contains the control member.
        """
        # Test the Attribute is Present
        try:
            cube_attribute = cube.attributes[attribute]
            cube_attribute_dict = json.loads(cube_attribute)
        except KeyError:
            return None
        control_key = None
        for key, value in cube_attribute_dict.items():
            # Test control member Exists
            if self.control_member in value:
                control_key = key
                break
            else:
                control_key = None

        return control_key

    @staticmethod
    def extract_realization_from_cube(key: int, cube: Cube) -> Cube:
        """Method extracts the realization containing the control member.
        Using a constraint created from the control_key.

        Args:
            key:
                The key referring to the realization,
                 that contains the control member.
            cube:
                The cube with the realization
                 containing the control member to be removed.

        Returns:
            control_realization_cube:
                Cube containing only the realization with the control member.
        """

        realization_constraint = iris.Constraint(realization=int(key))
        control_realization_cube = cube.extract(realization_constraint)

        return control_realization_cube

    def process(self, cubes: iris.cube.CubeList) ->  Cube:
        """Takes a cube and applies the different plugin methods.
         Used to identify and extract the realization with the control member.

        Args:
            cubes:
                A list of cubes containing forecasts and a cluster cube.
                The cluster cube will contain the attribute:
                "primary_input_realizations_to_clusters".
                This will be used to split the forecasts and cluster cube and
                determine which realizations to extract from the forecast cube.

        Returns:
            control_realization_cube:
                Forecast cube containing only the deterministic realization,
                 with the control member.
        """
        attribute = "primary_input_realizations_to_clusters"
        cluster_cube = None
        forecast_cube = None

        # Split the cubes into forecast and cluster cubes,
        # by the presence of the attribute
        for cube in cubes:
            if attribute in cube.attributes:
                cluster_cube = cube
            else:
                forecast_cube = cube

        if cluster_cube is None or forecast_cube is None:
            raise AttributeError(
                "Forecast and/or cluster cubes were not found in the input."
            )

        control_key = self.find_control_key(cluster_cube, attribute)
        if control_key is None:
            raise AttributeError("Control Member not found")

        control_realization_cube = self.extract_realization_from_cube(
            control_key, forecast_cube
        )
        return control_realization_cube