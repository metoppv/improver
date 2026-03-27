# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing a deterministic realization selector."""

# Load in Packages
import json

import iris
from iris.cube import Cube, CubeList

from improver import PostProcessingPlugin


# Define Plugin
class DeterministicRealizationSelector(PostProcessingPlugin):
    """Plugin to extract a deterministic realization
    from a set of realization ensembles that have been clustered
    using the improver.clustering.realization_clustering plugin.
    """

    def __init__(
        self,
        target_realization_id=0,
        attribute="primary_input_realizations_to_clusters",
    ) -> None:
        """Initialise the plugin.

        Args:
            target_realization_id:
                The numeric id of realization of intrest. Default value = 0.
            attribute:
                The attribute of the cluster cube,
                used to identify target realization, and it's associated cluster.
                Default value = "primary_input_realizations_to_clusters".
        """
        self.target_realization_id = target_realization_id
        self.attribute = attribute

    def split_input_cubelist(self, input_cubelist: CubeList) -> tuple[Cube, Cube]:
        """Splits the input cubelist into two cubes,
         depending on whether they contain the attribute:
         "primary_input_realizations_to_clusters".
        Args:
            input_cubelist:
                A list of cubes containing two cubes,
                with only one which contains the attribute:
                "primary_input_realizations_to_clusters".
        Returns:
            cube_with_attribute:
                Cube, to be the cluster_cube, which contains the attribute:
                "primary_input_realizations_to_clusters".
            cube_without_attribute:
                Cube, to be the forecast_cube, which doesn't contain the attribute:
                "primary_input_realizations_to_clusters".
        Raises:
            AttributeError:
                - If the input cubelist contains more than two cubes.
                - If the forecast_cube or cluster_cube cannot be found in the input.
                - If the target realization, does not exist or cannot be extracted.
        """
        cube_with_attribute = None
        cube_without_attribute = None

        # Check cubes only has two cubes
        if len(input_cubelist) != 2:
            raise AttributeError(
                f"Expected 2 cubes but found {len(input_cubelist)} cubes."
            )

        # Split the cubes by the presence of the attribute
        for cube in input_cubelist:
            if self.attribute in cube.attributes:
                cube_with_attribute = cube
            else:
                cube_without_attribute = cube

        if not (cube_with_attribute and cube_without_attribute):
            raise AttributeError(
                "Forecast and/or cluster cubes were not found in the input."
            )
        return cube_with_attribute, cube_without_attribute

    def find_target_key(self, cluster_cube: Cube) -> int | None:
        """Find the key (cluster) of the cluster cube,
        that contains the target realization.
        This cluster will become the deterministic realization.

        Args:
            cluster_cube:
                Clustered cube with the attribute :
                "primary_input_realizations_to_clusters", as a dictionary,
                to be searched through for the target realization.
        Returns:
            target_key:
                Key (cluster) that contains the target realization.
        Raises:
            KeyError: If the attribute does not exist,
             or cannot be converted into a dictionary.
        """
        # Extract the attribute and convert it into a dictionary
        try:
            cube_attribute = cluster_cube.attributes[self.attribute]
            cube_attribute_dict = json.loads(cube_attribute)
        except KeyError:
            # Return None as this case is handled in the process method.
            return None

        # Search through dictionary, to find target realization's key
        target_key = None
        for key, value in cube_attribute_dict.items():
            if self.target_realization_id in value:
                target_key = key
                break
            else:
                target_key = None

        return target_key

    @staticmethod
    def extract_cluster_from_cube(target_key: int, forecast_cube: Cube) -> Cube:
        """Extract the cluster containing the target realization.
        This cube becomes our deterministic realization.

        Args:
            target_key:
                The key corresponding to the cluster,
                 that contains the target realization.
            forecast_cube:
                The cube with the clusters
                 containing the target realization to be extracted.

        Returns:
            deterministic_realization_cube:
                Cube containing only the cluster with the target realization.
        """

        realization_constraint = iris.Constraint(realization=int(target_key))
        deterministic_realization_cube = forecast_cube.extract(realization_constraint)

        return deterministic_realization_cube

    def process(self, cubes: CubeList) -> Cube:
        """Extracts the target deterministic realization from the forecast cubes.
        Identifies the cluster cube
        (containing the attribute: "primary_input_realizations_to_clusters")
        and the forecast cube from the input cubelist.
        Determines the target realization, and it's cluster,
        extracts this from the forecast cube and
        returns the deterministic_realization_cube.

        Args:
            cubes:
                A list of two cubes containing a forecast and a cluster cube.
                The cluster cube will contain the attribute:
                "primary_input_realizations_to_clusters".
                This will be used to split the forecasts and cluster cube and
                determine which realizations to extract from the forecast cube.

        Returns:
            deterministic_realization_cube:
                Forecast cube containing only the cluster with the target realization.
                This cluster becomes our deterministic realization.

        Raises:
            AttributeError:
                - If the forecast_cube or cluster_cube cannot be found in the input.
                - If the target realization, does not exist or cannot be extracted.
        """

        cluster_cube, forecast_cube = self.split_input_cubelist(cubes)

        target_key = self.find_target_key(cluster_cube)
        if target_key is None:
            raise AttributeError("Target realization not found")

        deterministic_realization_cube = self.extract_cluster_from_cube(
            target_key, forecast_cube
        )
        return deterministic_realization_cube
