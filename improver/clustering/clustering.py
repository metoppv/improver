# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
""""""

import iris
import numpy as np
from iris.util import promote_aux_coord_to_dim_coord

from improver import BasePlugin
from improver.regrid.landsea import RegridLandSea


class Clustering(BasePlugin):
    """Class to perform clustering."""

    def __init__(self, clustering_method, **kwargs):
        """Initialize the clustering class.

        Args:
            clustering_method (str): The clustering method to use.
            **kwargs: Additional arguments for the clustering method.
        """
        self.clustering_method = clustering_method
        self.kwargs = kwargs

    def convert_3d_to_2d(array: np.ndarray) -> np.ndarray:
        """Convert a 3D array to a 2D array by maintaining the zeroth dimension
        and flattening all other dimensions.

        Args:
            array: The input array to convert.
        Returns:
            cube_2d: The converted 2D array.
        """
        array_2d = array.copy()
        if len(array.shape) > 2:
            target_shape = (array.shape[0], -1)
            array_2d = array.copy()
            array_2d = array_2d.reshape(target_shape)
        return array_2d

    def process(self, cube: iris.cube.Cube) -> np.ndarray:
        """Apply the clustering method to the cube.

        Args:
            cube (iris.cube.Cube): The input cube to cluster.

        Returns:
            clustering_result: The result of the clustering.
        """

        # If method requested is not available from sklearn,
        # try sklearn_extra (e.g. KMedoids)
        try:
            from sklearn import cluster

            clustering_class = getattr(cluster, self.clustering_method)
        except AttributeError:
            try:
                from sklearn_extra import cluster

                clustering_class = getattr(cluster, self.clustering_method)
            except AttributeError:
                msg = (
                    "The clustering method provided is not supported: "
                    f"{self.clustering_method}"
                )
                raise ValueError(msg)

        array_2d = self.convert_3d_to_2d(cube.data)

        clustering_result = clustering_class(**self.kwargs).fit(array_2d)
        return clustering_result


class SelectRealizationsForKMedoidClusters(BasePlugin):
    """Class to select clusters from the clustering result."""

    def __init__(self, clustering_result):
        """Initialize the SelectClusters class.

        Args:
            clustering_result: The result of the clustering.
        """
        self.clustering_result = clustering_result

    def process(self, cube: iris.cube.Cube) -> iris.cube.Cube:
        """Select clusters from the clustering result.

        Args:
            cube (iris.cube.Cube): The input cube to select
                clusters from.

        Returns:
            cube_clustered: The clustered cube.
        """
        n_realizations = len(cube.coord("realization").points)
        if len(self.clustering_result.medoid_indices_) > n_realizations:
            n_clusters = len(self.clustering_result.medoid_indices_)
            msg = (
                f"The number of clusters {n_clusters} is expected to be less than "
                "the number of realizations {}. Please reduce the number of clusters."
            )
            raise ValueError(msg)

        cube_clustered = cube[self.clustering_result.medoid_indices_]
        cube_clustered.coord("realization").points = range(len(n_realizations))
        promote_aux_coord_to_dim_coord(cube_clustered, "realization")
        return cube_clustered


class MeanSquaredErrorAssignment(BasePlugin):
    """Class to assign clusters based on mean squared error."""

    def __init__(self, n_realizations):
        """Initialize the MeanSquaredErrorAssignment class.

        Args:
            .
        """
        self.n_realizations = n_realizations

    def mean_squared_error(
        self,
        clustered_array,
        candidate_array,
    ):
        mse_list = []
        for index in range(self.n_realizations):
            mse_list.append(
                np.nanmean(
                    np.square(clustered_array - candidate_array[index]),
                    axis=1,
                )
            )
        return np.array(mse_list)

    def choose_clusters(self, mse_array):
        """Choose clusters based in the mean squared error.

        Args:
            mse_list (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Find minimum MSE for each realization. Therefore each realization now
        # has a single MSE value, rather than an MSE for each cluster.
        min_mse_array = np.min(mse_array, axis=1)
        # Subtract the minimum MSE for each realization for every cluster,
        # and sum across clusters.
        mse_array_cost = np.sum(mse_array - min_mse_array[:, None], axis=1)

        # Find the order (index) that sorts the MSE cost array from highest
        # MSE to the lowest MSE.
        index_across_realizations = np.argsort(mse_array_cost)[::-1]

        # Iterate over realizations from the realization with the largest MSE
        # cost to the realization with the smallest MSE cost.
        cluster_index_list = []
        for mse_values in mse_array[index_across_realizations]:
            if mse_values.ndim == 3:
                mse_values = np.nansum(mse_values, axis=-1)
                mse_values[np.isclose(mse_values, 0)] = np.nan

            # Identify the cluster with the minimum value for the MSE when comparing to a
            # specific realization.
            cluster_indices = np.argsort(mse_values)
            # assume_unique avoids sorted the cluster indices. The zeroth index then
            # gives the cluster with the minimum MSE.
            cluster_index = np.setdiff1d(
                cluster_indices, cluster_index_list, assume_unique=True
            )[0]
            cluster_index_list.append(cluster_index)
        return cluster_index_list

    def process(
        self, clustered_cube: iris.cube.Cube, candidate_cube: iris.cube.Cube
    ) -> iris.cube.Cube:
        """Assign clusters based on mean squared error.

        Args:
            clustered_cube (iris.cube.Cube): The clustered cube.
            cube (iris.cube.Cube): The input cube to assign clusters to.

        Returns:
            cube_assigned: The assigned cube.
        """
        mse_array = self.mean_squared_error(clustered_cube.data, candidate_cube.data)
        return self.choose_clusters(mse_array)


class ClusterAndClassify(BasePlugin):
    """Class to cluster and classify data."""

    def __init__(
        self, hierarchy, clustering_method, target_grid_name, forecast_period, **kwargs
    ):
        """Initialize the clustering and classification class.

        Args:
            hierarchy: The hierarchy of inputs defining the primary input, which is
                clustered, and secondary inputs, which is classified into each cluster.
                {"primary_input": "input1",
                 "secondary_inputs": {"input2": [0, 6], "input3": [0, 24]}]}
            clustering_method (str): The clustering method to use.
            **kwargs: Additional arguments for the clustering method.
        """
        self.hierarchy = hierarchy
        self.clustering_method = clustering_method
        self.kwargs = kwargs

    def process(self, cubes: iris.cube.CubeList) -> iris.cube.Cube:
        """Cluster and classify the data."""
        primary_cube = cubes.extract(self.hierarchy["primary_input"])

        target_grid_cube = cubes.extract(self.target_grid_name)

        clustering_result = Clustering(self.clustering_method, **self.kwargs)(
            primary_cube
        )

        primary_cube = primary_cube.extract(
            iris.Constraint(forecast_period=self.forecast_period)
        )
        clustered_cube = SelectRealizationsForKMedoidClusters(clustering_result)(
            primary_cube
        )
        upscaled_clustered_cube = RegridLandSea(
            regrid_mode="area-weighted",
        )(clustered_cube, target_grid_cube)

        for candidate_name, forecast_periods in self.hierarchy[
            "secondary_inputs"
        ].items():
            if self.forecast_period in list(range(*forecast_periods)):
                candidate_cube = cubes.extract(candidate_name)
                if candidate_cube is None:
                    continue
            upscaled_candidate_cube = RegridLandSea(
                regrid_mode="area-weighted",
            )(candidate_cube, target_grid_cube)
            classification_result = MeanSquaredErrorAssignment()(
                upscaled_clustered_cube, upscaled_candidate_cube
            )
            break
        else:
            msg = (
                "No secondary inputs found for the specified "
                f"forecast period: {self.forecast_period}. The secondary inputs "
                f"looked for were: {self.hierarchy['secondary_inputs']}"
            )
            raise ValueError(msg)
        return classification_result
