# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
""""""

from typing import Any

import iris
import numpy as np
import pandas as pd
from iris.util import promote_aux_coord_to_dim_coord

from improver import BasePlugin
from improver.clustering.clustering import FitClustering
from improver.regrid.landsea import RegridLandSea


class RealizationClustering(BasePlugin):
    """Class to perform clustering on realizations of a cube."""

    def __init__(self, clustering_method: str, **kwargs: Any) -> None:
        """Initialise the RealizationClustering class.

        Args:
            clustering_method: The clustering method to use.
            **kwargs: Additional arguments for the clustering method.
        """
        self.clustering_method = clustering_method
        self.kwargs = kwargs

    @staticmethod
    def convert_to_2d(array: np.ndarray) -> np.ndarray:
        """Convert an array with arbitrary dimensions to a 2D array by maintaining
        the zeroth dimension and flattening all other dimensions.

        Args:
            array: The input array to convert. Can have any number of dimensions.
        Returns:
            array_2d: The converted 2D array with shape (array.shape[0], -1).
        """
        if array.ndim < 2:
            msg = "Input array must have at least 2 dimensions."
            raise ValueError(msg)
        if array.ndim == 2:
            return array.copy()
        target_shape = (array.shape[0], -1)
        return array.reshape(target_shape)

    def process(self, cube: iris.cube.Cube) -> Any:
        """Apply the clustering method to the cube. 3d cubes are converted to 2d
        arrays before clustering. The leading dimension is assumed to be the realization
        dimension.

        Args:
            cube: The input cube to cluster.

        Returns:
            clustering_result: The result of the clustering.

        Raises:
            ValueError: If the leading dimension of the input cube is not
                the realization dimension.
        """
        if cube.dim_coords[0].name() != "realization":
            msg = (
                "The leading dimension of the input cube must be "
                "the realization dimension."
            )
            raise ValueError(msg)
        array_2d = self.convert_to_2d(cube.data)
        # The rows of the DataFrame correspond to realizations. The columns correspond
        # to the flattened spatial dimensions. These column values are the features that
        # the clustering algorithm will use to cluster the realizations.
        df = pd.DataFrame(
            array_2d,
            index=[f"realization_{p}" for p in cube.dim_coords[0].points],
        )
        return FitClustering(self.clustering_method, **self.kwargs)(df)


class RealizationToClusterMatcher(BasePlugin):
    """Match candidate realizations to clusters based on mean squared error.

    Supports both 3D cubes (realization, y, x) and 4D cubes
    (realization, forecast_period, y, x).
    """

    def __init__(self) -> None:
        """Initialise the plugin."""
        pass

    def mean_squared_error(
        self,
        clustered_array: np.ndarray,
        candidate_array: np.ndarray,
        n_realizations: int,
    ) -> np.ndarray:
        """Calculate mean squared error between clustered and candidate arrays.

        For 3D cubes, the MSE is calculated by averaging over spatial dimensions.
        For 4D cubes, the mean is calculated over spatial dimensions first, then
        the MSE is summed over forecast_period. This approach gives more weight
        to forecast periods in the matching process, as the MSE values are summed
        rather than averaged across forecast periods. This means that differences
        at multiple forecast periods accumulate, making the total MSE more
        sensitive to patterns across time.

        Args:
            clustered_array: The clustered array with shape (n_clusters, y, x) or
                (n_clusters, forecast_period, y, x).
            candidate_array: The candidate array with shape (n_realizations, y, x)
                or (n_realizations, forecast_period, y, x).
            n_realizations: The number of realizations in the candidate array.
        Returns:
            mse_array: The mean squared error array with shape
                (n_realizations, n_clusters).
        """
        mse_list = []
        for index in range(n_realizations):
            # Calculate squared differences
            squared_diff = np.square(clustered_array - candidate_array[index])

            if clustered_array.ndim == 3:
                # For 3D: average over spatial dimensions (y, x)
                mse = np.nanmean(squared_diff, axis=(1, 2))
            else:
                # For 4D: mean over spatial (y, x), then sum over forecast_period
                mse = np.nansum(np.nanmean(squared_diff, axis=(2, 3)), axis=1)

            mse_list.append(mse)
        return np.array(mse_list)

    def _validate_cube_dimensions(
        self, cube: iris.cube.Cube, candidate_cube: iris.cube.Cube
    ) -> None:
        """Validate that both cubes have matching dimensions.

        Args:
            cube: The clustered cube.
            candidate_cube: The candidate cube.

        Raises:
            ValueError: If cube dimensions don't match.
        """
        if cube.ndim != candidate_cube.ndim:
            msg = (
                f"Clustered cube has {cube.ndim} dimensions but candidate cube has "
                f"{candidate_cube.ndim} dimensions. Both cubes must have the same "
                "number of dimensions (either 3D or 4D)."
            )
            raise ValueError(msg)

    def _validate_forecast_period_coords(
        self, cube: iris.cube.Cube, candidate_cube: iris.cube.Cube
    ) -> None:
        """Validate matching forecast_period coordinates for 4D cubes.

        Args:
            cube: The clustered cube.
            candidate_cube: The candidate cube.

        Raises:
            ValueError: If forecast_period coords don't match or are missing.
        """
        if cube.ndim == 4:
            try:
                cube_fp = cube.coord("forecast_period")
                candidate_fp = candidate_cube.coord("forecast_period")
                if not np.array_equal(cube_fp.points, candidate_fp.points):
                    msg = (
                        "Forecast period coords must match between clustered and "
                        f"candidate cubes. Clustered: {cube_fp.points}, "
                        f"Candidate: {candidate_fp.points}"
                    )
                    raise ValueError(msg)
            except iris.exceptions.CoordinateNotFoundError as e:
                msg = (
                    "Both cubes must have forecast_period coordinate when using "
                    "4D cubes."
                )
                raise ValueError(msg) from e

    def choose_clusters(self, mse_array: np.ndarray) -> list[int]:
        """Choose clusters using a greedy assignment algorithm based on MSE.

        This method assigns each realization to its best-matching cluster by
        minimizing the mean squared error. When multiple realizations match the
        same cluster, the realization with the lowest MSE to that cluster is
        selected.

        The algorithm processes realizations in descending order of their
        "MSE cost" (the sum of differences between each cluster's MSE and the
        minimum MSE for that realization). Realizations with higher costs are
        assigned first. If a realization wants a cluster that's already been
        assigned to another realization with a lower MSE, the current realization
        must choose its next-best cluster.

        Note: This ensures that each cluster is assigned to the best-matching
        realization (lowest MSE), even when multiple realizations compete for the
        same cluster.

        Args:
            mse_array: The mean squared error array with shape
                (n_realizations, n_clusters).

        Returns:
            Tuple of (cluster_indices, realization_indices):
                cluster_indices: List of cluster indices that were assigned.
                realization_indices: List of realization indices, one per
                    assigned cluster. Each element is the index of the
                    candidate realization assigned to that cluster.
        """
        # Calculate cost for each realization (sum of differences from minimum MSE)
        min_mse_array = np.min(mse_array, axis=1, keepdims=True)
        mse_array_cost = np.sum(mse_array - min_mse_array, axis=1)

        # Process realizations in descending order of cost (highest cost first)
        realization_order = np.argsort(mse_array_cost)[::-1]

        n_clusters = mse_array.shape[1]
        n_realizations = mse_array.shape[0]
        cluster_to_realization = {}
        cluster_to_mse = {}

        for loop_idx, realization_idx in enumerate(realization_order):
            realizations_remaining = n_realizations - loop_idx
            assigned_clusters = list(cluster_to_realization.keys())
            clusters_remaining = n_clusters - len(assigned_clusters)
            mse_values = mse_array[realization_idx]
            if realizations_remaining <= clusters_remaining:
                mse_values[assigned_clusters] = np.inf
            cluster_idx = np.nanargmin(mse_values)
            if mse_values[cluster_idx] < cluster_to_mse.get(cluster_idx, np.inf):
                cluster_to_mse[cluster_idx] = mse_values[cluster_idx]
                cluster_to_realization[cluster_idx] = realization_idx

        # Sort by cluster index and return both cluster indices and realization indices
        sorted_items = sorted(cluster_to_realization.items())
        cluster_indices = [cluster_idx for cluster_idx, _ in sorted_items]
        realization_indices = [real_idx for _, real_idx in sorted_items]
        return cluster_indices, realization_indices

    def process(
        self,
        cube: iris.cube.Cube,
        candidate_cube: iris.cube.Cube,
    ) -> tuple[list[int], list[int]]:
        """Assign candidate realizations to clusters by mean squared error.

        This method takes a cube of clustered realizations and candidate
        realizations, then assigns each candidate to the cluster it best
        matches. The assignment uses the greedy MSE-based algorithm from
        choose_clusters. Multiple candidates can be assigned to the same
        cluster when the number of candidates exceeds the number of clusters.

        Supports both 3D cubes (realization, y, x) and 4D cubes
        (realization, forecast_period, y, x). When using 4D cubes, both input
        cubes must have matching forecast_period coordinates.

        Args:
            cube: The cube containing clustered realizations (e.g., from
                KMedoids). Shape: (n_clusters, y, x) or
                (n_clusters, forecast_period, y, x)
            candidate_cube: The input cube with realizations to assign to
                clusters. Shape: (n_realizations, y, x) or
                (n_realizations, forecast_period, y, x)

        Returns:
            Tuple of (cluster_indices, realization_indices):
                cluster_indices: List of cluster indices that were assigned.
                    May have length < n_clusters if there are fewer candidates
                    than clusters.
                realization_indices: List of candidate realization indices,
                    one per assigned cluster. Each element is the index of the
                    candidate realization assigned to that cluster.

        Raises:
            ValueError: If forecast_period coordinates don't match (for 4D
                cubes).
            ValueError: If cube dimensions don't match between clustered and
                candidate cubes.
        """
        n_candidates = len(candidate_cube.coord("realization").points)

        # Validate inputs
        self._validate_cube_dimensions(cube, candidate_cube)
        self._validate_forecast_period_coords(cube, candidate_cube)

        mse_array = self.mean_squared_error(
            cube.data,
            candidate_cube.data,
            n_candidates,
        )
        cluster_indices, realization_indices = self.choose_clusters(mse_array)

        return cluster_indices, realization_indices


class ClusterAndMatch(BasePlugin):
    """Class to cluster and match data."""

    def __init__(
        self,
        hierarchy: dict,
        model_id_attr: str,
        clustering_method,
        target_grid_name,
        **kwargs,
    ):
        """Initialise the clustering and matching class.

        Args:
            hierarchy: The hierarchy of inputs defining the primary input, which is
                clustered, and secondary inputs, which are matched to each cluster.
                The order of the secondary_inputs is used as the priority for matching:
                {"primary_input": "input1",
                 "secondary_inputs": {"input2": [0, 6], "input3": [0, 24]}]}
            model_id_attr: The model ID attribute used to identify different models
                within the input cubes.
            target_grid_name: The name of the target grid cube for regridding.
            clustering_method: The clustering method to use.
            **kwargs: Additional arguments for the clustering method.
        Raises:
            NotImplementedError: If the clustering method is not supported.
        """
        self.hierarchy = hierarchy
        self.model_id_attr = model_id_attr
        self.target_grid_name = target_grid_name
        self.clustering_method = clustering_method
        self.kwargs = kwargs
        if clustering_method != "KMedoids":
            msg = (
                "Currently only KMedoids clustering is supported for the clustering "
                "and matching of realizations."
            )
            raise NotImplementedError(msg)

    def cluster_primary_input(self, cube, target_grid_cube) -> iris.cube.Cube:
        """Cluster the primary input cube. The primary input cube is regridded
        to the target grid before clustering using area-weighted regridding.

        Args:
            primary_cube: The primary input cube to cluster.
            target_grid_cube: The target grid cube for regridding.
        Returns:
            Tuple of the clustered primary input cube and the regridded clustered
            primary input cube.
        """
        # Regrid the primary cube prior to clustering to speed up clustering and
        # emphasise key features of relevance for clustering.
        regridded_cube = RegridLandSea(
            regrid_mode="area-weighted",
        )(cube, target_grid_cube)

        clustering_result = RealizationClustering(
            self.clustering_method, **self.kwargs
        )(regridded_cube)
        clustered_cube = self._select_realizations_for_kmedoid_clusters(
            cube, clustering_result
        )
        clustered_regridded_cube = self._select_realizations_for_kmedoid_clusters(
            regridded_cube, clustering_result
        )
        return clustered_cube, clustered_regridded_cube

    def _select_realizations_for_kmedoid_clusters(
        self, cube: iris.cube.Cube, clustering_result
    ) -> iris.cube.Cube:
        """Select the realizations corresponding to the medoid indices from
        the clustering result.

        Args:
            cube: The input cube to select realizations from.
            clustering_result: The result of the clustering.
        Returns:
            cube_clustered: The clustered cube.
        """
        n_realizations = len(cube.coord("realization").points)
        if len(clustering_result.medoid_indices_) > n_realizations:
            n_clusters = len(clustering_result.medoid_indices_)
            msg = (
                f"The number of clusters {n_clusters} is expected to be less than "
                "the number of realizations {}. Please reduce the number of clusters."
            )
            raise ValueError(msg)

        # Select the realizations corresponding to the medoid indices.
        cube_clustered = cube[clustering_result.medoid_indices_]
        cube_clustered.coord("realization").points = range(len(n_realizations))
        promote_aux_coord_to_dim_coord(cube_clustered, "realization")
        return cube_clustered

    def _categorise_secondary_inputs(
        self, cubes: iris.cube.CubeList, n_clusters: int
    ) -> tuple[list[tuple[str, list]], list[tuple[str, list]]]:
        """Categorise secondary inputs by whether they have full or partial realizations.

        Args:
            cubes: The input CubeList containing all inputs.
            n_clusters: The number of clusters.

        Returns:
            Tuple of (full_realization_inputs, partial_realization_inputs):
                full_realization_inputs: List of (name, forecast_periods) tuples
                    for inputs with >= n_clusters realizations.
                partial_realization_inputs: List of (name, forecast_periods) tuples
                    for inputs with < n_clusters realizations.
        """
        full_realization_inputs = []
        partial_realization_inputs = []

        for candidate_name, forecast_periods in self.hierarchy[
            "secondary_inputs"
        ].items():
            model_id_constr = iris.AttributeConstraint(
                coord_values={self.model_id_attr: candidate_name}
            )
            # Check first forecast period to determine realization count
            test_fp = forecast_periods[0]
            test_constr = iris.Constraint(forecast_period=test_fp)
            test_cube = cubes.extract(model_id_constr & test_constr)
            n_realizations = len(test_cube.coord("realization").points)

            if n_realizations >= n_clusters:
                full_realization_inputs.append((candidate_name, forecast_periods))
            else:
                partial_realization_inputs.append((candidate_name, forecast_periods))

        return full_realization_inputs, partial_realization_inputs

    def _process_full_realization_inputs(
        self,
        full_realization_inputs: list[tuple[str, list]],
        cubes: iris.cube.CubeList,
        target_grid_cube: iris.cube.Cube,
        regridded_clustered_primary_cube: iris.cube.Cube,
        n_clusters: int,
        replaced_realizations: dict,
        matched_cubes: iris.cube.CubeList,
    ) -> None:
        """Process full realization inputs in reverse precedence order.

        This method ensures all clusters get assigned before selective replacement
        begins. It processes inputs with >= n_clusters realizations, working from
        lowest to highest precedence so that higher precedence inputs can overwrite.

        Args:
            full_realization_inputs: List of (name, forecast_periods) tuples for
                inputs with full realization sets.
            cubes: The input CubeList containing all data.
            target_grid_cube: The target grid cube for regridding.
            regridded_clustered_primary_cube: The regridded clustered primary cube.
            n_clusters: The number of clusters.
            replaced_realizations: Dictionary tracking which (forecast_period, cluster)
                pairs have been replaced. Modified in-place.
            matched_cubes: CubeList to append matched results to. Modified in-place.
        """
        # Process in reverse order (lowest precedence first)
        for candidate_name, forecast_periods in reversed(full_realization_inputs):
            model_id_constr = iris.AttributeConstraint(
                coord_values={self.model_id_attr: candidate_name}
            )
            # Only process forecast periods where not all realizations have been replaced
            fps_to_process = [
                fp
                for fp in forecast_periods
                if fp not in replaced_realizations
                or len(replaced_realizations[fp]) < n_clusters
            ]
            if not fps_to_process:
                continue

            fp_constr = iris.Constraint(
                forecast_period=lambda cell: cell in fps_to_process
            )
            candidate_cube = cubes.extract(model_id_constr & fp_constr)

            regridded_candidate_cube = RegridLandSea(
                regrid_mode="area-weighted",
            )(candidate_cube, target_grid_cube)
            cluster_indices, realization_indices = RealizationToClusterMatcher()(
                regridded_clustered_primary_cube.extract(fp_constr),
                regridded_candidate_cube,
            )

            # Index the candidate cube using the realization indices
            matched_cube = candidate_cube[realization_indices]
            matched_cube.coord("realization").points = cluster_indices

            matched_cube.attributes.pop(self.model_id_attr)
            matched_cubes.append(matched_cube)

            # Mark the assigned clusters at these forecast periods as replaced
            for fp in fps_to_process:
                if fp not in replaced_realizations:
                    replaced_realizations[fp] = set()
                replaced_realizations[fp].update(cluster_indices)

    def _process_partial_realization_inputs(
        self,
        partial_realization_inputs: list[tuple[str, list]],
        cubes: iris.cube.CubeList,
        target_grid_cube: iris.cube.Cube,
        regridded_clustered_primary_cube: iris.cube.Cube,
        replaced_realizations: dict,
        matched_cubes: iris.cube.CubeList,
    ) -> None:
        """Process partial realization inputs in reverse precedence order.

        This method selectively replaces specific realizations at specific forecast
        periods. It processes inputs with < n_clusters realizations, working from
        lowest to highest precedence so that higher precedence inputs can overwrite
        lower precedence ones.

        Args:
            partial_realization_inputs: List of (name, forecast_periods) tuples for
                inputs with partial realization sets.
            cubes: The input CubeList containing all data.
            target_grid_cube: The target grid cube for regridding.
            regridded_clustered_primary_cube: The regridded clustered primary cube.
            replaced_realizations: Dictionary tracking which (forecast_period, cluster)
                pairs have been replaced. Modified in-place.
            matched_cubes: CubeList to append/modify matched results. Modified in-place.
        """
        # Process in reverse order (lowest precedence first)
        for candidate_name, forecast_periods in reversed(partial_realization_inputs):
            model_id_constr = iris.AttributeConstraint(
                coord_values={self.model_id_attr: candidate_name}
            )

            for fp in forecast_periods:
                fp_constr = iris.Constraint(forecast_period=fp)
                candidate_cube = cubes.extract(model_id_constr & fp_constr)

                regridded_candidate_cube = RegridLandSea(
                    regrid_mode="area-weighted",
                )(candidate_cube, target_grid_cube)

                # Get the matching cluster indices from the matcher
                clustered_fp_cube = regridded_clustered_primary_cube.extract(fp_constr)
                cluster_indices, realization_indices = RealizationToClusterMatcher()(
                    clustered_fp_cube,
                    regridded_candidate_cube,
                )

                # Index the candidate cube using the realization indices
                matched_cube = candidate_cube[realization_indices]
                matched_cube.coord("realization").points = cluster_indices

                # Now we need to merge this with any existing data at this forecast period
                # Find which realizations to replace in the existing cube
                if fp in replaced_realizations:
                    # Get the existing cube for this forecast period
                    existing_cubes = [
                        cube
                        for cube in matched_cubes
                        if fp in cube.coord("forecast_period").points
                    ]
                    if existing_cubes:
                        # Remove the old cube for this forecast period
                        existing_cube = existing_cubes[0]
                        matched_cubes.remove(existing_cube)

                        # Extract just this forecast period
                        existing_fp_cube = existing_cube.extract(fp_constr)

                        # Replace data for the specific cluster indices with the new data
                        result_data = existing_fp_cube.data.copy()
                        for i, cluster_idx in enumerate(cluster_indices):
                            # Find which position cluster_idx is in the existing cube
                            existing_real_coords = existing_fp_cube.coord(
                                "realization"
                            ).points
                            pos = np.where(existing_real_coords == cluster_idx)[0]
                            if len(pos) > 0:
                                result_data[pos[0]] = matched_cube.data[i]

                        # Create a new cube with the merged data
                        merged_cube = existing_fp_cube.copy(data=result_data)
                        matched_cube = merged_cube

                matched_cube.attributes.pop(self.model_id_attr, None)
                matched_cubes.append(matched_cube)

                # Mark which cluster indices were replaced
                if fp not in replaced_realizations:
                    replaced_realizations[fp] = set()
                replaced_realizations[fp].update(cluster_indices)

    def _add_unreplaced_forecast_periods(
        self,
        clustered_primary_cube: iris.cube.Cube,
        replaced_realizations: dict,
        matched_cubes: iris.cube.CubeList,
    ) -> None:
        """Add forecast periods from clustered primary cube that weren't replaced.

        This method identifies forecast periods in the clustered primary cube that
        were not replaced by any secondary inputs and adds them to the output.

        Args:
            clustered_primary_cube: The clustered primary cube containing all
                forecast periods.
            replaced_realizations: Dictionary tracking which forecast periods
                have been replaced.
            matched_cubes: CubeList to append unreplaced periods to.
                Modified in-place.
        """
        all_forecast_periods = clustered_primary_cube.coord("forecast_period").points
        unreplaced_fps = [
            fp for fp in all_forecast_periods if fp not in replaced_realizations
        ]
        if unreplaced_fps:
            unreplaced_constr = iris.Constraint(
                forecast_period=lambda cell: cell.point in unreplaced_fps
            )
            unreplaced_cube = clustered_primary_cube.extract(unreplaced_constr)
            matched_cubes.append(unreplaced_cube)

    def process(self, cubes: iris.cube.CubeList) -> iris.cube.Cube:
        """Cluster and match the data.

        Args:
            cubes: The input CubeList containing all primary and secondary inputs.
                Different forecast sources must be identifiable using the model_id_attr
                attribute.
        Returns:
            The matched cube containing all secondary inputs matched to clusters.
        """
        constr = iris.AttributeConstraint(
            coord_values={self.model_id_attr: self.hierarchy["primary_input"]}
        )
        primary_cube = cubes.extract(constr)
        target_grid_cube = cubes.extract(self.target_grid_name)
        clustered_primary_cube, regridded_clustered_primary_cube = (
            self.cluster_primary_input(primary_cube, target_grid_cube)
        )

        n_clusters = len(clustered_primary_cube.coord("realization").points)

        # Categorise secondary inputs by whether they have full or partial realizations
        full_realization_inputs, partial_realization_inputs = (
            self._categorise_secondary_inputs(cubes, n_clusters)
        )

        # Track which (forecast_period, realization) pairs have been replaced
        # Key: forecast_period, Value: set of realization indices that have been replaced
        replaced_realizations = {}
        matched_cubes = iris.cube.CubeList()

        # First pass: Process full realization inputs
        self._process_full_realization_inputs(
            full_realization_inputs,
            cubes,
            target_grid_cube,
            regridded_clustered_primary_cube,
            n_clusters,
            replaced_realizations,
            matched_cubes,
        )

        # Second pass: Process partial realization inputs
        self._process_partial_realization_inputs(
            partial_realization_inputs,
            cubes,
            target_grid_cube,
            regridded_clustered_primary_cube,
            replaced_realizations,
            matched_cubes,
        )

        # Add any forecast periods from clustered_primary_cube that weren't replaced
        self._add_unreplaced_forecast_periods(
            clustered_primary_cube, replaced_realizations, matched_cubes
        )

        return matched_cubes.merge_cube()
