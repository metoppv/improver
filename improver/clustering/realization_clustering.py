# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugins to perform clustering on realizations within a cube."""

import json
from typing import Any

import iris
import numpy as np
import pandas as pd
from iris.coords import AuxCoord
from iris.util import equalise_attributes, new_axis, promote_aux_coord_to_dim_coord

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
        """Apply the clustering method to the cube.

        Cubes with more than 2 dimensions are converted to 2D arrays before
        clustering by flattening all dimensions except the leading dimension.
        The leading dimension is assumed to be the realization dimension.

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
        # to the flattened non-realization dimensions. These column values are the
        # features that the clustering algorithm will use to cluster the realizations.
        df = pd.DataFrame(
            array_2d,
            index=[f"realization_{p}" for p in cube.coord("realization").points],
        )
        return FitClustering(self.clustering_method, **self.kwargs)(df)


class RealizationToClusterMatcher(BasePlugin):
    """Match candidate realizations to clusters based on mean squared error.

    Each cluster is assigned to the candidate realization with the lowest MSE
    for that cluster. When multiple candidates compete for the same cluster,
    only the candidate with the lowest MSE is assigned; other candidates are
    not assigned to any cluster.

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
            realization_cluster_mse: The mean squared error array with shape
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

    def choose_clusters(self, realization_cluster_mse: np.ndarray) -> list[int]:
        """Choose clusters using a greedy assignment algorithm based on MSE.

        This method assigns clusters to candidate realizations by minimizing
        mean squared error. When multiple realizations compete for the same
        cluster, the cluster is assigned to the realization with the lowest
        MSE, and other competing realizations are not assigned to any cluster.

        Note: This greedy algorithm is chosen for its relative simplicity and
        computational efficiency. While optimal assignment algorithms (such as
        the Hungarian algorithm) could guarantee globally optimal solutions,
        this approach provides good results with O(n²) complexity and
        deterministic behavior.

        The algorithm processes realizations in descending order of their
        "MSE cost" (the sum of differences between each cluster's MSE and the
        minimum MSE for that realization). Realizations with higher costs
        (those with more uniform MSE across clusters and without a cluster that they
        are "well matched" to) are processed first.
        This ordering ensures that when realizations must be forced into
        clusters (when realizations_remaining <= clusters_remaining),
        realizations with strong cluster preferences are processed last and
        are more likely to get their preferred cluster.

        Each cluster is matched to the realization with the globally lowest
        MSE for that cluster. The number of assigned realizations may be
        less than the total number of input realizations, and may also be less
        than the number of clusters.

        Args:
            realization_cluster_mse: The mean squared error array with shape
                (n_realizations, n_clusters).

        Returns:
            Tuple of (cluster_indices, realization_indices):

            - cluster_indices: List of cluster indices that were assigned.
            - realization_indices: List of realization indices, one per
              assigned cluster. Each element is the index of the
              candidate realization assigned to that cluster.
        """
        # Calculate cost for each realization (sum of differences from minimum MSE)
        min_mse_array = np.min(realization_cluster_mse, axis=1, keepdims=True)
        mse_array_cost = np.sum(realization_cluster_mse - min_mse_array, axis=1)

        # Process realizations in descending order of cost (highest cost first)
        realization_order = np.argsort(mse_array_cost)[::-1]

        n_realizations = realization_cluster_mse.shape[0]
        n_clusters = realization_cluster_mse.shape[1]
        cluster_to_realization = {}
        cluster_to_mse = {}

        for loop_idx, realization_idx in enumerate(realization_order):
            realizations_remaining = n_realizations - loop_idx
            assigned_clusters = list(cluster_to_realization.keys())
            clusters_remaining = n_clusters - len(assigned_clusters)

            mse_values = realization_cluster_mse[realization_idx].copy()
            # If there are more clusters remaining than realizations,
            # allow this realization to compete for already-assigned clusters
            if realizations_remaining <= clusters_remaining:
                mse_values[assigned_clusters] = np.inf

            cluster_idx = np.nanargmin(mse_values)
            if mse_values[cluster_idx] < cluster_to_mse.get(cluster_idx, np.inf):
                cluster_to_mse[cluster_idx] = mse_values[cluster_idx]
                cluster_to_realization[cluster_idx] = realization_idx

        # Sort by cluster index and return both cluster indices and realization indices
        sorted_items = sorted(cluster_to_realization.items())
        cluster_indices, realization_indices = zip(*sorted_items)
        return list(cluster_indices), list(realization_indices)

    def process(
        self,
        cube: iris.cube.Cube,
        candidate_cube: iris.cube.Cube,
    ) -> tuple[list[int], list[int]]:
        """Assign candidate realizations to clusters by mean squared error.

        This method takes a cube of clustered realizations and candidate
        realizations, then assigns each cluster to the candidate realization
        with the lowest MSE for that cluster. When multiple candidates compete
        for the same cluster, only the one with the lowest MSE is assigned;
        other candidates are not assigned to any cluster.

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

        realization_cluster_mse = self.mean_squared_error(
            cube.data,
            candidate_cube.data,
            n_candidates,
        )
        cluster_indices, realization_indices = self.choose_clusters(
            realization_cluster_mse
        )

        return cluster_indices, realization_indices


class RealizationClusterAndMatch(BasePlugin):
    """Cluster primary input realizations and match secondary inputs to clusters.

    This plugin performs KMedoids clustering on a primary input, then matches
    secondary input realizations to the resulting clusters based on mean squared
    error, respecting a configurable precedence hierarchy for multiple secondary
    inputs.
    """

    def __init__(
        self,
        hierarchy: dict[str, str | dict[str, list[int]]],
        model_id_attr: str,
        clustering_method: str,
        target_grid_name: str | None = None,
        regrid_mode: str = "esmf-area-weighted",
        regrid_for_clustering: bool = True,
        regrid_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the clustering and matching class.

        Args:
            hierarchy: The hierarchy of inputs defining the primary input, which is
                clustered, and secondary inputs, which are matched to each cluster.
                The order of the secondary_inputs is used as the priority for matching.
                The list values for secondary inputs specify forecast periods in hours.
                A two-element list [start, end] will be expanded to include all hours
                in that range. Lists with other lengths are treated as explicit lists
                of forecast period hours. All values will be automatically converted
                to seconds to match the forecast_period coordinate units in the input
                cubes::

                    {
                        "primary_input": "input1",
                        "secondary_inputs": {"input2": [0, 6], "input3": [0, 24]},
                    }

                In this example, input2 will use forecast periods in the range
                0 to 6 hours inclusive (i.e., any forecast periods between 0 and
                21600 seconds), and input3 will use the range 0 to 24 hours
                (0 to 86400 seconds). Only forecast periods that actually exist in the
                input cubes within these ranges will be processed.
            model_id_attr: The model ID attribute used to identify different models
                within the input cubes.
            target_grid_name: The name of the target grid cube for regridding. Only
                required if regrid_for_clustering is True.
            clustering_method: The clustering method to use.
            regrid_mode: The regridding mode to use. Default is
                "esmf-area-weighted". See RegridLandSea for available modes.
            regrid_for_clustering: If True, regrid all cubes (primary and secondary)
                to the target grid before clustering and matching. If False, perform
                clustering and matching on the original grids without regridding.
                Default is True.
            regrid_kwargs: Additional keyword arguments to pass to RegridLandSea.
                Common options include:

                - mdtol (float): Tolerance of missing data (default 1)
                - extrapolation_mode (str): Mode to fill regions outside domain
                - landmask (Cube): Land-sea mask for mask-aware regridding
                - landmask_vicinity (float): Radius for coastline search

            **kwargs: Additional arguments for the clustering method.

        Raises:
            NotImplementedError: If the clustering method is not supported.
        """
        self.hierarchy = hierarchy
        self.model_id_attr = model_id_attr
        self.target_grid_name = target_grid_name
        self.clustering_method = clustering_method
        self.regrid_mode = regrid_mode
        self.regrid_for_clustering = regrid_for_clustering
        self.regrid_kwargs = regrid_kwargs if regrid_kwargs is not None else {}
        self.kwargs = kwargs

        if regrid_for_clustering and target_grid_name is None:
            msg = (
                "target_grid_name must be provided when regrid_for_clustering is True."
            )
            raise ValueError(msg)

        if clustering_method != "KMedoids":
            msg = (
                "Currently only KMedoids clustering is supported for the clustering "
                "and matching of realizations."
            )
            raise NotImplementedError(msg)

    @staticmethod
    def _expand_forecast_period_range(fp_range: list[int]) -> list[int]:
        """Expand a forecast period range [start, end] to a list of integers.

        Args:
            fp_range: A list containing either [start, end] values defining a range
                in hours, or a list of specific forecast period hours.

        Returns:
            If fp_range has 2 elements, returns integers from start to end inclusive
            in steps of 1 hour. Otherwise, returns the list as-is.

        Raises:
            ValueError: If start > end (when 2 elements provided).
        """
        if len(fp_range) == 2:
            start, end = fp_range
            if start > end:
                msg = f"Forecast period range start ({start}) must be <= end ({end})"
                raise ValueError(msg)
            return list(range(start, end + 1, 1))
        else:
            # Return as-is for lists with != 2 elements
            return fp_range

    @staticmethod
    def _convert_hours_to_seconds(hours: list[int]) -> list[int]:
        """Convert a list of hours to seconds.

        Args:
            hours: List of forecast period values in hours.

        Returns:
            List of forecast period values in seconds.
        """
        return [h * 3600 for h in hours]

    def cluster_primary_input(
        self, cube: iris.cube.Cube, target_grid_cube: iris.cube.Cube | None
    ) -> tuple[iris.cube.Cube, iris.cube.Cube]:
        """Cluster the primary input cube. If regridding is enabled, the primary
        input cube is regridded to the target grid before clustering using the
        specified regridding method.

        Args:
            primary_cube: The primary input cube to cluster.
            target_grid_cube: The target grid cube for regridding. Can be None if
                regrid_for_clustering is False.
        Returns:
            Tuple of the clustered primary input cube and the regridded clustered
            primary input cube (or the same cube twice if regridding is disabled).
        """
        # Regrid the primary cube prior to clustering to speed up clustering and
        # emphasise key features of relevance for clustering (if enabled).
        if self.regrid_for_clustering:
            regridded_cube = RegridLandSea(
                regrid_mode=self.regrid_mode,
                **self.regrid_kwargs,
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
        else:
            clustering_result = RealizationClustering(
                self.clustering_method, **self.kwargs
            )(cube)
            clustered_cube = self._select_realizations_for_kmedoid_clusters(
                cube, clustering_result
            )
            clustered_regridded_cube = clustered_cube

        return clustered_cube, clustered_regridded_cube

    def _select_realizations_for_kmedoid_clusters(
        self, cube: iris.cube.Cube, clustering_result: Any
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
        cube_clustered.coord("realization").points = range(
            len(clustering_result.medoid_indices_)
        )
        promote_aux_coord_to_dim_coord(cube_clustered, "realization")
        return cube_clustered

    def _categorise_secondary_inputs(
        self, cubes: iris.cube.CubeList, n_clusters: int
    ) -> tuple[list[tuple[str, list[int]]], list[tuple[str, list[int]]]]:
        """Categorise secondary inputs by full or partial realizations.

        Args:
            cubes: The input CubeList containing all inputs.
            n_clusters: The number of clusters.

        Returns:
            Tuple of (full_realization_inputs, partial_realization_inputs):
                full_realization_inputs: List of (name, forecast_periods) tuples
                    for inputs with >= n_clusters realizations. The forecast_periods
                    are the actual forecast periods (in seconds) that exist in the cubes
                    within the specified hour range.
                partial_realization_inputs: List of (name, forecast_periods) tuples
                    for inputs with < n_clusters realizations. The forecast_periods
                    are the actual forecast periods (in seconds) that exist in the cubes
                    within the specified hour range.
        """
        full_realization_inputs = []
        partial_realization_inputs = []

        for candidate_name, fp_range in self.hierarchy["secondary_inputs"].items():
            # Expand the forecast period range and convert from hours to seconds
            fp_hours = self._expand_forecast_period_range(fp_range)
            fp_seconds_range = self._convert_hours_to_seconds(fp_hours)

            # Create constraint to get all cubes for this model
            model_id_constr = iris.AttributeConstraint(
                **{self.model_id_attr: candidate_name}
            )

            # Extract all cubes for this model
            model_cubes = cubes.extract(model_id_constr)

            # Filter to only those within the forecast period range
            forecast_periods_in_range = []
            for cube in model_cubes:
                if cube.coords("forecast_period"):
                    fp_value = cube.coord("forecast_period").points[0]
                    if fp_value in fp_seconds_range:
                        forecast_periods_in_range.append(int(fp_value))

            if not forecast_periods_in_range:
                continue  # No cubes found in this range for this model

            # Check first forecast period to determine realization count
            test_fp = forecast_periods_in_range[0]
            test_constr = iris.Constraint(forecast_period=test_fp)
            test_cube = cubes.extract_cube(model_id_constr & test_constr)
            n_realizations = len(test_cube.coord("realization").points)

            if n_realizations >= n_clusters:
                full_realization_inputs.append(
                    (candidate_name, forecast_periods_in_range)
                )
            else:
                partial_realization_inputs.append(
                    (candidate_name, forecast_periods_in_range)
                )

        return full_realization_inputs, partial_realization_inputs

    def _ensure_realization_is_first_dimension(
        self, cube: iris.cube.Cube
    ) -> iris.cube.Cube:
        """Ensure realization is the leading dimension coordinate.

        Args:
            cube: The cube to check and potentially transpose.

        Returns:
            The cube with realization as the first dimension.
        """
        if cube.dim_coords[0].name() != "realization":
            real_dim = cube.coord_dims("realization")[0]
            new_order = [real_dim] + [i for i in range(cube.ndim) if i != real_dim]
            cube.transpose(new_order)
        return cube

    def _ensure_forecast_period_is_dimension(
        self, cube: iris.cube.Cube
    ) -> iris.cube.Cube:
        """Ensure forecast_period is a dimension coordinate and realization is first.

        If forecast_period exists but is not a dimension coordinate (i.e., it's scalar
        or auxiliary), promote it to a dimension coordinate using new_axis. Then ensure
        realization is the leading dimension. Also ensures that the time coordinate is
        associated with the forecast_period dimension to avoid it being scalar.

        Args:
            cube: The cube to check and potentially modify.

        Returns:
            The cube with forecast_period as a dimension coordinate (if it exists),
            time associated with the forecast_period dimension, and realization as
            the first dimension.
        """
        if cube.coords("forecast_period") and not cube.coord_dims("forecast_period"):
            cube = new_axis(cube, "forecast_period")
            cube = self._ensure_realization_is_first_dimension(cube)

        # Ensure time coordinate is associated with forecast_period dimension
        if cube.coords("time") and cube.coords("forecast_period"):
            fp_dim = cube.coord_dims("forecast_period")
            time_dims = cube.coord_dims("time")
            # If time is scalar or not associated with forecast_period dimension
            if not time_dims or time_dims != fp_dim:
                time_coord = cube.coord("time")
                fp_coord = cube.coord("forecast_period")
                # Only reassociate if time coord shape matches forecast_period shape
                if time_coord.shape == fp_coord.shape:
                    # Remove time as a coordinate and re-add it associated with
                    # forecast_period
                    cube.remove_coord("time")
                    cube.add_aux_coord(time_coord, fp_dim)

        return cube

    def _initialise_matched_cubes_with_primary(
        self, clustered_primary_cube: iris.cube.Cube
    ) -> iris.cube.CubeList:
        """Initialize matched_cubes with clustered primary cube for all periods.

        This ensures we always have a full set of realizations to work with
        as a base, which can then be selectively replaced by secondary inputs.

        Args:
            clustered_primary_cube: The clustered primary cube containing all
                forecast periods.

        Returns:
            A CubeList containing one cube per forecast period from the clustered
            primary cube, each with forecast_period as a dimension coordinate.
        """
        matched_cubes = iris.cube.CubeList()
        for fp in clustered_primary_cube.coord("forecast_period").points:
            fp_constr = iris.Constraint(forecast_period=fp)
            fp_cube = clustered_primary_cube.extract(fp_constr)
            fp_cube = self._ensure_forecast_period_is_dimension(fp_cube)
            matched_cubes.append(fp_cube)
        return matched_cubes

    def _update_cluster_sources(
        self,
        cluster_sources: dict[int, dict[str, list[int]]],
        cluster_indices: list[int],
        candidate_name: str,
        fp: int,
    ) -> None:
        """Update cluster sources tracking when replacing data.

        This method removes the forecast period from the primary input and adds
        it to the secondary input for the specified clusters.

        Args:
            cluster_sources: Dictionary tracking which input was used for each
                cluster at each forecast period. Modified in-place.
            cluster_indices: List of cluster indices being updated.
            candidate_name: Name of the secondary input being added.
            fp: Forecast period value in seconds.
        """
        primary_name = self.hierarchy["primary_input"]
        for cluster_idx in cluster_indices:
            if cluster_idx not in cluster_sources:
                cluster_sources[cluster_idx] = {}
            # Remove this forecast period from primary input
            if primary_name in cluster_sources[cluster_idx]:
                if fp in cluster_sources[cluster_idx][primary_name]:
                    cluster_sources[cluster_idx][primary_name].remove(fp)
                # Clean up empty lists
                if not cluster_sources[cluster_idx][primary_name]:
                    del cluster_sources[cluster_idx][primary_name]
            # Add to secondary input
            if candidate_name not in cluster_sources[cluster_idx]:
                cluster_sources[cluster_idx][candidate_name] = []
            if fp not in cluster_sources[cluster_idx][candidate_name]:
                cluster_sources[cluster_idx][candidate_name].append(fp)

    def _add_cluster_sources_coord(
        self,
        result_cube: iris.cube.Cube,
        cluster_sources: dict[int, dict[str, list[int]]],
    ) -> None:
        """Add cluster_sources coordinate to the result cube.

        Creates a 2D auxiliary coordinate that maps each (cluster, forecast_period)
        pair to the input name that was used for that cluster at that forecast period.

        Args:
            result_cube: The cube to add the coordinate to. Modified in-place.
            cluster_sources: Dictionary tracking which input was used for each
                cluster at each forecast period.
        """
        # Get dimensions
        realization_points = result_cube.coord("realization").points
        forecast_period_points = result_cube.coord("forecast_period").points

        # Build 2D array of input names (cluster x forecast_period)
        cluster_sources_array = np.empty(
            (len(realization_points), len(forecast_period_points)), dtype=object
        )

        for cluster_idx, cluster_num in enumerate(realization_points):
            for fp_idx, fp in enumerate(forecast_period_points):
                # Find which input was used for this cluster at this forecast period
                input_name = None
                if cluster_num in cluster_sources:
                    for name, fps in cluster_sources[cluster_num].items():
                        if fp in fps:
                            input_name = name
                            break
                # Fill with the input name or empty string if not found
                cluster_sources_array[cluster_idx, fp_idx] = (
                    input_name if input_name is not None else ""
                )

        # Create the auxiliary coordinate
        cluster_sources_coord = AuxCoord(
            cluster_sources_array,
            long_name="cluster_sources",
            units="no_unit",
        )

        # Add the coordinate associated with realization and forecast_period dimensions
        real_dim = result_cube.coord_dims("realization")[0]
        fp_dim = result_cube.coord_dims("forecast_period")[0]
        result_cube.add_aux_coord(cluster_sources_coord, (real_dim, fp_dim))

    def _process_full_realization_inputs(
        self,
        full_realization_inputs: list[tuple[str, list[int]]],
        cubes: iris.cube.CubeList,
        target_grid_cube: iris.cube.Cube,
        regridded_clustered_primary_cube: iris.cube.Cube,
        replaced_realizations: dict[int, set[int]],
        matched_cubes: iris.cube.CubeList,
        cluster_sources: dict[int, dict[str, list[int]]],
    ) -> None:
        """Process full realization inputs in reverse precedence order.

        This method replaces entire forecast period cubes with secondary inputs
        that have >= n_clusters realizations. It processes inputs working from
        lowest to highest precedence so that higher precedence inputs can overwrite.

        Args:
            full_realization_inputs: List of (name, forecast_periods) tuples for
                inputs with full realization sets.
            cubes: The input CubeList containing all data.
            target_grid_cube: The target grid cube for regridding.
            regridded_clustered_primary_cube: The regridded clustered primary cube.
            replaced_realizations: Dictionary tracking which (forecast_period, cluster)
                pairs have been replaced. Modified in-place.
            matched_cubes: CubeList containing cubes to modify. Modified in-place.
            cluster_sources: Dictionary tracking which input was used for each cluster
                at each forecast period. Modified in-place.
        """
        inputs_list = list(reversed(full_realization_inputs))

        # Process in reverse order (lowest precedence first)
        for idx, (candidate_name, forecast_periods) in enumerate(inputs_list):
            # Collect forecast periods from all higher-precedence inputs
            # (those that come after this one in inputs_list, i.e., earlier in
            # original list)
            higher_precedence_fps = set()
            for _, fps in inputs_list[idx + 1 :]:
                higher_precedence_fps.update(fps)

            # Skip forecast periods that will be overwritten by
            # higher-precedence inputs
            fps_to_process = [
                fp for fp in forecast_periods if fp not in higher_precedence_fps
            ]
            if not fps_to_process:
                continue

            model_id_constr = iris.AttributeConstraint(
                **{self.model_id_attr: candidate_name}
            )
            fp_constr = iris.Constraint(
                forecast_period=lambda cell: cell in fps_to_process
            )
            candidate_cubes = cubes.extract(model_id_constr & fp_constr)
            if len(candidate_cubes) > 1:
                candidate_cube = candidate_cubes.merge_cube()
                # Ensure realization is the leading dimension
                candidate_cube = self._ensure_realization_is_first_dimension(
                    candidate_cube
                )
            elif len(candidate_cubes) == 1:
                candidate_cube = candidate_cubes[0]
            else:
                continue  # No matching cubes for this forecast period

            if self.regrid_for_clustering:
                regridded_candidate_cube = RegridLandSea(
                    regrid_mode=self.regrid_mode,
                    **self.regrid_kwargs,
                )(candidate_cube, target_grid_cube)
            else:
                regridded_candidate_cube = candidate_cube

            cluster_indices, realization_indices = RealizationToClusterMatcher()(
                regridded_clustered_primary_cube.extract(fp_constr),
                regridded_candidate_cube,
            )

            # Index the candidate cube using the realization indices
            matched_cube = candidate_cube[realization_indices]
            matched_cube.coord("realization").points = cluster_indices

            matched_cube.attributes.pop(self.model_id_attr)

            # Ensure forecast_period is a dimension coordinate for consistent merging
            matched_cube = self._ensure_forecast_period_is_dimension(matched_cube)

            # Replace the existing forecast period cube(s) with the matched one
            for fp in fps_to_process:
                fp_constr_single = iris.Constraint(forecast_period=fp)
                # Remove existing cube for this forecast period
                existing_cubes = [
                    cube
                    for cube in matched_cubes
                    if fp in cube.coord("forecast_period").points
                ]
                for existing_cube in existing_cubes:
                    matched_cubes.remove(existing_cube)

                # Extract and add the new cube for this forecast period
                fp_matched_cube = matched_cube.extract(fp_constr_single)
                matched_cubes.append(fp_matched_cube)

                # Track which forecast periods have been fully replaced
                if fp not in replaced_realizations:
                    replaced_realizations[fp] = set()
                replaced_realizations[fp].update(cluster_indices)

                # Track cluster sources: update which input was used for each cluster
                self._update_cluster_sources(
                    cluster_sources, cluster_indices, candidate_name, fp
                )

    def _process_partial_realization_inputs(
        self,
        partial_realization_inputs: list[tuple[str, list[int]]],
        cubes: iris.cube.CubeList,
        target_grid_cube: iris.cube.Cube,
        regridded_clustered_primary_cube: iris.cube.Cube,
        replaced_realizations: dict[int, set[int]],
        matched_cubes: iris.cube.CubeList,
        cluster_sources: dict[int, dict[str, list[int]]],
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
            cluster_sources: Dictionary tracking which input was used for each cluster
                at each forecast period. Modified in-place.
        """
        inputs_list = list(reversed(partial_realization_inputs))

        # Process in reverse order (lowest precedence first)
        for candidate_name, forecast_periods in inputs_list:
            model_id_constr = iris.AttributeConstraint(
                **{self.model_id_attr: candidate_name}
            )

            for fp in forecast_periods:
                fp_constr = iris.Constraint(forecast_period=fp)
                candidate_cube = cubes.extract_cube(model_id_constr & fp_constr)

                if self.regrid_for_clustering:
                    regridded_candidate_cube = RegridLandSea(
                        regrid_mode=self.regrid_mode,
                        **self.regrid_kwargs,
                    )(candidate_cube, target_grid_cube)
                else:
                    regridded_candidate_cube = candidate_cube

                # Get the matching cluster indices from the matcher
                clustered_fp_cube = regridded_clustered_primary_cube.extract(fp_constr)
                cluster_indices, realization_indices = RealizationToClusterMatcher()(
                    clustered_fp_cube,
                    regridded_candidate_cube,
                )

                # Index the candidate cube using the realization indices
                matched_cube = candidate_cube[realization_indices]
                matched_cube.coord("realization").points = cluster_indices

                # Get the existing cube for this forecast period (guaranteed to exist)
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

                # Ensure forecast_period is a dimension coordinate for
                # consistent merging
                matched_cube = self._ensure_forecast_period_is_dimension(matched_cube)

                matched_cubes.append(matched_cube)

                # Mark which cluster indices were replaced
                if fp not in replaced_realizations:
                    replaced_realizations[fp] = set()
                replaced_realizations[fp].update(cluster_indices)

                # Track cluster sources: update which input was used for each cluster
                self._update_cluster_sources(
                    cluster_sources, cluster_indices, candidate_name, fp
                )

    def process(self, cubes: iris.cube.CubeList) -> iris.cube.Cube:
        """Cluster and match the data.

        Args:
            cubes: The input CubeList containing all primary and secondary inputs.
                Different forecast sources must be identifiable using the model_id_attr
                attribute.
        Returns:
            The matched cube containing all secondary inputs matched to clusters.
            The cube includes a 'cluster_sources' attribute (JSON string) that tracks
            which input source was used for each cluster at each forecast period.
            Format: {cluster_idx: {model_name: [fp1, fp2, ...]}}
            where cluster_idx is the cluster index (int), model_name is the value
            from model_id_attr (str), and the list contains forecast period values
            in seconds (int). Use json.loads() to parse the attribute value.
        """
        constr = iris.AttributeConstraint(
            **{self.model_id_attr: self.hierarchy["primary_input"]}
        )
        primary_cubes = cubes.extract(constr)
        if len(primary_cubes) > 1:
            equalise_attributes(primary_cubes)
            primary_cube = primary_cubes.merge_cube()
            # Ensure realization is the leading dimension after merge
            primary_cube = self._ensure_realization_is_first_dimension(primary_cube)
        elif len(primary_cubes) == 1:
            primary_cube = primary_cubes[0]
        else:
            raise ValueError(
                f"No primary cube found with {self.model_id_attr}="
                f"{self.hierarchy['primary_input']}"
            )

        target_grid_cube = None
        if self.regrid_for_clustering:
            target_grid_cube = cubes.extract_cube(self.target_grid_name)

        clustered_primary_cube, regridded_clustered_primary_cube = (
            self.cluster_primary_input(primary_cube, target_grid_cube)
        )

        n_clusters = len(clustered_primary_cube.coord("realization").points)

        # Categorise secondary inputs by whether they have full or partial realizations
        full_realization_inputs, partial_realization_inputs = (
            self._categorise_secondary_inputs(cubes, n_clusters)
        )

        # Track which (forecast_period, realization) pairs have been replaced
        # Key: forecast_period, Value: set of realization indices that have
        # been replaced
        replaced_realizations = {}

        # Track cluster sources: which input was used for each cluster at each
        # forecast period
        # Format: {cluster_idx: {model_name: [fp1, fp2, ...]}}
        cluster_sources = {}

        # Start with the clustered primary cube as the base for all forecast periods
        # This ensures we always have a full set of realizations to work with
        matched_cubes = self._initialise_matched_cubes_with_primary(
            clustered_primary_cube
        )

        # Initialize cluster_sources with primary input for all clusters and
        # forecast periods
        primary_name = self.hierarchy["primary_input"]
        for cluster_idx in range(n_clusters):
            cluster_sources[cluster_idx] = {}
            cluster_sources[cluster_idx][primary_name] = list(
                clustered_primary_cube.coord("forecast_period").points
            )

        # First pass: Process full realization inputs
        # These will replace entire forecast period cubes
        self._process_full_realization_inputs(
            full_realization_inputs,
            cubes,
            target_grid_cube,
            regridded_clustered_primary_cube,
            replaced_realizations,
            matched_cubes,
            cluster_sources,
        )

        # Second pass: Process partial realization inputs
        # These will selectively replace specific realizations within existing cubes
        self._process_partial_realization_inputs(
            partial_realization_inputs,
            cubes,
            target_grid_cube,
            regridded_clustered_primary_cube,
            replaced_realizations,
            matched_cubes,
            cluster_sources,
        )

        # Final safety check: ensure all cubes have forecast_period as a
        # dimension coordinate. This is necessary for successful concatenation.
        for i, cube in enumerate(matched_cubes):
            matched_cubes[i] = self._ensure_forecast_period_is_dimension(cube)

        # Equalise attributes across all cubes to ensure they can be concatenated
        equalise_attributes(matched_cubes)

        # Concatenate the cubes
        result_cube = matched_cubes.concatenate_cube()

        # Store cluster_sources as a cube attribute (as JSON string)
        # Format: {cluster_idx: {model_name: [fp1, fp2, ...]}}
        # Convert numpy int32 to native Python int for JSON serialization
        cluster_sources_serialisable = {
            int(k): {name: [int(fp) for fp in fps] for name, fps in v.items()}
            for k, v in cluster_sources.items()
        }
        result_cube.attributes["cluster_sources"] = json.dumps(
            cluster_sources_serialisable
        )

        return result_cube
