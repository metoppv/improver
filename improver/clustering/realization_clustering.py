# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugins to perform clustering on realizations within a cube."""

import json
import warnings
from collections import defaultdict
from typing import Any

import iris
import numpy as np
import pandas as pd
from iris.cube import Cube, CubeList
from iris.util import new_axis, promote_aux_coord_to_dim_coord

from improver import BasePlugin
from improver.clustering.clustering import FitClustering
from improver.regrid.landsea import RegridLandSea
from improver.utilities.cube_manipulation import (
    MergeCubes,
    enforce_coordinate_ordering,
    get_dim_coord_names,
)

try:
    import kmedoids
except ModuleNotFoundError:
    # Define empty class to avoid type hint errors.
    class KMedoids:
        pass


class RealizationClustering(BasePlugin):
    """Class to perform clustering on realizations of a cube. For example, this can be
    used to cluster a large number of ensemble members based on their spatial patterns
    into a smaller set of distinct clusters. If the input is precipitation forecasts,
    the resultant clusters could represent different types of precipitation events.
    """

    def __init__(self, clustering_method: str, **kwargs: Any) -> None:
        """Initialise the RealizationClustering class.

        Args:
            clustering_method: The clustering method to use. The clustering method
            to use (e.g. "KMedoids"). The method must be supported by the
            improver.clustering.FitClustering class.
            **kwargs: Additional arguments for the clustering method.
        """
        self.clustering_method = clustering_method
        self.kwargs = kwargs

    @staticmethod
    def _convert_to_2d(array: np.ndarray) -> np.ndarray:
        """Convert an array with arbitrary dimensions to a 2D array by maintaining
        the zeroth dimension and flattening all other dimensions.

        This prepares the data for clustering algorithms that expect 2D input where
        rows are samples (realizations) and columns are features
        (e.g. spatial points x forecast periods) so that an array of shape
        (18, 4, 100, 100) is converted to shape (18, 40000).

        Args:
            array: The input array to convert. Can have any number of dimensions.
        Returns:
            array_2d: The converted 2D array with shape (array.shape[0], -1).
        """
        if array.ndim < 2:
            msg = "Input array must have at least 2 dimensions."
            raise ValueError(msg)
        elif array.ndim == 2:
            return array.copy()
        else:
            target_shape = (array.shape[0], -1)
            return array.reshape(target_shape)

    def process(self, cube: Cube) -> Any:
        """Apply the clustering method to the cube.

        Cubes with more than 2 dimensions are converted to 2D arrays before
        clustering by flattening all dimensions except the leading dimension.
        The leading dimension is assumed to be the realization dimension.

        Args:
            cube: The input cube to cluster with the realization dimension
                as the leading dimension.

        Returns:
            The result of the clustering algorithm applied to the input data.

        Raises:
            ValueError: If the leading dimension of the input cube is not
                the realization dimension.
        """
        if cube.dim_coords[0].name() != "realization":
            msg = (
                "The leading dimension of the input cube must be "
                "the 'realization' dimension."
            )
            raise ValueError(msg)
        array_2d = self._convert_to_2d(cube.data)
        # The rows of the DataFrame correspond to realizations. The columns correspond
        # to the flattened non-realization dimensions. These column values are the
        # features that the clustering algorithm will use to cluster the realizations.
        df = pd.DataFrame(
            array_2d,
            index=[f"realization_{p}" for p in cube.coord("realization").points],
        )
        return FitClustering(self.clustering_method, **self.kwargs)(df)


class RealizationToClusterMatcher(BasePlugin):
    """Match candidate realizations to clusters based on mean squared error (MSE).
    In this context, 'candidate realizations' refers to the set of realizations being
    considered for assignment to clusters from a secondary input. These are matched
    to clusters derived from a primary input by minimizing mean squared error (MSE).

    Assigns realizations from a secondary input (e.g. a high-resolution regional
    ensemble model) to clusters defined by a primary input (e.g. a coarse global
    ensemble model). When multiple candidates compete for the same cluster, only the
    candidate with the lowest MSE is assigned; other candidates are not assigned to
    any cluster.

    Supports both 3D cubes and 4D cubes of dimensions (realization, y, x) and
    (realization, forecast_period, y, x) respectively only.
    """

    def __init__(self) -> None:
        """Initialise the plugin."""
        pass

    def _mean_squared_error_per_realization(
        self,
        clustered_array: np.ndarray,
        candidate_array: np.ndarray,
        n_realizations: int,
    ) -> np.ndarray:
        """Calculate MSE between clustered and candidate realization arrays. Lower MSE
        indicates a candidate realization better matches a cluster's representative
        member.

        For 3D cubes, the MSE is calculated by averaging over spatial dimensions (y, x).
        For 4D cubes, the mean is calculated over spatial dimensions first, then
        the MSE is averaged over forecast_period.

        Args:
            clustered_array: The clustered array with shape (n_clusters, y, x) or
                (n_clusters, forecast_period, y, x).
            candidate_array: The candidate array with shape (n_realizations, y, x)
                or (n_realizations, forecast_period, y, x).
            n_realizations: The number of realizations in the candidate array.
        Returns:
            Array of MSE values with shape (n_realizations, n_clusters) with
            element [i, j] containing the MSE between candidate realization i
            and cluster j.
        """
        mse_list = []
        for index in range(n_realizations):
            # Calculate squared differences between each candidate realization and
            # all cluster medoids
            squared_diff = np.square(clustered_array - candidate_array[index])

            if clustered_array.ndim == 3:
                # For 3D: average over spatial dimensions (y, x)
                mse = np.nanmean(squared_diff, axis=(1, 2))
            else:
                # For 4D: mean over spatial (y, x), then mean over forecast_period
                mse = np.nanmean(np.nanmean(squared_diff, axis=(2, 3)), axis=1)

            mse_list.append(mse)
        return np.array(mse_list)

    def _validate_cube_dimensions(
        self, clusters_cube: Cube, candidate_cube: Cube
    ) -> None:
        """Validate that both the clustered and candidate cubes have matching
        dimensions.

        Args:
            clusters_cube: The clustered cube.
            candidate_cube: The candidate cube.

        Raises:
            ValueError: If cube dimensions don't match.
            ValueError: If dimension coordinate names don't match.
        """
        if clusters_cube.ndim != candidate_cube.ndim:
            msg = (
                f"Clustered cube has {clusters_cube.ndim} dimensions but candidate "
                f"cube has {candidate_cube.ndim} dimensions. Both cubes must have "
                "the same number of dimensions (either 3D or 4D)."
            )
            raise ValueError(msg)
        if get_dim_coord_names(clusters_cube) != get_dim_coord_names(candidate_cube):
            msg = (
                "Clustered and candidate cubes must have the same dimension "
                "coordinates in the same order. "
                f"Clustered cube dimensions: {get_dim_coord_names(clusters_cube)}, "
                f"Candidate cube dimensions: {get_dim_coord_names(candidate_cube)}"
            )
            raise ValueError(msg)

    def _validate_forecast_period_coords(
        self, clusters_cube: Cube, candidate_cube: Cube
    ) -> None:
        """Validate matching forecast_period coordinates for 4D cubes.

        Args:
            clusters_cube: The clustered cube.
            candidate_cube: The candidate cube.

        Raises:
            ValueError: If forecast period coords do not match between clustered and
                candidate cubes.
        """
        if clusters_cube.ndim == 4:
            cube_fp = clusters_cube.coord("forecast_period")
            candidate_fp = candidate_cube.coord("forecast_period")
            if not np.array_equal(cube_fp.points, candidate_fp.points):
                msg = (
                    "Forecast period coords must match between clustered and "
                    f"candidate cubes. Clustered: {cube_fp.points}, "
                    f"Candidate: {candidate_fp.points}"
                )
                raise ValueError(msg)

    def assign_clusters(self, realization_cluster_mse: np.ndarray) -> list[int]:
        """Assign clusters to candidate realizations using greedy MSE minimization.

        This method assigns candidate realizations to clusters by minimizing mean
        squared error. The algorithm iterates through realizations in descending order
        of their "MSE cost" (the sum of differences between each cluster's MSE and
        the minimum MSE for that realization). Realizations with higher costs
        (those with more uniform MSE across clusters, i.e. without a cluster that they
        are "well matched" to) are processed first; low cost-realizations (those with
        a stronger "preference" for one cluster) are processed last. During each
        iteration, if the realization's MSE is better than the current holder's MSE
        (or the cluster is unassigned), it replaces assignment to that cluster;
        otherwise the cluster remains assigned to its current realization. This
        iterative process continues through all realizations, with early assignments
        by flexible realizations often being replaced by later-processed realizations
        that have stronger (lower MSE) matches to clusters

        Note: This greedy algorithm is chosen for its relative simplicity and
        computational efficiency. While optimal assignment algorithms (such as
        the Hungarian algorithm) could guarantee globally optimal solutions,
        this approach provides good results with O(n²) complexity and
        deterministic behavior.

        Args:
            realization_cluster_mse: The MSE array with shape
                (n_realizations, n_clusters).

        Returns:
            Tuple of (cluster_indices, realization_indices):

            - cluster_indices: List of cluster indices that were assigned
              (may be < n_clusters), sorted in ascending order.
            - realization_indices: List of realization indices assigned to each
              cluster (one per assigned cluster).
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

        # Iterate through realizations in order of descending cost. For example,
        # realization_order might be [3, 1, 0, 2].
        for loop_idx, realization_idx in enumerate(realization_order):
            # assigned_clusters is a list of cluster indices that have already
            # been assigned to a realization. In the first iteration, this will be
            # empty. In later iterations, this will contain the clusters that have
            # already been assigned to realizations in previous iterations.
            # clusters_remaining is the number of clusters that have not yet been
            # assigned to any realization.
            assigned_clusters = list(cluster_to_realization.keys())
            clusters_remaining = n_clusters - len(assigned_clusters)

            # If there are at least as many unassigned clusters as remaining
            # realizations, prevent competition for already-assigned clusters by
            # setting their MSE to inf. This forces each remaining realization to
            # select from unassigned clusters.
            mse_values = realization_cluster_mse[realization_idx].copy()
            n_realizations_remaining = n_realizations - loop_idx
            if n_realizations_remaining <= clusters_remaining:
                mse_values[assigned_clusters] = np.inf
            # Skip this realization if all MSE values are NaN
            if np.all(np.isnan(mse_values)):
                continue

            cluster_idx = np.nanargmin(mse_values)
            if mse_values[cluster_idx] < cluster_to_mse.get(cluster_idx, np.inf):
                # cluster_to_realization maps cluster indices to the currently assigned
                # realization index e.g. {1: 3}. cluster_to_mse maps cluster indices
                # to the MSE of the currently assigned realization e.g. {1: 10000}.
                cluster_to_mse[cluster_idx] = mse_values[cluster_idx]
                cluster_to_realization[cluster_idx] = realization_idx

        # Sort by cluster index and return both cluster indices and realization indices
        sorted_items = sorted(cluster_to_realization.items())
        cluster_indices, realization_indices = zip(*sorted_items)
        return list(cluster_indices), list(realization_indices)

    def process(
        self,
        clusters_cube: Cube,
        candidate_cube: Cube,
    ) -> tuple[list[int], list[int]]:
        """Assign candidate realizations to clusters by mean squared error (MSE).

        This method takes a cube of clustered realizations (e.g. from a global model)
        and candidate realizations (e.g. from a higher-resolution model), then assigns
        each cluster to the candidate realization with the lowest MSE for that cluster.
        When multiple candidates compete for the same cluster, only the one with the
        lowest MSE is assigned; other candidates are not assigned to any cluster.

        Supports both 3D cubes (realization, y, x) and 4D cubes
        (realization, forecast_period, y, x). When using 4D cubes, both input
        cubes must have matching forecast_period coordinates.

        Args:
            clusters_cube: The cube containing clustered realizations (e.g., from
                KMedoids clustering). Shape: (n_clusters, y, x) or
                (n_clusters, forecast_period, y, x).
            candidate_cube: The input cube with realizations to assign to
                clusters. Shape: (n_realizations, y, x) or
                (n_realizations, forecast_period, y, x).

        Returns:
            Tuple of (cluster_indices, realization_indices):
                cluster_indices: List of cluster indices that were assigned.
                    May have length < n_clusters if there are fewer candidates
                    than clusters.
                realization_indices: List of realization indices assigned to each
                    cluster (one per assigned cluster).
        """
        # Strictly enforce dimension order for both cubes
        enforce_coordinate_ordering(
            clusters_cube, ["realization", "forecast_period", "y", "x"]
        )
        enforce_coordinate_ordering(
            candidate_cube, ["realization", "forecast_period", "y", "x"]
        )

        n_candidates = len(candidate_cube.coord("realization").points)

        # Validate inputs
        self._validate_cube_dimensions(clusters_cube, candidate_cube)
        self._validate_forecast_period_coords(clusters_cube, candidate_cube)

        realization_cluster_mse = self._mean_squared_error_per_realization(
            clusters_cube.data,
            candidate_cube.data,
            n_candidates,
        )
        cluster_indices, realization_indices = self.assign_clusters(
            realization_cluster_mse
        )

        return cluster_indices, realization_indices


class RealizationClusterAndMatch(BasePlugin):
    """Cluster primary input realizations and match secondary inputs to clusters.

    This plugin performs KMedoids clustering on a primary input, then matches
    secondary input realizations to the resulting clusters based on mean squared
    error. When multiple secondary inputs are provided, their order in the hierarchy
    determines their precedence: inputs listed earlier (leftmost) in the
    secondary_inputs dictionary have higher priority and can overwrite matches from
    later (lower-priority) ones for overlapping forecast periods. In other words, the
    first (leftmost) secondary input in the dictionary has the highest precedence, and
    later ones have lower precedence. See the Args section of the __init__ docstring
    for details on how the hierarchy is specified and used.

    See Also:
        For a practical usage example, see:
        doc/source/examples/realization_cluster_and_match_example_data.py.
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
                in that inclusive range. Lists with other lengths are treated as
                explicit lists of forecast period hours. All values will be
                automatically converted to seconds to match the forecast_period
                coordinate units in the input cubes::

                    {
                        "primary_input": "input1",
                        "secondary_inputs": {"input2": [0, 24], "input3": [0, 6]},
                    }

                In this example, input2 will use forecast periods in the range
                0 to 24 hours inclusive (i.e., any forecast periods between 0 and
                86400 seconds), and input3 will use the range 0 to 6 hours
                (0 to 86400 seconds). For lead times, where secondary inputs are
                not provided the primary input will be used. Only forecast periods
                that actually exist in the input cubes within these ranges will be
                processed.
            model_id_attr: The model ID attribute used to identify different models
                within the input cubes.
            target_grid_name: The name of the target grid cube for regridding. Only
                required if regrid_for_clustering is True.
            clustering_method: The clustering method to use.
            regrid_mode: The regridding mode to use. Default is
                "esmf-area-weighted". See RegridLandSea for available modes.
            regrid_for_clustering: If True, regrid all cubes (primary and secondary)
                to the target grid before clustering and matching. This regridding
                step speeds up the computation by reducing the data size and,
                importantly, emphasises larger-scale spatial features in the data,
                rather than small-scale detail. This helps the clustering focus on the
                most relevant broad patterns rather than being dominated by
                fine-scale noise. If False, clustering and matching are performed
                on the original grids without regridding. Default is True.
            regrid_kwargs: Additional keyword arguments to pass to RegridLandSea.
                Common options include:

                - mdtol (float): Tolerance of missing data (default 1).
                - extrapolation_mode (str): Mode to fill regions outside domain.
                - landmask (Cube): Land-sea mask for mask-aware regridding.
                - landmask_vicinity (float): Radius for coastline search.

            **kwargs: Additional arguments for the clustering method.

        Raises:
            ValueError: If regrid_for_clustering is True but target_grid_name is None.
            NotImplementedError: If the clustering method is not supported
                (currently only KMedoids is supported).
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
        self, primary_cube: Cube, target_grid_cube: Cube | None
    ) -> tuple[Cube, Cube]:
        """Cluster the primary input cube. If regridding is enabled, the primary
        input cube is regridded to the target grid before clustering using the
        specified regridding method. Please see RegridLandSea for available modes.

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
            )(primary_cube, target_grid_cube)
        clustering_result = RealizationClustering(
            self.clustering_method, **self.kwargs
        )(regridded_cube if self.regrid_for_clustering else primary_cube)
        clustered_cube = self._select_realizations_for_kmedoid_clusters(
            primary_cube, clustering_result
        )
        clustered_regridded_cube = (
            self._select_realizations_for_kmedoid_clusters(
                regridded_cube, clustering_result
            )
            if self.regrid_for_clustering
            else clustered_cube
        )
        return clustered_cube, clustered_regridded_cube

    def _select_realizations_for_kmedoid_clusters(
        self, primary_cube: Cube, clustering_result: "kmedoids.KMedoids"
    ) -> Cube:
        """Select the realizations corresponding to the medoid indices from
        the clustering result.

        Args:
            primary_cube: The input cube to select realizations from.
            clustering_result: The result of the clustering.
        Returns:
            cube_clustered: The clustered cube.

        Raises:
            ValueError: If the number of clusters is greater than the number of
                realizations in the input cube.
        """
        n_realizations = len(primary_cube.coord("realization").points)
        if len(clustering_result.medoid_indices_) > n_realizations:
            n_clusters = len(clustering_result.medoid_indices_)
            msg = (
                f"The number of clusters {n_clusters} is expected to be less than "
                f"the number of realizations {n_realizations}. "
                "Please reduce the number of clusters."
            )
            raise ValueError(msg)

        # Select the realizations corresponding to the medoid indices.
        cube_clustered = primary_cube[clustering_result.medoid_indices_]
        cube_clustered.coord("realization").points = range(
            len(clustering_result.medoid_indices_)
        )
        promote_aux_coord_to_dim_coord(cube_clustered, "realization")

        cluster_to_realizations = defaultdict(list)
        for idx, cluster_num in enumerate(clustering_result.labels_):
            cluster_to_realizations[int(cluster_num)].append(
                int(primary_cube.coord("realization").points[idx])
            )

        # Convert defaultdict to regular dict for serialization
        cluster_to_realizations = {
            k: cluster_to_realizations[k] for k in sorted(cluster_to_realizations)
        }
        cube_clustered.attributes["clusters_to_primary_input_realizations"] = (
            cluster_to_realizations
        )
        return cube_clustered

    def _categorise_secondary_inputs(
        self, cubes: CubeList, n_clusters: int, primary_cube: Cube
    ) -> tuple[list[tuple[str, list[int]]], list[tuple[str, list[int]]]]:
        """
        Categorise secondary inputs by full or partial realizations.

        This method also validates that secondary inputs don't request forecast periods
        not present in the primary input. If such forecast periods are found, a warning
        is issued and those periods are filtered out.

        Args:
            cubes: CubeList containing all input cubes (primary and secondary)
                required for clustering and matching. Each cube should be identifiable
                by the model_id_attr.
            n_clusters: Number of clusters (realizations), created from the primary
                input, required to be considered a 'full' input.
            primary_cube: The primary input cube, used to filter forecast periods.

        Returns:
            Tuple (full_realization_inputs, partial_realization_inputs):
                full_realization_inputs: List of (model_name, forecast_periods) tuples
                    for secondary inputs with at least n_clusters realizations for
                    the relevant forecast periods. The forecast_periods are the
                    forecast periods (in seconds) that exist in the cubes within the
                    specified hour range and are present in the primary input.
                partial_realization_inputs: List of (model_name, forecast_periods)
                    tuples for secondary inputs with fewer than n_clusters realizations
                    for the relevant forecast periods. The forecast_periods are the
                    forecast periods (in seconds) that exist in the cubes within the
                    specified hour range and are present in the primary input.
        """
        full_realization_inputs = []
        partial_realization_inputs = []
        primary_fps = set(primary_cube.coord("forecast_period").points)

        for candidate_name, fp_range in self.hierarchy["secondary_inputs"].items():
            # Expand the forecast period range and convert from hours to seconds
            fp_hours = self._expand_forecast_period_range(fp_range)
            fp_seconds_range = self._convert_hours_to_seconds(fp_hours)

            # Create constraint to get all cubes for this model and these forecast periods
            model_id_constr = iris.AttributeConstraint(
                **{self.model_id_attr: candidate_name}
            )
            fp_constr = iris.Constraint(forecast_period=fp_seconds_range)
            model_cubes = cubes.extract(model_id_constr & fp_constr)
            if not model_cubes:
                continue  # No cubes found in this range for this model

            # Get all forecast periods present in the cubes
            forecast_periods_in_range = [
                int(cube.coord("forecast_period").points.item()) for cube in model_cubes
            ]

            # Check which forecast periods from secondary are not in primary
            secondary_fps = set(forecast_periods_in_range)
            missing_fps = secondary_fps - primary_fps

            if missing_fps:
                warnings.warn(
                    f"Secondary input '{candidate_name}' has forecast periods "
                    f"{sorted(missing_fps)} not present in primary input. "
                    "These will be ignored."
                )
                # Filter out missing forecast periods
                forecast_periods_in_range = [
                    fp for fp in forecast_periods_in_range if fp not in missing_fps
                ]

            if not forecast_periods_in_range:
                warnings.warn(
                    f"Secondary input '{candidate_name}' has no forecast periods "
                    "that overlap with primary input "
                    f"'{self.hierarchy['primary_input']}'. This input will be skipped.",
                    UserWarning,
                )
                continue  # No valid forecast periods after filtering

            # Check first forecast period to determine realization count
            n_realizations = len(model_cubes[0].coord("realization").points)

            if n_realizations >= n_clusters:
                full_realization_inputs.append(
                    (candidate_name, forecast_periods_in_range)
                )
            else:
                partial_realization_inputs.append(
                    (candidate_name, forecast_periods_in_range)
                )

        return full_realization_inputs, partial_realization_inputs

    def _ensure_forecast_period_is_dimension(self, cube: Cube) -> Cube:
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
            enforce_coordinate_ordering(cube, ["realization"])

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
        self, clustered_primary_cube: Cube
    ) -> CubeList:
        """Initialise matched_cubes with clustered primary cube for all periods.

        This ensures we always have a full set of realizations to work with
        as a base, which can then be selectively replaced by secondary inputs.

        Args:
            clustered_primary_cube: The clustered primary cube containing all
                forecast periods.

        Returns:
            A CubeList containing one cube per forecast period from the clustered
            primary cube, each with forecast_period as a dimension coordinate.
        """
        matched_cubes = CubeList()
        for fp_cube in clustered_primary_cube.slices_over("forecast_period"):
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
        """Update cluster sources tracking when replacing data from one model
        with another.

        This method removes the forecast period from the primary input's tracking
        and adds it to the secondary input for the specified clusters, maintaining a
        record of which model provided data for each cluster at each forecast_period.

        Args:
            cluster_sources: Dictionary tracking which input was used for each
                cluster at each forecast period. Modified in-place.
                Format: {cluster_idx: {model_name: [fp1, fp2, ...]}}
            cluster_indices: List of cluster indices being updated.
            candidate_name: Name of the secondary input being added
                e.g. 'secondary_input1'.
            fp: Forecast period value in seconds.
        """
        primary_name = self.hierarchy["primary_input"]
        for cluster_idx in cluster_indices:
            cluster_sources.setdefault(cluster_idx, {})
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

    def _maybe_regrid_candidate_cube(
        self, candidate_cube: Cube, target_grid_cube: Cube
    ) -> Cube:
        """Regrid the candidate cube if regrid_for_clustering is True, otherwise
        return as is.

        Args:
            candidate_cube: The input candidate Cube to potentially regrid.
            target_grid_cube: The target grid Cube to regrid onto if regridding
                is enabled.

        Returns:
            The regridded candidate Cube if regrid_for_clustering is True, otherwise
            the original candidate Cube.
        """
        if self.regrid_for_clustering:
            return RegridLandSea(
                regrid_mode=self.regrid_mode,
                **self.regrid_kwargs,
            )(candidate_cube, target_grid_cube)
        else:
            return candidate_cube

    def _process_full_realization_inputs(
        self,
        full_realization_inputs: list[tuple[str, list[int]]],
        cubes: CubeList,
        target_grid_cube: Cube,
        regridded_clustered_primary_cube: Cube,
        replaced_realizations: dict[int, set[int]],
        matched_cubes: CubeList,
        cluster_sources: dict[int, dict[str, list[int]]],
    ) -> None:
        """Process full realization inputs in reverse precedence order.

        This method replaces entire forecast period cubes with secondary inputs
        that have more realizations than the number of clusters to which they are being
        matched. It processes inputs working from lowest to highest precedence so
        that higher precedence inputs can overwrite.

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
                Format: {cluster_idx: {model_name: [fp1, fp2, ...]}}
        """
        inputs_list = list(reversed(full_realization_inputs))

        # Process in reverse order (lowest precedence first) so higher precedence
        # inputs can overwrite
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
            fp_constr = iris.Constraint(forecast_period=fps_to_process)
            candidate_cubes = cubes.extract(model_id_constr & fp_constr)

            candidate_cube = MergeCubes()(candidate_cubes)
            enforce_coordinate_ordering(candidate_cube, ["realization"])

            regridded_candidate_cube = self._maybe_regrid_candidate_cube(
                candidate_cube, target_grid_cube
            )

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
                # Extract and add the new cube for this forecast period
                fp_constr_single = iris.Constraint(forecast_period=fp)
                fp_matched_cube = matched_cube.extract(fp_constr_single)

                # Find index of the existing fp cube (guaranteed to exist)
                idx = next(
                    i
                    for i, c in enumerate(matched_cubes)
                    if fp in c.coord("forecast_period").points
                )
                # Replace in-place
                matched_cubes[idx] = fp_matched_cube

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
        cubes: CubeList,
        target_grid_cube: Cube,
        regridded_clustered_primary_cube: Cube,
        replaced_realizations: dict[int, set[int]],
        matched_cubes: CubeList,
        cluster_sources: dict[int, dict[str, list[int]]],
    ) -> None:
        """Process partial realization inputs in reverse precedence order.

        This method selectively replaces specific realizations at specific forecast
        periods. It processes inputs with fewer realizations than the number of
        clusters to which they are being matched. It works from lowest to highest
        precedence so that higher precedence inputs can overwrite lower precedence ones.

        Args:
            partial_realization_inputs: List of (name, forecast_periods) tuples for
                inputs with partial realization sets.
            cubes: CubeList containing all primary and secondary input cubes needed
                for clustering and matching. Each cube must have the model_id_attr
                attribute set, and all relevant models, forecast periods, and
                realizations to be processed or matched should be included.
            target_grid_cube: The target grid cube for regridding.
            regridded_clustered_primary_cube: The regridded clustered primary cube.
            replaced_realizations: Dictionary tracking which (forecast_period, cluster)
                pairs have been replaced. Modified in-place.
            matched_cubes: CubeList to append/modify matched results. Modified in-place.
            cluster_sources: Dictionary tracking which input was used for each cluster
                at each forecast period. Modified in-place.
                Format: {cluster_idx: {model_name: [fp1, fp2, ...]}}
        """
        # Process in reverse order (lowest precedence first)
        for candidate_name, forecast_periods in reversed(partial_realization_inputs):
            model_id_constr = iris.AttributeConstraint(
                **{self.model_id_attr: candidate_name}
            )

            for fp in forecast_periods:
                fp_constr = iris.Constraint(forecast_period=fp)
                candidate_cube = cubes.extract_cube(model_id_constr & fp_constr)

                regridded_candidate_cube = self._maybe_regrid_candidate_cube(
                    candidate_cube, target_grid_cube
                )

                # Get the matching cluster indices from the matcher
                clustered_fp_cube = regridded_clustered_primary_cube.extract(fp_constr)

                cluster_indices, realization_indices = RealizationToClusterMatcher()(
                    clustered_fp_cube,
                    regridded_candidate_cube,
                )

                # Index the candidate cube using the realization indices
                matched_cube = candidate_cube[realization_indices]
                matched_cube.coord("realization").points = cluster_indices

                # Extract existing fp cube and merge data for specified clusters
                idx = next(
                    i
                    for i, c in enumerate(matched_cubes)
                    if fp in c.coord("forecast_period").points
                )
                existing_fp_cube = matched_cubes[idx]

                # Replace data for the specific cluster indices with the new data
                result_data = existing_fp_cube.data.copy()
                for i, cluster_idx in enumerate(cluster_indices):
                    # Find which position cluster_idx is in the existing cube
                    pos = np.where(
                        existing_fp_cube.coord("realization").points == cluster_idx
                    )[0]
                    if len(pos) > 0:
                        result_data[pos[0]] = matched_cube.data[i]

                # Create a new cube with the merged data
                merged_cube = existing_fp_cube.copy(data=result_data)

                # Clean attributes and ensure coord shape/order, then replace in-place
                merged_cube.attributes.pop(self.model_id_attr, None)
                merged_cube = self._ensure_forecast_period_is_dimension(merged_cube)
                matched_cubes[idx] = merged_cube

                # Mark which cluster indices were replaced
                if fp not in replaced_realizations:
                    replaced_realizations[fp] = set()
                replaced_realizations[fp].update(cluster_indices)

                # Track cluster sources: update which input was used for each cluster
                self._update_cluster_sources(
                    cluster_sources, cluster_indices, candidate_name, fp
                )

    def process(self, cubes: CubeList) -> Cube:
        """Cluster and match the data.

        This method clusters the primary input realizations and matches secondary input
        realizations to the resulting clusters, according to the specified hierarchy
        and precedence.

        Args:
            cubes: The input CubeList containing all primary and secondary input
                cubes required for clustering and matching. Each cube must have the
                model_id_attr attribute set to identify its source/model. For each
                model (primary and secondary), include all forecast periods and
                realizations that should be considered for matching or replacement.

                Expected input shapes::

                    2D: (y, x)
                        for single realization, single forecast period fields.
                    3D: (realization, y, x)
                        for multiple realizations at a single forecast period.
                    4D: (realization, forecast_period, y, x)
                        for multiple realizations and multiple forecast periods.

                The leading dimension must always be realization if present.
                For 4D cubes, the second dimension must be forecast_period.

        Returns:
            The matched cube containing all secondary inputs matched to clusters.
            The output cube will have realization and forecast_period as leading
            dimensions (if present in the input), followed by spatial dimensions (y, x).
            The cube includes a 'cluster_sources' attribute (JSON string) that
            tracks which input source was used for each cluster at each
            forecast period. Format: {cluster_idx: {model_name: [fp1, fp2, ...]}},
            where cluster_idx is the cluster index (int), model_name is the value
            from model_id_attr (str), and the list contains forecast period values in
            seconds (int). Use json.loads() to parse the attribute value.

            Raises:
                ValueError: If no primary cube is found with the specified
                    model_id_attr.
        """
        constr = iris.AttributeConstraint(
            **{self.model_id_attr: self.hierarchy["primary_input"]}
        )
        primary_cubes = cubes.extract(constr)
        if primary_cubes:
            primary_cube = MergeCubes()(primary_cubes)
            enforce_coordinate_ordering(primary_cube, ["realization"])
        else:
            raise ValueError(
                f"No primary cube found with {self.model_id_attr}="
                f"{self.hierarchy['primary_input']}"
            )

        target_grid_cube = None
        if self.regrid_for_clustering:
            try:
                target_grid_cube = cubes.extract_cube(self.target_grid_name)
            except iris.exceptions.ConstraintMismatchError:
                msg = (
                    f"Target grid cube '{self.target_grid_name}' not found in input "
                    "cubes for regridding."
                )
                raise ValueError(msg)

        clustered_primary_cube, regridded_clustered_primary_cube = (
            self.cluster_primary_input(primary_cube, target_grid_cube)
        )
        # Store mapping for re-application to result.
        clusters_to_primary_input_realizations = clustered_primary_cube.attributes[
            "clusters_to_primary_input_realizations"
        ]

        n_clusters = len(clustered_primary_cube.coord("realization").points)

        # Categorise secondary inputs by whether they have full or partial realizations
        full_realization_inputs, partial_realization_inputs = (
            self._categorise_secondary_inputs(cubes, n_clusters, primary_cube)
        )
        # Check if we have any secondary inputs to process
        if not full_realization_inputs and not partial_realization_inputs:
            warnings.warn(
                "No secondary inputs have forecast periods that overlap with the "
                f"primary input '{self.hierarchy['primary_input']}'. "
                "Only the clustered primary input will be returned.",
                UserWarning,
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

        # Initialise cluster_sources with primary input for all clusters and
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

        result_cube = MergeCubes()(
            CubeList([iris.util.squeeze(c) for c in matched_cubes])
        )

        # Use json.dumps to store dictionary as attribute.
        result_cube.attributes["clusters_to_primary_input_realizations"] = json.dumps(
            clusters_to_primary_input_realizations
        )
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
