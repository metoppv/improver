# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the improver.clustering.realization_clustering module."""

import json
from datetime import datetime

import iris
import numpy as np
import pytest
from iris.util import promote_aux_coord_to_dim_coord

from improver.clustering.realization_clustering import (
    RealizationClusterAndMatch,
    RealizationClustering,
    RealizationToClusterMatcher,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


def _create_realization_cube(shape=(5, 10, 10), seed=42):
    """Create a cube with realization dimension for testing.

    Args:
        shape: Tuple of (n_realizations, dim1, dim2, ...). Must have at least
            2 dimensions.
        seed: Random seed for reproducibility.

    Returns:
        A cube with realization as the leading dimension.
    """
    np.random.seed(seed)
    n_realizations = shape[0]
    data = np.random.randn(*shape).astype(np.float32)

    cube = set_up_variable_cube(
        data,
        name="air_temperature",
        units="K",
        spatial_grid="equalarea",
        realizations=np.arange(n_realizations),
    )
    return cube


def _create_4d_realization_cube(
    n_realizations=5,
    forecast_periods=None,
    y_dim=8,
    x_dim=8,
    seed=42,
    model_id=None,
    base_value=None,
    realization_values=None,
    merge=True,
):
    """Create a 4D cube with realization and forecast_period dimensions.

    Args:
        n_realizations: Number of realizations.
        forecast_periods: List of forecast period hours. If None,
            uses range(n_forecast_periods).
        y_dim: Size of y dimension.
        x_dim: Size of x dimension.
        seed: Random seed for reproducibility (used if random data).
        model_id: Optional model_id attribute value.
        base_value: If set, use as base value for all data (plus forecast period offset).
        realization_values: Optional list/array of values for each realization
            (overrides base_value/random).

    Returns:
        If merge is True (default), returns a 4D cube with shape
        (n_realizations, n_forecast_periods, y_dim, x_dim).
        If merge is False, returns a list of cubes, one for each forecast_period,
        each with shape (n_realizations, y_dim, x_dim).
    """
    if forecast_periods is None:
        n_forecast_periods = 3
        forecast_periods = list(range(n_forecast_periods))
    else:
        n_forecast_periods = len(forecast_periods)

    cubes = iris.cube.CubeList()
    for i, fp_hours in enumerate(forecast_periods):
        if realization_values is not None:
            # Use per-realization values without forecast period offset
            data = np.array(
                [
                    np.full((y_dim, x_dim), val, dtype=np.float32)
                    for val in realization_values
                ]
            )
        elif base_value is not None:
            data = np.full(
                (n_realizations, y_dim, x_dim),
                base_value + fp_hours,
                dtype=np.float32,
            )
        else:
            np.random.seed(seed + i)  # Different seed per forecast period
            data = np.random.randn(n_realizations, y_dim, x_dim).astype(np.float32)

        cube = set_up_variable_cube(
            data,
            name="air_temperature",
            units="K",
            spatial_grid="equalarea",
            realizations=np.arange(n_realizations),
            time=datetime(2017, 1, 10, 3 + fp_hours),
            frt=datetime(2017, 1, 10, 3),
        )
        if model_id is not None:
            cube.attributes["model_id"] = model_id
        cubes.append(cube)

    if merge:
        merged_cube = cubes.merge_cube()
        # Promote the forecast_reference_time to a dimension coordinate.
        promote_aux_coord_to_dim_coord(merged_cube, "forecast_period")
        # Transpose so realization is the leading dimension
        # Order after merge is typically: time, realization, y, x
        # We want: realization, time (forecast_period), y, x
        merged_cube.transpose([1, 0, 2, 3])
        return merged_cube
    else:
        # Return the list of cubes, one per forecast_period, unmerged
        return list(cubes)


def _create_clusterable_realization_cube():
    """Create a cube with distinct clusterable realizations.

    Returns:
        A cube with 6 realizations that form 2 distinct clusters.
    """
    # Create two distinct patterns
    pattern1 = np.full((5, 5), 10.0, dtype=np.float32)
    pattern2 = np.full((5, 5), 20.0, dtype=np.float32)

    # Add small variations
    np.random.seed(42)
    data = np.array(
        [
            pattern1 + np.random.randn(5, 5) * 0.1,
            pattern1 + np.random.randn(5, 5) * 0.1,
            pattern1 + np.random.randn(5, 5) * 0.1,
            pattern2 + np.random.randn(5, 5) * 0.1,
            pattern2 + np.random.randn(5, 5) * 0.1,
            pattern2 + np.random.randn(5, 5) * 0.1,
        ],
        dtype=np.float32,
    )

    cube = set_up_variable_cube(
        data,
        name="air_temperature",
        units="K",
        spatial_grid="equalarea",
        realizations=np.arange(6),
    )
    return cube


def _create_uniform_cube(
    values,
    shape=(2, 2),
    name="air_temperature",
    units="K",
    spatial_grid="equalarea",
    dtype=np.float32,
):
    """Create a cube with uniform values for each realization.

    Helper function to reduce boilerplate in RealizationToClusterMatcher tests.

    Args:
        values: List/array of values, one per realization.
        shape: Spatial shape (y, x) for each realization.
        name: Cube name.
        units: Cube units.
        spatial_grid: Spatial grid type.
        dtype: Data type.

    Returns:
        Iris cube with specified uniform values per realization.
    """
    values = np.array(values)
    n_realizations = len(values)

    # Create data array with shape (n_realizations, y, x)
    data = np.array([np.full(shape, val, dtype=dtype) for val in values])

    return set_up_variable_cube(
        data,
        name=name,
        units=units,
        spatial_grid=spatial_grid,
        realizations=np.arange(n_realizations),
    )


def _assert_realization_matching(
    result, expected_cluster_indices, expected_realization_indices
):
    """Assert that result matches expected indices.

    Helper function to standardize verification logic in matcher tests.

    Args:
        result: Tuple of (cluster_indices, realization_indices) from matcher.
        expected_cluster_indices: Expected list of cluster indices.
        expected_realization_indices: Expected list of realization indices.
    """
    cluster_indices, realization_indices = result

    # Check cluster indices
    np.testing.assert_array_equal(cluster_indices, expected_cluster_indices)

    # Check realization indices
    np.testing.assert_array_equal(realization_indices, expected_realization_indices)


def _assert_cluster_sources_attribute(
    result_cube,
    expected_sources=None,
):
    """Assert that cluster_sources attribute exists and has expected content.

    Helper function to standardise cluster_sources attribute validation.
    The cluster_sources is stored as a JSON string in the cube attributes.

    Args:
        result_cube: The output cube from RealizationClusterAndMatch.process().
        expected_sources: Optional dict mapping (cluster_idx, fp_idx) tuples to
            expected source names. If provided, validates the actual sources match.
            The fp_idx is the index in the forecast_period coordinate.
            Example: {(0, 0): "secondary_model_1", (0, 1): "primary_model"}
    """
    assert (
        "cluster_sources" in result_cube.attributes
    ), "cluster_sources attribute should exist"

    # Parse the JSON string
    cluster_sources = json.loads(result_cube.attributes["cluster_sources"])

    # Convert string keys back to integers (JSON requires string keys)
    cluster_sources = {int(k): v for k, v in cluster_sources.items()}

    if expected_sources is not None:
        forecast_periods = result_cube.coord("forecast_period").points
        for (cluster_idx, fp_idx), expected_source in expected_sources.items():
            fp = forecast_periods[fp_idx]
            assert (
                cluster_idx in cluster_sources
            ), f"Cluster {cluster_idx} not found in cluster_sources"
            # Check if the forecast period is in any of the source lists
            found = False
            actual_source = None
            for source_name, fps in cluster_sources[cluster_idx].items():
                if fp in fps:
                    found = True
                    actual_source = source_name
                    break
            assert found, f"Forecast period {fp} not found for cluster {cluster_idx}"
            assert actual_source == expected_source, (
                f"Cluster {cluster_idx} at forecast period {fp} (index {fp_idx}) "
                f"should use {expected_source}, got {actual_source}"
            )


# Tests for RealizationClustering


@pytest.mark.parametrize(
    "clustering_method,n_clusters,random_state",
    [
        ("KMeans", 3, 42),  # Basic KMeans
        ("KMeans", 2, 42),  # KMeans with 2 clusters
        ("AgglomerativeClustering", 3, None),  # AgglomerativeClustering
    ],
)
def test_clustering_basic_clustering(clustering_method, n_clusters, random_state):
    """Test the RealizationClustering.process method with various clustering methods."""
    cube = _create_realization_cube(shape=(5, 10, 10))

    kwargs = {"n_clusters": n_clusters}
    if random_state is not None:
        kwargs["random_state"] = random_state

    plugin = RealizationClustering(clustering_method, **kwargs)
    result = plugin.process(cube)

    assert hasattr(result, "labels_")
    assert len(result.labels_) == 5  # 5 realizations
    assert np.all((result.labels_ >= 0) & (result.labels_ < n_clusters))


def test_clustering_kmeans_cluster_centers():
    """Test that KMeans clustering produces cluster centers."""
    cube = _create_realization_cube(shape=(6, 8, 8))

    plugin = RealizationClustering("KMeans", n_clusters=2, random_state=42)
    result = plugin.process(cube)

    assert hasattr(result, "cluster_centers_")
    # 2 clusters, 64 features (8x8 flattened)
    assert result.cluster_centers_.shape == (2, 64)


@pytest.mark.parametrize(
    "shape,expected_n_features,use_4d",
    [
        ((5, 10, 10), 100, False),  # 3D cube: 10x10 = 100 features
        ((3, 5, 5), 25, False),  # Smaller 3D cube: 5x5 = 25 features
        ((10, 20, 30), 600, False),  # Larger 3D cube: 20x30 = 600 features
        ((5, 50, 50), 2500, False),  # Large spatial dimensions: 50x50 = 2500 features
        ((8, 3, 7), 21, False),  # Different aspect ratio: 3x7 = 21 features
        (
            (5, 3, 8, 8),
            192,
            True,
        ),  # 4D cube: 3x8x8 = 192 features (with forecast_period)
    ],
)
def test_clustering_arbitrary_dimensions_to_2d_conversion(
    shape, expected_n_features, use_4d
):
    """Test that cubes with arbitrary dimensions are correctly converted to 2D.

    This implicitly tests the convert_to_2d method through the process method.
    The method flattens all dimensions except the leading (realization) dimension.
    This test includes both 3D cubes and 4D cubes (with forecast_period dimension)
    to verify that the convert_to_2d method handles arrays with any number of
    dimensions.
    """
    if use_4d:
        # For 4D: shape = (n_realizations, n_forecast_periods, y_dim, x_dim)
        n_realizations, n_forecast_periods, y_dim, x_dim = shape
        forecast_periods = list(range(n_forecast_periods))
        cube = _create_4d_realization_cube(
            n_realizations=n_realizations,
            forecast_periods=forecast_periods,
            y_dim=y_dim,
            x_dim=x_dim,
        )
    else:
        cube = _create_realization_cube(shape=shape)

    plugin = RealizationClustering("KMeans", n_clusters=2, random_state=42)
    result = plugin.process(cube)

    # The cluster centers should have the expected number of features
    assert hasattr(result, "cluster_centers_")
    assert result.cluster_centers_.shape == (2, expected_n_features)
    # The number of labels should match the number of realizations
    assert len(result.labels_) == shape[0]


def test_clustering_distinct_clusters():
    """Test clustering with clearly separable realizations."""
    cube = _create_clusterable_realization_cube()

    plugin = RealizationClustering("KMeans", n_clusters=2, random_state=42)
    result = plugin.process(cube)

    assert hasattr(result, "labels_")
    assert len(result.labels_) == 6
    # Should form 2 distinct clusters
    assert len(np.unique(result.labels_)) == 2


# fmt: off
@pytest.mark.parametrize(
    "method,kwargs,expected_attrs",
    [
        ("DBSCAN", {"eps": 50.0, "min_samples": 2}, ["labels_"]),
        ("KMedoids", {"n_clusters": 2, "random_state": 42}, ["labels_", "medoid_indices_"]),
    ],
)
# fmt: on
def test_clustering_specific_methods(method, kwargs, expected_attrs):
    """Test specific clustering methods."""
    if method == "KMedoids":
        pytest.importorskip("kmedoids")
    if method == "DBSCAN":
        cube = _create_realization_cube(shape=(8, 10, 10))
    elif method == "KMedoids":
        cube = _create_clusterable_realization_cube()
    plugin = RealizationClustering(method, **kwargs)
    result = plugin.process(cube)
    for attr in expected_attrs:
        assert hasattr(result, attr)


def test_clustering_invalid_clustering_method():
    """Test that an error is raised for unsupported clustering methods."""
    cube = _create_realization_cube()

    plugin = RealizationClustering("NonExistentMethod", n_clusters=3)

    with pytest.raises(
        ValueError, match="The clustering method 'NonExistentMethod' is not supported"
    ):
        plugin.process(cube)


def test_clustering_wrong_leading_dimension():
    """Test that an error is raised if realization is not the leading dimension."""
    # Create a cube with time as leading dimension
    data = np.random.randn(3, 5, 10, 10).astype(np.float32)
    cube = set_up_variable_cube(
        data[0],
        name="air_temperature",
        units="K",
        spatial_grid="equalarea",
        realizations=np.arange(5),
    )
    # Make latitude the leading dimension by transposing
    cube.transpose([1, 0, 2])

    plugin = RealizationClustering("KMeans", n_clusters=2)

    with pytest.raises(
        ValueError,
        match=(
            "The leading dimension of the input cube must be the "
            "'realization' dimension"
        ),
    ):
        plugin.process(cube)


def test_clustering_different_n_clusters():
    """Test that different n_clusters values produce different results."""
    cube = _create_realization_cube(shape=(8, 10, 10))

    plugin2 = RealizationClustering("KMeans", n_clusters=2, random_state=42)
    result2 = plugin2.process(cube)

    plugin4 = RealizationClustering("KMeans", n_clusters=4, random_state=42)
    result4 = plugin4.process(cube)

    assert len(np.unique(result2.labels_)) == 2
    assert len(np.unique(result4.labels_)) == 4


def test_clustering_preserves_cube():
    """Test that process does not modify the input cube."""
    cube = _create_realization_cube(shape=(5, 10, 10))
    original_data = cube.data.copy()

    plugin = RealizationClustering("KMeans", n_clusters=3, random_state=42)
    plugin.process(cube)

    np.testing.assert_array_equal(cube.data, original_data)


def test_clustering_with_single_realization():
    """Test process method with a single realization."""
    cube = _create_realization_cube(shape=(1, 10, 10))

    plugin = RealizationClustering("KMeans", n_clusters=1, random_state=42)
    result = plugin.process(cube)

    assert hasattr(result, "labels_")
    assert len(result.labels_) == 1
    assert result.labels_[0] == 0


# Tests for RealizationToClusterMatcher


def test_matcher_process_basic():
    """Test the process method with simple cubes."""
    clustered_cube = _create_uniform_cube([10.0, 20.0])
    candidate_cube = _create_uniform_cube([20.1, 10.1])

    plugin = RealizationToClusterMatcher()
    result = plugin.process(clustered_cube, candidate_cube)

    # Result should be (cluster_indices, realization_indices)
    # Expected: cluster 0 matches candidate 1, cluster 1 matches candidate 0
    _assert_realization_matching(result, [0, 1], [1, 0])


def test_matcher_process_multiple_realizations():
    """Test process with more realizations than clusters."""
    clustered_cube = _create_uniform_cube([0.0, 50.0, 100.0])
    candidate_cube = _create_uniform_cube([75.0, 99.0, 1.0, 2.0])

    plugin = RealizationToClusterMatcher()
    result = plugin.process(clustered_cube, candidate_cube)

    # Verify the specific matching based on the greedy algorithm:
    # The algorithm processes candidates in cost order and assigns:
    # - Candidate 2 (1.0) to cluster 0 with MSE=1.0
    # - Candidate 1 (99.0) to cluster 2 with MSE=1.0
    # - Candidate 0 (75.0) to cluster 1 with MSE=625.0
    # - Candidate 3 (2.0) tries cluster 0 but has MSE=4.0 > 1.0, so rejected
    # Result:
    # cluster 0 -> candidate 2, cluster 1 -> candidate 0, cluster 2 -> candidate 1
    _assert_realization_matching(result, [0, 1, 2], [2, 0, 1])


def test_matcher_process_with_nan_values():
    """Test that process handles NaN values correctly."""
    # Create cubes with some NaN values - manual setup required for NaN patterns
    clustered_data = np.array(
        [[[1.0, np.nan], [3.0, 4.0]], [[5.0, 6.0], [np.nan, 8.0]]], dtype=np.float32
    )
    clustered_cube = set_up_variable_cube(
        clustered_data,
        name="air_temperature",
        units="K",
        spatial_grid="equalarea",
        realizations=np.arange(2),
    )

    candidate_data = np.array(
        [[[1.0, np.nan], [3.0, 4.0]], [[5.0, 6.0], [np.nan, 8.0]]], dtype=np.float32
    )
    candidate_cube = set_up_variable_cube(
        candidate_data,
        name="air_temperature",
        units="K",
        spatial_grid="equalarea",
        realizations=np.arange(2),
    )

    plugin = RealizationToClusterMatcher()
    result = plugin.process(clustered_cube, candidate_cube)

    # Should return indices matching each cluster to its best candidate
    cluster_indices, realization_indices = result
    assert len(cluster_indices) == 2
    assert len(realization_indices) == 2
    # Both clusters should be matched
    np.testing.assert_array_equal(cluster_indices, [0, 1])


def test_matcher_process_identical_patterns():
    """Test matching with identical patterns to verify MSE calculation.

    This test verifies that the mean_squared_error calculation works correctly
    by using identical patterns that should match perfectly.
    """
    clustered_cube = _create_uniform_cube([0.0, 10.0])
    candidate_cube = _create_uniform_cube([0.0, 10.0])

    plugin = RealizationToClusterMatcher()
    result = plugin.process(clustered_cube, candidate_cube)

    # With identical patterns, each candidate should match its corresponding cluster
    cluster_indices, realization_indices = result
    assert len(cluster_indices) == 2
    assert len(realization_indices) == 2
    np.testing.assert_array_equal(cluster_indices, [0, 1])


def test_matcher_process_all_clusters_matched():
    """Test that all clusters are matched exactly once.

    This verifies the choose_clusters algorithm ensures one-to-one matching
    when n_realizations == n_clusters.
    """
    clustered_cube = _create_uniform_cube([0.0, 50.0, 100.0])
    # Create candidate with patterns in different order
    candidate_cube = _create_uniform_cube([100.0, 0.0, 50.0])

    plugin = RealizationToClusterMatcher()
    result = plugin.process(clustered_cube, candidate_cube)

    # All clusters should be matched: cluster 0->candidate 1, cluster 1->candidate 2,
    # cluster 2->candidate 0
    cluster_indices, realization_indices = result
    assert len(cluster_indices) == 3
    assert len(realization_indices) == 3
    np.testing.assert_array_equal(cluster_indices, [0, 1, 2])
    # Verify unique assignments
    assert len(set(realization_indices)) == 3, (
        "All realization indices should be unique"
    )


def test_matcher_choose_clusters_consistent_results():
    """Test that process produces consistent results for the same input."""
    clustered_cube = _create_realization_cube(shape=(3, 5, 5), seed=100)
    candidate_cube = _create_realization_cube(shape=(3, 5, 5), seed=200)

    plugin = RealizationToClusterMatcher()
    result1 = plugin.process(clustered_cube, candidate_cube)
    result2 = plugin.process(clustered_cube, candidate_cube)

    # Results should be identical for same input
    _assert_realization_matching(result1, result2[0], result2[1])


def test_matcher_process_single_cluster_multiple_candidates():
    """Test with a single cluster and multiple similar candidates.

    This test verifies that when multiple candidates compete for a single cluster,
    the algorithm selects the candidate with the lowest MSE.
    """
    clustered_cube = _create_uniform_cube([50.0])
    candidate_cube = _create_uniform_cube([51.0, 52.0, 53.0])

    plugin = RealizationToClusterMatcher()
    result = plugin.process(clustered_cube, candidate_cube)

    # Should select candidate 0 (closest to cluster) for cluster 0
    _assert_realization_matching(result, [0], [0])


def test_matcher_process_identical_mse_tie_breaking():
    """Test tie-breaking behaviour when multiple candidates have identical MSE values.

    This test verifies that when candidates have identical MSE costs, the algorithm
    processes them in a consistent order (based on the order they appear in the array).
    """
    clustered_cube = _create_uniform_cube([0.0, 100.0])
    candidate_cube = _create_uniform_cube([50.0, 50.0, 50.0])

    plugin = RealizationToClusterMatcher()
    result = plugin.process(clustered_cube, candidate_cube)

    # All candidates are identical, so any assignment is valid
    cluster_indices, realization_indices = result
    assert len(cluster_indices) == 2
    assert len(realization_indices) == 2

    # Verify determinism - running again should give same result
    result2 = plugin.process(clustered_cube, candidate_cube)
    _assert_realization_matching(result, result2[0], result2[1])


def test_matcher_process_metadata_and_datatype_preservation():
    """Test that the process method returns valid index arrays.

    This test verifies that the process method returns tuple of indices
    with correct types and values.
    """
    clustered_cube = _create_uniform_cube([10.0, 20.0])
    candidate_cube = _create_uniform_cube([20.1, 10.1])

    plugin = RealizationToClusterMatcher()
    result = plugin.process(clustered_cube, candidate_cube)

    # Verify result is a tuple
    assert isinstance(result, tuple), "Result should be a tuple"
    assert len(result) == 2, "Result should have 2 elements"

    cluster_indices, realization_indices = result

    # Verify indices are lists
    assert isinstance(cluster_indices, list), "cluster_indices should be a list"
    assert isinstance(realization_indices, list), "realization_indices should be a list"

    # Verify same length
    assert len(cluster_indices) == len(realization_indices), (
        "cluster_indices and realization_indices should have same length"
    )

    # Verify all indices are integers
    for idx in cluster_indices:
        assert isinstance(idx, (int, np.integer)), "Cluster indices should be integers"
    for idx in realization_indices:
        assert isinstance(idx, (int, np.integer)), (
            "Realization indices should be integers"
        )


# Tests for RealizationToClusterMatcher with 4D cubes


def test_matcher_process_4d_basic():
    """Test the process method with 4D cubes (realization, forecast_period, y, x)."""
    # Create 4D cubes with distinct patterns
    clustered_cube = _create_4d_realization_cube(
        n_realizations=2, forecast_periods=list(range(3)), y_dim=4, x_dim=4, seed=100
    )
    candidate_cube = _create_4d_realization_cube(
        n_realizations=2, forecast_periods=list(range(3)), y_dim=4, x_dim=4, seed=200
    )

    plugin = RealizationToClusterMatcher()
    result = plugin.process(clustered_cube, candidate_cube)

    # Verify result is indices tuple
    cluster_indices, realization_indices = result
    assert len(cluster_indices) == 2
    assert len(realization_indices) == 2
    # Both clusters should be matched
    np.testing.assert_array_equal(cluster_indices, [0, 1])


def test_matcher_process_4d_multiple_candidates():
    """Test 4D matching with more candidates than clusters.

    This test demonstrates the greedy MSE-based matching algorithm with 4D cubes.
    The algorithm calculates MSE by taking the mean over spatial dimensions (y, x)
    for each forecast period, then sums across forecast periods. This gives more
    weight to patterns that are consistently similar across multiple time points.
    """
    # Create clustered cube with 2 distinct clusters across 2 forecast periods
    # Cluster 0: Low values (around 10.0)
    # Cluster 1: High values (around 100.0)
    cluster_0_data = np.full((2, 3, 3), 10.0, dtype=np.float32)
    cluster_1_data = np.full((2, 3, 3), 100.0, dtype=np.float32)

    clustered_cubes = iris.cube.CubeList()
    for fp_idx in range(2):
        data = np.array(
            [cluster_0_data[fp_idx], cluster_1_data[fp_idx]], dtype=np.float32
        )
        cube = set_up_variable_cube(
            data,
            name="air_temperature",
            units="K",
            spatial_grid="equalarea",
            realizations=np.arange(2),
            time=datetime(2017, 1, 10, 3 + fp_idx),
            frt=datetime(2017, 1, 10, 3),
        )
        clustered_cubes.append(cube)
    clustered_cube = clustered_cubes.merge_cube()
    # Promote the forecast_reference_time to a dimension coordinate.
    promote_aux_coord_to_dim_coord(clustered_cube, "forecast_period")
    clustered_cube.transpose([1, 0, 2, 3])  # realization, forecast_period, y, x

    # Create candidate cube with 4 realizations that match clusters with varying quality
    # Candidate 0: Close to cluster 0 (MSE will be low)
    # Candidate 1: Close to cluster 1 (MSE will be low)
    # Candidate 2: Between clusters (MSE will be moderate for both)
    # Candidate 3: Far from both (MSE will be high for both)
    candidate_0_data = np.full((2, 3, 3), 11.0, dtype=np.float32)  # Close to cluster 0
    candidate_1_data = np.full((2, 3, 3), 99.0, dtype=np.float32)  # Close to cluster 1
    candidate_2_data = np.full((2, 3, 3), 55.0, dtype=np.float32)  # Middle
    candidate_3_data = np.full((2, 3, 3), 200.0, dtype=np.float32)  # Far from both

    candidate_cubes = iris.cube.CubeList()
    for fp_idx in range(2):
        data = np.array(
            [
                candidate_0_data[fp_idx],
                candidate_1_data[fp_idx],
                candidate_2_data[fp_idx],
                candidate_3_data[fp_idx],
            ],
            dtype=np.float32,
        )
        cube = set_up_variable_cube(
            data,
            name="air_temperature",
            units="K",
            spatial_grid="equalarea",
            realizations=np.arange(4),
            time=datetime(2017, 1, 10, 3 + fp_idx),
            frt=datetime(2017, 1, 10, 3),
        )
        candidate_cubes.append(cube)
    candidate_cube = candidate_cubes.merge_cube()
    promote_aux_coord_to_dim_coord(candidate_cube, "forecast_period")
    candidate_cube.transpose([1, 0, 2, 3])  # realization, forecast_period, y, x

    plugin = RealizationToClusterMatcher()
    result = plugin.process(clustered_cube, candidate_cube)

    # Expected: cluster 0 -> candidate 0, cluster 1 -> candidate 1
    # (based on MSE calculations described above)
    cluster_indices, realization_indices = result

    # Verify result has 2 clusters matched
    assert len(cluster_indices) == 2
    assert len(realization_indices) == 2
    np.testing.assert_array_equal(cluster_indices, [0, 1])

    # Expected: cluster 0 matches candidate 0 (11.0 vs 10.0), cluster 1 matches
    # candidate 1 (99.0 vs 100.0)
    np.testing.assert_array_equal(realization_indices, [0, 1])


@pytest.mark.parametrize(
    "clustered_cube_func, candidate_cube_func, expect_exception",
    [
        # Mismatched forecast periods: should raise ValueError
        (
            lambda: _create_4d_realization_cube(n_realizations=2, forecast_periods=list(range(2)), y_dim=3, x_dim=3, seed=10),
            lambda: _create_4d_realization_cube(n_realizations=2, forecast_periods=list(range(3)), y_dim=3, x_dim=3, seed=20),
            ValueError,
        ),
        # 4D vs 3D: should raise ValueError
        (
            lambda: _create_uniform_cube([10.0, 20.0]),
            lambda: _create_4d_realization_cube(n_realizations=2, forecast_periods=list(range(2)), y_dim=3, x_dim=3, seed=50),
            ValueError,
        ),
        # 4D consistent results: should not raise, should be consistent
        (
            lambda: _create_4d_realization_cube(n_realizations=3, forecast_periods=list(range(2)), y_dim=5, x_dim=5, seed=123),
            lambda: _create_4d_realization_cube(n_realizations=3, forecast_periods=list(range(2)), y_dim=5, x_dim=5, seed=456),
            None,
        ),
    ]
)
def test_matcher_4d_cases(clustered_cube_func, candidate_cube_func, expect_exception):
    clustered_cube = clustered_cube_func()
    candidate_cube = candidate_cube_func()
    plugin = RealizationToClusterMatcher()
    if expect_exception:
        with pytest.raises(expect_exception):
            plugin.process(clustered_cube, candidate_cube)
    else:
        result1 = plugin.process(clustered_cube, candidate_cube)
        result2 = plugin.process(clustered_cube, candidate_cube)
        _assert_realization_matching(result1, result2[0], result2[1])


def test_matcher_process_4d_metadata_preservation():
    """Test that the process method returns valid indices for 4D cubes."""
    clustered_cube = _create_4d_realization_cube(
        n_realizations=2, forecast_periods=list(range(2)), y_dim=3, x_dim=3, seed=99
    )
    candidate_cube = _create_4d_realization_cube(
        n_realizations=2, forecast_periods=list(range(2)), y_dim=3, x_dim=3, seed=88
    )

    plugin = RealizationToClusterMatcher()
    result = plugin.process(clustered_cube, candidate_cube)

    # Verify result is a tuple with indices
    assert isinstance(result, tuple), "Result should be a tuple"
    cluster_indices, realization_indices = result

    # Verify indices are lists of integers
    assert isinstance(cluster_indices, list)
    assert isinstance(realization_indices, list)
    assert len(cluster_indices) == len(realization_indices)

    # Both clusters should be matched for 2-cluster case
    assert len(cluster_indices) == 2
    np.testing.assert_array_equal(cluster_indices, [0, 1])


# Tests for RealizationClusterAndMatch


def _create_target_grid_cube(spatial_shape=(3, 3)):
    """Create a target grid cube for RealizationClusterAndMatch tests.

    Args:
        spatial_shape: Shape of spatial dimensions (y, x).

    Returns:
        Target grid cube.
    """
    target_data = np.zeros(spatial_shape, dtype=np.float32)
    target_cube = set_up_variable_cube(
        target_data,
        name="target_grid",
        units="1",
        spatial_grid="equalarea",
    )
    target_cube.rename("target_grid")
    return target_cube


def test_clusterandmatch_init_basic():
    """Test that RealizationClusterAndMatch initialization sets all attributes
    correctly."""
    hierarchy = {
        "primary_input": "model_a",
        "secondary_inputs": {"model_b": [0, 6], "model_c": [12, 18]},
    }
    model_id_attr = "model_id"
    target_grid_name = "target_grid"
    clustering_method = "KMedoids"

    plugin = RealizationClusterAndMatch(
        hierarchy=hierarchy,
        model_id_attr=model_id_attr,
        clustering_method=clustering_method,
        target_grid_name=target_grid_name,
        n_clusters=3,
        random_state=42,
    )

    assert plugin.hierarchy == hierarchy
    assert plugin.model_id_attr == model_id_attr
    assert plugin.target_grid_name == target_grid_name
    assert plugin.clustering_method == clustering_method
    assert plugin.kwargs == {"n_clusters": 3, "random_state": 42}


def test_clusterandmatch_init_invalid_clustering_method():
    """Test that non-KMedoids clustering methods raise NotImplementedError."""
    hierarchy = {
        "primary_input": "model_a",
        "secondary_inputs": {"model_b": [0, 6]},
    }

    with pytest.raises(
        NotImplementedError,
        match=(
            "Currently only KMedoids clustering is supported for the clustering and "
            "matching"
        ),
    ):
        RealizationClusterAndMatch(
            hierarchy=hierarchy,
            model_id_attr="model_id",
            clustering_method="KMeans",
            target_grid_name="target_grid",
            n_clusters=3,
        )


def test_clusterandmatch_process_basic():
    """Test basic end-to-end processing with simple hierarchy."""
    pytest.importorskip("kmedoids")

    # Create cubes with distinct, verifiable values
    cubes = iris.cube.CubeList()
    spatial_shape = (5, 5)

    # Primary input with 6 realizations, value 100
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[0, 6, 12, 18],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            base_value=100.0,
            model_id="primary_model",
            merge=False
        )
    )

    # Secondary input 1 for fp=[0, 6] with value 200
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[0, 6],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            base_value=200.0,
            model_id="secondary_model_1",
            merge=False
        )
    )

    # Secondary input 2 for fp=[12, 18] with value 300
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=4,
            forecast_periods=[12, 18],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            base_value=300.0,
            model_id="secondary_model_2",
            merge=False
        )
    )

    # Target grid
    cubes.append(_create_target_grid_cube())

    hierarchy = {
        "primary_input": "primary_model",
        "secondary_inputs": {
            "secondary_model_1": [0, 6],
            "secondary_model_2": [12, 18],
        },
    }

    plugin = RealizationClusterAndMatch(
        hierarchy=hierarchy,
        model_id_attr="model_id",
        clustering_method="KMedoids",
        target_grid_name="target_grid",
        n_clusters=3,
        random_state=42,
    )

    result = plugin.process(cubes)

    # Check basic structure
    assert isinstance(result, iris.cube.Cube)
    assert result.name() == "air_temperature"
    assert result.units == "K"

    # Check that we have the expected number of clusters
    n_clusters = len(result.coord("realization").points)
    assert n_clusters == 3

    # Check that all forecast periods are present (in seconds)
    forecast_periods = result.coord("forecast_period").points
    assert len(forecast_periods) == 4
    np.testing.assert_array_equal(forecast_periods, [0, 6 * 3600, 12 * 3600, 18 * 3600])

    # Check cluster_sources attribute validates correctly
    # Note: We're not checking specific sources here, just ensuring the attribute exists
    # and can be validated via the helper function
    _assert_cluster_sources_attribute(result)

    # Check that model_id attribute is removed from result
    assert "model_id" not in result.attributes

    # Check data values to verify correct inputs were used
    # fp=0,6 should use secondary_model_1 (highest precedence, values ~200)
    fp_0_data = result.extract(iris.Constraint(forecast_period=0)).data
    assert np.allclose(fp_0_data, 200.0, atol=5.0), "fp=0 should use secondary_model_1"

    fp_6_data = result.extract(iris.Constraint(forecast_period=6 * 3600)).data
    assert np.allclose(fp_6_data, 206.0, atol=5.0), "fp=6 should use secondary_model_1"

    # fp=12,18 should use secondary_model_2 (highest precedence, values ~300)
    fp_12_data = result.extract(iris.Constraint(forecast_period=12 * 3600)).data
    assert np.allclose(fp_12_data, 312.0, atol=5.0), (
        "fp=12 should use secondary_model_2"
    )

    fp_18_data = result.extract(iris.Constraint(forecast_period=18 * 3600)).data
    assert np.allclose(fp_18_data, 318.0, atol=5.0), (
        "fp=18 should use secondary_model_2"
    )

    # Check cluster_sources attribute exists and has correct structure
    expected_sources = {}
    for cluster_idx in range(3):
        # All clusters at fp=0,6 should use secondary_model_1
        expected_sources[(cluster_idx, 0)] = "secondary_model_1"
        expected_sources[(cluster_idx, 1)] = "secondary_model_1"
        # fp=12,18 may use secondary_model_2 or fall back to primary_model
        # (secondary_model_2 has only 4 realizations for 3 clusters)
    _assert_cluster_sources_attribute(result, expected_sources)


def test_clusterandmatch_cluster_primary_input():
    """Test the clustering of primary input returns correct shape and structure.

    This test verifies that the primary input is correctly clustered by checking
    that the output has the expected number of clusters and forecast periods.
    Uses distinct values to verify primary input is used when no secondary inputs
    cover a forecast period.
    """
    pytest.importorskip("kmedoids")

    cubes = iris.cube.CubeList()
    spatial_shape = (5, 5)

    # Primary input with 6 realizations, value 100
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[0, 6, 12],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="primary_model",
            base_value=100.0,
            merge=False
        )
    )

    # Secondary input only for fp=[0, 6], value 200
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[0, 6],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="secondary_model_1",
            base_value=200.0,
            merge=False
        )
    )

    # Target grid
    cubes.append(_create_target_grid_cube())

    hierarchy = {
        "primary_input": "primary_model",
        "secondary_inputs": {"secondary_model_1": [0, 6]},
    }

    plugin = RealizationClusterAndMatch(
        hierarchy=hierarchy,
        model_id_attr="model_id",
        clustering_method="KMedoids",
        target_grid_name="target_grid",
        n_clusters=3,
        random_state=42,
    )

    result = plugin.process(cubes)

    # Check that clustering produced the expected number of clusters
    assert len(result.coord("realization").points) == 3

    # Check that forecast periods are present (in seconds)
    assert result.coord("forecast_period") is not None
    np.testing.assert_array_equal(
        result.coord("forecast_period").points, [0, 6 * 3600, 12 * 3600]
    )

    # Check basic structure
    assert isinstance(result, iris.cube.Cube)
    assert result.name() == "air_temperature"

    # Check data values: fp=12 should use primary (value ~112),
    # others secondary model 1 (~200, ~206)
    fp_0_data = result.extract(iris.Constraint(forecast_period=0)).data
    assert np.allclose(fp_0_data, 200.0, atol=5.0), "fp=0 should use secondary_model_1"

    fp_6_data = result.extract(iris.Constraint(forecast_period=6 * 3600)).data
    assert np.allclose(fp_6_data, 206.0, atol=5.0), "fp=6 should use secondary_model_1"

    fp_12_data = result.extract(iris.Constraint(forecast_period=12 * 3600)).data
    assert np.allclose(fp_12_data, 112.0, atol=5.0), (
        "fp=12 should use clustered primary_model"
    )


def test_clusterandmatch_precedence_order():
    """Test that secondary inputs are processed in correct precedence order.

    First in hierarchy = highest precedence, should overwrite earlier inputs.
    """
    pytest.importorskip("kmedoids")

    # Create cubes with distinct patterns
    cubes = iris.cube.CubeList()
    spatial_shape = (5, 5)

    # Primary input with 6 realizations at fp=0
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[0],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="primary_model",
            base_value=100.0,
            merge=False
        )
    )

    # Secondary input 1 (higher precedence) with 6 realizations, distinct value
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[0],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="secondary_model_1",
            base_value=200.0,
            merge=False
        )
    )

    # Secondary input 2 (lower precedence) with 6 realizations, distinct value
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[0],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="secondary_model_2",
            base_value=300.0,
            merge=False
        )
    )

    # Target grid
    cubes.append(_create_target_grid_cube())

    # Hierarchy: secondary_model_1 listed first (higher precedence),
    # secondary_model_2 listed second (lower precedence)
    hierarchy = {
        "primary_input": "primary_model",
        "secondary_inputs": {
            "secondary_model_1": [0],
            "secondary_model_2": [0],
        },
    }

    plugin = RealizationClusterAndMatch(
        hierarchy=hierarchy,
        model_id_attr="model_id",
        clustering_method="KMedoids",
        target_grid_name="target_grid",
        n_clusters=3,
        random_state=42,
    )

    result = plugin.process(cubes)

    # Result should contain data from secondary_model_1 (highest precedence,
    # listed first). All values should be close to 200.0
    assert np.allclose(result.data, 200.0, atol=1.0), (
        "Result should use highest precedence input (secondary_model_1)"
    )

    # Check cluster_sources attribute reflects precedence order
    expected_sources = {}
    for cluster_idx in range(3):
        expected_sources[(cluster_idx, 0)] = "secondary_model_1"
    _assert_cluster_sources_attribute(result, expected_sources)


def test_clusterandmatch_overlapping_forecast_periods():
    """Test handling of overlapping forecast periods with different precedence."""
    pytest.importorskip("kmedoids")

    cubes = iris.cube.CubeList()
    spatial_shape = (5, 5)

    # Primary input with 6 realizations, value 100
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[0, 6, 12],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="primary_model",
            base_value=100.0,
            merge=False
        )
    )

    # Secondary input 1 for fp=[0, 6], value 200 (higher precedence - listed first)
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[0, 6],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="secondary_model_1",
            base_value=200.0,
            merge=False
        )
    )

    # Secondary input 2 for fp=[6, 12], value 300 (lower precedence, overlaps at fp=6)
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[6, 12],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="secondary_model_2",
            base_value=300.0,
            merge=False
        )
    )

    # Target grid
    cubes.append(_create_target_grid_cube())

    hierarchy = {
        "primary_input": "primary_model",
        "secondary_inputs": {
            "secondary_model_1": [0, 6],
            "secondary_model_2": [6, 12],
        },
    }

    plugin = RealizationClusterAndMatch(
        hierarchy=hierarchy,
        model_id_attr="model_id",
        clustering_method="KMedoids",
        target_grid_name="target_grid",
        n_clusters=3,
        random_state=42,
    )

    result = plugin.process(cubes)

    # Check all forecast periods present
    forecast_periods = result.coord("forecast_period").points
    assert len(forecast_periods) == 3
    np.testing.assert_array_equal(forecast_periods, [0, 6 * 3600, 12 * 3600])

    # At fp=6, secondary_model_1 should have overwritten secondary_model_2
    # (secondary_model_1 is listed first, so has higher precedence)
    assert result.coord("realization").shape[0] == 3

    # Check data values to verify precedence
    fp_0_data = result.extract(iris.Constraint(forecast_period=0)).data
    assert np.allclose(fp_0_data, 200.0, atol=5.0), "fp=0 should use secondary_model_1"

    fp_6_data = result.extract(iris.Constraint(forecast_period=6 * 3600)).data
    assert np.allclose(fp_6_data, 206.0, atol=5.0), (
        "fp=6 should use secondary_model_1 (higher precedence), not secondary_model_2"
    )

    fp_12_data = result.extract(iris.Constraint(forecast_period=12 * 3600)).data
    assert np.allclose(fp_12_data, 312.0, atol=5.0), (
        "fp=12 should use secondary_model_2"
    )


def test_clusterandmatch_single_secondary_input():
    """Test with only one secondary input."""
    pytest.importorskip("kmedoids")

    cubes = iris.cube.CubeList()
    spatial_shape = (5, 5)

    # Primary input with value 100
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[0, 6, 12],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="primary_model",
            base_value=100.0,
            merge=False
        )
    )

    # Secondary input only for fp=[0, 6], value 200
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[0, 6],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="secondary_model_1",
            base_value=200.0,
            merge=False
        )
    )

    # Target grid
    cubes.append(_create_target_grid_cube())

    hierarchy = {
        "primary_input": "primary_model",
        "secondary_inputs": {"secondary_model_1": [0, 6]},
    }

    plugin = RealizationClusterAndMatch(
        hierarchy=hierarchy,
        model_id_attr="model_id",
        clustering_method="KMedoids",
        target_grid_name="target_grid",
        n_clusters=3,
        random_state=42,
    )

    result = plugin.process(cubes)

    # Check structure
    assert isinstance(result, iris.cube.Cube)
    assert len(result.coord("realization").points) == 3

    # Should have all 3 forecast periods (0, 6 from secondary, 12 from primary)
    forecast_periods = result.coord("forecast_period").points
    assert len(forecast_periods) == 3
    np.testing.assert_array_equal(forecast_periods, [0, 6 * 3600, 12 * 3600])

    # Check data values: fp=[0,6] should use secondary (~200),
    # fp=12 should use primary (~112)
    fp_0_data = result.extract(iris.Constraint(forecast_period=0)).data
    assert np.allclose(fp_0_data, 200.0, atol=5.0), "fp=0 should use secondary_model_1"

    fp_6_data = result.extract(iris.Constraint(forecast_period=6 * 3600)).data
    assert np.allclose(fp_6_data, 206.0, atol=5.0), "fp=6 should use secondary_model_1"

    fp_12_data = result.extract(iris.Constraint(forecast_period=12 * 3600)).data
    assert np.allclose(fp_12_data, 112.0, atol=5.0), (
        "fp=12 should use clustered primary_model"
    )

    # Check cluster_sources attribute
    expected_sources = {}
    for cluster_idx in range(3):
        expected_sources[(cluster_idx, 0)] = "secondary_model_1"  # fp=0
        expected_sources[(cluster_idx, 1)] = "secondary_model_1"  # fp=6
        expected_sources[(cluster_idx, 2)] = "primary_model"  # fp=12
    _assert_cluster_sources_attribute(result, expected_sources)


def test_clusterandmatch_categorise_full_realizations():
    """Test categorisation with all inputs having >= n_clusters realizations.

    This test verifies that when all secondary inputs have enough realizations
    to fill all clusters, they are correctly used in the output.
    """
    pytest.importorskip("kmedoids")

    cubes = iris.cube.CubeList()
    spatial_shape = (5, 5)

    # Primary input with value 100
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[0, 6],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="primary_model",
            base_value=100.0,
            merge=False
        )
    )

    # Secondary input 1 for fp=0, value 200 (6 realizations >= 3 clusters)
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[0],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="secondary_model_1",
            base_value=200.0,
            merge=False
        )
    )

    # Secondary input 2 for fp=6, value 300 (4 realizations >= 3 clusters)
    # Note: Using 294.0 as base so that 294.0 + 6 (fp_hours) = 300.0
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=4,
            forecast_periods=[6],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="secondary_model_2",
            base_value=294.0,
            merge=False
        )
    )

    # Target grid
    cubes.append(_create_target_grid_cube())

    hierarchy = {
        "primary_input": "primary_model",
        "secondary_inputs": {
            "secondary_model_1": [0],
            "secondary_model_2": [6],
        },
    }

    plugin = RealizationClusterAndMatch(
        hierarchy=hierarchy,
        model_id_attr="model_id",
        clustering_method="KMedoids",
        target_grid_name="target_grid",
        n_clusters=3,
        random_state=42,
    )

    result = plugin.process(cubes)

    # Both forecast periods should be present and use secondary inputs
    forecast_periods = result.coord("forecast_period").points
    assert len(forecast_periods) == 2
    np.testing.assert_array_equal(forecast_periods, [0, 6 * 3600])

    # Should have correct number of clusters
    assert len(result.coord("realization").points) == 3

    # Check data: both fps should use their respective secondary inputs
    fp_0_data = result.extract(iris.Constraint(forecast_period=0)).data
    assert np.allclose(fp_0_data, 200.0, atol=5.0), (
        "fp=0 should use secondary_model_1 (full realizations)"
    )

    fp_6_data = result.extract(iris.Constraint(forecast_period=6 * 3600)).data
    assert np.allclose(fp_6_data, 300.0, atol=5.0), (
        "fp=6 should use secondary_model_2 (full realizations)"
    )


def test_clusterandmatch_categorise_partial_realizations():
    """Test categorisation with all inputs having < n_clusters realizations.

    This test verifies that when all secondary inputs have fewer realizations
    than clusters, they selectively replace clusters rather than filling all.
    The result should merge partial secondary inputs with the clustered primary.
    """
    pytest.importorskip("kmedoids")

    cubes = iris.cube.CubeList()
    spatial_shape = (5, 5)

    # Primary input with value 100
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[0, 6, 12],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="primary_model",
            base_value=100.0,
            merge=False)
    )

    # Secondary input 1 for fp=0, value 200 (2 realizations < 3 clusters)
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=2,
            forecast_periods=[0],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="secondary_model_1",
            base_value=200.0,
            merge=False
        )
    )

    # Secondary input 2 for fp=6, value 300 (2 realizations < 3 clusters)
    # Note: Using 294.0 as base so that 294.0 + 6 (fp_hours) = 300.0
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=2,
            forecast_periods=[6],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="secondary_model_2",
            base_value=294.0,
            merge=False
        )
    )

    # Target grid
    cubes.append(_create_target_grid_cube())

    hierarchy = {
        "primary_input": "primary_model",
        "secondary_inputs": {
            "secondary_model_1": [0],
            "secondary_model_2": [6],
        },
    }

    plugin = RealizationClusterAndMatch(
        hierarchy=hierarchy,
        model_id_attr="model_id",
        clustering_method="KMedoids",
        target_grid_name="target_grid",
        n_clusters=3,
        random_state=42,
    )

    result = plugin.process(cubes)

    # All forecast periods should be present
    # fp=0 uses secondary_model_1 (2 realizations) merged with primary
    # fp=6 uses secondary_model_2 (2 realizations) merged with primary
    # fp=12 uses primary only
    forecast_periods = result.coord("forecast_period").points
    assert len(forecast_periods) == 3
    np.testing.assert_array_equal(forecast_periods, [0, 6 * 3600, 12 * 3600])

    # Should still have 3 clusters
    assert len(result.coord("realization").points) == 3

    # Check data: With random_state=42 and uniform input data,
    # the matching is deterministic.
    # Clusters [0, 1] to be replaced by the 2 secondary realizations, leaving
    # cluster [2] with primary data.
    # Since we have 3 clusters total, 2/3 of realizations get secondary,
    # 1/3 gets primary.

    fp_0_data = result.extract(iris.Constraint(forecast_period=0)).data
    # Check cluster 0 and 1 get secondary (~200), cluster 2 gets primary (~100)
    assert np.allclose(fp_0_data[0], 200.0, atol=5.0), (
        "Realization 0 should use secondary_model_1"
    )
    assert np.allclose(fp_0_data[1], 200.0, atol=5.0), (
        "Realization 1 should use secondary_model_1"
    )
    assert np.allclose(fp_0_data[2], 100.0, atol=5.0), (
        "Realization 2 should use clustered primary_model"
    )

    fp_6_data = result.extract(iris.Constraint(forecast_period=6 * 3600)).data
    assert np.allclose(fp_6_data[0], 300.0, atol=5.0), (
        "Realization 0 should use secondary_model_2"
    )
    assert np.allclose(fp_6_data[1], 300.0, atol=5.0), (
        "Realization 1 should use secondary_model_2"
    )
    assert np.allclose(fp_6_data[2], 106.0, atol=5.0), (
        "Realization 2 should use clustered primary_model"
    )

    # fp=12 should be entirely from primary
    fp_12_data = result.extract(iris.Constraint(forecast_period=12 * 3600)).data
    assert np.allclose(fp_12_data, 112.0, atol=5.0), (
        "fp=12 should use clustered primary_model only"
    )


def test_clusterandmatch_multiple_partial_secondary_same_forecast_period():
    """Test categorisation with multiple partial secondary inputs for same forecast period.

    This test verifies that when multiple secondary inputs have partial realizations
    (< n_clusters) for the same forecast period, they are both processed via MSE
    matching. Since partial inputs don't know in advance which clusters they'll match
    to, both are processed in reverse precedence order (lowest first).

    With appropriate data values, the MSE matching will assign each partial input to
    different clusters based on which clusters they match best. This demonstrates that
    multiple partial inputs can successfully contribute different clusters to the same
    forecast period.
    """
    pytest.importorskip("kmedoids")

    cubes = iris.cube.CubeList()
    spatial_shape = (5, 5)

    # Primary input with 6 realizations at fp=0 with varying values
    # Create distinct realizations that will cluster into 3 groups
    # Group 1 (realizations 0, 1): ~90
    # Group 2 (realizations 2, 3): ~100
    # Group 3 (realizations 4, 5): ~110
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[0],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="primary_model",
            realization_values=[90.0, 91.0, 100.0, 101.0, 110.0, 111.0],
            merge=False
        )
    )

    # Secondary input 1 for fp=0, value 90 (2 realizations < 3 clusters)
    # This has highest precedence (listed first)
    # Value chosen to match the low cluster (~90)
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=2,
            forecast_periods=[0],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="secondary_model_1",
            base_value=90.0,
            merge=False
        )
    )

    # Secondary input 2 for fp=0, value 110 (1 realization < 3 clusters)
    # This has lower precedence (listed second) but will still be processed
    # Value chosen to match the high cluster (~110)
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=1,
            forecast_periods=[0],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="secondary_model_2",
            base_value=110.0,
            merge=False
        )
    )

    # Target grid
    cubes.append(_create_target_grid_cube())

    hierarchy = {
        "primary_input": "primary_model",
        "secondary_inputs": {
            "secondary_model_1": [0],
            "secondary_model_2": [0],
        },
    }

    plugin = RealizationClusterAndMatch(
        hierarchy=hierarchy,
        model_id_attr="model_id",
        clustering_method="KMedoids",
        target_grid_name="target_grid",
        n_clusters=3,
        random_state=42,
    )

    result = plugin.process(cubes)

    # Should have 1 forecast period
    forecast_periods = result.coord("forecast_period").points
    assert len(forecast_periods) == 1
    np.testing.assert_array_equal(forecast_periods, [0])

    # Should have 3 clusters
    assert len(result.coord("realization").points) == 3

    # Check data: Both secondary inputs are processed (lowest precedence first).
    # With the chosen data values (primary clusters ~90, ~100, ~110;
    # secondary_1=90, secondary_2=110), the MSE matching algorithm will match
    # them to different clusters.
    #
    # Expected result with random_state=42:
    # - Clustering creates 3 clusters with medoid values ~90, ~100, ~110
    # - secondary_model_2 processes first, matches to cluster with value ~110
    # - secondary_model_1 processes second, matches to clusters with values ~90 and ~100
    # - Result: All 3 clusters filled by secondary inputs
    fp_0_data = result.extract(iris.Constraint(forecast_period=0)).data

    # Verify the final result after precedence resolution
    unique_values = np.unique(fp_0_data.round())

    # Should see values from both secondary inputs
    assert any(np.isclose(unique_values, 90.0, atol=5.0)), (
        "Should contain values from secondary_model_1 (~90)"
    )
    assert any(np.isclose(unique_values, 110.0, atol=5.0)), (
        "Should contain values from secondary_model_2 (~110)"
    )

    # Count how many clusters got each value
    n_from_secondary_1 = np.sum(np.isclose(fp_0_data, 90.0, atol=5.0))
    n_from_secondary_2 = np.sum(np.isclose(fp_0_data, 110.0, atol=5.0))

    # Should have 2 clusters from secondary_model_1 (its 2 realizations)
    # and 1 cluster from secondary_model_2 (its 1 realization)
    assert n_from_secondary_1 == 2 * spatial_shape[0] * spatial_shape[1], (
        "2 clusters should use secondary_model_1"
    )
    assert n_from_secondary_2 == 1 * spatial_shape[0] * spatial_shape[1], (
        "1 cluster should use secondary_model_2"
    )


def test_clusterandmatch_categorise_mixed_realizations():
    """Test categorisation with mix of full and partial realizations.

    This test verifies correct handling when some secondary inputs have enough
    realizations to fill all clusters while others have fewer.
    """
    pytest.importorskip("kmedoids")

    cubes = iris.cube.CubeList()
    spatial_shape = (5, 5)

    # Primary input with value 100
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[0, 6, 12],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="primary_model",
            base_value=100.0,
            merge=False
        )
    )

    # Secondary input 1 for fp=0, value 200 (6 realizations >= 3 clusters, full)
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[0],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="secondary_model_1",
            base_value=200.0,
            merge=False
        )
    )

    # Secondary input 2 for fp=6, value 300 (2 realizations < 3 clusters, partial)
    # Note: Using 294.0 as base so that 294.0 + 6 (fp_hours) = 300.0
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=2,
            forecast_periods=[6],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="secondary_model_2",
            base_value=294.0,
            merge=False
        )
    )

    # Target grid
    cubes.append(_create_target_grid_cube())

    hierarchy = {
        "primary_input": "primary_model",
        "secondary_inputs": {
            "secondary_model_1": [0],
            "secondary_model_2": [6],
        },
    }

    plugin = RealizationClusterAndMatch(
        hierarchy=hierarchy,
        model_id_attr="model_id",
        clustering_method="KMedoids",
        target_grid_name="target_grid",
        n_clusters=3,
        random_state=42,
    )

    result = plugin.process(cubes)

    # All forecast periods should be present
    # fp=0 uses secondary_model_1 (full 3 clusters)
    # fp=6 uses secondary_model_2 (2 realizations) merged with primary
    # fp=12 uses primary only
    forecast_periods = result.coord("forecast_period").points
    assert len(forecast_periods) == 3
    np.testing.assert_array_equal(forecast_periods, [0, 6 * 3600, 12 * 3600])

    # Should have 3 clusters
    assert len(result.coord("realization").points) == 3

    # Check data:
    # fp=0 should be entirely from secondary_model_1
    # (full realizations, 6 >= 3 clusters)
    fp_0_data = result.extract(iris.Constraint(forecast_period=0)).data
    assert np.allclose(fp_0_data, 200.0, atol=5.0), (
        "fp=0 should use secondary_model_1 (full realizations)"
    )

    # fp=6: With random_state=42, clusters [0, 1] get replaced by secondary_model_2,
    # cluster [2] keeps primary data
    fp_6_data = result.extract(iris.Constraint(forecast_period=6 * 3600)).data
    assert np.allclose(fp_6_data[0], 300.0, atol=5.0), (
        "Realization 0 should use secondary_model_2"
    )
    assert np.allclose(fp_6_data[1], 300.0, atol=5.0), (
        "Realization 1 should use secondary_model_2"
    )
    assert np.allclose(fp_6_data[2], 106.0, atol=5.0), (
        "Realization 2 should use clustered primary_model"
    )

    # fp=12 should be entirely from primary
    fp_12_data = result.extract(iris.Constraint(forecast_period=12 * 3600)).data
    assert np.allclose(fp_12_data, 112.0, atol=5.0), (
        "fp=12 should use clustered primary_model only"
    )


def test_clusterandmatch_regrid_for_clustering_false():
    """Test that regrid_for_clustering=False works without requiring target grid.

    This test verifies that when regrid_for_clustering is set to False:
    1. No target_grid_name is required
    2. Clustering and matching proceed without regridding
    3. Results are produced successfully with original grid data
    """
    pytest.importorskip("kmedoids")

    cubes = iris.cube.CubeList()
    spatial_shape = (5, 5)

    # Primary input with 6 realizations, value 100
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[0, 6, 12],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="primary_model",
            base_value=100.0,
            merge=False
        )
    )

    # Secondary input 1 for fp=[0, 6], value 200
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=6,
            forecast_periods=[0, 6],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="secondary_model_1",
            base_value=200.0,
            merge=False
        )
    )

    # Secondary input 2 for fp=[12], value 300
    cubes.extend(
        _create_4d_realization_cube(
            n_realizations=4,
            forecast_periods=[12],
            y_dim=spatial_shape[0],
            x_dim=spatial_shape[1],
            model_id="secondary_model_2",
            base_value=300.0,
            merge=False
        )
    )

    # Note: No target grid cube added

    hierarchy = {
        "primary_input": "primary_model",
        "secondary_inputs": {
            "secondary_model_1": [0, 6],
            "secondary_model_2": [12],
        },
    }

    # Create plugin without target_grid_name and with regrid_for_clustering=False
    plugin = RealizationClusterAndMatch(
        hierarchy=hierarchy,
        model_id_attr="model_id",
        clustering_method="KMedoids",
        regrid_for_clustering=False,
        n_clusters=3,
        random_state=42,
    )

    result = plugin.process(cubes)

    # Check basic structure
    assert isinstance(result, iris.cube.Cube)
    assert result.name() == "air_temperature"
    assert result.units == "K"

    # Check that we have the expected number of clusters
    n_clusters = len(result.coord("realization").points)
    assert n_clusters == 3

    # Check that all forecast periods are present (in seconds)
    forecast_periods = result.coord("forecast_period").points
    assert len(forecast_periods) == 3
    np.testing.assert_array_equal(forecast_periods, [0, 6 * 3600, 12 * 3600])

    # Check cluster_sources attribute exists even when regrid_for_clustering=False
    expected_sources = {}
    for cluster_idx in range(3):
        expected_sources[(cluster_idx, 0)] = "secondary_model_1"
        expected_sources[(cluster_idx, 1)] = "secondary_model_1"
    _assert_cluster_sources_attribute(result, expected_sources)

    # Check that model_id attribute is removed from result
    assert "model_id" not in result.attributes

    # Check data values to verify correct inputs were used
    # fp=0,6 should use secondary_model_1 (highest precedence, values ~200)
    fp_0_data = result.extract(iris.Constraint(forecast_period=0)).data
    assert np.allclose(fp_0_data, 200.0, atol=5.0), "fp=0 should use secondary_model_1"

    fp_6_data = result.extract(iris.Constraint(forecast_period=6 * 3600)).data
    assert np.allclose(fp_6_data, 206.0, atol=5.0), "fp=6 should use secondary_model_1"

    # fp=12 should use secondary_model_2 (highest precedence, values ~300)
    fp_12_data = result.extract(iris.Constraint(forecast_period=12 * 3600)).data
    assert np.allclose(fp_12_data, 312.0, atol=5.0), (
        "fp=12 should use secondary_model_2"
    )


@pytest.mark.parametrize(
    "regrid_for_clustering,target_grid_name,should_raise",
    [
        (False, None, False),
        (True, None, True),
    ],
)
def test_clusterandmatch_regrid_for_clustering_and_target_grid_name(
        regrid_for_clustering, target_grid_name, should_raise):
    """Test the interaction between regrid_for_clustering and target_grid_name.

    If regrid_for_clustering is True, target_grid_name must be provided,
    otherwise ValueError is raised.
    If regrid_for_clustering is False, target_grid_name can be omitted (None).
    """
    pytest.importorskip("kmedoids")

    hierarchy = {
        "primary_input": "model_a",
        "secondary_inputs": {"model_b": [0, 6]},
    }

    if should_raise:
        with pytest.raises(
            ValueError,
            match="target_grid_name must be provided when regrid_for_clustering is True",
        ):
            RealizationClusterAndMatch(
                hierarchy=hierarchy,
                model_id_attr="model_id",
                clustering_method="KMedoids",
                target_grid_name=target_grid_name,
                regrid_for_clustering=regrid_for_clustering,
                n_clusters=3,
            )
    else:
        plugin = RealizationClusterAndMatch(
            hierarchy=hierarchy,
            model_id_attr="model_id",
            clustering_method="KMedoids",
            target_grid_name=target_grid_name,
            regrid_for_clustering=regrid_for_clustering,
            n_clusters=3,
            random_state=42,
        )
        assert plugin.regrid_for_clustering is False
        assert plugin.target_grid_name is None
