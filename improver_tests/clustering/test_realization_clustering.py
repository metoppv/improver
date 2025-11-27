# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the improver.clustering.realization_clustering module."""

from datetime import datetime

import iris
import numpy as np
import pytest

from improver.clustering.realization_clustering import (
    ClusterAndMatch,
    RealizationClustering,
    RealizationToClusterMatcher,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


def _create_realization_cube(shape=(5, 10, 10), seed=42):
    """Create a cube with realization dimension for testing.

    Args:
        shape: Tuple of (n_realizations, dim1, dim2, ...). Must have at least 2 dimensions.
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
    n_realizations=5, n_forecast_periods=3, y_dim=8, x_dim=8, seed=42
):
    """Create a 4D cube with realization and forecast_period dimensions.

    Args:
        n_realizations: Number of realizations.
        n_forecast_periods: Number of forecast periods.
        y_dim: Size of y dimension.
        x_dim: Size of x dimension.
        seed: Random seed for reproducibility.

    Returns:
        A 4D cube with shape (n_realizations, n_forecast_periods, y_dim, x_dim).
    """
    np.random.seed(seed)

    cubes = iris.cube.CubeList()
    for fp_hours in range(n_forecast_periods):
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
        cubes.append(cube)

    # Merge the cubes along forecast_period dimension
    merged_cube = cubes.merge_cube()

    # Transpose so realization is the leading dimension
    # Order after merge is typically: time, realization, y, x
    # We want: realization, time (forecast_period), y, x
    merged_cube.transpose([1, 0, 2, 3])

    return merged_cube


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
    result, expected_indices, candidate_data, n_expected=None, atol=0.1
):
    """Assert that result matches expected candidate indices.

    Helper function to standardize verification logic in matcher tests.

    Args:
        result: Result cube from matcher.
        expected_indices: List of candidate indices that should appear in result.
        candidate_data: Original candidate data array.
        n_expected: Expected number of realizations (defaults to len(expected_indices)).
        atol: Absolute tolerance for value comparison.
    """
    if n_expected is None:
        n_expected = len(expected_indices)

    # Check number of realizations
    assert result.coord("realization").shape[0] == n_expected

    # Check realization numbering
    np.testing.assert_array_equal(
        result.coord("realization").points, np.arange(n_expected)
    )

    # Check data matching
    for i, candidate_idx in enumerate(expected_indices):
        np.testing.assert_allclose(
            result.data[i], candidate_data[candidate_idx], atol=atol
        )


@pytest.mark.parametrize(
    "clustering_method,n_clusters,random_state",
    [
        ("KMeans", 3, 42),  # Basic KMeans
        ("KMeans", 2, 42),  # KMeans with 2 clusters
        ("AgglomerativeClustering", 3, None),  # AgglomerativeClustering
    ],
)
def test_process_basic_clustering(clustering_method, n_clusters, random_state):
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


def test_process_kmeans_cluster_centers():
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
def test_process_arbitrary_dimensions_to_2d_conversion(
    shape, expected_n_features, use_4d
):
    """Test that cubes with arbitrary dimensions are correctly converted to 2D.

    This implicitly tests the convert_to_2d method through the process method.
    The method flattens all dimensions except the leading (realization) dimension.
    This test includes both 3D cubes and 4D cubes (with forecast_period dimension)
    to verify that the convert_to_2d method handles arrays with any number of dimensions.
    """
    if use_4d:
        # For 4D: shape = (n_realizations, n_forecast_periods, y_dim, x_dim)
        n_realizations, n_forecast_periods, y_dim, x_dim = shape
        cube = _create_4d_realization_cube(
            n_realizations=n_realizations,
            n_forecast_periods=n_forecast_periods,
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


def test_process_distinct_clusters():
    """Test clustering with clearly separable realizations."""
    cube = _create_clusterable_realization_cube()

    plugin = RealizationClustering("KMeans", n_clusters=2, random_state=42)
    result = plugin.process(cube)

    assert hasattr(result, "labels_")
    assert len(result.labels_) == 6
    # Should form 2 distinct clusters
    assert len(np.unique(result.labels_)) == 2


def test_process_dbscan():
    """Test the RealizationClustering.process method with DBSCAN clustering."""
    cube = _create_realization_cube(shape=(8, 10, 10))

    plugin = RealizationClustering("DBSCAN", eps=50.0, min_samples=2)
    result = plugin.process(cube)

    assert hasattr(result, "labels_")
    assert len(result.labels_) == 8


def test_process_kmedoids():
    """Test the RealizationClustering.process method with KMedoids."""
    pytest.importorskip("kmedoids")
    cube = _create_clusterable_realization_cube()

    plugin = RealizationClustering("KMedoids", n_clusters=2, random_state=42)
    result = plugin.process(cube)

    assert hasattr(result, "labels_")
    assert hasattr(result, "medoid_indices_")
    assert len(result.labels_) == 6
    assert len(result.medoid_indices_) == 2


def test_process_invalid_clustering_method():
    """Test that an error is raised for unsupported clustering methods."""
    cube = _create_realization_cube()

    plugin = RealizationClustering("NonExistentMethod", n_clusters=3)

    with pytest.raises(
        ValueError, match="The clustering method provided is not supported"
    ):
        plugin.process(cube)


def test_process_wrong_leading_dimension():
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
        match="The leading dimension of the input cube must be the realization dimension",
    ):
        plugin.process(cube)


def test_process_different_n_clusters():
    """Test that different n_clusters values produce different results."""
    cube = _create_realization_cube(shape=(8, 10, 10))

    plugin2 = RealizationClustering("KMeans", n_clusters=2, random_state=42)
    result2 = plugin2.process(cube)

    plugin4 = RealizationClustering("KMeans", n_clusters=4, random_state=42)
    result4 = plugin4.process(cube)

    assert len(np.unique(result2.labels_)) == 2
    assert len(np.unique(result4.labels_)) == 4


def test_process_preserves_cube():
    """Test that process does not modify the input cube."""
    cube = _create_realization_cube(shape=(5, 10, 10))
    original_data = cube.data.copy()

    plugin = RealizationClustering("KMeans", n_clusters=3, random_state=42)
    plugin.process(cube)

    np.testing.assert_array_equal(cube.data, original_data)


def test_process_with_single_realization():
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

    # Shape should match input
    assert result.shape == candidate_cube.shape
    # Expected: candidate[1] -> cluster[0], candidate[0] -> cluster[1]
    _assert_realization_matching(result, [1, 0], candidate_cube.data, atol=0.01)


def test_matcher_process_multiple_realizations():
    """Test process with more realizations than clusters."""
    clustered_cube = _create_uniform_cube([0.0, 50.0, 100.0])
    candidate_cube = _create_uniform_cube([75.0, 99.0, 1.0, 2.0])

    plugin = RealizationToClusterMatcher()
    result = plugin.process(clustered_cube, candidate_cube)

    # Verify the specific data matching:
    # Based on the actual algorithm behaviour:
    # The algorithm processes candidates in cost order [2, 1, 3, 0] and assigns:
    # - Candidate 2 (1.0) to cluster 0 with MSE=1.0
    # - Candidate 1 (99.0) to cluster 2 with MSE=1.0
    # - Candidate 3 (2.0) tries cluster 0 but has MSE=4.0 > 1.0, so rejected
    # - Candidate 0 (75.0) to cluster 1 with MSE=625.0
    # Result[0] should be candidate[2] (1.0) - matched to cluster 0
    # Result[1] should be candidate[0] (75.0) - matched to cluster 1
    # Result[2] should be candidate[1] (99.0) - matched to cluster 2
    _assert_realization_matching(result, [2, 0, 1], candidate_cube.data, n_expected=3)


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

    # Should complete without error
    assert result.coord("realization").shape[0] == 2
    # Verify each result matches one candidate (with NaN handling)
    for i in range(2):
        matched = any(
            np.allclose(result.data[i], candidate_data[j], equal_nan=True)
            for j in range(2)
        )
        assert matched, f"Result realization {i} doesn't match any candidate"


def test_matcher_process_identical_patterns():
    """Test matching with identical patterns to verify MSE calculation.

    This test verifies that the mean_squared_error calculation works correctly
    by using identical patterns that should match perfectly.
    """
    clustered_cube = _create_uniform_cube([0.0, 10.0])
    candidate_cube = _create_uniform_cube([0.0, 10.0])

    plugin = RealizationToClusterMatcher()
    result = plugin.process(clustered_cube, candidate_cube)

    # With identical patterns, each candidate should match to a cluster
    assert result.coord("realization").shape[0] == 2
    np.testing.assert_array_equal(result.coord("realization").points, [0, 1])
    # Each realization in the result should be one of the candidate patterns
    assert np.allclose(result.data[0], candidate_cube.data[0]) or np.allclose(
        result.data[0], candidate_cube.data[1]
    )
    assert np.allclose(result.data[1], candidate_cube.data[0]) or np.allclose(
        result.data[1], candidate_cube.data[1]
    )


def test_matcher_process_greedy_assignment():
    """Test that greedy assignment works correctly through the process method.

    This test verifies that the choose_clusters greedy algorithm prioritises
    realizations with higher MSE cost (more distinctive preferences).
    """
    clustered_cube = _create_uniform_cube([0.0, 100.0])
    candidate_cube = _create_uniform_cube([0.01, 50.0])

    plugin = RealizationToClusterMatcher()
    result = plugin.process(clustered_cube, candidate_cube)

    # Expected: candidate[0] -> cluster[0], candidate[1] -> cluster[1]
    _assert_realization_matching(result, [0, 1], candidate_cube.data, atol=0.1)


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

    # All clusters should be used - result should have 3 realizations
    assert result.coord("realization").shape[0] == 3
    np.testing.assert_array_equal(result.coord("realization").points, [0, 1, 2])
    # Verify each result matches exactly one candidate uniquely
    matched_indices = set()
    for i in range(3):
        for j in range(3):
            if (
                np.allclose(result.data[i], candidate_cube.data[j])
                and j not in matched_indices
            ):
                matched_indices.add(j)
                break
    assert len(matched_indices) == 3, "Not all candidates were matched uniquely"


def test_matcher_choose_clusters_consistent_results():
    """Test that process produces consistent results for the same input."""
    clustered_cube = _create_realization_cube(shape=(3, 5, 5), seed=100)
    candidate_cube = _create_realization_cube(shape=(3, 5, 5), seed=200)

    plugin = RealizationToClusterMatcher()
    result1 = plugin.process(clustered_cube, candidate_cube)
    result2 = plugin.process(clustered_cube, candidate_cube)

    # Results should be identical for same input
    np.testing.assert_array_equal(result1.data, result2.data)
    np.testing.assert_array_equal(
        result1.coord("realization").points, result2.coord("realization").points
    )


def test_matcher_process_single_cluster_multiple_candidates():
    """Test with a single cluster and multiple similar candidates.

    This test verifies that when multiple candidates compete for a single cluster,
    the algorithm selects the candidate with the lowest MSE.
    """
    clustered_cube = _create_uniform_cube([50.0])
    candidate_cube = _create_uniform_cube([51.0, 52.0, 53.0])

    plugin = RealizationToClusterMatcher()
    result = plugin.process(clustered_cube, candidate_cube)

    # Should select candidate 0 (closest to cluster)
    _assert_realization_matching(result, [0], candidate_cube.data, n_expected=1)


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
    assert result.coord("realization").shape[0] == 2
    assert np.allclose(result.data[0], 50.0)
    assert np.allclose(result.data[1], 50.0)

    # Verify determinism - running again should give same result
    result2 = plugin.process(clustered_cube, candidate_cube)
    np.testing.assert_array_equal(result.data, result2.data)


def test_matcher_process_metadata_and_datatype_preservation():
    """Test that metadata and data type are preserved in the output.

    This test verifies that the process method preserves the cube's metadata
    (name, units, coordinates) and data type from the candidate cube.
    """
    clustered_cube = _create_uniform_cube([10.0, 20.0])
    candidate_cube = _create_uniform_cube([20.1, 10.1])

    # Store original metadata for comparison
    original_name = candidate_cube.name()
    original_units = candidate_cube.units
    original_dtype = candidate_cube.data.dtype
    original_spatial_coords = [
        coord.name()
        for coord in candidate_cube.coords()
        if coord.name() != "realization"
    ]

    plugin = RealizationToClusterMatcher()
    result = plugin.process(clustered_cube, candidate_cube)

    # Verify metadata preservation
    assert result.name() == original_name, "Cube name should be preserved"
    assert result.units == original_units, "Cube units should be preserved"

    # Verify data type preservation
    assert (
        result.data.dtype == original_dtype
    ), f"Data type should be preserved: expected {original_dtype}, got {result.data.dtype}"

    # Verify spatial coordinates are preserved
    result_spatial_coords = [
        coord.name() for coord in result.coords() if coord.name() != "realization"
    ]
    assert set(result_spatial_coords) == set(
        original_spatial_coords
    ), "Spatial coordinates should be preserved"

    # Verify realization coordinate exists and is correctly numbered
    assert (
        result.coord("realization") is not None
    ), "Realization coordinate should exist"
    (
        np.testing.assert_array_equal(result.coord("realization").points, [0, 1]),
        "Realization points should be renumbered sequentially",
    )

    # Verify shape is correct
    assert result.shape == candidate_cube.shape, "Output shape should match input shape"


# Tests for RealizationToClusterMatcher with 4D cubes


def test_matcher_process_4d_basic():
    """Test the process method with 4D cubes (realization, forecast_period, y, x)."""
    # Create 4D cubes with distinct patterns
    clustered_cube = _create_4d_realization_cube(
        n_realizations=2, n_forecast_periods=3, y_dim=4, x_dim=4, seed=100
    )
    candidate_cube = _create_4d_realization_cube(
        n_realizations=2, n_forecast_periods=3, y_dim=4, x_dim=4, seed=200
    )

    plugin = RealizationToClusterMatcher()
    result = plugin.process(clustered_cube, candidate_cube)

    # Verify output shape
    assert result.ndim == 4, "Result should be 4D"
    assert result.shape == candidate_cube.shape, "Shape should match input"

    # Verify coordinates
    assert result.coord("realization").shape[0] == 2
    np.testing.assert_array_equal(result.coord("realization").points, [0, 1])
    assert result.coord("forecast_period") is not None
    np.testing.assert_array_equal(
        result.coord("forecast_period").points,
        candidate_cube.coord("forecast_period").points,
    )


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
    candidate_cube.transpose([1, 0, 2, 3])  # realization, forecast_period, y, x

    plugin = RealizationToClusterMatcher()
    result = plugin.process(clustered_cube, candidate_cube)

    # Expected MSE calculations (mean over spatial, sum over forecast_period):
    # For each candidate to each cluster, MSE = sum_over_fp(mean_over_spatial((cluster - candidate)^2))
    # Candidate 0 (11.0) to cluster 0 (10.0): sum_fp(mean((10-11)^2)) = 2 * 1.0 = 2.0
    # Candidate 0 (11.0) to cluster 1 (100.0): sum_fp(mean((100-11)^2)) = 2 * 7921.0 = 15842.0
    # Candidate 1 (99.0) to cluster 0 (10.0): sum_fp(mean((10-99)^2)) = 2 * 7921.0 = 15842.0
    # Candidate 1 (99.0) to cluster 1 (100.0): sum_fp(mean((100-99)^2)) = 2 * 1.0 = 2.0
    # Candidate 2 (55.0) to cluster 0 (10.0): sum_fp(mean((10-55)^2)) = 2 * 2025.0 = 4050.0
    # Candidate 2 (55.0) to cluster 1 (100.0): sum_fp(mean((100-55)^2)) = 2 * 2025.0 = 4050.0
    # Candidate 3 (200.0) to cluster 0 (10.0): sum_fp(mean((10-200)^2)) = 2 * 36100.0 = 72200.0
    # Candidate 3 (200.0) to cluster 1 (100.0): sum_fp(mean((100-200)^2)) = 2 * 10000.0 = 20000.0

    # Greedy algorithm assigns based on MSE cost (sum of differences from minimum):
    # Candidate 0: cost = (15842.0 - 2.0) = 15840.0 (prefers cluster 0)
    # Candidate 1: cost = (15842.0 - 2.0) = 15840.0 (prefers cluster 1)
    # Candidate 2: cost = (4050.0 - 4050.0) = 0.0 (no preference)
    # Candidate 3: cost = (72200.0 - 20000.0) = 52200.0 (prefers cluster 1 but much worse)

    # Processing order (descending cost): [3, 0, 1, 2]
    # Step 1: Candidate 3 assigns to cluster 1 (best match, MSE=20000.0)
    # Step 2: Candidate 0 assigns to cluster 0 (best match, MSE=2.0)
    # Step 3: Candidate 1 wants cluster 1 but it's taken by candidate 3 with MSE=20000.0
    #         Candidate 1 has MSE=2.0 < 20000.0, so it overwrites: cluster 1 gets candidate 1
    # Result: cluster 0 -> candidate 0, cluster 1 -> candidate 1

    # Verify result has 2 realizations (matching 2 clusters)
    assert result.coord("realization").shape[0] == 2
    np.testing.assert_array_equal(result.coord("realization").points, [0, 1])

    # Verify the specific assignments:
    # Result[0] should be candidate[0] (11.0) - matched to cluster 0
    # Result[1] should be candidate[1] (99.0) - matched to cluster 1
    np.testing.assert_allclose(result.data[0], candidate_cube.data[0], rtol=1e-5)
    np.testing.assert_allclose(result.data[1], candidate_cube.data[1], rtol=1e-5)

    # Verify forecast_period coordinate is preserved
    np.testing.assert_array_equal(
        result.coord("forecast_period").points,
        candidate_cube.coord("forecast_period").points,
    )


def test_matcher_process_4d_mismatched_forecast_periods():
    """Test that mismatched forecast_period coordinates raise an error."""
    clustered_cube = _create_4d_realization_cube(
        n_realizations=2, n_forecast_periods=2, y_dim=3, x_dim=3, seed=10
    )

    # Create candidate with different forecast periods
    candidate_cube = _create_4d_realization_cube(
        n_realizations=2, n_forecast_periods=3, y_dim=3, x_dim=3, seed=20
    )

    plugin = RealizationToClusterMatcher()

    with pytest.raises(ValueError, match="Forecast period coords must match"):
        plugin.process(clustered_cube, candidate_cube)


def test_matcher_process_4d_vs_3d_dimension_mismatch():
    """Test that mixing 3D and 4D cubes raises an error."""
    clustered_cube_3d = _create_uniform_cube([10.0, 20.0])
    candidate_cube_4d = _create_4d_realization_cube(
        n_realizations=2, n_forecast_periods=2, y_dim=3, x_dim=3, seed=50
    )

    plugin = RealizationToClusterMatcher()

    with pytest.raises(
        ValueError, match="Both cubes must have the same number of dimensions"
    ):
        plugin.process(clustered_cube_3d, candidate_cube_4d)


def test_matcher_process_4d_consistent_results():
    """Test that 4D matching produces consistent results for repeated calls."""
    clustered_cube = _create_4d_realization_cube(
        n_realizations=3, n_forecast_periods=2, y_dim=5, x_dim=5, seed=123
    )
    candidate_cube = _create_4d_realization_cube(
        n_realizations=3, n_forecast_periods=2, y_dim=5, x_dim=5, seed=456
    )

    plugin = RealizationToClusterMatcher()
    result1 = plugin.process(clustered_cube, candidate_cube)
    result2 = plugin.process(clustered_cube, candidate_cube)

    # Results should be identical
    np.testing.assert_array_equal(result1.data, result2.data)
    np.testing.assert_array_equal(
        result1.coord("realization").points, result2.coord("realization").points
    )


def test_matcher_process_4d_metadata_preservation():
    """Test that metadata is preserved for 4D cubes."""
    clustered_cube = _create_4d_realization_cube(
        n_realizations=2, n_forecast_periods=2, y_dim=3, x_dim=3, seed=99
    )
    candidate_cube = _create_4d_realization_cube(
        n_realizations=2, n_forecast_periods=2, y_dim=3, x_dim=3, seed=88
    )

    # Store original metadata
    original_name = candidate_cube.name()
    original_units = candidate_cube.units
    original_dtype = candidate_cube.data.dtype
    original_fp_points = candidate_cube.coord("forecast_period").points.copy()

    plugin = RealizationToClusterMatcher()
    result = plugin.process(clustered_cube, candidate_cube)

    # Verify metadata preservation
    assert result.name() == original_name
    assert result.units == original_units
    assert result.data.dtype == original_dtype
    np.testing.assert_array_equal(
        result.coord("forecast_period").points, original_fp_points
    )

    # Verify all necessary coordinates exist
    assert result.coord("realization") is not None
    assert result.coord("forecast_period") is not None
    assert result.coord("projection_y_coordinate") is not None
    assert result.coord("projection_x_coordinate") is not None


# Tests for ClusterAndMatch


def _create_test_cubes_for_cluster_and_match(
    n_realizations_primary=6,
    n_realizations_secondary1=6,
    n_realizations_secondary2=4,
    forecast_periods_primary=None,
    forecast_periods_secondary1=None,
    forecast_periods_secondary2=None,
    spatial_shape=(5, 5),
    seed=42,
):
    """Create test cubes for ClusterAndMatch tests.

    Args:
        n_realizations_primary: Number of realizations for primary input.
        n_realizations_secondary1: Number of realizations for first secondary input.
        n_realizations_secondary2: Number of realizations for second secondary input.
        forecast_periods_primary: Forecast periods (hours) for primary input.
        forecast_periods_secondary1: Forecast periods for first secondary input.
        forecast_periods_secondary2: Forecast periods for second secondary input.
        spatial_shape: Shape of spatial dimensions (y, x).
        seed: Random seed for reproducibility.

    Returns:
        CubeList containing primary, secondary, and target grid cubes.
    """
    if forecast_periods_primary is None:
        forecast_periods_primary = [0, 6, 12, 18]
    if forecast_periods_secondary1 is None:
        forecast_periods_secondary1 = [0, 6]
    if forecast_periods_secondary2 is None:
        forecast_periods_secondary2 = [12, 18]

    np.random.seed(seed)
    cubes = iris.cube.CubeList()

    # Create primary input cubes
    for fp_hours in forecast_periods_primary:
        data = np.random.randn(n_realizations_primary, *spatial_shape).astype(
            np.float32
        )
        cube = set_up_variable_cube(
            data,
            name="air_temperature",
            units="K",
            spatial_grid="equalarea",
            realizations=np.arange(n_realizations_primary),
            time=datetime(2024, 1, 1, 0),
            frt=datetime(2024, 1, 1, 0),
        )
        # Add forecast_period coordinate
        forecast_period = iris.coords.DimCoord(
            np.array([fp_hours], dtype=np.int32),
            standard_name="forecast_period",
            units="hours",
        )
        cube.add_aux_coord(forecast_period)
        cube.attributes["model_id"] = "primary_model"
        cubes.append(cube)

    # Create first secondary input cubes
    for fp_hours in forecast_periods_secondary1:
        data = np.random.randn(n_realizations_secondary1, *spatial_shape).astype(
            np.float32
        )
        cube = set_up_variable_cube(
            data,
            name="air_temperature",
            units="K",
            spatial_grid="equalarea",
            realizations=np.arange(n_realizations_secondary1),
            time=datetime(2024, 1, 1, 0),
            frt=datetime(2024, 1, 1, 0),
        )
        forecast_period = iris.coords.DimCoord(
            np.array([fp_hours], dtype=np.int32),
            standard_name="forecast_period",
            units="hours",
        )
        cube.add_aux_coord(forecast_period)
        cube.attributes["model_id"] = "secondary_model_1"
        cubes.append(cube)

    # Create second secondary input cubes
    for fp_hours in forecast_periods_secondary2:
        data = np.random.randn(n_realizations_secondary2, *spatial_shape).astype(
            np.float32
        )
        cube = set_up_variable_cube(
            data,
            name="air_temperature",
            units="K",
            spatial_grid="equalarea",
            realizations=np.arange(n_realizations_secondary2),
            time=datetime(2024, 1, 1, 0),
            frt=datetime(2024, 1, 1, 0),
        )
        forecast_period = iris.coords.DimCoord(
            np.array([fp_hours], dtype=np.int32),
            standard_name="forecast_period",
            units="hours",
        )
        cube.add_aux_coord(forecast_period)
        cube.attributes["model_id"] = "secondary_model_2"
        cubes.append(cube)

    # Create target grid cube (smaller spatial dimensions for faster clustering)
    target_data = np.zeros((3, 3), dtype=np.float32)
    target_cube = set_up_variable_cube(
        target_data,
        name="target_grid",
        units="1",
        spatial_grid="equalarea",
    )
    target_cube.rename("target_grid")
    cubes.append(target_cube)

    return cubes


def test_clusterandmatch_init_basic():
    """Test that ClusterAndMatch initialization sets all attributes correctly."""
    hierarchy = {
        "primary_input": "model_a",
        "secondary_inputs": {"model_b": [0, 6], "model_c": [12, 18]},
    }
    model_id_attr = "model_id"
    target_grid_name = "target_grid"
    clustering_method = "KMedoids"

    plugin = ClusterAndMatch(
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
        match="Currently only KMedoids clustering is supported for the clustering and matching",
    ):
        ClusterAndMatch(
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

    # Primary input with 6 realizations
    for fp_hours in [0, 6, 12, 18]:
        data_primary = np.full((6, *spatial_shape), 100.0 + fp_hours, dtype=np.float32)
        cube = set_up_variable_cube(
            data_primary,
            name="air_temperature",
            units="K",
            spatial_grid="equalarea",
            realizations=np.arange(6),
            time=datetime(2024, 1, 1, 0),
            frt=datetime(2024, 1, 1, 0),
        )
        forecast_period = iris.coords.DimCoord(
            np.array([fp_hours], dtype=np.int32),
            standard_name="forecast_period",
            units="hours",
        )
        cube.add_aux_coord(forecast_period)
        cube.attributes["model_id"] = "primary_model"
        cubes.append(cube)

    # Secondary input 1 for fp=[0, 6] with value 200
    for fp_hours in [0, 6]:
        data = np.full((6, *spatial_shape), 200.0 + fp_hours, dtype=np.float32)
        cube = set_up_variable_cube(
            data,
            name="air_temperature",
            units="K",
            spatial_grid="equalarea",
            realizations=np.arange(6),
            time=datetime(2024, 1, 1, 0),
            frt=datetime(2024, 1, 1, 0),
        )
        forecast_period = iris.coords.DimCoord(
            np.array([fp_hours], dtype=np.int32),
            standard_name="forecast_period",
            units="hours",
        )
        cube.add_aux_coord(forecast_period)
        cube.attributes["model_id"] = "secondary_model_1"
        cubes.append(cube)

    # Secondary input 2 for fp=[12, 18] with value 300
    for fp_hours in [12, 18]:
        data = np.full((4, *spatial_shape), 300.0 + fp_hours, dtype=np.float32)
        cube = set_up_variable_cube(
            data,
            name="air_temperature",
            units="K",
            spatial_grid="equalarea",
            realizations=np.arange(4),
            time=datetime(2024, 1, 1, 0),
            frt=datetime(2024, 1, 1, 0),
        )
        forecast_period = iris.coords.DimCoord(
            np.array([fp_hours], dtype=np.int32),
            standard_name="forecast_period",
            units="hours",
        )
        cube.add_aux_coord(forecast_period)
        cube.attributes["model_id"] = "secondary_model_2"
        cubes.append(cube)

    # Target grid
    target_data = np.zeros((3, 3), dtype=np.float32)
    target_cube = set_up_variable_cube(
        target_data,
        name="target_grid",
        units="1",
        spatial_grid="equalarea",
    )
    target_cube.rename("target_grid")
    cubes.append(target_cube)

    hierarchy = {
        "primary_input": "primary_model",
        "secondary_inputs": {
            "secondary_model_1": [0, 6],
            "secondary_model_2": [12, 18],
        },
    }

    plugin = ClusterAndMatch(
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

    # Check that all forecast periods are present
    forecast_periods = result.coord("forecast_period").points
    assert len(forecast_periods) == 4
    np.testing.assert_array_equal(forecast_periods, [0, 6, 12, 18])

    # Check that model_id attribute is removed from result
    assert "model_id" not in result.attributes

    # Check data values to verify correct inputs were used
    # fp=0,6 should use secondary_model_1 (highest precedence, values ~200)
    fp_0_data = result.extract(iris.Constraint(forecast_period=0)).data
    assert np.allclose(fp_0_data, 200.0, atol=5.0), "fp=0 should use secondary_model_1"

    fp_6_data = result.extract(iris.Constraint(forecast_period=6)).data
    assert np.allclose(fp_6_data, 206.0, atol=5.0), "fp=6 should use secondary_model_1"

    # fp=12,18 should use secondary_model_2 (highest precedence, values ~300)
    fp_12_data = result.extract(iris.Constraint(forecast_period=12)).data
    assert np.allclose(
        fp_12_data, 312.0, atol=5.0
    ), "fp=12 should use secondary_model_2"

    fp_18_data = result.extract(iris.Constraint(forecast_period=18)).data
    assert np.allclose(
        fp_18_data, 318.0, atol=5.0
    ), "fp=18 should use secondary_model_2"


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
    for fp_hours in [0, 6, 12]:
        data = np.full((6, *spatial_shape), 100.0 + fp_hours, dtype=np.float32)
        cube = set_up_variable_cube(
            data,
            name="air_temperature",
            units="K",
            spatial_grid="equalarea",
            realizations=np.arange(6),
            time=datetime(2024, 1, 1, 0),
            frt=datetime(2024, 1, 1, 0),
        )
        forecast_period = iris.coords.DimCoord(
            np.array([fp_hours], dtype=np.int32),
            standard_name="forecast_period",
            units="hours",
        )
        cube.add_aux_coord(forecast_period)
        cube.attributes["model_id"] = "primary_model"
        cubes.append(cube)

    # Secondary input only for fp=[0, 6], value 200
    for fp_hours in [0, 6]:
        data = np.full((6, *spatial_shape), 200.0 + fp_hours, dtype=np.float32)
        cube = set_up_variable_cube(
            data,
            name="air_temperature",
            units="K",
            spatial_grid="equalarea",
            realizations=np.arange(6),
            time=datetime(2024, 1, 1, 0),
            frt=datetime(2024, 1, 1, 0),
        )
        forecast_period = iris.coords.DimCoord(
            np.array([fp_hours], dtype=np.int32),
            standard_name="forecast_period",
            units="hours",
        )
        cube.add_aux_coord(forecast_period)
        cube.attributes["model_id"] = "secondary_model_1"
        cubes.append(cube)

    # Target grid
    target_data = np.zeros((3, 3), dtype=np.float32)
    target_cube = set_up_variable_cube(
        target_data,
        name="target_grid",
        units="1",
        spatial_grid="equalarea",
    )
    target_cube.rename("target_grid")
    cubes.append(target_cube)

    hierarchy = {
        "primary_input": "primary_model",
        "secondary_inputs": {"secondary_model_1": [0, 6]},
    }

    plugin = ClusterAndMatch(
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

    # Check that forecast periods are present
    assert result.coord("forecast_period") is not None
    np.testing.assert_array_equal(result.coord("forecast_period").points, [0, 6, 12])

    # Check basic structure
    assert isinstance(result, iris.cube.Cube)
    assert result.name() == "air_temperature"

    # Check data values: fp=12 should use primary (value ~112), others secondary (~200, ~206)
    fp_0_data = result.extract(iris.Constraint(forecast_period=0)).data
    assert np.allclose(fp_0_data, 200.0, atol=5.0), "fp=0 should use secondary_model_1"

    fp_6_data = result.extract(iris.Constraint(forecast_period=6)).data
    assert np.allclose(fp_6_data, 206.0, atol=5.0), "fp=6 should use secondary_model_1"

    fp_12_data = result.extract(iris.Constraint(forecast_period=12)).data
    assert np.allclose(
        fp_12_data, 112.0, atol=5.0
    ), "fp=12 should use clustered primary_model"


def test_clusterandmatch_precedence_order():
    """Test that secondary inputs are processed in correct precedence order.

    Last in hierarchy = highest precedence, should overwrite earlier inputs.
    """
    pytest.importorskip("kmedoids")

    # Create cubes with distinct patterns
    cubes = iris.cube.CubeList()
    spatial_shape = (5, 5)

    # Primary input with 6 realizations at fp=0
    data_primary = np.full((6, *spatial_shape), 100.0, dtype=np.float32)
    cube_primary = set_up_variable_cube(
        data_primary,
        name="air_temperature",
        units="K",
        spatial_grid="equalarea",
        realizations=np.arange(6),
        time=datetime(2024, 1, 1, 0),
        frt=datetime(2024, 1, 1, 0),
    )
    forecast_period = iris.coords.DimCoord(
        np.array([0], dtype=np.int32),
        standard_name="forecast_period",
        units="hours",
    )
    cube_primary.add_aux_coord(forecast_period)
    cube_primary.attributes["model_id"] = "primary_model"
    cubes.append(cube_primary)

    # Secondary input 1 (lower precedence) with 6 realizations, distinct value
    data_secondary1 = np.full((6, *spatial_shape), 200.0, dtype=np.float32)
    cube_secondary1 = set_up_variable_cube(
        data_secondary1,
        name="air_temperature",
        units="K",
        spatial_grid="equalarea",
        realizations=np.arange(6),
        time=datetime(2024, 1, 1, 0),
        frt=datetime(2024, 1, 1, 0),
    )
    cube_secondary1.add_aux_coord(forecast_period.copy())
    cube_secondary1.attributes["model_id"] = "secondary_model_1"
    cubes.append(cube_secondary1)

    # Secondary input 2 (higher precedence) with 6 realizations, distinct value
    data_secondary2 = np.full((6, *spatial_shape), 300.0, dtype=np.float32)
    cube_secondary2 = set_up_variable_cube(
        data_secondary2,
        name="air_temperature",
        units="K",
        spatial_grid="equalarea",
        realizations=np.arange(6),
        time=datetime(2024, 1, 1, 0),
        frt=datetime(2024, 1, 1, 0),
    )
    cube_secondary2.add_aux_coord(forecast_period.copy())
    cube_secondary2.attributes["model_id"] = "secondary_model_2"
    cubes.append(cube_secondary2)

    # Target grid
    target_data = np.zeros((3, 3), dtype=np.float32)
    target_cube = set_up_variable_cube(
        target_data,
        name="target_grid",
        units="1",
        spatial_grid="equalarea",
    )
    target_cube.rename("target_grid")
    cubes.append(target_cube)

    # Hierarchy: secondary_model_1 listed first (higher precedence),
    # secondary_model_2 listed second (lower precedence)
    hierarchy = {
        "primary_input": "primary_model",
        "secondary_inputs": {
            "secondary_model_1": [0],
            "secondary_model_2": [0],
        },
    }

    plugin = ClusterAndMatch(
        hierarchy=hierarchy,
        model_id_attr="model_id",
        clustering_method="KMedoids",
        target_grid_name="target_grid",
        n_clusters=3,
        random_state=42,
    )

    result = plugin.process(cubes)

    # Result should contain data from secondary_model_1 (highest precedence, listed first)
    # All values should be close to 200.0
    assert np.allclose(
        result.data, 200.0, atol=1.0
    ), "Result should use highest precedence input (secondary_model_1)"


def test_clusterandmatch_overlapping_forecast_periods():
    """Test handling of overlapping forecast periods with different precedence."""
    pytest.importorskip("kmedoids")

    cubes = iris.cube.CubeList()
    spatial_shape = (5, 5)

    # Primary input with 6 realizations, value 100
    for fp_hours in [0, 6, 12]:
        data = np.full((6, *spatial_shape), 100.0 + fp_hours, dtype=np.float32)
        cube = set_up_variable_cube(
            data,
            name="air_temperature",
            units="K",
            spatial_grid="equalarea",
            realizations=np.arange(6),
            time=datetime(2024, 1, 1, 0),
            frt=datetime(2024, 1, 1, 0),
        )
        forecast_period = iris.coords.DimCoord(
            np.array([fp_hours], dtype=np.int32),
            standard_name="forecast_period",
            units="hours",
        )
        cube.add_aux_coord(forecast_period)
        cube.attributes["model_id"] = "primary_model"
        cubes.append(cube)

    # Secondary input 1 for fp=[0, 6], value 200 (higher precedence - listed first)
    for fp_hours in [0, 6]:
        data = np.full((6, *spatial_shape), 200.0 + fp_hours, dtype=np.float32)
        cube = set_up_variable_cube(
            data,
            name="air_temperature",
            units="K",
            spatial_grid="equalarea",
            realizations=np.arange(6),
            time=datetime(2024, 1, 1, 0),
            frt=datetime(2024, 1, 1, 0),
        )
        forecast_period = iris.coords.DimCoord(
            np.array([fp_hours], dtype=np.int32),
            standard_name="forecast_period",
            units="hours",
        )
        cube.add_aux_coord(forecast_period)
        cube.attributes["model_id"] = "secondary_model_1"
        cubes.append(cube)

    # Secondary input 2 for fp=[6, 12], value 300 (lower precedence, overlaps at fp=6)
    for fp_hours in [6, 12]:
        data = np.full((6, *spatial_shape), 300.0 + fp_hours, dtype=np.float32)
        cube = set_up_variable_cube(
            data,
            name="air_temperature",
            units="K",
            spatial_grid="equalarea",
            realizations=np.arange(6),
            time=datetime(2024, 1, 1, 0),
            frt=datetime(2024, 1, 1, 0),
        )
        forecast_period = iris.coords.DimCoord(
            np.array([fp_hours], dtype=np.int32),
            standard_name="forecast_period",
            units="hours",
        )
        cube.add_aux_coord(forecast_period)
        cube.attributes["model_id"] = "secondary_model_2"
        cubes.append(cube)

    # Target grid
    target_data = np.zeros((3, 3), dtype=np.float32)
    target_cube = set_up_variable_cube(
        target_data,
        name="target_grid",
        units="1",
        spatial_grid="equalarea",
    )
    target_cube.rename("target_grid")
    cubes.append(target_cube)

    hierarchy = {
        "primary_input": "primary_model",
        "secondary_inputs": {
            "secondary_model_1": [0, 6],
            "secondary_model_2": [6, 12],
        },
    }

    plugin = ClusterAndMatch(
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
    np.testing.assert_array_equal(forecast_periods, [0, 6, 12])

    # At fp=6, secondary_model_1 should have overwritten secondary_model_2
    # (secondary_model_1 is listed first, so has higher precedence)
    assert result.coord("realization").shape[0] == 3

    # Check data values to verify precedence
    fp_0_data = result.extract(iris.Constraint(forecast_period=0)).data
    assert np.allclose(fp_0_data, 200.0, atol=5.0), "fp=0 should use secondary_model_1"

    fp_6_data = result.extract(iris.Constraint(forecast_period=6)).data
    assert np.allclose(
        fp_6_data, 206.0, atol=5.0
    ), "fp=6 should use secondary_model_1 (higher precedence), not secondary_model_2"

    fp_12_data = result.extract(iris.Constraint(forecast_period=12)).data
    assert np.allclose(
        fp_12_data, 312.0, atol=5.0
    ), "fp=12 should use secondary_model_2"


def test_clusterandmatch_single_secondary_input():
    """Test with only one secondary input."""
    pytest.importorskip("kmedoids")

    cubes = iris.cube.CubeList()
    spatial_shape = (5, 5)

    # Primary input with value 100
    for fp_hours in [0, 6, 12]:
        data = np.full((6, *spatial_shape), 100.0 + fp_hours, dtype=np.float32)
        cube = set_up_variable_cube(
            data,
            name="air_temperature",
            units="K",
            spatial_grid="equalarea",
            realizations=np.arange(6),
            time=datetime(2024, 1, 1, 0),
            frt=datetime(2024, 1, 1, 0),
        )
        forecast_period = iris.coords.DimCoord(
            np.array([fp_hours], dtype=np.int32),
            standard_name="forecast_period",
            units="hours",
        )
        cube.add_aux_coord(forecast_period)
        cube.attributes["model_id"] = "primary_model"
        cubes.append(cube)

    # Secondary input only for fp=[0, 6], value 200
    for fp_hours in [0, 6]:
        data = np.full((6, *spatial_shape), 200.0 + fp_hours, dtype=np.float32)
        cube = set_up_variable_cube(
            data,
            name="air_temperature",
            units="K",
            spatial_grid="equalarea",
            realizations=np.arange(6),
            time=datetime(2024, 1, 1, 0),
            frt=datetime(2024, 1, 1, 0),
        )
        forecast_period = iris.coords.DimCoord(
            np.array([fp_hours], dtype=np.int32),
            standard_name="forecast_period",
            units="hours",
        )
        cube.add_aux_coord(forecast_period)
        cube.attributes["model_id"] = "secondary_model_1"
        cubes.append(cube)

    # Target grid
    target_data = np.zeros((3, 3), dtype=np.float32)
    target_cube = set_up_variable_cube(
        target_data,
        name="target_grid",
        units="1",
        spatial_grid="equalarea",
    )
    target_cube.rename("target_grid")
    cubes.append(target_cube)

    hierarchy = {
        "primary_input": "primary_model",
        "secondary_inputs": {"secondary_model_1": [0, 6]},
    }

    plugin = ClusterAndMatch(
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
    np.testing.assert_array_equal(forecast_periods, [0, 6, 12])

    # Check data values: fp=0,6 should use secondary (~200), fp=12 should use primary (~112)
    fp_0_data = result.extract(iris.Constraint(forecast_period=0)).data
    assert np.allclose(fp_0_data, 200.0, atol=5.0), "fp=0 should use secondary_model_1"

    fp_6_data = result.extract(iris.Constraint(forecast_period=6)).data
    assert np.allclose(fp_6_data, 206.0, atol=5.0), "fp=6 should use secondary_model_1"

    fp_12_data = result.extract(iris.Constraint(forecast_period=12)).data
    assert np.allclose(
        fp_12_data, 112.0, atol=5.0
    ), "fp=12 should use clustered primary_model"


def test_clusterandmatch_categorise_full_realizations():
    """Test categorization with all inputs having >= n_clusters realizations.

    This test verifies that when all secondary inputs have enough realizations
    to fill all clusters, they are correctly used in the output.
    """
    pytest.importorskip("kmedoids")

    cubes = iris.cube.CubeList()
    spatial_shape = (5, 5)

    # Primary input with value 100
    for fp_hours in [0, 6]:
        data = np.full((6, *spatial_shape), 100.0 + fp_hours, dtype=np.float32)
        cube = set_up_variable_cube(
            data,
            name="air_temperature",
            units="K",
            spatial_grid="equalarea",
            realizations=np.arange(6),
            time=datetime(2024, 1, 1, 0),
            frt=datetime(2024, 1, 1, 0),
        )
        forecast_period = iris.coords.DimCoord(
            np.array([fp_hours], dtype=np.int32),
            standard_name="forecast_period",
            units="hours",
        )
        cube.add_aux_coord(forecast_period)
        cube.attributes["model_id"] = "primary_model"
        cubes.append(cube)

    # Secondary input 1 for fp=0, value 200 (6 realizations >= 3 clusters)
    data = np.full((6, *spatial_shape), 200.0, dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        name="air_temperature",
        units="K",
        spatial_grid="equalarea",
        realizations=np.arange(6),
        time=datetime(2024, 1, 1, 0),
        frt=datetime(2024, 1, 1, 0),
    )
    forecast_period = iris.coords.DimCoord(
        np.array([0], dtype=np.int32),
        standard_name="forecast_period",
        units="hours",
    )
    cube.add_aux_coord(forecast_period)
    cube.attributes["model_id"] = "secondary_model_1"
    cubes.append(cube)

    # Secondary input 2 for fp=6, value 300 (4 realizations >= 3 clusters)
    data = np.full((4, *spatial_shape), 300.0, dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        name="air_temperature",
        units="K",
        spatial_grid="equalarea",
        realizations=np.arange(4),
        time=datetime(2024, 1, 1, 0),
        frt=datetime(2024, 1, 1, 0),
    )
    forecast_period = iris.coords.DimCoord(
        np.array([6], dtype=np.int32),
        standard_name="forecast_period",
        units="hours",
    )
    cube.add_aux_coord(forecast_period)
    cube.attributes["model_id"] = "secondary_model_2"
    cubes.append(cube)

    # Target grid
    target_data = np.zeros((3, 3), dtype=np.float32)
    target_cube = set_up_variable_cube(
        target_data,
        name="target_grid",
        units="1",
        spatial_grid="equalarea",
    )
    target_cube.rename("target_grid")
    cubes.append(target_cube)

    hierarchy = {
        "primary_input": "primary_model",
        "secondary_inputs": {
            "secondary_model_1": [0],
            "secondary_model_2": [6],
        },
    }

    plugin = ClusterAndMatch(
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
    np.testing.assert_array_equal(forecast_periods, [0, 6])

    # Should have correct number of clusters
    assert len(result.coord("realization").points) == 3

    # Check data: both fps should use their respective secondary inputs
    fp_0_data = result.extract(iris.Constraint(forecast_period=0)).data
    assert np.allclose(
        fp_0_data, 200.0, atol=5.0
    ), "fp=0 should use secondary_model_1 (full realizations)"

    fp_6_data = result.extract(iris.Constraint(forecast_period=6)).data
    assert np.allclose(
        fp_6_data, 300.0, atol=5.0
    ), "fp=6 should use secondary_model_2 (full realizations)"


def test_clusterandmatch_categorise_partial_realizations():
    """Test categorization with all inputs having < n_clusters realizations.

    This test verifies that when all secondary inputs have fewer realizations
    than clusters, they selectively replace clusters rather than filling all.
    The result should merge partial secondary inputs with the clustered primary.
    """
    pytest.importorskip("kmedoids")

    cubes = iris.cube.CubeList()
    spatial_shape = (5, 5)

    # Primary input with value 100
    for fp_hours in [0, 6, 12]:
        data = np.full((6, *spatial_shape), 100.0 + fp_hours, dtype=np.float32)
        cube = set_up_variable_cube(
            data,
            name="air_temperature",
            units="K",
            spatial_grid="equalarea",
            realizations=np.arange(6),
            time=datetime(2024, 1, 1, 0),
            frt=datetime(2024, 1, 1, 0),
        )
        forecast_period = iris.coords.DimCoord(
            np.array([fp_hours], dtype=np.int32),
            standard_name="forecast_period",
            units="hours",
        )
        cube.add_aux_coord(forecast_period)
        cube.attributes["model_id"] = "primary_model"
        cubes.append(cube)

    # Secondary input 1 for fp=0, value 200 (2 realizations < 3 clusters)
    data = np.full((2, *spatial_shape), 200.0, dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        name="air_temperature",
        units="K",
        spatial_grid="equalarea",
        realizations=np.arange(2),
        time=datetime(2024, 1, 1, 0),
        frt=datetime(2024, 1, 1, 0),
    )
    forecast_period = iris.coords.DimCoord(
        np.array([0], dtype=np.int32),
        standard_name="forecast_period",
        units="hours",
    )
    cube.add_aux_coord(forecast_period)
    cube.attributes["model_id"] = "secondary_model_1"
    cubes.append(cube)

    # Secondary input 2 for fp=6, value 300 (2 realizations < 3 clusters)
    data = np.full((2, *spatial_shape), 300.0, dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        name="air_temperature",
        units="K",
        spatial_grid="equalarea",
        realizations=np.arange(2),
        time=datetime(2024, 1, 1, 0),
        frt=datetime(2024, 1, 1, 0),
    )
    forecast_period = iris.coords.DimCoord(
        np.array([6], dtype=np.int32),
        standard_name="forecast_period",
        units="hours",
    )
    cube.add_aux_coord(forecast_period)
    cube.attributes["model_id"] = "secondary_model_2"
    cubes.append(cube)

    # Target grid
    target_data = np.zeros((3, 3), dtype=np.float32)
    target_cube = set_up_variable_cube(
        target_data,
        name="target_grid",
        units="1",
        spatial_grid="equalarea",
    )
    target_cube.rename("target_grid")
    cubes.append(target_cube)

    hierarchy = {
        "primary_input": "primary_model",
        "secondary_inputs": {
            "secondary_model_1": [0],
            "secondary_model_2": [6],
        },
    }

    plugin = ClusterAndMatch(
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
    np.testing.assert_array_equal(forecast_periods, [0, 6, 12])

    # Should still have 3 clusters
    assert len(result.coord("realization").points) == 3

    # Check data: With random_state=42 and uniform input data, the matching is deterministic.
    # Clusters [0, 1] to be replaced by the 2 secondary realizations, leaving
    # cluster [2] with primary data.
    # Since we have 3 clusters total, 2/3 of realizations get secondary, 1/3 gets primary.

    fp_0_data = result.extract(iris.Constraint(forecast_period=0)).data
    # Check cluster 0 and 1 get secondary (~200), cluster 2 gets primary (~100)
    assert np.allclose(
        fp_0_data[0], 200.0, atol=5.0
    ), "Realization 0 should use secondary_model_1"
    assert np.allclose(
        fp_0_data[1], 200.0, atol=5.0
    ), "Realization 1 should use secondary_model_1"
    assert np.allclose(
        fp_0_data[2], 100.0, atol=5.0
    ), "Realization 2 should use clustered primary_model"

    fp_6_data = result.extract(iris.Constraint(forecast_period=6)).data
    assert np.allclose(
        fp_6_data[0], 300.0, atol=5.0
    ), "Realization 0 should use secondary_model_2"
    assert np.allclose(
        fp_6_data[1], 300.0, atol=5.0
    ), "Realization 1 should use secondary_model_2"
    assert np.allclose(
        fp_6_data[2], 106.0, atol=5.0
    ), "Realization 2 should use clustered primary_model"

    # fp=12 should be entirely from primary
    fp_12_data = result.extract(iris.Constraint(forecast_period=12)).data
    assert np.allclose(
        fp_12_data, 112.0, atol=5.0
    ), "fp=12 should use clustered primary_model only"


def test_clusterandmatch_categorise_mixed_realizations():
    """Test categorization with mix of full and partial realizations.

    This test verifies correct handling when some secondary inputs have enough
    realizations to fill all clusters while others have fewer.
    """
    pytest.importorskip("kmedoids")

    cubes = iris.cube.CubeList()
    spatial_shape = (5, 5)

    # Primary input with value 100
    for fp_hours in [0, 6, 12]:
        data = np.full((6, *spatial_shape), 100.0 + fp_hours, dtype=np.float32)
        cube = set_up_variable_cube(
            data,
            name="air_temperature",
            units="K",
            spatial_grid="equalarea",
            realizations=np.arange(6),
            time=datetime(2024, 1, 1, 0),
            frt=datetime(2024, 1, 1, 0),
        )
        forecast_period = iris.coords.DimCoord(
            np.array([fp_hours], dtype=np.int32),
            standard_name="forecast_period",
            units="hours",
        )
        cube.add_aux_coord(forecast_period)
        cube.attributes["model_id"] = "primary_model"
        cubes.append(cube)

    # Secondary input 1 for fp=0, value 200 (6 realizations >= 3 clusters, full)
    data = np.full((6, *spatial_shape), 200.0, dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        name="air_temperature",
        units="K",
        spatial_grid="equalarea",
        realizations=np.arange(6),
        time=datetime(2024, 1, 1, 0),
        frt=datetime(2024, 1, 1, 0),
    )
    forecast_period = iris.coords.DimCoord(
        np.array([0], dtype=np.int32),
        standard_name="forecast_period",
        units="hours",
    )
    cube.add_aux_coord(forecast_period)
    cube.attributes["model_id"] = "secondary_model_1"
    cubes.append(cube)

    # Secondary input 2 for fp=6, value 300 (2 realizations < 3 clusters, partial)
    data = np.full((2, *spatial_shape), 300.0, dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        name="air_temperature",
        units="K",
        spatial_grid="equalarea",
        realizations=np.arange(2),
        time=datetime(2024, 1, 1, 0),
        frt=datetime(2024, 1, 1, 0),
    )
    forecast_period = iris.coords.DimCoord(
        np.array([6], dtype=np.int32),
        standard_name="forecast_period",
        units="hours",
    )
    cube.add_aux_coord(forecast_period)
    cube.attributes["model_id"] = "secondary_model_2"
    cubes.append(cube)

    # Target grid
    target_data = np.zeros((3, 3), dtype=np.float32)
    target_cube = set_up_variable_cube(
        target_data,
        name="target_grid",
        units="1",
        spatial_grid="equalarea",
    )
    target_cube.rename("target_grid")
    cubes.append(target_cube)

    hierarchy = {
        "primary_input": "primary_model",
        "secondary_inputs": {
            "secondary_model_1": [0],
            "secondary_model_2": [6],
        },
    }

    plugin = ClusterAndMatch(
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
    np.testing.assert_array_equal(forecast_periods, [0, 6, 12])

    # Should have 3 clusters
    assert len(result.coord("realization").points) == 3

    # Check data:
    # fp=0 should be entirely from secondary_model_1 (full realizations, 6 >= 3 clusters)
    fp_0_data = result.extract(iris.Constraint(forecast_period=0)).data
    assert np.allclose(
        fp_0_data, 200.0, atol=5.0
    ), "fp=0 should use secondary_model_1 (full realizations)"

    # fp=6: With random_state=42, clusters [0, 1] get replaced by secondary_model_2,
    # cluster [2] keeps primary data
    fp_6_data = result.extract(iris.Constraint(forecast_period=6)).data
    assert np.allclose(
        fp_6_data[0], 300.0, atol=5.0
    ), "Realization 0 should use secondary_model_2"
    assert np.allclose(
        fp_6_data[1], 300.0, atol=5.0
    ), "Realization 1 should use secondary_model_2"
    assert np.allclose(
        fp_6_data[2], 106.0, atol=5.0
    ), "Realization 2 should use clustered primary_model"

    # fp=12 should be entirely from primary
    fp_12_data = result.extract(iris.Constraint(forecast_period=12)).data
    assert np.allclose(
        fp_12_data, 112.0, atol=5.0
    ), "fp=12 should use clustered primary_model only"
