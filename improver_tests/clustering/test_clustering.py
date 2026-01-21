# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the improver.clustering.clustering module."""

import numpy as np
import pandas as pd
import pytest

from improver.clustering.clustering import FitClustering


def _create_sample_dataframe():
    """Create a sample DataFrame for testing.

    Returns:
        A DataFrame with 20 rows and 5 features.
    """
    np.random.seed(42)
    data = np.random.randn(20, 5)
    return pd.DataFrame(data, columns=[f"feature_{i}" for i in range(5)])


def _create_clusterable_dataframe():
    """Create a simple DataFrame with clearly separable clusters.

    Returns:
        A DataFrame with two distinct clusters.
    """
    # Create two distinct clusters
    cluster1 = np.random.randn(10, 2) + np.array([0, 0])
    cluster2 = np.random.randn(10, 2) + np.array([10, 10])
    data = np.vstack([cluster1, cluster2])
    return pd.DataFrame(data, columns=["x", "y"])


@pytest.mark.parametrize(
    "clustering_method,n_clusters,random_state,expected_n_labels,expected_shape",
    [
        ("KMeans", 3, 42, 3, (3, 5)),  # Basic KMeans
        ("KMeans", 2, 42, 2, (2, 5)),  # KMeans with 2 clusters
        ("KMeans", 5, 100, 5, (5, 5)),  # KMeans with different random state
        ("AgglomerativeClustering", 4, None, 4, None),  # AgglomerativeClustering
    ],
)
def test_process_clustering_combined(
    clustering_method,
    n_clusters,
    random_state,
    expected_n_labels,
    expected_shape,
):
    """Test FitClustering.process with various clustering methods available in
    scikit-learn and cluster center assertions (excluding KMedoids)."""
    df = _create_sample_dataframe()

    kwargs = {"n_clusters": n_clusters}
    if random_state is not None:
        kwargs["random_state"] = random_state

    plugin = FitClustering(clustering_method, **kwargs)
    result = plugin.process(df)

    assert hasattr(result, "labels_")
    assert len(result.labels_) == len(df)
    assert len(np.unique(result.labels_)) == expected_n_labels
    assert np.all((result.labels_ >= 0) & (result.labels_ < n_clusters))

    # Cluster center assertions for KMeans only
    if expected_shape is not None:
        assert hasattr(result, "cluster_centers_")
        assert result.cluster_centers_.shape == expected_shape


def test_process_dbscan():
    """Test the FitClustering.process method with DBSCAN clustering."""
    sample_df = _create_sample_dataframe()

    plugin = FitClustering("DBSCAN", eps=0.5, min_samples=5)
    result = plugin.process(sample_df)

    assert hasattr(result, "labels_")
    assert len(result.labels_) == len(sample_df)


def test_process_kmedoids():
    """Test the FitClustering.process method with KMedoids from kmedoids package."""
    pytest.importorskip("kmedoids")
    clusterable_df = _create_clusterable_dataframe()

    plugin = FitClustering("KMedoids", n_clusters=2, random_state=42)
    result = plugin.process(clusterable_df)

    assert hasattr(result, "labels_")
    assert hasattr(result, "cluster_centers_")
    assert len(result.labels_) == len(clusterable_df)
    assert result.cluster_centers_.shape == (2, 2)


def test_process_returns_fitted_model():
    """Test that process returns a fitted clustering model that can predict."""
    sample_df = _create_sample_dataframe()

    plugin = FitClustering("KMeans", n_clusters=2, random_state=42)
    result = plugin.process(sample_df)

    # Verify the model is fitted by checking it can predict
    assert hasattr(result, "predict")
    predictions = result.predict(sample_df)
    assert len(predictions) == len(sample_df)


def test_process_invalid_clustering_method():
    """Test that an error is raised for unsupported clustering methods."""
    sample_df = _create_sample_dataframe()

    plugin = FitClustering("NonExistentMethod", n_clusters=3)

    with pytest.raises(
        ValueError,
        match="The clustering method 'NonExistentMethod' is not supported",
    ):
        plugin.process(sample_df)


def test_process_with_empty_dataframe():
    """Test process method with an empty DataFrame raises an error."""
    plugin = FitClustering("KMeans", n_clusters=2)
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError):
        plugin.process(empty_df)


def test_process_with_single_sample():
    """Test process method with a single sample."""
    plugin = FitClustering("KMeans", n_clusters=1, random_state=42)
    single_sample_df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
    result = plugin.process(single_sample_df)

    assert hasattr(result, "labels_")
    assert len(result.labels_) == 1
    assert result.labels_[0] == 0


def test_process_preserves_dataframe():
    """Test that process does not modify the input DataFrame."""
    sample_df = _create_sample_dataframe()
    original_data = sample_df.copy()

    plugin = FitClustering("KMeans", n_clusters=3, random_state=42)
    plugin.process(sample_df)

    pd.testing.assert_frame_equal(sample_df, original_data)
