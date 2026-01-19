# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugins to perform clustering on DataFrames using scikit-learn or kmedoids."""

from typing import Any

import pandas as pd

from improver import BasePlugin


class FitClustering(BasePlugin):
    """Class to perform clustering on DataFrames using scikit-learn or kmedoids.

    This plugin provides a unified interface for applying various clustering algorithms
    to pandas DataFrames. It supports clustering methods from scikit-learn's cluster
    module as well as the KMedoids algorithm from the kmedoids package.
    The plugin automatically selects the appropriate package based on the specified
    clustering method:
    - "KMedoids": Uses the kmedoids package
    - All other methods: Uses sklearn.cluster
    """

    def __init__(self, clustering_method: str, **kwargs: Any) -> None:
        """Initialise the clustering plugin.

        Args:
            clustering_method: The name of the clustering method to use.
                Must be either "KMedoids" (from kmedoids package) or a valid
                clustering class name from sklearn.cluster (e.g., "KMeans",
                "DBSCAN", "AgglomerativeClustering").
            **kwargs: Additional keyword arguments to pass to the clustering
                algorithm. These are method-specific parameters. Common examples:
                - n_clusters (int): Number of clusters (for KMeans,
                AgglomerativeClustering)
                - random_state (int): Random seed for reproducibility
                Refer to the scikit-learn or kmedoids documentation for the complete
                list of parameters for each clustering method.
        Raises:
            ValueError: If the specified clustering method is not found in
                sklearn.cluster or kmedoids packages.
        """
        self.clustering_method = clustering_method
        self.kwargs = kwargs

    def process(self, df: pd.DataFrame) -> Any:
        """Apply the clustering method to the DataFrame. Fits the specified clustering
        algorithm to the input DataFrame and returns the fitted clustering model.

        Args:
            df: The input DataFrame to cluster. Each row represents
                a sample and each column represents a feature. The DataFrame should
                contain numeric data suitable for the chosen clustering algorithm.
        Returns:
            A fitted clustering model object from either sklearn.cluster or kmedoids.
            The returned object will have at minimum a `labels_` attribute containing
            the cluster assignment for each sample. Additional attributes depend on
            the specific clustering method used (e.g., `cluster_centers_` for KMeans,
            `core_sample_indices_` for DBSCAN).
        Raises:
            ValueError: If the specified clustering method is not found in
                sklearn.cluster or is not "KMedoids".
        """
        # Use kmedoids directly if requested
        if self.clustering_method == "KMedoids":
            import kmedoids

            # Set default metric to euclidean if not specified
            kwargs = self.kwargs.copy()
            if "metric" not in kwargs:
                kwargs["metric"] = "euclidean"

            clustering_class = getattr(kmedoids, self.clustering_method)
            # Convert DataFrame to numpy array for kmedoids
            return clustering_class(**kwargs).fit(df.values)

        # Otherwise, use sklearn
        from sklearn import cluster

        if hasattr(cluster, self.clustering_method):
            clustering_class = getattr(cluster, self.clustering_method)
            return clustering_class(**self.kwargs).fit(df)
        else:
            msg = (
                f"The clustering method '{self.clustering_method}' is not supported. "
                "Please check sklearn.cluster documentation for available methods."
            )
            raise ValueError(msg)
