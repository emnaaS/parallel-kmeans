#!/usr/bin/env python
# coding: utf-8

# In[2]:
import glob
import os
import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

#my sequential kmeans

class cust_K_Means(object):
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol  # Convergence tolerance
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.number_of_iter = 0

    def _initialize_centroids(self, X):
        """Initialize centroids to zero"""
        return np.zeros((self.n_clusters, X.shape[1]))

    def _compute_distances(self, X, centers):
        """
        Vectorized distance computation - MUCH faster!
        Returns: (n_samples, n_clusters) array of distances
        """
        # Expand dimensions for broadcasting
        # X: (n_samples, n_features) -> (n_samples, 1, n_features)
        # centers: (n_clusters, n_features) -> (1, n_clusters, n_features)
        distances = np.sqrt(((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
        return distances

    def _assign_clusters(self, X, centers):
        """Assign each point to nearest cluster - VECTORIZED"""
        distances = self._compute_distances(X, centers)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, labels):
        """Compute new centroids - VECTORIZED"""
        new_centers = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centers[k] = cluster_points.mean(axis=0)
            else:
                # Handle empty cluster - le centroide sera affecté le k--ième point
                new_centers[k] = X[k % X.shape[0]]  # ← DÉTERMINISTE
        return new_centers


    def fit(self, X):
        """Train K-Means on data X"""
        X = np.array(X)

        # Initialize centroids
        self.cluster_centers_ = self._initialize_centroids(X)

        for i in range(self.max_iter):
            # Assign points to clusters
            labels = self._assign_clusters(X, self.cluster_centers_)

            # Compute new centroids
            new_centers = self._compute_centroids(X, labels)

            # Check convergence (centroid shift)
            center_shift = np.sqrt(((new_centers - self.cluster_centers_) ** 2).sum())

            # Update centroids
            self.cluster_centers_ = new_centers
            self.labels_ = labels

            # Check convergence
            if center_shift < self.tol:
                self.number_of_iter = i + 1
                print(f'Converged after {i + 1} iterations')
                break
        else:
            self.number_of_iter = self.max_iter
            print(f'Max iterations ({self.max_iter}) reached')


        return self

    def predict(self, X):
        """Predict cluster labels for new data"""
        X = np.array(X)
        return self._assign_clusters(X, self.cluster_centers_)


# In[ ]:


get_ipython().system('jupyter nbconvert --to python /home/user/sequent.ipynb')

