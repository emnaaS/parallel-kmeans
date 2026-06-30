#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
from dask.distributed import Client
import time

def _worker_compute(chunk, centroids):
    """
    Pure-Python worker function (runs on worker).
    Returns: labels (np.ndarray), partial_sums (np.ndarray), counts (np.ndarray)
    """
    t_start = time.time()
    
    chunk = np.array(chunk)
    centroids = np.array(centroids)

    # distances: (n_points_in_chunk, n_clusters)
    t_distance_start = time.time()
    distances = np.sqrt(((chunk[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
    labels = np.argmin(distances, axis=1)
    t_distance_end = time.time()

    n_clusters = centroids.shape[0]
    partial_sums = np.zeros((n_clusters, chunk.shape[1]))
    counts = np.zeros(n_clusters, dtype=int)

    t_aggregation_start = time.time()
    for k in range(n_clusters):
        mask = labels == k
        counts[k] = mask.sum()
        if counts[k] > 0:
            partial_sums[k] = chunk[mask].sum(axis=0)
    t_aggregation_end = time.time()
    
    t_total = t_aggregation_end - t_start
    
    #print(f'Worker - Distance computation: {t_distance_end-t_distance_start:.3f}s | '
          #f'Aggregation: {t_aggregation_end-t_aggregation_start:.3f}s | '
          #f'Total: {t_total:.3f}s')

    return labels, partial_sums, counts


class KMeansDistributedDask:
    """
    Distributed KMeans using dask.distributed futures (scatter / submit / gather).
    Caller should create a dask Client before calling fit(), e.g.:
        client = Client()  # or Client(<scheduler-address>)
    """
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, n_workers=4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_workers = n_workers
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.number_of_iter = 0

    def _initialize_centroids(self, X):
        """Initialize centroids to zero"""
        return np.zeros((self.n_clusters, X.shape[1]))

    def fit(self, X, client: Client):
        """
        Fit KMeans using an existing dask.distributed Client.
        - X: array-like (n_samples, n_features)
        - client: dask.distributed.Client (must be running)
        """
        t_fit_start = time.time()
        
        X = np.array(X)
        n_samples, n_features = X.shape

        # initialize centroids
        self.cluster_centers_ = self._initialize_centroids(X)

        # split and scatter data chunks to workers
        print(f'\n=== Setup ===')
        t_split_start = time.time()
        X_chunks = np.array_split(X, self.n_workers)
        t_split_end = time.time()
        print(f'Data splitting: {t_split_end-t_split_start:.3f}s')
        
        t_scatter_start = time.time()
        scattered = client.scatter(X_chunks, broadcast=False)
        t_scatter_end = time.time()
        print(f'Scatter to workers: {t_scatter_end-t_scatter_start:.3f}s')
        
        # Timing collectors
        submit_times = []
        gather_times = []
        aggregation_times = []
        centroid_update_times = []
        convergence_check_times = []

        print(f'\n=== Training Started ===')
        for iteration in range(self.max_iter):
            print(f'\n--- Iteration {iteration + 1} ---')
            
            # Submit jobs for each chunk
            t_submit_start = time.time()
            futures = [
                client.submit(_worker_compute, chunk_future, self.cluster_centers_)
                for chunk_future in scattered
            ]
            t_submit_end = time.time()
            submit_time = t_submit_end - t_submit_start
            submit_times.append(submit_time)
            print(f'Submit tasks: {submit_time:.3f}s')

            # Wait and gather results (this is where parallel work happens)
            t_gather_start = time.time()
            results = client.gather(futures)
            t_gather_end = time.time()
            gather_time = t_gather_end - t_gather_start
            gather_times.append(gather_time)
            print(f'Gather results: {gather_time:.3f}s')

            # Aggregate results
            t_agg_start = time.time()
            all_labels = []
            total_sums = np.zeros((self.n_clusters, n_features))
            total_counts = np.zeros(self.n_clusters, dtype=int)

            for labels_chunk, partial_sums, counts in results:
                all_labels.extend(labels_chunk.tolist() if isinstance(labels_chunk, np.ndarray) else labels_chunk)
                total_sums += partial_sums
                total_counts += counts

            self.labels_ = np.array(all_labels)
            t_agg_end = time.time()
            aggregation_times.append(t_agg_end - t_agg_start)
            print(f'Result aggregation: {t_agg_end-t_agg_start:.3f}s')

            # Compute new centers
            t_centroid_start = time.time()
            new_centers = np.zeros((self.n_clusters, n_features))
            for k in range(self.n_clusters):
                if total_counts[k] > 0:
                    new_centers[k] = total_sums[k] / total_counts[k]
                else:
                    # reinit empty cluster
                    new_centers[k] = X[k % X.shape[0]]
            t_centroid_end = time.time()
            centroid_update_times.append(t_centroid_end - t_centroid_start)
            print(f'Centroid update: {t_centroid_end-t_centroid_start:.3f}s')

            # Compute centroid shift
            t_conv_start = time.time()
            center_shift = np.sqrt(((new_centers - self.cluster_centers_) ** 2).sum())
            self.cluster_centers_ = new_centers
            t_conv_end = time.time()
            convergence_check_times.append(t_conv_end - t_conv_start)
            print(f'Convergence check: {t_conv_end-t_conv_start:.3f}s | Center shift: {center_shift:.6f}')

            if center_shift < self.tol:
                self.number_of_iter = iteration + 1
                print(f'\n✓ Converged after {iteration + 1} iterations')
                break
        else:
            self.number_of_iter = self.max_iter
            print(f'\n⚠ Max iterations ({self.max_iter}) reached')

        t_fit_end = time.time()
        
        # Print summary
        print(f'\n=== Training Summary ===')
        print(f'Total training time: {t_fit_end-t_fit_start:.3f}s')
        print(f'Number of iterations: {self.number_of_iter}')
        print(f'\nSubmit times per iteration: {[f"{t:.3f}s" for t in submit_times]}')
        print(f'Somme submit time: {np.sum(submit_times):.3f}s')
        print(f'\nGather times per iteration: {[f"{t:.3f}s" for t in gather_times]}')
        print(f'Somme gather time: {np.sum(gather_times):.3f}s')
        print(f'\nSomme aggregation time: {np.sum(aggregation_times):.3f}s')
        print(f'Somme centroid update time: {np.sum(centroid_update_times):.3f}s')
        print(f'Somme convergence check time: {np.sum(convergence_check_times):.3f}s')
        
        # Breakdown of time spent
        total_submit = sum(submit_times)
        total_gather = sum(gather_times)
        total_agg = sum(aggregation_times)
        total_centroid = sum(centroid_update_times)
        total_conv = sum(convergence_check_times)
        total_iteration = total_submit + total_gather + total_agg + total_centroid + total_conv
        
        print(f'\n=== Time Breakdown ===')
        print(f'Submit tasks: {total_submit:.3f}s ({100*total_submit/total_iteration:.1f}%)')
        print(f'Gather (parallel work): {total_gather:.3f}s ({100*total_gather/total_iteration:.1f}%)')
        print(f'Aggregation: {total_agg:.3f}s ({100*total_agg/total_iteration:.1f}%)')
        print(f'Centroid updates: {total_centroid:.3f}s ({100*total_centroid/total_iteration:.1f}%)')
        print(f'Convergence checks: {total_conv:.3f}s ({100*total_conv/total_iteration:.1f}%)')

        return self

    def predict(self, X):
        X = np.array(X)
        distances = np.sqrt(((X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def fit_predict(self, X, client: Client):
        self.fit(X, client)
        return self.labels_


