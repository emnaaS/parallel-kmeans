#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from multiprocessing import Pool, cpu_count, shared_memory
import ctypes
import time


def _worker_compute_shared(args):
    t_start = time.time()
    
    shm_name, shape, dtype, start_idx, end_idx, centroids, n_clusters = args

    # Attach to existing shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    X_shared = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    # Work on assigned slice
    X_chunk = X_shared[start_idx:end_idx]
    centroids = np.array(centroids)

    # Vectorized distance computation
    t_distance_start = time.time()
    distances = np.sqrt(((X_chunk[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
    labels = np.argmin(distances, axis=1)
    t_distance_end = time.time()

    # Compute partial sums and counts for each cluster
    t_aggregation_start = time.time()
    partial_sums = np.zeros((n_clusters, X_chunk.shape[1]))
    counts = np.zeros(n_clusters, dtype=int)

    for k in range(n_clusters):
        mask = labels == k
        counts[k] = np.sum(mask)
        if counts[k] > 0:
            partial_sums[k] = X_chunk[mask].sum(axis=0)
    t_aggregation_end = time.time()

    # Clean up (don't unlink - just detach)
    shm.close()
    
    t_total = t_aggregation_end - t_start

    return labels, partial_sums, counts


class KMeansMultiprocessing:

    def __init__(self, n_clusters, max_iter=300, tol=1e-4, n_workers=None,
                 random_state=None, verbose=True, use_shared_memory=True):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_workers = n_workers if n_workers is not None else cpu_count()
        self.random_state = random_state
        self.verbose = verbose
        self.use_shared_memory = use_shared_memory

        self.cluster_centers_ = None
        self.labels_ = None
        self.number_of_iter = 0

    def _initialize_centroids(self, X):
        """Initialize centroids to zero"""
        return np.zeros((self.n_clusters, X.shape[1]))

    def _fit_shared_memory(self, X):
        """Fit using shared memory (memory efficient)"""
        t_fit_start = time.time()
        
        n_samples, n_features = X.shape

        print(f'\n=== Setup ===')
        # Create shared memory block
        t_shm_start = time.time()
        shm = shared_memory.SharedMemory(create=True, size=X.nbytes)
        X_shared = np.ndarray(X.shape, dtype=X.dtype, buffer=shm.buf)
        X_shared[:] = X[:]  # Copy data once into shared memory
        t_shm_end = time.time()
        print(f'Shared memory creation and data copy: {t_shm_end-t_shm_start:.3f}s')

        # Calculate chunk boundaries (indices instead of copying data)
        t_split_start = time.time()
        chunk_sizes = [len(chunk) for chunk in np.array_split(range(n_samples), self.n_workers)]
        chunk_indices = []
        start_idx = 0
        for size in chunk_sizes:
            end_idx = start_idx + size
            chunk_indices.append((start_idx, end_idx))
            start_idx = end_idx
        t_split_end = time.time()
        print(f'Chunk boundary calculation: {t_split_end-t_split_start:.3f}s')

        # Timing collectors
        iteration_total_times = []
        args_prep_times = []
        map_times = []
        aggregation_times = []
        centroid_update_times = []
        convergence_check_times = []
        iteration_overhead_times = []

        try:
            with Pool(processes=self.n_workers) as pool:
                print(f'\n=== Training Started ===')
                for iteration in range(self.max_iter):
                    t_iteration_start = time.time()
                    
                    if self.verbose:
                        print(f'\n--- Iteration {iteration + 1} ---')
                    
                    # Prepare arguments: each worker gets indices, not data
                    t_args_start = time.time()
                    args_list = [
                        (shm.name, X.shape, X.dtype, start_idx, end_idx,
                         self.cluster_centers_, self.n_clusters)
                        for start_idx, end_idx in chunk_indices
                    ]
                    t_args_end = time.time()
                    args_prep_time = t_args_end - t_args_start
                    args_prep_times.append(args_prep_time)
                    if self.verbose:
                        print(f'Arguments preparation: {args_prep_time:.3f}s')

                    # Distribute work
                    t_map_start = time.time()
                    results = pool.map(_worker_compute_shared, args_list)
                    t_map_end = time.time()
                    map_time = t_map_end - t_map_start
                    map_times.append(map_time)
                    if self.verbose:
                        print(f'Parallel computation (pool.map): {map_time:.3f}s')

                    # Aggregate results
                    t_agg_start = time.time()
                    all_labels = []
                    total_sums = np.zeros((self.n_clusters, n_features))
                    total_counts = np.zeros(self.n_clusters, dtype=int)

                    for labels_chunk, partial_sums, counts in results:
                        all_labels.extend(labels_chunk)
                        total_sums += partial_sums
                        total_counts += counts

                    self.labels_ = np.array(all_labels)
                    t_agg_end = time.time()
                    aggregation_time = t_agg_end - t_agg_start
                    aggregation_times.append(aggregation_time)
                    if self.verbose:
                        print(f'Result aggregation: {aggregation_time:.3f}s')

                    # Compute new centroids
                    t_centroid_start = time.time()
                    new_centers = np.zeros((self.n_clusters, n_features))
                    for k in range(self.n_clusters):
                        if total_counts[k] > 0:
                            new_centers[k] = total_sums[k] / total_counts[k]
                        else:
                            # Deterministic handling of empty clusters
                            new_centers[k] = X[k % n_samples]  # Use k-th point instead of random
                    t_centroid_end = time.time()
                    centroid_time = t_centroid_end - t_centroid_start
                    centroid_update_times.append(centroid_time)
                    if self.verbose:
                        print(f'Centroid update: {centroid_time:.3f}s')

                    # Check convergence
                    t_conv_start = time.time()
                    center_shift = np.sqrt(((new_centers - self.cluster_centers_) ** 2).sum())
                    self.cluster_centers_ = new_centers
                    t_conv_end = time.time()
                    convergence_time = t_conv_end - t_conv_start
                    convergence_check_times.append(convergence_time)
                    if self.verbose:
                        print(f'Convergence check: {convergence_time:.3f}s | Center shift: {center_shift:.6f}')

                    # Calculate iteration overhead (time not accounted for)
                    t_iteration_end = time.time()
                    iteration_total = t_iteration_end - t_iteration_start
                    iteration_total_times.append(iteration_total)
                    
                    accounted_time = args_prep_time + map_time + aggregation_time + centroid_time + convergence_time
                    overhead = iteration_total - accounted_time
                    iteration_overhead_times.append(overhead)
                    
                    if self.verbose:
                        print(f'Iteration overhead (print, misc): {overhead:.3f}s')
                        print(f'Total iteration time: {iteration_total:.3f}s')

                    if center_shift < self.tol:
                        self.number_of_iter = iteration + 1
                        if self.verbose:
                            print(f'\n✓ Converged after {iteration + 1} iterations')
                        break
                else:
                    self.number_of_iter = self.max_iter
                    if self.verbose:
                        print(f'\n⚠ Max iterations ({self.max_iter}) reached')
        finally:
            # Clean up shared memory
            t_cleanup_start = time.time()
            shm.close()
            shm.unlink()
            t_cleanup_end = time.time()
            if self.verbose:
                print(f'\nShared memory cleanup: {t_cleanup_end-t_cleanup_start:.3f}s')

        t_fit_end = time.time()
        
        if self.verbose:
            # Print summary
            print(f'\n=== Training Summary ===')
            print(f'Total training time: {t_fit_end-t_fit_start:.3f}s')
            print(f'Number of iterations: {self.number_of_iter}')
            
            print(f'\nArguments prep times per iteration: {[f"{t:.3f}s" for t in args_prep_times]}')
            print(f'Sum args prep time: {np.sum(args_prep_times):.3f}s')
            
            print(f'\nMap times per iteration: {[f"{t:.3f}s" for t in map_times]}')
            print(f'Sum map time: {np.sum(map_times):.3f}s')
            
            print(f'\nSum aggregation time: {np.sum(aggregation_times):.3f}s')
            print(f'Sum centroid update time: {np.sum(centroid_update_times):.3f}s')
            print(f'Sum convergence check time: {np.sum(convergence_check_times):.3f}s')
            print(f'Sum iteration overhead time: {np.sum(iteration_overhead_times):.3f}s')
            
            # Breakdown of time spent
            total_args_prep = sum(args_prep_times)
            total_map = sum(map_times)
            total_agg = sum(aggregation_times)
            total_centroid = sum(centroid_update_times)
            total_conv = sum(convergence_check_times)
            total_overhead = sum(iteration_overhead_times)
            total_iteration = total_args_prep + total_map + total_agg + total_centroid + total_conv + total_overhead
            
            print(f'\n=== Time Breakdown ===')
            print(f'Arguments preparation: {total_args_prep:.3f}s ({100*total_args_prep/total_iteration:.1f}%)')
            print(f'Parallel computation: {total_map:.3f}s ({100*total_map/total_iteration:.1f}%)')
            print(f'Aggregation: {total_agg:.3f}s ({100*total_agg/total_iteration:.1f}%)')
            print(f'Centroid updates: {total_centroid:.3f}s ({100*total_centroid/total_iteration:.1f}%)')
            print(f'Convergence checks: {total_conv:.3f}s ({100*total_conv/total_iteration:.1f}%)')
            print(f'Iteration overhead (print, misc): {total_overhead:.3f}s ({100*total_overhead/total_iteration:.1f}%)')
            
            # Verification
            print(f'\n=== Verification ===')
            print(f'Sum of all measured times: {total_iteration:.3f}s')
            print(f'Actual total training time: {t_fit_end-t_fit_start:.3f}s')
            print(f'Setup + cleanup time: {(t_shm_end-t_shm_start) + (t_split_end-t_split_start) + (t_cleanup_end-t_cleanup_start):.3f}s')
            print(f'Expected total: {total_iteration + (t_shm_end-t_shm_start) + (t_split_end-t_split_start) + (t_cleanup_end-t_cleanup_start):.3f}s')

    def fit(self, X):
        X = np.array(X)

        # Initialize centroids
        self.cluster_centers_ = self._initialize_centroids(X)

        # Choose fitting method based on use_shared_memory flag
        self._fit_shared_memory(X)

        return self

