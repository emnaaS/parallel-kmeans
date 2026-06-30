#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
from parsl.app.app import python_app
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
import parsl
import time

@python_app
def compute_distances_and_assign(X_chunk, centroids, n_clusters):
    """
    Parsl app: Compute distances and assign clusters for a data chunk.
    Returns: labels, partial_sums, counts
    """
    import numpy as np
    import time
    
    t_start = time.time()

    X_chunk = np.array(X_chunk)
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
    
    t_total = t_aggregation_end - t_start

    return labels, partial_sums, counts


class KMeansDistributedParsl:

    def __init__(self, n_clusters, max_iter=300, tol=1e-4, n_workers=4, random_state=None, verbose=True):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_workers = n_workers
        self.random_state = random_state
        self.verbose = verbose

        self.cluster_centers_ = None
        self.labels_ = None
        self.number_of_iter = 0

    def _initialize_centroids(self, X):
        """Initialize centroids to zero"""
        return np.zeros((self.n_clusters, X.shape[1]))

    def fit(self, X):
        t_fit_start = time.time()

        X = np.array(X)
        n_samples, n_features = X.shape

        # Initialize centroids
        self.cluster_centers_ = self._initialize_centroids(X)

        print(f'\n=== Setup ===')
        # Split data into chunks for parallel processing
        t_split_start = time.time()
        X_chunks = np.array_split(X, self.n_workers)
        t_split_end = time.time()
        print(f'Data splitting: {t_split_end-t_split_start:.3f}s')

        # Timing collectors
        iteration_total_times = []
        submit_times = []
        gather_times = []
        aggregation_times = []
        centroid_update_times = []
        convergence_check_times = []
        iteration_overhead_times = []

        print(f'\n=== Training Started ===')
        for iteration in range(self.max_iter):
            t_iteration_start = time.time()
            
            if self.verbose:
                print(f'\n--- Iteration {iteration + 1} ---')

            # Submit parallel tasks - one per chunk
            t_submit_start = time.time()
            futures = [
                compute_distances_and_assign(
                    chunk,
                    self.cluster_centers_,
                    self.n_clusters
                )
                for chunk in X_chunks
            ]
            t_submit_end = time.time()
            submit_time = t_submit_end - t_submit_start
            submit_times.append(submit_time)
            if self.verbose:
                print(f'Submit tasks: {submit_time:.3f}s')

            # Wait for all tasks to complete and gather results
            t_gather_start = time.time()
            results = [future.result() for future in futures]
            t_gather_end = time.time()
            gather_time = t_gather_end - t_gather_start
            gather_times.append(gather_time)
            if self.verbose:
                print(f'Gather results (parallel work): {gather_time:.3f}s')

            # Aggregate results from all workers
            t_agg_start = time.time()
            all_labels = []
            total_sums = np.zeros((self.n_clusters, n_features))
            total_counts = np.zeros(self.n_clusters, dtype=int)

            for labels_chunk, partial_sums, counts in results:
                labels_chunk = np.array(labels_chunk)
                partial_sums = np.array(partial_sums)
                counts = np.array(counts)

                all_labels.extend(labels_chunk.tolist())
                total_sums += partial_sums
                total_counts += counts

            self.labels_ = np.array(all_labels)
            t_agg_end = time.time()
            aggregation_time = t_agg_end - t_agg_start
            aggregation_times.append(aggregation_time)
            if self.verbose:
                print(f'Result aggregation: {aggregation_time:.3f}s')

            # Compute new centroids from aggregated data
            t_centroid_start = time.time()
            new_centers = np.zeros((self.n_clusters, n_features))
            for k in range(self.n_clusters):
                if total_counts[k] > 0:
                    new_centers[k] = total_sums[k] / total_counts[k]
                else:
                    # Handle empty cluster 
                    new_centers[k] = X[k % X.shape[0]]  # Deterministic
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

            # Calculate iteration overhead
            t_iteration_end = time.time()
            iteration_total = t_iteration_end - t_iteration_start
            iteration_total_times.append(iteration_total)
            
            accounted_time = submit_time + gather_time + aggregation_time + centroid_time + convergence_time
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

        t_fit_end = time.time()

        if self.verbose:
            # Print summary
            print(f'\n=== Training Summary ===')
            print(f'Total training time: {t_fit_end-t_fit_start:.3f}s')
            print(f'Number of iterations: {self.number_of_iter}')
            
            print(f'\nSubmit times per iteration: {[f"{t:.3f}s" for t in submit_times]}')
            print(f'Sum submit time: {np.sum(submit_times):.3f}s')
            
            print(f'\nGather times per iteration: {[f"{t:.3f}s" for t in gather_times]}')
            print(f'Sum gather time: {np.sum(gather_times):.3f}s')
            
            print(f'\nSum aggregation time: {np.sum(aggregation_times):.3f}s')
            print(f'Sum centroid update time: {np.sum(centroid_update_times):.3f}s')
            print(f'Sum convergence check time: {np.sum(convergence_check_times):.3f}s')
            print(f'Sum iteration overhead time: {np.sum(iteration_overhead_times):.3f}s')
            
            # Breakdown of time spent
            total_submit = sum(submit_times)
            total_gather = sum(gather_times)
            total_agg = sum(aggregation_times)
            total_centroid = sum(centroid_update_times)
            total_conv = sum(convergence_check_times)
            total_overhead = sum(iteration_overhead_times)
            total_iteration = total_submit + total_gather + total_agg + total_centroid + total_conv + total_overhead
            
            print(f'\n=== Time Breakdown ===')
            print(f'Submit tasks: {total_submit:.3f}s ({100*total_submit/total_iteration:.1f}%)')
            print(f'Gather (parallel work): {total_gather:.3f}s ({100*total_gather/total_iteration:.1f}%)')
            print(f'Aggregation: {total_agg:.3f}s ({100*total_agg/total_iteration:.1f}%)')
            print(f'Centroid updates: {total_centroid:.3f}s ({100*total_centroid/total_iteration:.1f}%)')
            print(f'Convergence checks: {total_conv:.3f}s ({100*total_conv/total_iteration:.1f}%)')
            print(f'Iteration overhead (print, misc): {total_overhead:.3f}s ({100*total_overhead/total_iteration:.1f}%)')
            
            # Verification
            print(f'\n=== Verification ===')
            print(f'Sum of all measured times: {total_iteration:.3f}s')
            print(f'Actual total training time: {t_fit_end-t_fit_start:.3f}s')
            print(f'Setup time: {t_split_end-t_split_start:.3f}s')
            print(f'Expected total: {total_iteration + (t_split_end-t_split_start):.3f}s')

        return self

    def predict(self, X):
        """Predict cluster labels for new data"""
        X = np.array(X)
        distances = np.sqrt(((X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        """Fit and return cluster labels"""
        self.fit(X)
        return self.labels_



