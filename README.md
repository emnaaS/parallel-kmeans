# Parallel kmeans Clustering

## Abstract
Although the k-means algorithm is a popular technique for unsupervised clustering, when applied to large-scale datasets, its com- putational cost becomes a significant drawback. Despite the fact that the parallelization provides a practical way to increase scalability and performance, the effects of various parallel computing frameworks are still unknown. Five parallel k-means implementations are compared in this paper: a GPU-based implementation using CUDA and four CPU- based frameworks (Multiprocessing, Ray, Dask, and Parsl). To guarantee uniformity and equitable comparison, every implementation is created from the ground up. A medical imaging dataset associated with diabetic retinopathy is used for experiments. The GPU implementation outper- forms sequential execution by 42.1×, according to the results. Parsl out- performs all CPU frameworks with a speedup 3.7× at 8 threads. How- ever, parallel eﬀiciency drastically declines from 87% to 27% for Parsl. This highlights important trade-offs when choosing the right paralleliza- tion frameworks for large-scale clustering tasks.


## Publication
This work was presented at **AINA 2026**.


## Acknowledgment
This work was conducted under the supervision of Prof. Yosr Slama, 
LIPSIC Lab, Faculty of Sciences of Tunis El Manar.


## Full Paper
See `Comparative Performance Analysis of Parallelization Frameworks for K-Means : Implementation on CPU and GPU.pdf` for complete methodology, results, and discussion.


## Contact
If you have any questions or feedback, feel free to reach out to me at emnaa.sellami@gmail.com.
