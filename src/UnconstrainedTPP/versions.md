
## Unconstrained Touring Polygons Problem - Legacy Versions

This directory contains legacy versions of the Unconstrained Touring Polygons Problem (TPP) solutions. These versions were developed over time, each improving upon the previous in terms of performance and correctness.

These versions are maintained for reference and comparison purposes. They may not represent the most efficient or correct implementations available today.

### Versions

- `u_tpp_naive.py`: The initial naive implementation of the Unconstrained TPP solution. Locates points in `query` by using brute force. This version serves as a baseline for performance and correctness.

- `u_tpp_fast_locate.py`: An improved version that optimizes the point location process in `query`. This version uses binary search to speed up the location of points, significantly reducing computation time compared to the naive approach.

- `u_tpp_filtered.py`: This version filters out unnecessary vertices by computing the first contact region. This version is faster than previous versions for larger inputs, as it reduces the number of vertices that need to be considered in the solution process.

- `u_tpp_jit1.py`: A Just-In-Time (JIT) approach to the Unconstrained TPP solution. This version computes the first contact region in `log(|P_i|)` time for each polygon, then only computes the vertex regions that are necessary for answering a query. This results in significant performance improvements for large datasets.

- `u_tpp_jit2.py`: A faster and simpler JIT implementation. This version oonly computes vertex regions and determines if a edge is visible if necessary for answering a query. This is the most efficient version available in this directory and is recommended for use in performance-critical applications.

- `u_tpp.py`: The current main implementation of the Unconstrained TPP solution. This version incorporates the best practices and optimizations from previous versions, providing a robust and efficient solution for the Unconstrained TPP. This approach is mostly based on `u_tpp_jit2.py` but with further refinements and improvements.
