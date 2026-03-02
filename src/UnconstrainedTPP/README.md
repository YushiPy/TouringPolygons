
## Unconstrained Touring Polygons Problem

This directory contains multiple files related to the unconstrained touring polygons problem. 

### Versions

There are 12 different versions of the solution, each with different optimizations and approaches. The main file is `u_tpp_final.py`, which contains the final version of the solution. The other files are intermediate versions that were used during the development process.

This `README.md` will probably be outdated, so remind me to update it when I make significant changes to the codebase.

- `u_tpp_fast_jit.py`: A version of the solution that uses binary search and computes only cones of visibility that are needed for the point location queries (JIT query).
- `u_tpp_fast_locate.py`: A version of the solution that uses binary search to query, but computes all cones of visibility in advance. It's straight up worse than `u_tpp_fast_jit.py`.
- `u_tpp_filtered.py`: It's the same as `u_tpp_fast_locate.py`, but it computes the first contact region for each polygon and redefined the polygons using it. It's slightly faster than `u_tpp_fast_locate.py`, but still worse than `u_tpp_fast_jit.py`.
- `u_tpp_final.py`: The final version of the solution, it uses JIT computation of cones of visibility and chooses between binary search and linear search for point location based on the number of vertices of the polygon.
- `u_tpp_first.py`: This is the first version of the solution ever created. It uses linear search for point location and computes all cones of visibility in advance. The first contact region is calculated in `O(|P_i|^2)` time by checking each edge for each vertex. It's VERY slow, but it was a good starting point for the development process and it has sentimental value to me, so I decided to keep it in the codebase.
- `u_tpp_jit1.py`: This version is the first attempt to use JIT computation of cones of visibility. It computes the first contact region in `O(|P_i|)` time. The algorithm is absolutely insane and I am proud of it. However, it is objectivelly worse than all other implementation of JIT which are much simpler, so I decided to not use it as the final version. Still, it's a very interesting implementation and I recommend checking it out.
- `u_tpp_jit2.py`: This version is a simplified version of `u_tpp_jit1.py` that is easier to understand and implement. It only determines whether an edge is in the first contact region or not if necessary by making an extra query.
- `u_tpp_naive.py`: A naive implementation that uses linear seach for point location and computes all cones of visibility in advance. It's slightly faster than `u_tpp_first.py`, but still very slow for large inputs.
- `u_tpp_naive_cpp.py`: Same as `u_tpp_naive.py`, but implemented in C++ and exposed to Python using some other files. It's faster than `u_tpp_naive.py`, but doesn't solve the main problem.
- `u_tpp_naive_jit.py`: Same as `u_tpp_naive.py`, but with JIT computation of cones of visibility. It's actually very fast for small inputs and it's a good example of how JIT can be used to speed up a solution, but it still doesn't solve the main problem for large inputs. For this reason, we combined it with the binary search optimization to create `u_tpp_final.py`, which is the final version of the solution.
- `u_tpp_sympy.py`: Treats the problem as a non linear optimization problem and uses the sympy library to solve it. It's very slow and isn't very precise. However, it can deal with non-convex polygons which no other solution can. While it isn't very useful for the convex case, it is interesting and it is an important point of comparison for our solutions.
- `u_tpp.py`: You would think this is the main solution, but I haven't updated it yet. It is just `u_tpp_jit2.py` but micro optimized.

