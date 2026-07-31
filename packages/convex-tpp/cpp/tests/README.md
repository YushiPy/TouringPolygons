
## Tests directory

This directory contains test cases for the TPP for convex polygons C++ implementation.
These test cases are used to verify the correctness of the implementation and to ensure that it produces the expected results for various input scenarios. The test cases are stored in binary format in the `tests` directory.

## Test case format

Every file in the `tests` directory with a `.bin` extension contains one or more test cases. Each test case is structured as follows:


- `16` bytes: Start point (`2 doubles`)
- `16` bytes: Target point (`2 doubles`)
- `8` bytes: Number of polygons (`size_t`)
- For each polygon:
	- `8` bytes: Number of vertices (`size_t`)
	- For each vertex:
		- `16` bytes: Vertex position (`2 doubles`)
- `8 `bytes: Number of points in the solution (size_t)
- For each point in the solution:
	- `16` bytes: Point position (`2 doubles`)

The file may contains multiple test cases, one after the other, following the same format. The number of test cases in the file can be determined by reading until the end of the file.