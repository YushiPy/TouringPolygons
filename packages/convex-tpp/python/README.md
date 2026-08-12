# Convex TPP Python

This directory keeps Python code that is still useful for convex TPP reference,
validation, visualization, or comparison with the maintained C++ solver.

Current conventions:

- Keep reusable modules and named tests here.
- Do not keep `_tmp*.py`, `_test*.py`, or other scratch files here.
- Move historical attempts and abandoned prototypes to `experiments/`.
- Keep generated binary fixtures in package `tests/` directories only when they
  are intentional regression inputs.

The maintained implementation surface is the C++ package in
`packages/convex-tpp/cpp`. Python files here are support code, not the primary
solver API.
