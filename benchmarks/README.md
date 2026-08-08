# Benchmarking

This folder is for benchmark outputs, split test sets, and notes about how to interpret them.

The current benchmark focuses on the non-convex branch-and-bound solver. Each B&B node calls the convex TPP solver on either a lower-bound instance or a leaf instance, so the main cost drivers are the number of convex calls and the cost per convex call.

## Build

From the repository root:

```bash
cmake --preset nonconvex-release -DTARGET=main-bnb_workload_benchmark
cmake --build --preset nonconvex-release
```

The benchmark binary is:

```bash
./build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp
```

## Run A Full Benchmark

```bash
./build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp \
  packages/nonconvex-tpp/cpp/tests/test_cases_simplified2.bin \
  -1 -1 1000000 -1 \
  benchmarks/results.csv \
  benchmarks/summary.md
```

Arguments:

| Argument | Meaning |
|---|---|
| `input_file` | Binary test set. |
| `max_polygons` | Skip instances with more polygons than this. Use `-1` for no limit. |
| `max_instances` | Stop after this many accepted instances. Use `-1` for no limit. |
| `max_calls_per_instance` | Cap convex solver calls per instance. This is the main timeout guard. |
| `max_branching` | Cap explored children per polygon during B&B. Use `-1` for no limit. |
| `repeat_count` | Optional repeated runs per accepted instance. |
| `csv_output_file` | Optional per-instance CSV output path. |
| `summary_md_file` | Optional markdown summary output path. |

By default the benchmark uses the machine's hardware thread count. To force a specific number of workers:

```bash
TPP_BENCH_THREADS=1 ./build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp ...
```

Use `TPP_BENCH_THREADS=1` when you want a single-thread baseline.

## Read The Summary

Important fields:

| Field | Meaning |
|---|---|
| `Fully solved runs` | Instances where the B&B search finished under the configured limits. |
| `Capped by calls runs` | Instances stopped by `max_calls_per_instance`; these are usually the hard cases. |
| `Best updates` | How often B&B improved the approximation/incumbent. Low values mean the approximation is already strong. |
| `Initial gap %` | Relative improvement from the initial approximation to the final result. |
| `Incumbent gap %` | Relative improvement from the first selected-piece incumbent to the final result. |
| `Failed-prune bound/incumbent` | How close non-pruning lower bounds are to the incumbent. Values near `1.0` mean proving optimality is hard. |
| `Measured work` | Sum of per-instance measured times. In parallel runs this can exceed wall-clock time. |
| `Wall-clock total` | Actual elapsed time. |
| `Parallel speedup estimate` | `measured work / wall-clock total`. |

If `Initial gap %` is tiny but calls are high, the solver is finding good solutions quickly but spending time proving optimality.

## Split A Test Set By Difficulty

First run a benchmark and keep its CSV. Then build the splitter:

```bash
cmake --preset nonconvex-release -DTARGET=main-split_benchmark_cases
cmake --build --preset nonconvex-release
```

Split the original binary test set using the measured CSV:

```bash
./build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp \
  packages/nonconvex-tpp/cpp/tests/test_cases_simplified2.bin \
  benchmarks/results.csv \
  benchmarks/splits
```

This writes:

| File | Contents |
|---|---|
| `benchmarks/splits/easy.bin` | Lowest measured-difficulty instances. |
| `benchmarks/splits/medium.bin` | Middle measured-difficulty instances. |
| `benchmarks/splits/hard.bin` | Highest measured-difficulty instances. |
| `benchmarks/splits/manifest.csv` | Original case indices and ranking metrics. |

Default split sizes are `34%` easy, `33%` medium, and the rest hard. You can override the first two fractions:

```bash
./build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp \
  packages/nonconvex-tpp/cpp/tests/test_cases_simplified2.bin \
  benchmarks/results.csv \
  benchmarks/splits \
  0.50 0.25
```

The splitter ranks instances by measured behavior from the CSV:

1. fully solved cases before capped or branch-limited cases;
2. fewer convex calls before more convex calls;
3. lower B&B runtime before higher runtime;
4. fewer decomposed pieces and lower branching as tie-breakers.

This means the difficulty split is tied to the benchmark configuration used to generate the CSV. If you change `max_calls_per_instance`, solver implementation, hardware, or the B&B policy, regenerate the CSV and split again.
