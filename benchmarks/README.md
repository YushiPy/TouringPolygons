# Benchmarking

This folder is for benchmark outputs, split test sets, and notes about how to interpret them.

The current benchmark focuses on the non-convex branch-and-bound solver. Each B&B node calls the convex TPP solver on either a lower-bound instance or a leaf instance, so the main cost drivers are the number of convex calls and the cost per convex call.

## Recommended Workflow

`tpp.py` is the main entry point for instance generation and benchmarking. A generated parameter sweep is stored as a campaign so its inputs, generation settings, benchmark results, and progress stay together.

Activate the project virtual environment before generating OSM instances so `osmium` and `shapely` are available:

```bash
source .venv/bin/activate
```

Create a campaign:

```bash
python3 benchmarks/tpp.py generate-matrix sao-paulo \
  packages/instance-generation/regions/sao-paulo.osm.pbf \
  --instances 100 \
  --sample-size 40 \
  --seed 42
```

This creates:

```text
benchmarks/campaigns/sao-paulo/
├── campaign.json
├── inputs/
└── results/              # created when the benchmark starts
```

Run or resume all campaign inputs, then inspect progress:

```bash
python3 benchmarks/tpp.py run sao-paulo --max-calls 1000000 --timeout 3600
python3 benchmarks/tpp.py status sao-paulo
```

Each input receives a CSV, Markdown summary, and log under `results/`. `results/run-index.csv` tracks pending, completed, failed, timed-out, and interrupted files. Completed files are skipped when a run resumes. The benchmark parameters and status totals are also appended to `campaign.json`.

Generate one standalone binary with the same entry point:

```bash
python3 benchmarks/tpp.py generate \
  packages/instance-generation/regions/sao-paulo.osm.pbf \
  --output-bin /tmp/sao-paulo.bin \
  --instances 100 \
  --polygons-per-instance 20 \
  --no-preview \
  --no-manifest
```

Run `python3 benchmarks/tpp.py --help` to see all workflow commands. The older scripts remain available as compatible lower-level entry points.

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
| `Polygons` | Distribution of polygon counts across benchmarked runs. Useful when comparing generated workloads with different nominal sizes. |
| `Decomposed pieces` | Total convex pieces after decomposing all non-convex polygons in an instance. |
| `Pieces per polygon` | Average decomposition branching density per polygon. |
| `log2(total combinations)` | Base-2 logarithm of the product of per-polygon decomposition piece counts. This is easier to compare than raw combination counts when branch spaces are huge. |
| `Initial gap %` | Relative improvement from the initial approximation to the final result. |
| `Incumbent gap %` | Relative improvement from the first selected-piece incumbent to the final result. |
| `Failed-prune bound/incumbent` | How close non-pruning lower bounds are to the incumbent. Values near `1.0` mean proving optimality is hard. |
| `Measured work` | Sum of per-instance measured times. In parallel runs this can exceed wall-clock time. |
| `Wall-clock total` | Actual elapsed time. |
| `Parallel speedup estimate` | `measured work / wall-clock total`. |

If `Initial gap %` is tiny but calls are high, the solver is finding good solutions quickly but spending time proving optimality.

## Split A Test Set By Difficulty

First run a benchmark and keep its CSV. Then split the original binary test set using the measured CSV:

```bash
python3 benchmarks/bench.py split \
  --input packages/nonconvex-tpp/cpp/tests/test_cases_simplified2.bin \
  --csv benchmarks/results.csv \
  --output benchmarks/splits
```

This builds and runs the C++ splitter, then restores the benchmark target. To run the splitter directly:

```bash
cmake --preset nonconvex-release -DTARGET=main-split_benchmark_cases
cmake --build --preset nonconvex-release

./build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp \
  packages/nonconvex-tpp/cpp/tests/test_cases_simplified2.bin \
  benchmarks/results.csv \
  benchmarks/splits
```

This writes:

| File | Contents |
|---|---|
| `benchmarks/splits/under_1ms.bin` | Cases with measured B&B time under `1ms`. |
| `benchmarks/splits/under_10ms.bin` | Cases with measured B&B time from `1ms` to under `10ms`. |
| `benchmarks/splits/under_100ms.bin` | Cases with measured B&B time from `10ms` to under `100ms`. |
| `benchmarks/splits/under_1s.bin` | Cases with measured B&B time from `100ms` to under `1s`. |
| `benchmarks/splits/under_10s.bin` | Cases with measured B&B time from `1s` to under `10s`. |
| `benchmarks/splits/over_10s_or_capped.bin` | Cases that took at least `10s`, hit the call cap, or were branch-limited. |
| `benchmarks/splits/manifest.csv` | Original case indices and measured metrics for every split case. |
| `benchmarks/splits/instances.json` | Machine-readable index used by `bench.py`. |

Inside each bucket, cases are ordered from easier to harder using calls, B&B runtime, decomposed pieces, branching, and original case index as tie-breakers.

This means the difficulty split is tied to the benchmark configuration used to generate the CSV. If you change `max_calls_per_instance`, solver implementation, hardware, or the B&B policy, regenerate the CSV and split again.

## Use The Helper

List available groups:

```bash
python3 benchmarks/bench.py list --index benchmarks/splits/instances.json
```

Run all groups whose bucket upper bound is at most `1s`:

```bash
python3 benchmarks/bench.py run \
  --index benchmarks/splits/instances.json \
  --max-time 1s
```

By default, selected groups are concatenated into one temporary `.bin` file and benchmarked in a single run. This produces one CSV and one markdown summary for the whole selected workload.

Run one explicit group:

```bash
python3 benchmarks/bench.py run \
  --index benchmarks/splits/instances.json \
  --group under_100ms
```

Useful run options:

| Option | Meaning |
|---|---|
| `--group NAME` | Run a specific group. May be passed more than once. |
| `--max-time 1s` | Run groups whose JSON `upper_seconds` is at most this limit. |
| `--include-overflow` | Include `over_10s_or_capped` when using `--max-time`. |
| `--name NAME` | Output basename for the combined run. |
| `--separate-groups` | Run each selected group separately instead of concatenating them. |
| `--no-build` | Skip CMake configure/build and use the existing binary. |
| `--threads N` | Set `TPP_BENCH_THREADS=N` for this run. |
| `--max-calls N` | Override the per-instance convex call cap. |
| `--max-instances N` | Limit instances from the selected input. Useful for smoke tests. |

Run outputs are written under `benchmarks/runs/<timestamp>/`. Combined runs write:

| File | Meaning |
|---|---|
| `<name>.bin` | Concatenated temporary benchmark input. |
| `<name>.csv` | Per-instance benchmark rows. |
| `<name>.md` | Single summary for the whole selected workload. |
| `<name>.groups.json` | Names of the groups included in the combined input. |

The helper only reconfigures CMake when the configured target is different from the required target. It still runs the incremental build by default so source changes are not missed. Use `--no-build` only when you know the binary is already current.

## Run Generated Test Sets

For existing generated directories that predate campaigns, run every `.bin` file directly:

```bash
python3 benchmarks/run_generated.py
```

Results are written to `benchmarks/generated-runs/`, with one CSV, Markdown summary, and log per input file. `run-index.csv` records the status and elapsed time of the whole batch. Completed inputs are skipped when the command is run again; pass `--force` to rerun them.

Common controls:

```bash
python3 benchmarks/run_generated.py \
  --input packages/nonconvex-tpp/cpp/tests/generated/sao-paulo \
  --threads 1 \
  --max-calls 1000000 \
  --timeout 3600
```

Use `--dry-run` to list the selected files without building or running them, and `--pattern 'p20_*.bin'` to run only matching generated configurations.
