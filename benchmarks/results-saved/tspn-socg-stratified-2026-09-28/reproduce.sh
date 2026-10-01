#!/usr/bin/env bash
# Campaign reproduction only. Benchmark entry point remains benchmarks/tpp.py.
set -euo pipefail
campaign="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$campaign/../../.." && pwd)"
cd "$repo"
mkdir -p .build
output="$(mktemp -d "$repo/.build/tspn-socg-2026-09-28.XXXXXX")"
printf 'Validation and results: %s\n' "$output"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/private/tmp/tpp-python-cache}"
python_bin="${TPP_PYTHON:-/opt/homebrew/bin/python3}"
comparison=/private/tmp/tpp-tspn-comparison-build
vendor=/Users/gabriel/Documents/Scripts/TouringPolygons/third_party/tspn-socg
frozen_fekete=/private/tmp/tpp-tspn-baseline-b70cda3/tpp-fekete-cycle
if [[ ! -x "$frozen_fekete" ]]; then
    printf 'Missing frozen reference binary: %s\n' "$frozen_fekete" >&2
    exit 1
fi
git rev-parse HEAD > "$output/base-commit.txt"
git diff --binary -- packages benchmarks/_internal docs/algorithms DEVELOPMENT.md > "$output/source.patch"
for backend in ON OFF; do
    for test in cycle cycle_certificate; do
        if [[ "$backend" == ON ]]; then
            build="/private/tmp/tpp-tspn-gmp-${test//_/-}-tests"
            # Existing certificate build uses this shorter name.
            [[ "$test" != cycle_certificate ]] || build=/private/tmp/tpp-tspn-gmp-certificate-tests
        else
            build=/private/tmp/tpp-tspn-fallback-off-tests
        fi
        log="$output/${test}-${backend}.log"
        printf 'Building and testing %s, GMP=%s\n' "$test" "$backend"
        cmake -S packages/convex-tpp/cpp -B "$build" -DCMAKE_BUILD_TYPE=Release \
            -DTPP_ENABLE_GMP_RATIONAL="$backend" -DTARGET="main-${test}_tests" > "$log" 2>&1
        cmake --build "$build" -j 4 >> "$log" 2>&1
        "$build/tpp-convex" >> "$log" 2>&1
    done
done
printf 'Building and testing TSPN and endpoint B&B\n'
cmake -S benchmarks/_internal/tspn_native -B "$comparison" \
    -DFEKETE_SOURCE="$vendor" -DTPP_ENABLE_GMP_RATIONAL=ON -DTARGET=main-unordered \
    > "$output/tspn-build.log" 2>&1
cmake --build "$comparison" --target tpp-unordered tpp-tspn-tests tpp-unordered-tests -j 4 \
    >> "$output/tspn-build.log" 2>&1
"$comparison/touring_polygons/tpp-tspn-tests" > "$output/tspn-tests.log" 2>&1
"$comparison/touring_polygons/tpp-unordered-tests" > "$output/unordered-tests.log" 2>&1
for sample in screening holdout; do
    if [[ "$sample" == screening ]]; then
        inputs="$campaign/baseline/instances.json"
        repetitions=1
    else
        inputs="$campaign/holdout/instances.json"
        repetitions=3
    fi
    if ! "$python_bin" benchmarks/tpp.py tspn-benchmark --skip-build \
        --ours-binary "$comparison/touring_polygons/tpp-unordered" --fekete-binary "$frozen_fekete" \
        --fekete-source "$vendor" --build-dir "$comparison" --inputs "$inputs" \
        --output "$output/$sample" --seconds 2 --external-timeout 12 --repetitions "$repetitions" \
        > "$output/$sample.log" 2>&1; then
        [[ -f "$output/$sample/summary.json" ]] || { cat "$output/$sample.log"; exit 1; }
        printf '%s completed with censored, invalid or failed runs; inspect summary.json.\n' "$sample"
    fi
done
printf 'All native tests passed. Benchmark artifacts: %s\n' "$output"
