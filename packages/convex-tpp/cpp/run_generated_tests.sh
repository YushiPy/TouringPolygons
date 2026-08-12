#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
generator_build_dir="$repo_root/build/convex-generate-tests"
verifier_build_dir="$repo_root/build/convex-verify-tests"
generated_dir="$repo_root/build/convex-generated-tests/tests"

cmake -S "$script_dir" -B "$generator_build_dir" -DTARGET=main-generate_tests
cmake --build "$generator_build_dir"
"$generator_build_dir/tpp-convex" "$generated_dir"

cmake -S "$script_dir" -B "$verifier_build_dir" -DTARGET=main-verify_solutions
cmake --build "$verifier_build_dir"
TPP_TEST_DIR="$script_dir/tests" "$verifier_build_dir/tpp-convex"
TPP_TEST_DIR="$generated_dir" "$verifier_build_dir/tpp-convex"
