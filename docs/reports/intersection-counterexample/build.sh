#!/usr/bin/env bash
set -euo pipefail

report_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$report_dir/../../.." && pwd)"
build_dir="$repo_dir/.build/intersection-report"
output_dir="$repo_dir/output/pdf"

if ! command -v latexmk >/dev/null 2>&1 && [[ -x /Library/TeX/texbin/latexmk ]]; then
  export PATH="/Library/TeX/texbin:$PATH"
fi

mkdir -p "$build_dir" "$output_dir"
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -jobname=intersection-counterexample -outdir="$build_dir" "$report_dir/report.tex"
cp "$build_dir/intersection-counterexample.pdf" "$output_dir/intersection-counterexample.pdf"
printf 'Built %s\n' "$output_dir/intersection-counterexample.pdf"
