#!/usr/bin/env bash
set -euo pipefail

report_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$report_dir/../../.." && pwd)"
build_dir="$repo_dir/.build/directional-tpp-tutorial"
mkdir -p "$build_dir"

if command -v latexmk >/dev/null 2>&1; then
  (cd "$report_dir" && latexmk -pdf -interaction=nonstopmode -halt-on-error \
    -file-line-error -outdir="$build_dir" main.tex)
else
  (cd "$report_dir" && pdflatex -interaction=nonstopmode -halt-on-error \
    -file-line-error -output-directory="$build_dir" main.tex)
  (cd "$report_dir" && pdflatex -interaction=nonstopmode -halt-on-error \
    -file-line-error -output-directory="$build_dir" main.tex)
fi

cp "$build_dir/main.pdf" "$report_dir/directional-tpp-tutorial.pdf"
printf 'Built %s\n' "$report_dir/directional-tpp-tutorial.pdf"
