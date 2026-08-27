#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
app_dir="$(cd "$script_dir/.." && pwd)"
repo_root="$(cd "$app_dir/../.." && pwd)"
output_dir="$app_dir/static/wasm"

mkdir -p "$output_dir"

export EM_CONFIG="${EM_CONFIG:-/tmp/tpp-emscripten-config}"

if [[ -z "${EMSDK_PYTHON:-}" && -x /opt/homebrew/bin/python3 ]]; then
	export EMSDK_PYTHON=/opt/homebrew/bin/python3
fi

if [[ -z "${BINARYEN_ROOT:-}" && -x /opt/homebrew/opt/emscripten/libexec/binaryen/bin/wasm-opt ]]; then
	export BINARYEN_ROOT=/opt/homebrew/opt/emscripten/libexec/binaryen
fi

if [[ -z "${LLVM_ROOT:-}" && -x /opt/homebrew/opt/emscripten/libexec/llvm/bin/clang++ ]]; then
	export LLVM_ROOT=/opt/homebrew/opt/emscripten/libexec/llvm
fi

if [[ -d /opt/homebrew/opt/emscripten/libexec/binaryen/bin ]]; then
	PATH="/opt/homebrew/opt/emscripten/libexec/binaryen/bin:$PATH"
fi

if [[ -d /opt/homebrew/opt/emscripten/libexec/llvm/bin ]]; then
	PATH="/opt/homebrew/opt/emscripten/libexec/llvm/bin:$PATH"
fi

export EM_CACHE="${EM_CACHE:-$app_dir/.emscripten-cache/cache}"
mkdir -p "$EM_CACHE"
ln -sfn /opt "$(dirname "$EM_CACHE")/opt"

if [[ ! -f "$EM_CONFIG" || "${TPP_REWRITE_EM_CONFIG:-0}" == "1" ]]; then
	cat > "$EM_CONFIG" <<'EOF'
LLVM_ROOT = '/opt/homebrew/opt/emscripten/libexec/llvm/bin'
BINARYEN_ROOT = '/opt/homebrew/opt/emscripten/libexec/binaryen'
EMSCRIPTEN_ROOT = '/opt/homebrew/opt/emscripten/libexec'
NODE_JS = '/opt/homebrew/bin/node'
EOF
fi

cd /

em++ \
	--cache "$EM_CACHE" \
	-std=c++23 \
	-O3 \
	-fexceptions \
	-I"$repo_root/packages/common-geometry/cpp/include" \
	-I"$repo_root/packages/nonconvex-tpp/cpp/include" \
	-I"$repo_root/packages/convex-tpp/cpp/include" \
	-I"$repo_root/packages/optimal-convex-partition/cpp/include" \
	"$script_dir/tpp_convex_wasm.cpp" \
	"$repo_root/packages/common-geometry/cpp/src/common.cpp" \
	"$repo_root/packages/convex-tpp/cpp/src/core/solution.cpp" \
	"$repo_root/packages/convex-tpp/cpp/src/solvers/binary_search.cpp" \
	"$repo_root/packages/convex-tpp/cpp/src/solvers/linear_search.cpp" \
	"$repo_root/packages/convex-tpp/cpp/src/solvers/tan_jiang.cpp" \
	"$repo_root/packages/nonconvex-tpp/cpp/src/common.cpp" \
	"$repo_root/packages/optimal-convex-partition/cpp/src/optimal_convex_partition.cpp" \
	-sMODULARIZE=1 \
	-sEXPORT_ES6=1 \
	-sENVIRONMENT=web,node \
	-sALLOW_MEMORY_GROWTH=1 \
	-sASSERTIONS=0 \
	-sDISABLE_EXCEPTION_CATCHING=0 \
	-sMIN_WEBGL_VERSION=0 \
	-sMAX_WEBGL_VERSION=0 \
	-sEXPORTED_FUNCTIONS='["_malloc","_free","_tpp_solve","_tpp_solve_convex","_tpp_get_path_points","_tpp_solution_exact","_tpp_solution_calls","_tpp_solution_seconds"]' \
	-sEXPORTED_RUNTIME_METHODS='["HEAPF64","HEAP32"]' \
	-o "$output_dir/tpp_convex_wasm.js"
