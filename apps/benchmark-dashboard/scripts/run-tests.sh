#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${APP_ROOT}/../.." && pwd)"
PYTHON="${APP_ROOT}/.venv/bin/python3"
BROWSER_PORT="${BROWSER_PORT:-8018}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/touringpolygons-uv-cache}"
TMP_FILES=()

cleanup() {
	for file in "${TMP_FILES[@]+"${TMP_FILES[@]}"}"; do
		rm -f "${file}"
	done
}

trap cleanup EXIT

make_log() {
	local file
	file="$(mktemp)"
	TMP_FILES+=("${file}")
	printf '%s\n' "${file}"
}

extract_unittest_count() {
	sed -nE 's/^Ran ([0-9]+) tests? .*/\1/p' "$1" | tail -n 1
}

extract_node_test_count() {
	sed -nE 's/^.*tests ([0-9]+)$/\1/p' "$1" | tail -n 1
}

cd "${APP_ROOT}"

python_unittest_log="$(make_log)"
frontend_utils_log="$(make_log)"
camera_log="$(make_log)"
browser_status="skipped"

uv run ruff check .
"${PYTHON}" -m py_compile main.py dashboard/*.py tests/*.py
"${PYTHON}" -m unittest discover -s tests 2>&1 | tee "${python_unittest_log}"
npm run check:js
npm run lint:js
node --test tests/frontend-utils.test.mjs | tee "${frontend_utils_log}"
node --test tests/manual-editor-camera.test.mjs | tee "${camera_log}"

if [[ "${RUN_BROWSER:-0}" == "1" ]]; then
	uv run uvicorn main:app --host 127.0.0.1 --port "${BROWSER_PORT}" &
	server_pid="$!"
	trap 'kill "${server_pid}" 2>/dev/null || true; cleanup' EXIT
	sleep 1
	DASHBOARD_URL="http://127.0.0.1:${BROWSER_PORT}" npm run test:browser
	kill "${server_pid}" 2>/dev/null || true
	trap cleanup EXIT
	browser_status="1 smoke test"
fi

cd "${REPO_ROOT}"
cmake --preset convex-release -DTARGET=main-intersection_tests
cmake --build --preset convex-release
./build/convex-release/packages/convex-tpp/cpp/tpp-convex
git diff --check

python_unittest_count="$(extract_unittest_count "${python_unittest_log}")"
frontend_utils_count="$(extract_node_test_count "${frontend_utils_log}")"
camera_count="$(extract_node_test_count "${camera_log}")"

printf '\n'
printf 'Test summary: all checks passed.\n'
printf '  Python lint: ruff check\n'
printf '  Python compile: main.py, dashboard/*.py, tests/*.py\n'
printf '  Python unittest: %s tests\n' "${python_unittest_count:-unknown}"
printf '  JavaScript syntax: static/*.js\n'
printf '  JavaScript lint: eslint static tests\n'
printf '  Frontend utility tests: %s tests\n' "${frontend_utils_count:-unknown}"
printf '  Manual editor camera tests: %s tests\n' "${camera_count:-unknown}"
printf '  Browser smoke: %s\n' "${browser_status}"
printf '  Native convex regression: main-intersection_tests\n'
printf '  Diff hygiene: git diff --check\n'
