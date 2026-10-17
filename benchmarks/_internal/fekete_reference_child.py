"""One run of Fekete et al.'s unmodified solver, called as its experiment script calls it.

``fekete_reference.py`` executes this file with the Python of the upstream build,
so it imports only the standard library and the upstream package ``tspn_bnb2``.
``solve`` copies ``run_solve_with_order_pruning`` of
``evaluation/rq4_scalability/01_run_solver.py`` at the pinned commit: the same
call, arguments and ``time.time()`` measurement around it. Left out are only the
cluster scheduler (slurminade) and the result database (algbench) of that script.

The result goes to ``--output`` as JSON; the solver's own console output goes to
this process's stdout. ``--check`` only runs the warm-up and reports success.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import zipfile
from pathlib import Path


def solve(instance, *, root: str, threads: int, time_limit: int, eps: float) -> tuple[object, float, float]:
	"""The solver call of the paper's experiment script, timed the same way."""
	from tspn_bnb2.operations import solve_annotated_instance

	start_time = time.time()
	started = time.perf_counter()
	solution = solve_annotated_instance(
		instance,
		timelimit=time_limit,
		branching="FarthestPoly",
		search="DfsBfs",
		root=root,
		rules=[],
		node_simplification=False,
		MAX_TE_TOGGLE_COUNT=1,
		MAX_GEOM_TOGGLE_COUNT=1,
		num_threads=threads,
		decomposition_branch=True,
		eps=eps,
		callback=None,
		initial_solution=None,
	)
	end_time = time.time()
	return solution, end_time - start_time, time.perf_counter() - started


def warm_up() -> None:
	"""Start Gurobi before timing anything.

	The solver keeps one static Gurobi environment per process; the first SOCP
	creates it and checks the license. The paper's jobs ran several instances per
	process, so only their first instance paid that; here every run has its own
	process, and a three-square instance pays it outside the measurement.
	"""
	from tspn_bnb2.schemas import AnnotatedInstance

	squares = [f"POLYGON(({x} 0, {x + 1} 0, {x + 1} 1, {x} 1, {x} 0))" for x in (0, 3, 6)]
	solution, _, _ = solve(AnnotatedInstance(polygons=squares), root="LongestEdgePlusFurthestSite",
		threads=1, time_limit=10, eps=0.001)
	if not solution.is_optimal:
		raise RuntimeError("the warm-up instance did not close its gap")


def load_instance(archive: Path, name: str):
	from tspn_bnb2.schemas import AnnotatedInstance

	with zipfile.ZipFile(archive) as file, file.open(f"{name}.json") as member:
		return AnnotatedInstance.model_validate_json(member.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
	parser.add_argument("--archive", type=Path)
	parser.add_argument("--instance")
	parser.add_argument("--root")
	parser.add_argument("--threads", type=int)
	parser.add_argument("--time-limit", type=int)
	parser.add_argument("--eps", type=float)
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--check", action="store_true", help="only start Gurobi and solve the warm-up instance")
	args = parser.parse_args(argv)

	result: dict = {}
	try:
		warm_up()
		if not args.check:
			instance = load_instance(args.archive, args.instance)
			result.update(polygons=instance.num_polygons(), type=instance.derive_instance_type())
			solution, total_time, perf_seconds = solve(instance, root=args.root, threads=args.threads,
				time_limit=args.time_limit, eps=args.eps)
			result.update(solve_time=total_time, solve_time_perf=perf_seconds,
				solution=json.loads(solution.model_dump_json()))
		result["ok"] = True
	# The experiment script records RuntimeError and ValueError as failed runs;
	# anything else is reported too instead of losing the row.
	except Exception as error:  # noqa: BLE001
		result.update(ok=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(limit=8))
	sys.stdout.flush()
	temporary = args.output.with_name(args.output.name + ".tmp")
	temporary.write_text(json.dumps(result) + "\n")
	temporary.replace(args.output)
	return 0 if result["ok"] else 1


if __name__ == "__main__":
	raise SystemExit(main())
