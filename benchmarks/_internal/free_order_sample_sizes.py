"""Run nested random polygon subsets of a free-order TPP instance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from unordered_runner import run_unordered_solver


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "benchmarks/campaigns/usp-campus-85/instance.txt"
DEFAULT_SIZES = (10, 20, 30, 40, 50, 60, 70, 80)
SUMMARY_FIELDS = (
	"sample_size",
	"status",
	"termination",
	"exact",
	"calls",
	"solver_seconds",
	"wall_seconds",
	"lower_bound",
	"upper_bound",
	"gap",
	"nodes",
	"peak_queue",
	"fallback_calls",
	"sample_indices",
	"sample_fids",
	"sample_ids",
	"error",
)


def sha256_file(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def parse_sizes(value: str) -> tuple[int, ...]:
	try:
		sizes = tuple(int(piece.strip()) for piece in value.split(","))
	except ValueError as error:
		raise argparse.ArgumentTypeError("use tamanhos inteiros separados por vírgula") from error
	if not sizes or any(size <= 0 for size in sizes):
		raise argparse.ArgumentTypeError("todos os tamanhos precisam ser positivos")
	if tuple(sorted(set(sizes))) != sizes:
		raise argparse.ArgumentTypeError("os tamanhos precisam ser únicos e estar em ordem crescente")
	return sizes


def parse_positive_int(value: str) -> int:
	try:
		parsed = int(value)
	except ValueError as error:
		raise argparse.ArgumentTypeError("use um inteiro positivo") from error
	if parsed <= 0:
		raise argparse.ArgumentTypeError("use um inteiro positivo")
	return parsed


def parse_positive_float(value: str) -> float:
	try:
		parsed = float(value)
	except ValueError as error:
		raise argparse.ArgumentTypeError("use um número positivo") from error
	if not math.isfinite(parsed) or parsed <= 0:
		raise argparse.ArgumentTypeError("use um número positivo e finito")
	return parsed


def read_text_instance(path: Path) -> tuple[list[float], list[float], list[list[list[float]]], dict[str, Any]]:
	tokens = path.read_text().split()
	if len(tokens) < 7:
		raise ValueError("cabeçalho incompleto: esperava início, fim, quantidade e limites")
	try:
		start = [float(tokens[0]), float(tokens[1])]
		target = [float(tokens[2]), float(tokens[3])]
		polygon_count = int(tokens[4])
		embedded_max_calls = int(tokens[5])
		embedded_max_seconds = float(tokens[6])
	except ValueError as error:
		raise ValueError("cabeçalho inválido na instância TPP") from error
	if any(not math.isfinite(value) for value in (*start, *target, embedded_max_seconds)):
		raise ValueError("início, fim e limite de tempo precisam ser finitos")
	if polygon_count <= 0 or embedded_max_calls <= 0 or embedded_max_seconds <= 0:
		raise ValueError("a instância precisa ter polígonos e limites positivos")

	polygons: list[list[list[float]]] = []
	position = 7
	for polygon_index in range(polygon_count):
		if position >= len(tokens):
			raise ValueError(f"faltam os vértices do polígono {polygon_index}")
		try:
			vertex_count = int(tokens[position])
		except ValueError as error:
				raise ValueError(f"número de vértices inválido no polígono {polygon_index}") from error
		position += 1
		if vertex_count < 3:
			raise ValueError(f"o polígono {polygon_index} tem menos de 3 vértices")
		needed = 2 * vertex_count
		if position + needed > len(tokens):
			raise ValueError(f"faltam coordenadas no polígono {polygon_index}")
		try:
			values = [float(token) for token in tokens[position:position + needed]]
		except ValueError as error:
			raise ValueError(f"coordenada inválida no polígono {polygon_index}") from error
		if any(not math.isfinite(value) for value in values):
			raise ValueError(f"coordenada não finita no polígono {polygon_index}")
		polygons.append([[values[i], values[i + 1]] for i in range(0, needed, 2)])
		position += needed
	if position != len(tokens):
		raise ValueError(f"a instância contém {len(tokens) - position} valores inesperados no final")
	metadata = {
		"embedded_max_calls": embedded_max_calls,
		"embedded_max_seconds": embedded_max_seconds,
	}
	return start, target, polygons, metadata


def load_feature_manifest(input_path: Path, polygon_count: int) -> list[dict[str, Any]]:
	manifest_path = input_path.with_name("manifest.json")
	if not manifest_path.exists():
		return [{"index": index, "fid": None, "id": None} for index in range(polygon_count)]
	try:
		data = json.loads(manifest_path.read_text())
		features = data["polygons"]
	except (json.JSONDecodeError, KeyError, TypeError) as error:
		raise ValueError(f"manifesto inválido: {manifest_path}") from error
	if len(features) != polygon_count:
		raise ValueError(f"o manifesto tem {len(features)} polígonos, mas a instância tem {polygon_count}")
	for index, feature in enumerate(features):
		if feature.get("index") != index:
			raise ValueError("os índices no manifesto não correspondem à ordem dos polígonos")
	return features


def config_signature(config: dict[str, Any]) -> str:
	encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
	return hashlib.sha256(encoded).hexdigest()


def write_summary(output_dir: Path) -> None:
	runs_dir = output_dir / "runs"
	rows = []
	for path in sorted(runs_dir.glob("n-*.json")):
		rows.append(json.loads(path.read_text()))
	rows.sort(key=lambda row: row["sample_size"])
	with (output_dir / "summary.csv").open("w", newline="") as stream:
		writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
		writer.writeheader()
		for row in rows:
			writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Executa o solver de ordem livre em amostras aleatórias cumulativas. "
			"Uma única permutação aleatória define todas as amostras."
		)
	)
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"instância TPP em texto (padrão: {DEFAULT_INPUT})")
	parser.add_argument("--solver", type=Path, default=REPO_ROOT / ".build/unordered/tpp", help="executável C++ de ordem livre")
	parser.add_argument("--output", type=Path, help="pasta local de saída; por padrão, benchmarks/results/<instância>-sample-sizes-seed-<seed>")
	parser.add_argument("--sizes", type=parse_sizes, default=DEFAULT_SIZES, help="tamanhos crescentes separados por vírgula (padrão: 10,20,30,40,50,60,70,80)")
	parser.add_argument("--seed", type=int, default=1, help="semente da permutação aleatória (padrão: 1)")
	parser.add_argument("--seconds", type=parse_positive_float, default=5.0, help="limite de tempo por tamanho, em segundos (padrão: 5)")
	parser.add_argument("--max-calls", type=parse_positive_int, default=100_000, help="limite de chamadas ao oráculo por tamanho (padrão: 100000)")
	parser.add_argument("--resume", action="store_true", help="retoma resultados concluídos na mesma pasta se a configuração for idêntica")
	return parser


def main(argv: list[str] | None = None) -> int:
	parser = make_parser()
	args = parser.parse_args(argv)
	input_path = args.input.expanduser().resolve()
	solver_path = args.solver.expanduser().resolve()
	if not input_path.is_file():
		parser.error(f"arquivo de instância não encontrado: {input_path}")
	if not solver_path.is_file():
		parser.error(f"solver não encontrado: {solver_path}; compile-o antes de rodar")

	try:
		start, target, polygons, embedded_limits = read_text_instance(input_path)
		features = load_feature_manifest(input_path, len(polygons))
	except (OSError, ValueError) as error:
		parser.error(str(error))

	if args.sizes[-1] > len(polygons):
		parser.error(f"a maior amostra solicitada ({args.sizes[-1]}) excede os {len(polygons)} polígonos da instância")

	output_dir = args.output.expanduser().resolve() if args.output else (
		REPO_ROOT / "benchmarks/results" / f"{input_path.stem}-sample-sizes-seed-{args.seed}"
	)
	config = {
		"input_path": str(input_path),
		"input_sha256": sha256_file(input_path),
		"solver_path": str(solver_path),
		"solver_sha256": sha256_file(solver_path),
		"polygon_count": len(polygons),
		"sizes": list(args.sizes),
		"seed": args.seed,
		"sampling": "one random permutation; each sample is its nested prefix",
		"max_calls_per_run": args.max_calls,
		"max_seconds_per_run": args.seconds,
		"embedded_input_limits": embedded_limits,
	}
	signature = config_signature(config)
	config_path = output_dir / "campaign.json"
	if output_dir.exists():
		if not args.resume:
			parser.error(f"a pasta de saída já existe: {output_dir}; use outra pasta ou passe --resume")
		if not config_path.is_file():
			parser.error(f"não encontrei a configuração da campanha para retomar: {config_path}")
		existing = json.loads(config_path.read_text())
		if existing.get("config_signature") != signature:
			parser.error("a configuração atual difere da campanha existente; use outra pasta de saída")
	else:
		output_dir.mkdir(parents=True)
		(output_dir / "runs").mkdir()
		(config_path).write_text(json.dumps({**config, "config_signature": signature}, indent=2) + "\n")
		permutation = random.Random(args.seed).sample(range(len(polygons)), len(polygons))
		plan = {
			"seed": args.seed,
			"permutation": permutation,
			"samples": [
				{
					"size": size,
					"features": [features[index] for index in permutation[:size]],
				}
				for size in args.sizes
			],
		}
		(output_dir / "sample-plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")

	plan = json.loads((output_dir / "sample-plan.json").read_text())
	if plan.get("seed") != args.seed or len(plan.get("permutation", [])) != len(polygons):
		parser.error("o plano de amostragem existente não corresponde à configuração")
	permutation = plan["permutation"]
	runs_dir = output_dir / "runs"

	print(f"Instância: {input_path} ({len(polygons)} polígonos)", flush=True)
	print(f"Amostragem: prefixos cumulativos; seed={args.seed}; limite={args.seconds:g}s ou {args.max_calls} chamadas por tamanho", flush=True)
	print(f"Saída: {output_dir}", flush=True)

	for run_index, size in enumerate(args.sizes, start=1):
		run_path = runs_dir / f"n-{size:03}.json"
		selected_indices = permutation[:size]
		if run_path.exists():
			if not args.resume:
				parser.error(f"resultado já existe: {run_path}; use --resume")
			previous = json.loads(run_path.read_text())
			if previous.get("config_signature") != signature or previous.get("sample_indices") != selected_indices:
				parser.error(f"resultado existente não corresponde à amostra atual: {run_path}")
			print(f"[{run_index}/{len(args.sizes)}] n={size}: já concluído ({previous.get('termination') or previous.get('status')})", flush=True)
			continue

		sample_polygons = [polygons[index] for index in selected_indices]
		selected_features = [features[index] for index in selected_indices]
		print(f"[{run_index}/{len(args.sizes)}] n={size}: iniciando...", flush=True)
		started = time.perf_counter()
		try:
			result = run_unordered_solver(
				solver_path,
				start,
				target,
				sample_polygons,
				args.max_calls,
				args.seconds,
			)
			status = "completed"
			error_text = ""
		except subprocess.TimeoutExpired as error:
			result = {}
			status = "runner_timeout"
			error_text = f"process exceeded external timeout ({error.timeout}s)"
		except (RuntimeError, json.JSONDecodeError, OSError) as error:
			result = {}
			status = "solver_error"
			error_text = str(error)
		wall_seconds = time.perf_counter() - started
		lower = result.get("lower_bound")
		upper = result.get("upper_bound")
		row = {
			"sample_size": size,
			"status": status,
			"termination": result.get("termination", ""),
			"exact": result.get("exact", ""),
			"calls": result.get("calls", ""),
			"solver_seconds": result.get("seconds", ""),
			"wall_seconds": round(wall_seconds, 6),
			"lower_bound": lower if lower is not None else "",
			"upper_bound": upper if upper is not None else "",
			"gap": upper - lower if isinstance(lower, (int, float)) and isinstance(upper, (int, float)) else "",
			"nodes": result.get("nodes", ""),
			"peak_queue": result.get("peak_queue", ""),
			"fallback_calls": result.get("fallback_calls", ""),
			"sample_indices": json.dumps(selected_indices),
			"sample_fids": json.dumps([feature.get("fid") for feature in selected_features]),
			"sample_ids": json.dumps([feature.get("id") for feature in selected_features], ensure_ascii=False),
			"error": error_text,
			"config_signature": signature,
		}
		run_path.write_text(json.dumps({**row, "solver_result": result}, ensure_ascii=False, indent=2) + "\n")
		write_summary(output_dir)
		if status == "completed":
			print(
				f"[{run_index}/{len(args.sizes)}] n={size}: {result.get('termination')} | "
				f"{result.get('seconds', wall_seconds):.3f}s | calls={result.get('calls')} | exact={result.get('exact')}",
				flush=True,
			)
		else:
			print(f"[{run_index}/{len(args.sizes)}] n={size}: {status}: {error_text}", flush=True)

	write_summary(output_dir)
	print(f"Resumo: {output_dir / 'summary.csv'}", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
