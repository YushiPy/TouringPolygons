#!/usr/bin/env python3
"""Recover source labels and classify polygon contacts for this campaign.

The source archive is sorted in the same order as instances.bin. Before using
that ordering, this script verifies every case by an exact, order-independent
polygon-coordinate fingerprint. Polygon contact statistics are computed by
the companion Boost.Geometry program.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import shlex
import struct
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CAMPAIGN = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "third_party/tspn-socg/instances/instances_socg_simplified.zip"
NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def canonical_ring(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
	if points and points[0] == points[-1]:
		points = points[:-1]
	if len(points) < 3:
		raise ValueError("polygon has fewer than three distinct vertices")
	forward = points
	backward = list(reversed(points))
	rotations = []
	for sequence in (forward, backward):
		minimum = min(range(len(sequence)), key=sequence.__getitem__)
		rotations.append(tuple(sequence[minimum:] + sequence[:minimum]))
	return min(rotations)


def parse_wkt_polygon(wkt: str) -> tuple[tuple[float, float], ...]:
	if not wkt.startswith("POLYGON ((") or not wkt.endswith("))") or "), (" in wkt:
		raise ValueError(f"expected a simple exterior-only polygon WKT, got {wkt[:80]!r}")
	values = [float(value) for value in NUMBER.findall(wkt)]
	if len(values) % 2:
		raise ValueError("odd number of polygon coordinates")
	return canonical_ring(list(zip(values[::2], values[1::2])))


def read_native_cases(path: Path) -> tuple[list[dict], list[str]]:
	data = path.read_bytes()
	offset = 0
	cases: list[dict] = []
	hashes: list[str] = []
	while offset < len(data):
		start = offset
		offset += 32  # start and target points
		polygon_count = struct.unpack_from("<Q", data, offset)[0]
		offset += 8
		polygons = []
		for _ in range(polygon_count):
			vertex_count = struct.unpack_from("<Q", data, offset)[0]
			offset += 8
			points = [struct.unpack_from("<dd", data, offset + i * 16) for i in range(vertex_count)]
			polygons.append(canonical_ring(points))
			offset += vertex_count * 16
		solution_count = struct.unpack_from("<Q", data, offset)[0]
		offset += 8 + solution_count * 16
		hashes.append(hashlib.sha256(data[start:offset]).hexdigest())
		cases.append({"polygons": polygons})
	return cases, hashes


def source_label(metadata: dict, archive_entry: str) -> tuple[str, str]:
	if metadata.get("source") == "random":
		return "random", Path(archive_entry).stem
	if "geo_information" in metadata:
		return "OSM", str(metadata["source_file"])
	if metadata.get("source") == "public_instance_set":
		return "tessellation", str(metadata.get("instance_uid", Path(archive_entry).stem))
	raise ValueError(f"unrecognized source metadata: {metadata}")


def load_archive(archive: Path) -> list[dict]:
	with zipfile.ZipFile(archive) as source_zip:
		entries = []
		for name in sorted(path for path in source_zip.namelist() if path.endswith(".json")):
			payload = json.loads(source_zip.read(name))
			metadata = payload["meta"]
			source_type, source_name = source_label(metadata, name)
			polygons = [parse_wkt_polygon(wkt) for wkt in payload["polygons"]]
			entries.append({
				"archive_entry": name,
				"source_type": source_type,
				"source_name": source_name,
				"polygons": polygons,
				"wkts": payload["polygons"],
			})
	return entries


def canonical_instance(polygons: list[tuple]) -> tuple:
	return tuple(sorted(polygons))


def run_geometry_classifier(source_entries: list[dict], compiler: str) -> list[dict]:
	tool_dir = Path(__file__).resolve().parent
	cpp_source = tool_dir / "classify_geometry.cpp"
	with tempfile.TemporaryDirectory(prefix="german-classification-") as temp_dir:
		temp = Path(temp_dir)
		input_path = temp / "polygons.tsv"
		binary_path = temp / "classify_geometry"
		with input_path.open("w", encoding="utf-8") as output:
			for case_index, entry in enumerate(source_entries):
				for polygon_index, wkt in enumerate(entry["wkts"]):
					output.write(f"{case_index}|{entry['source_type']}|{entry['source_name']}|{polygon_index}|{wkt}\n")
		compile_command = shlex.split(compiler) + ["-std=c++17", "-O2"]
		homebrew_boost = Path("/opt/homebrew/include/boost/geometry.hpp")
		if homebrew_boost.exists():
			compile_command += ["-I/opt/homebrew/include"]
		compile_command += [str(cpp_source), "-o", str(binary_path)]
		compiled = subprocess.run(compile_command, capture_output=True, text=True)
		if compiled.returncode:
			raise RuntimeError(f"Boost.Geometry compilation failed:\n{compiled.stderr}")
		result = subprocess.run([str(binary_path), str(input_path)], check=True, capture_output=True, text=True)
	return list(csv.DictReader(result.stdout.splitlines()))


def read_csv(path: Path) -> dict[int, dict[str, str]]:
	with path.open(newline="", encoding="utf-8") as source:
		return {int(row["case_index"]): row for row in csv.DictReader(source)}


def geometric_mean(rows: list[dict]) -> float | None:
	values = [float(row["speedup_fekete_over_ours"]) for row in rows if row["fekete_completed"] == "True"]
	return math.exp(statistics.fmean(math.log(value) for value in values)) if values else None


def paired(rows: list[dict]) -> list[dict]:
	return [row for row in rows if row["fekete_completed"] == "True"]


def summary_cells(rows: list[dict]) -> tuple[int, str, str, str, str]:
	values = [float(row["speedup_fekete_over_ours"]) for row in rows]
	if not values:
		return 0, "—", "—", "—", "—"
	return (
		len(values),
		f"{math.exp(statistics.fmean(math.log(value) for value in values)):.3f}×",
		f"{statistics.median(values):.3f}×",
		f"{statistics.fmean(values):.3f}×",
		f"{sum(value > 1 for value in values)}/{len(values)}",
	)


def source_report(rows: list[dict], archive: Path) -> str:
	all_rows = rows
	common = paired(rows)
	submodule_root = archive.resolve().parents[1]
	commit_result = subprocess.run(
		["git", "-C", str(submodule_root), "rev-parse", "HEAD"],
		capture_output=True, text=True,
	)
	submodule_commit = commit_result.stdout.strip() if commit_result.returncode == 0 else "indisponível"
	lines = [
		"# Análise estratificada — German comparison",
		"",
		"## Escopo e rótulos",
		"",
		"A análise de runtime usa apenas as **550 instâncias concluídas por ambos**; as 8 sem resultado do Fekete foram excluídas, como na comparação anterior. As oito são OSM, todas com 60 polígonos. `speedup = runtime(Fekete) / runtime(nosso)`, então valores acima de 1 favorecem nosso solver.",
		"",
		"Os 558 rótulos foram recuperados do ZIP simplificado do submódulo e associados ao `instances.bin` por uma impressão digital exata das coordenadas: sem depender da ordem dos polígonos, do vértice inicial ou da orientação. A verificação passou para 558/558 instâncias. `random` vem do metadado `source=random`; `tessellation` vem de `source=public_instance_set`; `OSM` vem da presença de `geo_information`.",
		"",
		"A recuperação encontrou 320 OSM, 160 aleatórias e 78 tessellations, exatamente os totais descritos na Seção 4.2 do artigo (`docs/bibliography/TSPN-B&B-Michael/original.pdf`). O artigo diz que as tessellations são regiões de Voronoi derivadas de conjuntos CG:SHOP, TSPLIB e Salzburg; os casos OSM são pegadas de edifícios de 20 cidades. A tabela de runtime abaixo usa somente os pares comuns concluídos.",
		"",
		"## Runtime por fonte",
		"",
		"A média geométrica é a métrica principal para comparar razões multiplicativas. A média aritmética aparece como complemento e é mais afetada pelos grandes speedups. `Vitórias` conta instâncias em que nosso runtime foi menor.",
		"",
		"| Fonte | Corpus | Casos pareados | Sem Fekete | Speedup geométrico | Mediana | Média aritmética | Vitórias do nosso |",
		"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
	]
	for source in ("OSM", "random", "tessellation"):
		group = [row for row in common if row["source_type"] == source]
		missing = sum(row["source_type"] == source and row["fekete_completed"] != "True" for row in all_rows)
		total = sum(row["source_type"] == source for row in all_rows)
		n, geo, median, mean, wins = summary_cells(group)
		lines.append(f"| {source} | {total} | {n} | {missing} | {geo} | {median} | {mean} | {wins} |")
	lines += [
		"",
		"A separação por fonte muda a leitura do efeito do tamanho. Em especial, a coorte tessellation/Voronoi passa a favorecer Fekete nas faixas maiores, enquanto OSM segue favorecendo nosso solver; a tendência agregada mistura esses perfis.",
		"",
		"## Speedup geométrico por fonte e tamanho",
		"",
		"`n` é o número de pares concluídos na célula. Células vazias não têm instâncias.",
		"",
		"| Faixa de polígonos | OSM | Aleatórias | Voronoi/tessellation |",
		"| --- | ---: | ---: | ---: |",
	]
	bins = ((4, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60))
	for low, high in bins:
		cells = []
		for source in ("OSM", "random", "tessellation"):
			group = [row for row in common if row["source_type"] == source and low <= int(row["polygons"]) <= high]
			gm = geometric_mean(group)
			cells.append(f"{gm:.3f}× (n={len(group)})" if gm is not None else "—")
		lines.append(f"| {low}–{high} | " + " | ".join(cells) + " |")
	lines += [
		"",
		"Por fonte, OSM fica relativamente estável depois da primeira faixa; aleatórias caem até 41–50 polígonos e têm leve recuperação em 51–60; tessellation cai de 6.80× para abaixo de 1× nas faixas 31–40, 41–50 e 51–60. Portanto, a impressão de que o speedup cresce com o número de polígonos não é geral: ela depende fortemente da origem da instância. As oito falhas do Fekete em OSM com 60 polígonos não entram no speedup, então a última faixa OSM representa apenas os 32 casos concluídos.",
		"",
		"## Relação entre sobreposição e speedup",
		"",
		"A classificação usa a geometria simplificada efetivamente presente na campanha. `interior_disjoint` permite compartilhamento de arestas ou vértices, desde que não haja interseção de área positiva; `strictly_disjoint` também exclui esses contatos. As classes abaixo são mutuamente exclusivas.",
		"",
		"| Relação geométrica | Composição | Pares | Speedup geométrico | Mediana | Vitórias do nosso |",
		"| --- | --- | ---: | ---: | ---: | ---: |",
	]
	classes = (
		("Estritamente disjuntas", lambda row: row["strictly_disjoint"] == "1"),
		("Sem sobreposição de área, com contato", lambda row: row["interior_overlap_pairs"] == "0" and row["boundary_contact_pairs"] != "0"),
		("Com sobreposição de área", lambda row: row["interior_overlap_pairs"] != "0"),
	)
	for label, predicate in classes:
		group = [row for row in common if predicate(row)]
		composition = Counter(row["source_type"] for row in group)
		n, geo, median, _mean, wins = summary_cells(group)
		counts = ", ".join(f"{source} {composition[source]}" for source in ("OSM", "random", "tessellation") if composition[source])
		lines.append(f"| {label} | {counts} | {n} | {geo} | {median} | {wins} |")
	lines += ["", "Separando por fonte, o speedup geométrico é:", "", "| Fonte | Sem sobreposição de área | Com sobreposição de área |", "| --- | ---: | ---: |"]
	for source in ("OSM", "random", "tessellation"):
		cells = []
		for has_overlap in (False, True):
			group = [row for row in common if row["source_type"] == source and ((row["interior_overlap_pairs"] != "0") == has_overlap)]
			gm = geometric_mean(group)
			cells.append(f"{gm:.3f}× (n={len(group)})" if gm is not None else "—")
		lines.append(f"| {source} | " + " | ".join(cells) + " |")
	lines += [
		"",
		"No conjunto pareado, os casos sem sobreposição de área têm média geométrica maior que os casos com sobreposição (5.43× contra 3.43×). Mas `disjunto` não explica sozinho o resultado: os 78 casos Voronoi não têm sobreposição de área e mesmo assim ficam em 1.11×; todos têm contatos de fronteira entre células. Já as 177 instâncias estritamente disjuntas — 156 OSM e 21 aleatórias — ficam em 10.49×. As 21 aleatórias estritamente disjuntas são pequenas (18 com 5 polígonos, uma com 9 e duas com 10), então tamanho e fonte também confundem esse contraste.",
		"",
		"## Menores speedups",
		"",
		"Os 12 menores speedups das 550 instâncias pareadas são todos tessellation/Voronoi. O pior é o Caso 452 (`sbgdb-20200507-pntset-0000060`), com 60 polígonos: 246.63 s no nosso solver, 18.83 s no Fekete e speedup de 0.076× (Fekete cerca de 13.1× mais rápido).",
		"",
		"## Método e limitações geométricas",
		"",
		"O classificador testa pares de polígonos com Boost.Geometry. Vértices consecutivos separados por até `1e-12 × max(1, extensão da caixa delimitadora)` são removidos antes da validação; a interseção é considerada de área positiva somente se superar `max(1e-12, 1e-10 × menor área dos dois polígonos)`. Áreas menores que isso são registradas como contato de fronteira. Essa tolerância evita tratar ruído numérico como sobreposição, mas casos com áreas na vizinhança do limite dependem dela.",
		"",
		"A classificação de disjunção é descritiva e correlacional. As classes diferem também em fonte e tamanho; ela não isola causalmente o efeito da sobreposição. Para Voronoi, compartilhar fronteiras é esperado, portanto disjunção estrita não é uma categoria aplicável a essas regiões fechadas.",
		"",
		"O CSV por instância mantém os rótulos da fonte, nomes/IDs originais, status, runtimes, speedup e classificação de contatos. A proveniência do artigo está documentada junto à campanha.",
		"",
		f"SHA-256 do ZIP de instâncias simplificadas: `{hashlib.sha256(archive.read_bytes()).hexdigest()}`.",
		f"Revisão do submódulo: `{submodule_commit}`.",
	]
	return "\n".join(lines) + "\n"


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--instances", type=Path, default=CAMPAIGN / "instances.bin")
	parser.add_argument("--per-instance", type=Path, default=CAMPAIGN / "analysis/per-instance.csv")
	parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
	parser.add_argument("--output", type=Path, default=CAMPAIGN / "analysis/instance-classification.csv")
	parser.add_argument("--report", type=Path, default=CAMPAIGN / "analysis/source-stratified-report.md")
	parser.add_argument("--compiler", default="c++", help="C++ compiler command; Boost.Geometry headers must be available")
	args = parser.parse_args()

	cases, hashes = read_native_cases(args.instances)
	sources = load_archive(args.archive)
	if len(cases) != 558 or len(sources) != len(cases):
		raise SystemExit(f"expected 558 cases in binary and archive, found {len(cases)} and {len(sources)}")
	for case_index, (native, source) in enumerate(zip(cases, sources)):
		if canonical_instance(native["polygons"]) != canonical_instance(source["polygons"]):
			raise SystemExit(f"exact geometry fingerprint mismatch at case_index={case_index}, archive={source['archive_entry']}")

	geometry = run_geometry_classifier(sources, args.compiler)
	per_instance = read_csv(args.per_instance)
	if set(per_instance) != set(range(len(cases))):
		raise SystemExit("per-instance CSV does not contain exactly case_index 0..557")
	columns = [
		"case_index", "case_number", "input_sha256", "polygons", "source_type", "source_name", "archive_entry",
		"interior_overlap_pairs", "boundary_contact_pairs", "interior_disjoint", "strictly_disjoint",
		"overlap_area_sum", "max_pair_overlap_fraction", "ours_seconds", "fekete_seconds", "fekete_status",
		"fekete_completed", "speedup_fekete_over_ours",
	]
	args.output.parent.mkdir(parents=True, exist_ok=True)
	with args.output.open("w", newline="", encoding="utf-8") as output:
		writer = csv.DictWriter(output, fieldnames=columns)
		writer.writeheader()
		for geom in geometry:
			case_index = int(geom["case_index"])
			benchmark = per_instance[case_index]
			if int(geom["polygons"]) != int(benchmark["polygons"]):
				raise SystemExit(f"polygon count mismatch at case_index={case_index}")
			writer.writerow({
				"case_index": case_index,
				"case_number": case_index + 1,
				"input_sha256": hashes[case_index],
				"polygons": geom["polygons"],
				"source_type": geom["source_type"],
				"source_name": geom["source_name"],
				"archive_entry": sources[case_index]["archive_entry"],
				"interior_overlap_pairs": geom["interior_overlap_pairs"],
				"boundary_contact_pairs": geom["boundary_contact_pairs"],
				"interior_disjoint": geom["interior_disjoint"],
				"strictly_disjoint": geom["strictly_disjoint"],
				"overlap_area_sum": geom["overlap_area_sum"],
				"max_pair_overlap_fraction": geom["max_pair_overlap_fraction"],
				"ours_seconds": benchmark["ours_seconds"],
				"fekete_seconds": benchmark["fekete_seconds"],
				"fekete_status": benchmark["fekete_status"],
				"fekete_completed": benchmark["fekete_completed"],
				"speedup_fekete_over_ours": benchmark["speedup_fekete_over_ours"],
			})
	with args.output.open(newline="", encoding="utf-8") as source:
		classified_rows = list(csv.DictReader(source))
	args.report.parent.mkdir(parents=True, exist_ok=True)
	args.report.write_text(source_report(classified_rows, args.archive), encoding="utf-8")
	print(f"wrote {len(geometry)} classified cases to {args.output} and report to {args.report}")


if __name__ == "__main__":
	main()
