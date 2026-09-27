"""Compare paired one-thread and multi-thread free-order solver runs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SUCCESS_STATUSES = {"optimal", "complete", "completed", "success", "solved"}


@dataclass(frozen=True)
class Result:
    case: int
    sha256: str
    polygons: int | None
    status: str
    optimal: bool | None
    seconds: float | None
    threads: int | None
    workers: int | None
    time_limit_seconds: float | None
    gap: str


@dataclass
class Dataset:
    label: str
    path: Path
    rows: dict[int, Result]
    metadata: dict[str, Any]
    input_sha256: str

    @property
    def config(self) -> dict[str, Any]:
        value = self.metadata.get("config", {})
        return value if isinstance(value, dict) else {}


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def parse_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_int(value: Any) -> int | None:
    number = parse_float(value)
    return int(number) if number is not None else None


def first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
                    rows.append(value)
        return rows, {}

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value, {}
    if isinstance(value, dict) and isinstance(value.get("rows"), list):
        return value["rows"], value
    raise ValueError(f"No result rows found in JSON input: {path}")


def csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        sample = file.read(8192)
        file.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            delimiter = ";" if sample.splitlines() and sample.splitlines()[0].count(";") > sample.splitlines()[0].count(",") else ","
        else:
            delimiter = dialect.delimiter
        return list(csv.DictReader(file, delimiter=delimiter))


def report_has_solver(path: Path, solver: str) -> bool:
    try:
        rows, _ = json_rows(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return any(str(row.get("solver", "")).lower() == solver.lower() for row in rows)


def latest_file(paths: Iterable[Path]) -> Path | None:
    values = list(paths)
    return max(values, key=lambda item: item.stat().st_mtime_ns) if values else None


def resolve_input(value: Path, solver: str) -> Path:
    path = value.expanduser().resolve()
    if path.is_file():
        if path.name == "report.json" and solver == "tspn" and not report_has_solver(path, solver):
            partial = latest_file(path.parent.rglob("*-tspn-path.csv"))
            if partial:
                return partial
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Result input does not exist: {path}")

    reports = sorted(path.rglob("report.json"), key=lambda item: item.stat().st_mtime_ns, reverse=True)
    for report in reports:
        if report_has_solver(report, solver):
            return report
    if solver == "tspn":
        partial = latest_file(path.rglob("*-tspn-path.csv"))
        if partial:
            return partial

    preferred = ["runs.csv", "final.csv", "ours.csv", "fekete.csv"] if solver == "unordered" else ["final.csv", "fekete.csv", "ours.csv", "runs.csv"]
    for name in preferred:
        candidate = path / name
        if candidate.is_file():
            return candidate
    candidates = [*path.glob("*.csv"), *path.glob("*.json"), *path.glob("*.jsonl")]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not choose a result file in {path}; pass the CSV or report JSON directly.")


def sidecar_metadata(path: Path, embedded: dict[str, Any]) -> dict[str, Any]:
    if embedded:
        return embedded
    for parent in (path.parent, *list(path.parents)[:4]):
        for name in ("manifest.json", "report.json"):
            candidate = parent / name
            if candidate == path or not candidate.is_file():
                continue
            try:
                value = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and ("config" in value or "rows" in value):
                return value
    return {}


def config_value(metadata: dict[str, Any], *keys: str) -> Any:
    config = metadata.get("config", {})
    if not isinstance(config, dict):
        config = {}
    for key in keys:
        if config.get(key) is not None:
            return config[key]
        if metadata.get(key) is not None:
            return metadata[key]
    return None


def gap_label(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    eps = first(row, "eps", "external_optimality_eps")
    if eps is not None:
        return f"eps={eps}"
    absolute = first(row, "absolute_gap")
    relative = first(row, "relative_gap")
    if absolute is not None or relative is not None:
        return f"abs={absolute or 0}; rel={relative or 0}"
    variant = first(row, "variant")
    if variant is not None:
        config = metadata.get("config", {})
        variants = config.get("variants", {}) if isinstance(config, dict) else {}
        if variant in variants:
            return f"{variant}: {json.dumps(variants[variant], sort_keys=True)}"
        return str(variant)
    config = metadata.get("config", {})
    if isinstance(config, dict):
        ours = config.get("ours_optimality")
        if isinstance(ours, dict):
            return f"ours abs={ours.get('absolute_gap')}; rel={ours.get('relative_gap')}"
        if config.get("external_optimality_eps") is not None:
            return f"eps={config['external_optimality_eps']}"
        mapping = config.get("fekete_gap_mapping")
        if isinstance(mapping, dict):
            return f"rel={mapping.get('our_relative_gap')}"
    return "unknown"


def make_result(
    row: dict[str, Any], metadata: dict[str, Any], thread_override: int | None,
) -> Result | None:
    case = parse_int(first(row, "case", "case_index", "index"))
    if case is None:
        return None
    status = str(first(row, "termination", "status", "result") or "unknown").strip().lower()
    optimal = parse_bool(first(row, "exact", "is_optimal", "optimal", "gap_closed"))
    if optimal is None:
        optimal = status in SUCCESS_STATUSES
    config = metadata.get("config", {})
    config = config if isinstance(config, dict) else {}
    threads = parse_int(first(row, "threads_per_instance", "threads", "num_threads"))
    if threads is None:
        threads = thread_override
    if threads is None:
        threads = parse_int(config_value(metadata, "threads_per_instance", "num_threads"))
    if threads is None:
        worker_note = str(metadata.get("workers", "")).lower()
        if "single-thread" in worker_note or "single threaded" in worker_note:
            threads = 1
    workers = parse_int(first(row, "instance_workers", "workers", "worker_count"))
    if workers is None:
        workers = parse_int(config_value(metadata, "instance_workers", "workers"))
    limit = parse_float(first(row, "time_limit_seconds", "max_seconds", "time_limit"))
    if limit is None:
        limit = parse_float(config_value(metadata, "max_seconds", "time_limit_seconds"))
    return Result(
        case=case,
        sha256=str(first(row, "sha256", "geometry_sha256", "instance_sha256") or "").strip().lower(),
        polygons=parse_int(first(row, "polygons", "polygon_count")),
        status=status,
        optimal=optimal,
        seconds=parse_float(first(row, "seconds", "solve_seconds", "runtime_seconds")),
        threads=threads,
        workers=workers,
        time_limit_seconds=limit,
        gap=gap_label(row, metadata),
    )


def load_dataset(
    value: Path,
    label: str,
    solver: str,
    variant: str | None,
    thread_override: int | None,
) -> Dataset:
    path = resolve_input(value, solver)
    embedded: dict[str, Any] = {}
    if path.suffix.lower() in {".json", ".jsonl"}:
        rows, embedded = json_rows(path)
    elif path.suffix.lower() == ".csv":
        rows = csv_rows(path)
    else:
        raise ValueError(f"Expected a report JSON/JSONL or CSV: {path}")
    metadata = sidecar_metadata(path, embedded)
    chosen: dict[int, tuple[str, int, Result]] = {}
    for position, row in enumerate(rows):
        row_solver = str(row.get("solver", "")).strip().lower()
        if row_solver and row_solver != solver.lower():
            continue
        row_variant = str(row.get("variant", "")).strip()
        if variant is not None and row_variant and row_variant != variant:
            continue
        result = make_result(row, metadata, thread_override)
        if result is None:
            continue
        stamp = str(first(row, "attempted_at", "created_at", "finished_at") or "")
        previous = chosen.get(result.case)
        if previous is None or (stamp, position) >= (previous[0], previous[1]):
            chosen[result.case] = (stamp, position, result)
    result_rows = {case: value[2] for case, value in chosen.items()}
    if not result_rows:
        filter_text = f" for solver={solver!r}" + (f", variant={variant!r}" if variant else "")
        raise ValueError(f"No matching result rows found in {path}{filter_text}.")
    return Dataset(label, path, result_rows, metadata, sha256_file(path))


def hashes_match(left: Result | None, right: Result | None) -> bool:
    return not (left and right and left.sha256 and right.sha256 and left.sha256 != right.sha256)


def all_hashes_match(results: Iterable[Result | None]) -> bool:
    hashes = {row.sha256 for row in results if row and row.sha256}
    return len(hashes) <= 1


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def paired_speedups(single: Dataset, multi: Dataset) -> tuple[list[tuple[int, float]], dict[str, int]]:
    ratios: list[tuple[int, float]] = []
    counts = {"paired": 0, "both_optimal": 0, "single_only": 0, "multi_only": 0, "neither": 0, "hash_mismatch": 0}
    for case in sorted(set(single.rows) | set(multi.rows)):
        one, many = single.rows.get(case), multi.rows.get(case)
        if not one or not many:
            continue
        counts["paired"] += 1
        if not hashes_match(one, many):
            counts["hash_mismatch"] += 1
            continue
        one_ok, many_ok = one.optimal is True, many.optimal is True
        if one_ok and many_ok:
            counts["both_optimal"] += 1
            if one.seconds is not None and many.seconds is not None and one.seconds > 0 and many.seconds > 0:
                ratios.append((case, one.seconds / many.seconds))
        elif one_ok:
            counts["single_only"] += 1
        elif many_ok:
            counts["multi_only"] += 1
        else:
            counts["neither"] += 1
    return ratios, counts


def summarize_ratios(ratios: list[tuple[int, float]]) -> dict[str, Any]:
    values = [value for _, value in ratios]
    if not values:
        return {"n": 0, "median": None, "geomean": None, "multi_faster": 0, "single_faster": 0, "equal": 0}
    return {
        "n": len(values),
        "median": statistics.median(values),
        "geomean": math.exp(statistics.mean(math.log(value) for value in values)),
        "multi_faster": sum(value > 1 and not math.isclose(value, 1, rel_tol=1e-9) for value in values),
        "single_faster": sum(value < 1 and not math.isclose(value, 1, rel_tol=1e-9) for value in values),
        "equal": sum(math.isclose(value, 1, rel_tol=1e-9) for value in values),
    }


def cross_solver_ratios(left: Dataset, right: Dataset) -> list[tuple[int, float]]:
    result = []
    for case in sorted(set(left.rows) & set(right.rows)):
        a, b = left.rows[case], right.rows[case]
        if not hashes_match(a, b) or a.optimal is not True or b.optimal is not True:
            continue
        if a.seconds is not None and b.seconds is not None and a.seconds > 0 and b.seconds > 0:
            result.append((case, a.seconds / b.seconds))
    return result


def summarize_cross_solver(ratios: list[tuple[int, float]]) -> dict[str, Any]:
    values = [value for _, value in ratios]
    if not values:
        return {"n": 0, "median": None, "geomean": None, "ours_faster": 0, "fekete_faster": 0, "equal": 0}
    return {
        "n": len(values),
        "median": statistics.median(values),
        "geomean": math.exp(statistics.mean(math.log(value) for value in values)),
        "ours_faster": sum(value < 1 and not math.isclose(value, 1, rel_tol=1e-9) for value in values),
        "fekete_faster": sum(value > 1 and not math.isclose(value, 1, rel_tol=1e-9) for value in values),
        "equal": sum(math.isclose(value, 1, rel_tol=1e-9) for value in values),
    }


def dataset_setup(dataset: Dataset) -> str:
    rows = list(dataset.rows.values())
    threads = sorted({row.threads for row in rows if row.threads is not None})
    workers = sorted({row.workers for row in rows if row.workers is not None})
    limits = sorted({row.time_limit_seconds for row in rows if row.time_limit_seconds is not None})
    gaps = sorted({row.gap for row in rows if row.gap and row.gap != "unknown"})
    return "; ".join([
        f"threads={threads or 'unknown'}",
        f"instance workers={workers or 'unknown'}",
        f"time limits={limits or 'unknown'}",
        f"gap={gaps[:3] or ['unknown']}" + (" …" if len(gaps) > 3 else ""),
    ])


def speedup_table_row(label: str, single: Dataset, multi: Dataset) -> tuple[str, dict[str, Any]]:
    ratios, counts = paired_speedups(single, multi)
    stats = summarize_ratios(ratios)
    median = f"{stats['median']:.3f}×" if stats["median"] is not None else "n/a"
    geomean = f"{stats['geomean']:.3f}×" if stats["geomean"] is not None else "n/a"
    text = (f"| {label} | {stats['n']} | {median} | {geomean} | {stats['multi_faster']} | "
            f"{stats['single_faster']} | {stats['equal']} | {counts['single_only']} | {counts['multi_only']} |")
    return text, {"counts": counts, "speedup": stats}


def collect_warnings(datasets: dict[str, Dataset]) -> list[str]:
    warnings: list[str] = []
    for solver in ("ours", "fekete"):
        single = datasets[f"{solver}_single"]
        multi = datasets[f"{solver}_multi"]
        one_rows, many_rows = list(single.rows.values()), list(multi.rows.values())
        one_workers = {row.workers for row in one_rows if row.workers is not None}
        many_workers = {row.workers for row in many_rows if row.workers is not None}
        if one_workers and many_workers and one_workers != many_workers:
            warnings.append(
                f"{solver}: instance-level worker counts differ ({sorted(one_workers)} vs {sorted(many_workers)}); "
                "the runtime ratio does not isolate solver threads."
            )
        paired_cases = set(single.rows) & set(multi.rows)
        limit_mismatches = [
            case for case in paired_cases
            if single.rows[case].time_limit_seconds is not None
            and multi.rows[case].time_limit_seconds is not None
            and single.rows[case].time_limit_seconds != multi.rows[case].time_limit_seconds
        ]
        if limit_mismatches:
            warnings.append(f"{solver}: per-instance time limits differ on {len(limit_mismatches)} paired cases.")
        gap_mismatches = [
            case for case in paired_cases
            if single.rows[case].gap != "unknown" and multi.rows[case].gap != "unknown"
            and single.rows[case].gap != multi.rows[case].gap
        ]
        if gap_mismatches:
            warnings.append(f"{solver}: stopping-gap metadata differs on {len(gap_mismatches)} paired cases.")
    ours_single = datasets["ours_single"]
    ours_multi = datasets["ours_multi"]
    old_hash = config_value(ours_single.metadata, "solver_sha256", "solver_binary_sha256")
    new_hash = config_value(ours_multi.metadata, "solver_sha256", "solver_binary_sha256")
    if old_hash and new_hash and old_hash != new_hash:
        warnings.append("our solver binary hashes differ between the one-thread and multi-thread sources.")
    return warnings


def write_outputs(output: Path, datasets: dict[str, Dataset]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    all_cases = sorted(set().union(*(set(data.rows) for data in datasets.values())))
    columns = [
        "case", "sha256", "polygons", "hashes_match",
        "ours_single_status", "ours_single_optimal", "ours_single_seconds", "ours_single_threads", "ours_single_workers", "ours_single_time_limit_seconds", "ours_single_gap",
        "ours_multi_status", "ours_multi_optimal", "ours_multi_seconds", "ours_multi_threads", "ours_multi_workers", "ours_multi_time_limit_seconds", "ours_multi_gap",
        "ours_observed_speedup_1t_over_multi", "ours_optimal_pair_speedup_1t_over_multi",
        "fekete_single_status", "fekete_single_optimal", "fekete_single_seconds", "fekete_single_threads", "fekete_single_workers", "fekete_single_time_limit_seconds", "fekete_single_gap",
        "fekete_multi_status", "fekete_multi_optimal", "fekete_multi_seconds", "fekete_multi_threads", "fekete_multi_workers", "fekete_multi_time_limit_seconds", "fekete_multi_gap",
        "fekete_observed_speedup_1t_over_multi", "fekete_optimal_pair_speedup_1t_over_multi",
        "ours_over_fekete_single_thread", "ours_over_fekete_multi_thread",
    ]
    comparison_path = output / "comparison.csv"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output, prefix=".comparison.", delete=False) as file:
        temporary = Path(file.name)
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for case in all_cases:
            values = {name: datasets[name].rows.get(case) for name in datasets}
            present = [value for value in values.values() if value]
            hashes_ok = all_hashes_match(present)
            digest = next((value.sha256 for value in present if value.sha256), "")
            polygons = next((value.polygons for value in present if value.polygons is not None), None)
            row: dict[str, Any] = {"case": case, "sha256": digest, "polygons": polygons, "hashes_match": hashes_ok}
            for name, value in values.items():
                if not value:
                    continue
                prefix = name
                row[f"{prefix}_status"] = value.status
                row[f"{prefix}_optimal"] = value.optimal
                row[f"{prefix}_seconds"] = value.seconds
                row[f"{prefix}_threads"] = value.threads
                row[f"{prefix}_workers"] = value.workers
                row[f"{prefix}_time_limit_seconds"] = value.time_limit_seconds
                row[f"{prefix}_gap"] = value.gap
            for solver in ("ours", "fekete"):
                one, many = values[f"{solver}_single"], values[f"{solver}_multi"]
                if one and many and hashes_match(one, many) and one.seconds is not None and many.seconds is not None and one.seconds > 0 and many.seconds > 0:
                    row[f"{solver}_observed_speedup_1t_over_multi"] = one.seconds / many.seconds
                    if one.optimal is True and many.optimal is True:
                        row[f"{solver}_optimal_pair_speedup_1t_over_multi"] = one.seconds / many.seconds
            for thread_label, one_name, fekete_name in (
                ("single_thread", "ours_single", "fekete_single"),
                ("multi_thread", "ours_multi", "fekete_multi"),
            ):
                ours, fekete = values[one_name], values[fekete_name]
                if ours and fekete and hashes_match(ours, fekete) and ours.optimal is True and fekete.optimal is True and ours.seconds and fekete.seconds:
                    row[f"ours_over_fekete_{thread_label}"] = ours.seconds / fekete.seconds
            writer.writerow({key: fmt(value) for key, value in row.items()})
    temporary.replace(comparison_path)

    summaries: dict[str, Any] = {}
    table_rows = []
    for solver, label in (("ours", "Nosso solver"), ("fekete", "Fekete")):
        line, summary = speedup_table_row(label, datasets[f"{solver}_single"], datasets[f"{solver}_multi"])
        table_rows.append(line)
        summaries[solver] = summary

    warnings = collect_warnings(datasets)
    common_all = set.intersection(*(set(data.rows) for data in datasets.values()))
    all_hash_match = sum(all_hashes_match(data.rows[case] for data in datasets.values()) for case in common_all)
    cross = {}
    for name, one_name, multi_name in (
        ("one_thread", "ours_single", "fekete_single"),
        ("multi_thread", "ours_multi", "fekete_multi"),
    ):
        ratios = cross_solver_ratios(datasets[one_name], datasets[multi_name])
        stats = summarize_cross_solver(ratios)
        cross[name] = stats

    ours_gain = summaries["ours"]["speedup"]["geomean"]
    fekete_gain = summaries["fekete"]["speedup"]["geomean"]
    if ours_gain is not None and fekete_gain is not None:
        relative_verdict = "Nosso solver" if ours_gain > fekete_gain else "Fekete" if fekete_gain > ours_gain else "empate"
    else:
        relative_verdict = "dados insuficientes para comparar os speedups"

    lines = [
        "# Comparação do efeito de multithreading", "",
        f"- Casos no CSV: {len(all_cases)}; casos presentes nas quatro fontes: {len(common_all)}; hashes iguais nesses casos: {all_hash_match}.",
        "- Speedup = tempo com 1 thread / tempo com múltiplas threads. Acima de 1×, o run multithread foi mais rápido.",
        "- Os speedups resumidos usam apenas pares com o mesmo hash em que ambas as execuções fecharam a tolerância solicitada.", "",
        "## Fontes e configurações detectadas", "",
        "| Fonte | Arquivo | Casos | Configuração por registro |", "|---|---|---:|---|",
    ]
    for name, dataset in datasets.items():
        lines.append(f"| {name} | `{dataset.path}` | {len(dataset.rows)} | {dataset_setup(dataset)} |")
    lines.extend([
        "", "## Speedup por solver", "",
        "| Solver | Pares concluídos | Mediana 1t/nt | Média geométrica 1t/nt | nt mais rápido | 1t mais rápido | Iguais | Só 1t fechou gap | Só nt fechou gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *table_rows,
        "", f"Pelo speedup geométrico observado, a aceleração relativa maior foi de: **{relative_verdict}**.",
        "", "## Comparação de runtime entre solvers", "",
        "A razão abaixo é nosso tempo / tempo de Fekete; abaixo de 1×, nosso solver foi mais rápido.", "",
        "| Threads | Pares concluídos | Mediana nosso/Fekete | Média geométrica | Nosso mais rápido | Fekete mais rápido | Iguais |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for label, key in (("1 thread", "one_thread"), ("Múltiplas threads", "multi_thread")):
        stats = cross[key]
        median = f"{stats['median']:.3f}×" if stats["median"] is not None else "n/a"
        geo = f"{stats['geomean']:.3f}×" if stats["geomean"] is not None else "n/a"
        lines.append(f"| {label} | {stats['n']} | {median} | {geo} | {stats['ours_faster']} | {stats['fekete_faster']} | {stats['equal']} |")
    lines.extend(["", "## Limitações", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- Nenhuma diferença de workers, limite ou gap foi detectada nos metadados disponíveis; código, hardware e carga do sistema ainda podem diferir entre campanhas.")
    lines.extend([
        "- Instâncias em que uma execução atingiu o limite ou não fechou o gap não entram na estatística de speedup; os status e tempos observados continuam no CSV.",
        "- Este é um comparativo pareado observacional. Para atribuir causalmente o speedup às threads, as quatro campanhas precisam usar as mesmas versões/configurações, mesmo número de instâncias concorrentes e condições de máquina equivalentes.",
        "", f"CSV: `{comparison_path}`", "",
    ])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    provenance_metadata = {
        name: {
            key: dataset.metadata[key]
            for key in ("schema_version", "created_at", "status", "title", "host", "config", "workers")
            if key in dataset.metadata
        }
        for name, dataset in datasets.items()
    }
    provenance = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "metric": "one_thread_seconds / multi_thread_seconds; only same-hash pairs closing the requested gap are summarized",
        "sources": {
            name: {
                "path": str(dataset.path),
                "sha256": dataset.input_sha256,
                "rows": len(dataset.rows),
                "metadata": provenance_metadata[name],
            }
            for name, dataset in datasets.items()
        },
        "summary": summaries,
        "cross_solver": cross,
        "warnings": warnings,
    }
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return {"cases": len(all_cases), "sources": {name: len(data.rows) for name, data in datasets.items()}, "summary": summaries, "warnings": warnings}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Compare one-thread and multi-thread runtimes on matching free-order TPP instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    result.add_argument("--ours-single", type=Path, required=True, help="Our one-thread CSV, JSON report, or run directory.")
    result.add_argument("--ours-multi", type=Path, required=True, help="Our multi-thread CSV, JSON report, or run directory.")
    result.add_argument("--fekete-single", type=Path, required=True, help="Fekete one-thread CSV, JSON report, or run directory.")
    result.add_argument("--fekete-multi", type=Path, required=True, help="Fekete multi-thread CSV, JSON report, or run directory.")
    result.add_argument("--output", type=Path, required=True, help="Directory for comparison.csv, summary.md, and manifest.json.")
    result.add_argument("--ours-single-variant", default="fekete_gap", help="Variant filter for an attempt log such as runs.csv; empty disables filtering.")
    result.add_argument("--ours-single-threads", type=int, help="Override missing one-thread count metadata.")
    result.add_argument("--ours-multi-threads", type=int, help="Override missing multi-thread count metadata.")
    result.add_argument("--fekete-single-threads", type=int, help="Override missing one-thread count metadata.")
    result.add_argument("--fekete-multi-threads", type=int, help="Override missing multi-thread count metadata.")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    datasets = {
        "ours_single": load_dataset(args.ours_single, "ours_single", "unordered", args.ours_single_variant or None, args.ours_single_threads),
        "ours_multi": load_dataset(args.ours_multi, "ours_multi", "unordered", None, args.ours_multi_threads),
        "fekete_single": load_dataset(args.fekete_single, "fekete_single", "tspn", None, args.fekete_single_threads),
        "fekete_multi": load_dataset(args.fekete_multi, "fekete_multi", "tspn", None, args.fekete_multi_threads),
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    summary = write_outputs(output, datasets)
    print(f"Compared {summary['cases']} case indices.")
    for name, count in summary["sources"].items():
        print(f"{name}: {count} result rows")
    print(f"Summary: {output / 'summary.md'}")
    print(f"CSV: {output / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
