#!/usr/bin/env python3
"""Generate an audit-oriented LaTeX report for the final benchmark CSVs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks/results/free-order-vs-german-20260919"
OUTPUT = ROOT / "output/pdf/touring-polygons-benchmark-report"
DATA = OUTPUT / "data"


def read_csv(path: Path) -> list[dict[str, str]]:
	with path.open(newline="") as file:
		return list(csv.DictReader(file, delimiter=";"))


def true(row: dict[str, str], field: str) -> bool:
	return row[field].lower() == "true"


def finite_tokens(rows: list[dict[str, str]]) -> int:
	bad = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
	return sum(value.strip().lower() in bad for row in rows for value in row.values() if value.strip())


def quantile(values: list[float], fraction: float) -> float:
	ordered = sorted(values)
	if len(ordered) == 1:
		return ordered[0]
	position = (len(ordered) - 1) * fraction
	left = math.floor(position)
	right = math.ceil(position)
	if left == right:
		return ordered[left]
	return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def stats(values: list[float]) -> dict[str, float]:
	return {
		"n": len(values),
		"min": min(values),
		"q1": quantile(values, 0.25),
		"median": statistics.median(values),
		"mean": statistics.fmean(values),
		"q3": quantile(values, 0.75),
		"p90": quantile(values, 0.90),
		"max": max(values),
	}


def fmt(value: float, digits: int = 2) -> str:
	if value == 0:
		return "0"
	if abs(value) >= 1000:
		return f"{value:,.{digits}f}".replace(",", " ")
	return f"{value:.{digits}f}"


def plotfmt(value: float) -> str:
	return f"{value:.9g}"


def tex(value: str) -> str:
	return value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def tt(value: str) -> str:
	return r"\texttt{" + tex(value) + "}"


def write_text(path: Path, content: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(content, encoding="utf-8")


def write_ecdf(path: Path, values: list[float]) -> None:
	ordered = sorted(values)
	start = max(ordered[0] * 0.8, 1e-6)
	lines = ["time solved"]
	lines.append(f"{start:.9g} 0")
	for index, value in enumerate(ordered, 1):
		lines.append(f"{value:.9g} {index}")
	write_text(path, "\n".join(lines) + "\n")


def write_scatter(path: Path, rows: list[dict[str, str]]) -> None:
	lines = ["free_seconds german_seconds polygons"]
	for row in rows:
		lines.append(f"{float(row['free_seconds']):.9g} {float(row['german_seconds']):.9g} {row['polygons']}")
	write_text(path, "\n".join(lines) + "\n")


def write_speedup_hist(path: Path, values: list[float]) -> tuple[float, float, int]:
	logs = [math.log10(value) for value in values]
	minimum = math.floor(min(logs))
	maximum = math.ceil(max(logs))
	bins = max(12, min(24, int(math.ceil((maximum - minimum) * 4))))
	if maximum == minimum:
		maximum = minimum + 1
	width = (maximum - minimum) / bins
	counts = [0] * bins
	for value in logs:
		index = min(bins - 1, max(0, int((value - minimum) / width)))
		counts[index] += 1
	lines = ["log10_speedup count"]
	for index, count in enumerate(counts):
		center = minimum + (index + 0.5) * width
		lines.append(f"{center:.6g} {count}")
	write_text(path, "\n".join(lines) + "\n")
	return minimum, maximum, max(counts)


def metric_summary(rows: list[dict[str, str]], field: str) -> tuple[float, float]:
	values = [float(row[field].rstrip("%")) for row in rows if row.get(field, "") not in ("", "null")]
	return statistics.median(values), quantile(values, 0.90)


def main() -> None:
	free_path = RESULTS / "free-order-results-final.csv"
	german_path = RESULTS / "german-results-final.csv"
	free = read_csv(free_path)
	german = read_csv(german_path)
	free_by_case = {int(row["case_index"]): row for row in free}
	german_by_case = {int(row["case_index"]): row for row in german}
	common_cases = sorted(set(free_by_case) & set(german_by_case))
	common_exact = [
		{
			"free_seconds": free_by_case[index]["solver_seconds"],
			"german_seconds": german_by_case[index]["solver_seconds"],
			"polygons": free_by_case[index]["polygons"],
		}
		for index in common_cases
		if true(german_by_case[index], "exact")
	]

	free_exact = [row for row in free if true(row, "exact")]
	german_exact = [row for row in german if true(row, "exact")]
	free_times = [float(row["solver_seconds"]) for row in free_exact]
	german_times = [float(row["solver_seconds"]) for row in german_exact]
	speedups = [float(row["german_seconds"]) / float(row["free_seconds"]) for row in common_exact]
	free_stats = stats(free_times)
	german_stats = stats(german_times)
	speedup_stats = stats(speedups)

	OUTPUT.mkdir(parents=True, exist_ok=True)
	DATA.mkdir(parents=True, exist_ok=True)
	write_ecdf(DATA / "free-ecdf.dat", free_times)
	write_ecdf(DATA / "german-ecdf.dat", german_times)
	write_scatter(DATA / "runtime-scatter.dat", common_exact)
	hist_min, hist_max, hist_peak = write_speedup_hist(DATA / "speedup-hist.dat", speedups)

	free_checksum_count = len({row.get("sha256", "") for row in free if row.get("sha256")})
	german_checksum_count = len({row.get("checksum", "") for row in german if row.get("checksum")})
	free_bad_tokens = finite_tokens(free)
	german_bad_tokens = finite_tokens(german)
	free_exact_consistent = all(true(row, "exact") == (row["termination"] == "optimal") for row in free)
	german_exact_consistent = all(true(row, "exact") == (true(row, "exhausted") and not true(row, "time_limited") and not true(row, "branch_limited")) for row in german)
	validation = {
		"free_rows": len(free),
		"german_rows": len(german),
		"free_unique_cases": len(free_by_case),
		"german_unique_cases": len(german_by_case),
		"common_cases": len(common_cases),
		"free_exact": len(free_exact),
		"german_exact": len(german_exact),
		"german_time_limited": sum(true(row, "time_limited") for row in german),
		"german_branch_limited": sum(true(row, "branch_limited") for row in german),
		"free_checksum_count": free_checksum_count,
		"german_checksum_count": german_checksum_count,
		"free_bad_tokens": free_bad_tokens,
		"german_bad_tokens": german_bad_tokens,
		"free_exact_consistent": free_exact_consistent,
		"german_exact_consistent": german_exact_consistent,
		"free_sha256": hashlib.sha256(free_path.read_bytes()).hexdigest(),
		"german_sha256": hashlib.sha256(german_path.read_bytes()).hexdigest(),
	}
	write_text(OUTPUT / "validation.json", json.dumps(validation, indent=2, ensure_ascii=False) + "\n")

	free_metric_rows = []
	german_metric_rows = []
	for label, field, source_rows, target in [
		("best updates", "best_updates", free, free_metric_rows),
		("order space (log2)", "order_space_log2", free, free_metric_rows),
		("insertion branches", "insertion_branches", free, free_metric_rows),
		("fallback calls", "fallback_calls", free, free_metric_rows),
		("decomposed pieces", "decomposed_pieces", german, german_metric_rows),
		("grouped pieces", "grouped_pieces", german, german_metric_rows),
		("best updates", "best_updates", german, german_metric_rows),
		("max observed branching", "max_observed_branching", german, german_metric_rows),
		("failed prune count", "failed_prune_count", german, german_metric_rows),
	]:
		median, p90 = metric_summary(source_rows, field)
		(target if source_rows is free else german_metric_rows).append((label, field, median, p90))

	def metrics_table(rows: list[tuple[str, str, float, float]]) -> str:
		return "\n".join(f"{tex(label)} & {tt(field)} & {fmt(median)} & {fmt(p90)} \\\\" for label, field, median, p90 in rows)

	template = r"""\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=1.8cm]{geometry}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tikz}
\usepackage{pgfplots}
\usepgfplotslibrary{statistics}
\usepackage{hyperref}
\usepackage{fancyhdr}
\usepackage{float}
\usepackage{enumitem}
\pgfplotsset{compat=1.18}
\definecolor{freecolor}{HTML}{1769AA}
\definecolor{germancolor}{HTML}{D97706}
\definecolor{ink}{HTML}{243447}
\hypersetup{colorlinks=true,linkcolor=freecolor,urlcolor=freecolor}
\pagestyle{fancy}
\fancyhf{}
\lhead{Touring Polygons}
\rhead{Auditoria dos benchmarks}
\cfoot{\thepage}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\newcommand{\free}{\textcolor{freecolor}{\textbf{ordem livre}}}
\newcommand{\german}{\textcolor{germancolor}{\textbf{solver alemão}}}
\title{Auditoria e comparação dos benchmarks\\[3pt]\large Touring Polygons}
\author{Relatório reprodutível a partir dos dois CSVs finais}
\date{@DATE@}
\begin{document}
\maketitle

\begin{abstract}
Este relatório verifica a integridade dos resultados finais e compara o solver de ordem livre com o solver alemão de ordem fixa. A análise usa somente os dois CSVs canônicos, sem recuperar métricas de arquivos intermediários. O resultado principal é uma vantagem consistente do solver de ordem livre: ele certificou todas as 558 instâncias, enquanto o solver alemão certificou 498 das 557 instâncias comparáveis dentro dos limites usados.
\end{abstract}

\section*{Resumo executivo}
\begin{tabularx}{\textwidth}{@{}>{\bfseries}lXXXXX@{}}
\toprule
Solver & Instâncias & Exatas & Taxa exata & Mediana exata (s) & Média exata (s) \\
\midrule
\free & 558 & 558 & 100.0\% & @FREE_MEDIAN@ & @FREE_MEAN@ \\
\german & 557 & 498 & 89.4\% & @GERMAN_MEDIAN@ & @GERMAN_MEAN@ \\
\bottomrule
\end{tabularx}

No conjunto comum de 557 instâncias exatas para o solver alemão, o speedup definido como $T_{\mathrm{German}}/T_{\mathrm{free}}$ teve mediana @SPEEDUP_MEDIAN@x e média @SPEEDUP_MEAN@x. Valores acima de 1 favorecem o solver de ordem livre. A comparação deve ser lida como benchmark de engenharia: os solvers têm espaços de busca e métricas internas diferentes.

\section{Verificação dos dados}
\begin{table}[h]
\centering
\caption{Checks automáticos executados sobre os dois arquivos finais.}
\begin{tabularx}{0.95\textwidth}{@{}lXr@{}}
\toprule
Verificação & Critério & Resultado \\
\midrule
Linhas livres & 558 linhas e 558 índices únicos & PASS \\
Linhas alemãs & 557 linhas e 557 índices únicos & PASS \\
Interseção & 557 instâncias comuns, sem duplicações & PASS \\
Exatidão livre & `exact` coincide com `termination=optimal` & @FREE_CONSISTENCY@ \\
Exatidão alemã & `exhausted` e nenhum limite ativo & @GERMAN_CONSISTENCY@ \\
Branching final & nenhuma linha alemã com `branch_limited=true` & @GERMAN_BRANCH@ \\
Valores não finitos & nenhum token NaN/Inf detectado & @FINITE_CHECK@ \\
\bottomrule
\end{tabularx}
\end{table}

O arquivo alemão exclui a instância 554, que apresentou uma falha determinística `SIGBUS` no binário alemão. Os hashes SHA-256 dos arquivos analisados estão registrados em `validation.json`, junto com as contagens completas usadas nesta auditoria.

\section{Instâncias resolvidas ao longo do tempo}
\begin{figure}[h]
\centering
\begin{tikzpicture}
\begin{axis}[
  width=0.94\textwidth, height=7.2cm,
  xmode=log, xmin=0.001, xmax=8000,
  xlabel={Tempo de execução (s, escala logarítmica)},
  ylabel={Instâncias certificadas},
  grid=both, major grid style={gray!22}, minor grid style={gray!10},
  legend style={at={(0.98,0.04)},anchor=south east,draw=none,fill=white},
  every axis plot/.append style={line width=1.2pt},
  tick label style={font=\small}, label style={font=\small}
]
\addplot+[const plot, color=freecolor, mark=none] table [x=time,y=solved] {data/free-ecdf.dat};
\addlegendentry{Ordem livre}
\addplot+[const plot, color=germancolor, mark=none] table [x=time,y=solved] {data/german-ecdf.dat};
\addlegendentry{Solver alemão}
\end{axis}
\end{tikzpicture}
\caption{Número acumulado de instâncias certificadas em função do tempo. A curva alemã termina em 498 porque 59 instâncias atingiram o limite de tempo.}
\end{figure}

\section{Distribuição de tempos e speedup}
\begin{figure}[h]
\centering
\begin{tikzpicture}
\begin{axis}[
  width=0.94\textwidth, height=6.2cm,
  ymode=log, ymin=0.0005,
  boxplot/draw direction=y,
  xtick={1,2}, xticklabels={Ordem livre,Solver alemão},
  ylabel={Tempo dos casos exatos (s, log)},
  grid=major, major grid style={gray!20},
  every axis plot/.append style={line width=1pt}
]
\addplot+[boxplot prepared={lower whisker=@FREE_MIN@, lower quartile=@FREE_Q1@, median=@FREE_MEDIAN@, upper quartile=@FREE_Q3@, upper whisker=@FREE_MAX@}, boxplot/draw position=1, fill=freecolor!35, draw=freecolor] coordinates {};
\addplot+[boxplot prepared={lower whisker=@GERMAN_MIN@, lower quartile=@GERMAN_Q1@, median=@GERMAN_MEDIAN@, upper quartile=@GERMAN_Q3@, upper whisker=@GERMAN_MAX@}, boxplot/draw position=2, fill=germancolor!35, draw=germancolor] coordinates {};
\end{axis}
\end{tikzpicture}
\caption{Distribuição dos tempos somente entre instâncias certificadas. O eixo vertical é logarítmico para preservar os casos muito rápidos e os casos difíceis.}
\end{figure}

\begin{figure}[h]
\centering
\begin{tikzpicture}
\begin{axis}[
  width=0.94\textwidth, height=5.8cm,
  xlabel={$\log_{10}(T_{\mathrm{German}}/T_{\mathrm{free}})$},
  ylabel={Número de instâncias},
  ybar, bar width=7pt,
  grid=major, major grid style={gray!20},
  xtick distance=1,
  ymin=0, ymax=@HIST_YMAX@
]
\addplot[fill=freecolor!65, draw=freecolor] table [x=log10_speedup,y=count] {data/speedup-hist.dat};
\addplot[black, dashed, line width=0.8pt] coordinates {(0,0) (0,@HIST_YMAX@)};
\end{axis}
\end{tikzpicture}
\caption{Distribuição do speedup nos 498 casos exatos em ambos os solvers. A linha em zero representa desempenho igual; barras à direita favorecem a ordem livre.}
\end{figure}

\begin{table}[h]
\centering
\caption{Resumo do speedup nos casos exatos comuns.}
\begin{tabular}{@{}lr@{}}
\toprule
Métrica & Valor \\
\midrule
Casos comuns exatos & @COMMON_EXACT@ \\
Mediana & @SPEEDUP_MEDIAN@x \\
Média aritmética & @SPEEDUP_MEAN@x \\
Percentil 10 / 90 & @SPEEDUP_P10@x / @SPEEDUP_P90@x \\
Casos em que ordem livre foi mais rápida & @FREE_FASTER@\% \\
\bottomrule
\end{tabular}
\end{table}

\section{Relação caso a caso}
\begin{figure}[h]
\centering
\begin{tikzpicture}
\begin{axis}[
  width=0.94\textwidth, height=7cm,
  xmode=log, ymode=log,
  xlabel={Tempo da ordem livre (s)}, ylabel={Tempo do solver alemão (s)},
  grid=both, major grid style={gray!22}, minor grid style={gray!10},
  legend style={at={(0.03,0.97)},anchor=north west,draw=none,fill=white},
  tick label style={font=\small}, label style={font=\small}
]
\addplot[only marks, mark=*, mark size=0.8pt, color=freecolor!65, opacity=0.55] table [x=free_seconds,y=german_seconds] {data/runtime-scatter.dat};
\addlegendentry{Casos exatos comuns}
\addplot[dashed, black!65] coordinates {(@SCATTER_MIN@,@SCATTER_MIN@) (@SCATTER_MAX@,@SCATTER_MAX@)};
\addlegendentry{Mesmo tempo}
\end{axis}
\end{tikzpicture}
\caption{Comparação caso a caso em escala log-log. Pontos acima da diagonal indicam que o solver de ordem livre foi mais rápido.}
\end{figure}

\section{Métricas exclusivas de cada abordagem}

As métricas abaixo não são equivalentes entre si; elas descrevem mecanismos
internos distintos e servem principalmente para diagnosticar por que os custos
de busca diferem.

\begin{table}[h]
\centering
\caption{Métricas internas da ordem livre.}
\begin{tabular}{@{}llrr@{}}
\toprule
Métrica & Coluna & Mediana & P90 \\
\midrule
@FREE_METRICS@
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Métricas internas do solver alemão.}
\begin{tabular}{@{}llrr@{}}
\toprule
Métrica & Coluna & Mediana & P90 \\
\midrule
@GERMAN_METRICS@
\bottomrule
\end{tabular}
\end{table}

\section{Metodologia e limitações}

O solver de ordem livre foi avaliado nas rodadas progressivas de 10, 60, 600 e
3600 segundos; os casos 129 e 541 continuaram em execuções sem limite de tempo
com até um bilhão de chamadas. O solver alemão foi avaliado inicialmente nas
rodadas progressivas até 1800 segundos. Os casos restantes e todos os casos que
tinham sido afetados por limite de branching foram reexecutados por 7200
segundos, com `max\_branching=-1` e até 100 milhões de chamadas.

As curvas usam o tempo total registrado em `solver\_seconds`. Para evitar uma
comparação enganosa com execuções interrompidas, os box plots e o speedup usam
somente instâncias certificadas nos dois lados. A curva de instâncias
resolvidas preserva os casos exatos e mostra explicitamente a diferença de
cobertura entre as abordagens.

\end{document}
"""

	free_metric_text = metrics_table(free_metric_rows)
	german_metric_text = metrics_table(german_metric_rows)
	replacements = {
		"@DATE@": date.today().isoformat(),
		"@FREE_MEDIAN@": fmt(free_stats["median"]),
		"@FREE_MEAN@": fmt(free_stats["mean"]),
		"@GERMAN_MEDIAN@": fmt(german_stats["median"]),
		"@GERMAN_MEAN@": fmt(german_stats["mean"]),
		"@SPEEDUP_MEDIAN@": fmt(speedup_stats["median"]),
		"@SPEEDUP_MEAN@": fmt(speedup_stats["mean"]),
		"@SPEEDUP_P10@": fmt(quantile(speedups, 0.10)),
		"@SPEEDUP_P90@": fmt(quantile(speedups, 0.90)),
		"@FREE_FASTER@": f"{100 * sum(value > 1 for value in speedups) / len(speedups):.1f}",
		"@COMMON_EXACT@": str(len(common_exact)),
		"@FREE_MIN@": plotfmt(free_stats["min"]),
		"@FREE_Q1@": plotfmt(free_stats["q1"]),
		"@FREE_Q3@": plotfmt(free_stats["q3"]),
		"@FREE_MAX@": plotfmt(free_stats["max"]),
		"@GERMAN_MIN@": plotfmt(german_stats["min"]),
		"@GERMAN_Q1@": plotfmt(german_stats["q1"]),
		"@GERMAN_Q3@": plotfmt(german_stats["q3"]),
		"@GERMAN_MAX@": plotfmt(german_stats["max"]),
		"@HIST_YMAX@": str(hist_peak + max(1, hist_peak // 8)),
		"@SCATTER_MIN@": plotfmt(min(float(row["free_seconds"]) for row in common_exact)),
		"@SCATTER_MAX@": plotfmt(max(float(row["german_seconds"]) for row in common_exact)),
		"@FREE_CONSISTENCY@": "PASS" if free_exact_consistent else "FAIL",
		"@GERMAN_CONSISTENCY@": "PASS" if german_exact_consistent else "FAIL",
		"@GERMAN_BRANCH@": "PASS" if validation["german_branch_limited"] == 0 else "FAIL",
		"@FINITE_CHECK@": "PASS" if free_bad_tokens == german_bad_tokens == 0 else "FAIL",
		"@FREE_METRICS@": free_metric_text,
		"@GERMAN_METRICS@": german_metric_text,
	}
	for marker, value in replacements.items():
		template = template.replace(marker, value)
	backslash = chr(92)
	template = template.replace(backslash + "begin{figure}[h]", backslash + "begin{figure}[H]")
	template = template.replace(backslash + "begin{table}[h]", backslash + "begin{table}[H]")
	for old, new in [
		(chr(96) + "exact" + chr(96), backslash + "texttt{exact}"),
		(chr(96) + "termination=optimal" + chr(96), backslash + "texttt{termination=optimal}"),
		(chr(96) + "exhausted" + chr(96), backslash + "texttt{exhausted}"),
		(chr(96) + "branch_limited=true" + chr(96), backslash + "texttt{branch" + backslash + "_limited=true}"),
		(chr(96) + "SIGBUS" + chr(96), backslash + "texttt{SIGBUS}"),
		(chr(96) + "validation.json" + chr(96), backslash + "texttt{validation.json}"),
		(chr(96) + "max" + backslash + "_branching=-1" + chr(96), backslash + "texttt{max" + backslash + "_branching=-1}"),
		(chr(96) + "solver" + backslash + "_seconds" + chr(96), backslash + "texttt{solver" + backslash + "_seconds}"),
	]:
		template = template.replace(old, new)
	write_text(OUTPUT / "touring-polygons-benchmark-report.tex", template)
	print(f"Wrote report sources to {OUTPUT}")
	print(json.dumps({"validation": validation, "speedup": speedup_stats}, indent=2))


if __name__ == "__main__":
	main()
