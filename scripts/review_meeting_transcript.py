from __future__ import annotations

import argparse
import re
from pathlib import Path


REPLACEMENTS = {
	"prorrogação da WoWs": "prorrogação da bolsa",
	"regapto": "recapitulo",
	"regação da bolsa": "prorrogação da bolsa",
	"algoritmo de Greenee": "algoritmo de Greene",
	"algoritmo de Green,": "algoritmo de Greene,",
	"decomposição complexa": "decomposição convexa",
	"biblioteca SEGAL": "biblioteca CGAL",
	"biblioteca Segal": "biblioteca CGAL",
	"biblioteca segal": "biblioteca CGAL",
	"biblioteca cegal": "biblioteca CGAL",
	"standard points": "Steiner points",
	"adestas": "arestas",
	"maresta": "uma aresta",
	"NPR Hard": "NP-hard",
	"NPR hard": "NP-hard",
	"problema é NPR": "problema é NP-hard",
	"não tem NPR": "não tem como evitar o comportamento exponencial",
	"sobrepossição": "sobreposição",
	"bounds piores do Bernard Chazelle": "bounds piores no Branch and Bound",
	"composto convexa": "problema de decomposição convexa",
	"web sem mulher": "WebAssembly",
	"Brandtian Bound": "Branch and Bound",
	"brancha de bão": "Branch and Bound",
	"branch no bottom": "Branch and Bound",
	"branch about": "Branch and Bound",
	"branch on bound": "Branch and Bound",
	"Branchenbaum": "Branch and Bound",
	"pedímetro": "perímetro",
	"colígono": "polígono",
	"colígonos": "polígonos",
	"polígão": "polígono",
	"polígãos": "polígonos",
	"poligonocinhos": "polígonos convexos menores",
	"casco com vexo": "casco convexo",
	"casinhos convexos": "cascos convexos",
	"intercessões": "interseções",
	"intercessão": "interseção",
	"DUROR": "Dror",
	"DOR": "Dror",
	"do door": "de Dror",
	"do DUROR": "de Dror",
	"do TANENJANG": "de Tan e Jiang",
	"TANENJANG": "Tan e Jiang",
	"Tanen Zhang": "Tan e Jiang",
	"UROB": "Gurobi",
	"UROBI": "Gurobi",
	"Guroby": "Gurobi",
	"Sikuspi": "SIICUSP",
	"Sikusp": "SIICUSP",
	"bolsa do BEP": "bolsa do BEPE",
	"pedido BEP": "pedido de BEPE",
	"bolsa de Bepi": "bolsa de BEPE",
	"reconsideração da Beppi": "reconsideração do BEPE",
	"bolsa de papel": "bolsa do BEPE",
	"viciâncias": "vizinhanças",
	"viciância": "vizinhança",
	"poligonárias": "poligonais",
	"poligonária": "poligonal",
	"FRP": "VRP",
	"closing up TSP": "Close-Enough TSP",
	"closing up": "Close-Enough",
	"GTSP Libby": "GTSP-LIB",
	"GTSP Libre": "GTSP-LIB",
	"GTSP LIBE": "GTSP-LIB",
	"MOM Libby": "MOM-LIB",
	"Mom Libre": "MOM-LIB",
	"MOM LIBE": "MOM-LIB",
	"sessões do livro": "seções do livro",
	"sessões que eu ensinei": "seções que eu ensinei",
	"C mais mais": "C++",
	"c mais mais": "C++",
	"A-Star": "A*",
	"breadth for search": "breadth-first search",
	"Universidade de Laval": "Université Laval",
	"instação científica": "iniciação científica",
	"10 mil dores canadenses": "10 mil dólares canadenses",
	"indiferimento": "indeferimento",
	"indiferir": "indeferir",
	"Fabespi": "FAPESP",
	"vê quanto custa nos passagens": "vê quanto custam as passagens",
	"a consideração": "a reconsideração",
}


def revise(text: str) -> str:
	for source, target in REPLACEMENTS.items():
		text = text.replace(source, target)
	text = re.sub(r"(?:O que eu acho legal\?\s*){2,}", "O que eu acho legal? ", text)
	text = re.sub(r"\bTCP\b", "TSP", text)
	text = re.sub(r"\bSEGAL\b|\bcegal\b", "CGAL", text)
	text = re.sub(r"\bDUROR\b|\bDOR\b", "Dror", text)
	text = re.sub(r"\bBEP(?:I)?\b|\bBeppi\b", "BEPE", text)
	text = re.sub(r"\s+([,.?!:;])", r"\1", text)
	return text


def limit_duration(text: str, end_time: float | None) -> str:
	if end_time is None:
		return text
	blocks = text.strip().split("\n\n")
	kept: list[str] = []
	for block in blocks:
		match = re.match(r"\[(\d{2}):(\d{2}):(\d{2}),\d{3}", block)
		if match is None:
			continue
		hours, minutes, seconds = map(int, match.groups())
		start = hours * 3600 + minutes * 60 + seconds
		if start <= end_time:
			kept.append(block)
	return "\n\n".join(kept) + "\n"


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Aplica correções contextuais recorrentes à transcrição de uma reunião."
	)
	parser.add_argument("transcript", type=Path)
	parser.add_argument("--output", type=Path)
	parser.add_argument(
		"--end-time",
		type=float,
		help="Ignora segmentos iniciados após este instante, em segundos.",
	)
	args = parser.parse_args()

	output = args.output or args.transcript.with_name("transcrição-revisada.txt")
	raw = limit_duration(args.transcript.read_text(encoding="utf-8"), args.end_time)
	header = (
		"TRANSCRIÇÃO REVISADA\n"
		"Revisão contextual de nomes próprios e termos técnicos. Os timestamps foram "
		"preservados. Trechos ainda pouco claros devem ser conferidos no áudio.\n"
	)
	if args.end_time is not None:
		header += (
			f"Conteúdo limitado aos primeiros {args.end_time:g} segundos, para excluir "
			"áudio posterior que não pertence à reunião.\n"
		)
	header += "\n"
	output.write_text(header + revise(raw), encoding="utf-8")


if __name__ == "__main__":
	main()
