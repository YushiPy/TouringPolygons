# Pôster A0 — 34º SIICUSP

## Versão atual

A fonte atual está em
[`poster.tex`](poster.tex); o PDF local correspondente é
[`poster.pdf`](poster.pdf). O PDF é ignorado pelo Git e pode
ser regenerado a partir da fonte.

As propostas e versões anteriores ficam no histórico Git.
Os arquivos `*.aux`, `*.fdb_latexmk`, `*.fls`,
`*.log` e `*.out` são auxiliares descartáveis do LaTeX; compile em `/tmp` para
mantê-los fora desta pasta.

## Compilar a versão revisada

Requer XeLaTeX, Arial, Latin Modern Math e os pacotes do preâmbulo. A partir
desta pasta:

```sh
mkdir -p /tmp/siicusp34-poster-build
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/siicusp34-poster-build poster.tex
cp /tmp/siicusp34-poster-build/poster.pdf poster.pdf
```

O pôster é uma página A0 em retrato. Corpo e referências usam 26 pt. As figuras
e sua proveniência estão documentadas em [`figures/README.md`](figures/README.md);
a captura do mapa vem do app estático congelado em `apps/siicusp34` e não é
refeita ao compilar.

## Fontes dos dados

- A demonstração da USP e seus dados: [`USP-DEMO.md`](../../../../../apps/siicusp34/data/USP-DEMO.md).
- Os resultados comparativos: campanha preservada em
  [`benchmarks/results-saved/fekete-comparison`](../../../../../benchmarks/results-saved/fekete-comparison/README.md).
- A formulação do TPP de ordem livre: [`unordered-tpp.md`](../../../../algorithms/unordered-tpp.md).

As métricas, tolerâncias e qualificações exibidas são mantidas na fonte do
pôster para evitar uma segunda cópia que possa ficar desatualizada.
