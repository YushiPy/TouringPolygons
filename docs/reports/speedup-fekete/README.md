# Speedup sobre o solver de Fekete et al.

Nota curta (em português, 3 páginas) com a razão de tempo entre o solver de
Fekete et al. e o nosso no TPP de ordem livre com extremos fixos, separada por
fonte das instâncias (OSM, aleatórias, Voronoi) e pela geometria dos polígonos
(disjuntas, só toques de fronteira, sobreposição com área), e com a
distribuição das chamadas ao polimento por pontos interiores entre essas
classes.

Compilar a partir de qualquer diretório:

```bash
bash docs/reports/speedup-fekete/build.sh
```

O script grava `speedup-fekete.pdf` nesta pasta (ignorado pelo Git) e os
arquivos auxiliares em `.build/speedup-fekete/`.

- `main.tex`: texto, figuras pgfplots e tabelas.
- `dados.tex`: pontos dos gráficos (um por caso) e médias geométricas por
  faixa de 10 polígonos.

Origem dos dados: tempos do Fekete em
`benchmarks/results-saved/free-order-dantzig-2026-10-06/per-case.csv`; fonte e
geometria em
`benchmarks/results-saved/fekete-comparison/analysis/instance-classification.csv`;
nossos tempos e contagens de chamadas na rodada da dantzig de 08/10
(`float-recovery-dantzig`, campanha local, resumida na seção
`float-recovery-dantzig-2026-10-08` de `benchmarks/results-saved/README.md`).
Os casos são casados pelo índice e conferidos pelo SHA-256 da geometria e pelo
número de polígonos. A seção "Dados e limitações" do PDF registra a
formulação, as tolerâncias e as ressalvas da comparação.

As paletas dos dois gráficos foram validadas para daltonismo (diferença
perceptual entre cores vizinhas) e são diferentes entre si, para que a cor não
sugira correspondência entre fonte e geometria.
