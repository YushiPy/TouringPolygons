# O oráculo convexo certificado

Relatório técnico (em português) sobre o oráculo convexo de ordem fixa usado
pelo branch-and-bound de ordem livre: mapas de último passo, por que o
algoritmo é sensível à aritmética, a certificação primal-dual em `binary64`
com arredondamento dirigido, o polimento por pontos interiores, o fallback
exato e as medições.

Compilar a partir de qualquer diretório:

```bash
bash docs/reports/oraculo-convexo/build.sh
```

O script grava `oraculo-convexo.pdf` nesta pasta (ignorado pelo Git) e os
arquivos auxiliares em `.build/oraculo-convexo/`.

- `main.tex`: texto, pseudocódigo e bibliografia.
- `figures.tex`: figuras TikZ/pgfplots. Os desenhos geométricos usam
  coordenadas exatas dos exemplos do texto, com a mesma escala nos dois eixos.
  O mapa de último passo e o desdobramento vêm do tutorial
  `docs/reports/directional-tpp-tutorial`, e o mapa foi conferido por força
  bruta numa grade de 5.117 pontos.
- `dados.tex`: tabelas das figuras de medição, extraídas das campanhas locais
  (`free-order-history`, `float-recovery-dantzig`). Os resumos e as
  limitações de cada campanha estão em `benchmarks/results-saved/README.md`.

Os exemplos numéricos (quadrado e alvo t_A) foram conferidos em aritmética
exata e reexecutados no oráculo com `tpp-convex-path-oracle-replay`.
