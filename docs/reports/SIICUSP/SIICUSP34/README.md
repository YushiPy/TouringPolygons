# 34º SIICUSP

O edital e os critérios de avaliação (`edital.pdf`, `criterios.pdf`, locais e
ignorados) e o [plano de trabalho](PLANO-DE-TRABALHO.md) valem para as duas
etapas.

- [`fase-local/`](fase-local/): Primeira Etapa no IME (apresentada até 16/10/2026).
  Resumos em português e inglês como submetidos, pôster, slides e roteiro.
  É um registro do que foi apresentado; não atualize os números.
- [`fase-internacional/`](fase-internacional/): Etapa Internacional, presencial,
  em 25 e 26/11/2026 (sessão de pôster). O material é todo em inglês (edital,
  item 4.5). O resumo novo vai para cpqi@ime.usp.br até 30/10/2026, com o
  assunto "SIICUSP Internacional: Materiais". As orientações finais chegam até
  10/11/2026.

O resumo internacional reaproveita o logo de `fase-local/resumo/assets/`. A
figura 1 (`assets/tspn-figure.pdf`, ignorada) é gerada por
`fase-internacional/resumo/generate_tspn_figure.py`, que decompõe as regiões
com a nossa implementação de Greene e resolve os tours com `tpp-unordered`:

A figura 2 (`assets/speedup-figure.pdf`, ignorada) é gerada por
`generate_speedup_figure.py` a partir de
`benchmarks/results-saved/fekete-reference-armijo-2026-10-30/per-case.csv`:
o código original de Fekete et al. com uma thread (configuração B, raiz CHR)
contra o nosso com uma thread (F), nas 541 instâncias que os dois resolvem,
com o mesmo critério U−L ≤ 10⁻³U + 10⁻³.

```sh
cd fase-internacional/resumo
../../../../../../apps/benchmark-dashboard/.venv/bin/python generate_tspn_figure.py
../../../../../../apps/benchmark-dashboard/.venv/bin/python generate_speedup_figure.py
```
