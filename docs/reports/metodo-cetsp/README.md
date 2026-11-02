# O CETSP por geometria

Texto curto (em português, 5 páginas) pedido por Ernesto em 28/10. Explica o
oráculo de discos de ordem fixa e o branch-and-bound sobre ele, com
pseudocódigo, para um caminho de s a t (o depósito, se s = t):
- o certificado por dualidade fraca, em forma fechada;
- a atualização de Alhazen;
- as varreduras, o Newton que fecha o tipo, os blocos e os duais;
- os limites de inserção;
- o branch-and-bound;
- a relação com Coutinho et al. e com a busca por blocos de coordenadas;
- os resultados.

Compilar a partir de qualquer diretório:

```bash
bash docs/reports/metodo-cetsp/build.sh
```

O script grava `metodo-cetsp.pdf` nesta pasta (ignorado pelo Git) e os
arquivos auxiliares em `.build/metodo-cetsp/`.

Origem dos números: `benchmarks/results-saved/README.md`, seções
`cetsp-table2-mac-2026-10-27` e `cetsp-campaign2-mac-2026-10-26`, e
`docs/algorithms/convex-cycle.md`, seção *Disks*.
