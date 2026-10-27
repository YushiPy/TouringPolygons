# O oráculo do ciclo convexo de ordem fixa

Texto curto (em português, 5 páginas) que explica o oráculo do ciclo por partes,
para a reunião de 30/10: o certificado (Teorema A), a construção por reflexões e
a forma fechada de um tipo, o polimento, o caminho exato com âncoras, o passo em
forma fechada e a bissecção (Teorema B). Responde às perguntas de 23/10 com
medições: quantas sondas a bissecção faz e se os pontos interiores podem sair.

Compilar a partir de qualquer diretório:

```bash
bash docs/reports/metodo-ciclo/build.sh
```

O script grava `metodo-ciclo.pdf` nesta pasta (ignorado pelo Git) e os
arquivos auxiliares em `.build/metodo-ciclo/`.

Origem dos números: `benchmarks/results-saved/README.md`, seções
`fekete-reference-armijo-2026-10-17` (frações das chamadas),
`anchor-step-mac-2026-10-23` (sondas) e `no-polish-mac-2026-10-23` (pontos
interiores). Provas completas em `docs/algorithms/convex-cycle.md` e
`convex-cycle-certificate.md`.
