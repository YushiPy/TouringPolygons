# Comparação alemã: Gurobi 13 versus solver exato

Data da execução: 2026-09-18  
Corpus: `benchmarks/campaigns/german-instances/inputs/german-instances.bin`  
Instâncias: 558  
Limite por instância: 10 s  
Paralelismo: oito processos concorrentes, um thread do solver por processo

## Ambiente Gurobi

O solver nativo foi recompilado contra a instalação local:

`/Library/gurobi1303/macos_universal2`

O binding Python recompilado é:

`tspn-comparison/solver/python/tspn_bnb2/core/_tspn_bindings.cpython-312-darwin.so`

Também foi instalado `gurobipy==13.0.3` no ambiente usado pelo runner. A
licença acadêmica foi reconhecida e o smoke test resolveu uma instância real.
O runner foi ajustado para aceitar tanto a API antiga do checkout alemão
(sem os argumentos opcionais do oráculo) quanto a API nova.

## Resultado pareado

| Métrica | Solver exato/híbrido | Baseline Gurobi 13 |
|---|---:|---:|
| Resultado certificado | 475/558 (85,1%) | 437/558 (78,3%) dentro de 0,1% numérico |
| Instâncias no limite | 83/558 (14,9%) | 121/558 (21,7%) |
| Mediana do tempo | 0,0772 s | 0,2441 s |
| Percentil 95 | 10,0025 s | 10,0133 s |
| Tempo somado dos processos | 1.236,974 s | 1.591,246 s |

O resultado Gurobi é uma certificação numérica baseada em limite inferior e
superior; ele não é uma prova exata. Os 437 casos marcados como `optimal`
satisfazem a tolerância relativa de 0,1% do runner. No solver exato, os 475
casos certificados são provas exatas; os casos interrompidos não são contados
como certificados, mesmo quando já possuem uma trajetória válida.

Na comparação por índice:

| Situação | Casos |
|---|---:|
| Ambos certificados | 434 |
| Apenas solver exato certificado | 41 |
| Apenas Gurobi dentro da tolerância | 3 |
| Ambos no limite | 80 |

Nos 434 casos certificados por ambos, a mediana foi 0,0185 s para o solver
exato e 0,0779 s para o Gurobi. O solver exato foi mais rápido em 392/434
casos (90,3%); a razão mediana `tempo Gurobi / tempo exato` foi 5,27.

## Robustez geométrica

As trajetórias do Gurobi foram auditadas independentemente. Com a tolerância
numérica usada pelo próprio solver (`1e-3`), 558/558 trajetórias passam. Com
a tolerância estrita `1e-7`, apenas 128/558 passam após o ajuste dos extremos
(48/558 passam sem esse ajuste). Isso é esperado de uma saída numérica e não
deve ser apresentado como exatidão. No solver exato, a auditoria estrita
`1e-7` passou para 558/558 trajetórias, com distância máxima de `3,53e-11`.

## Artefatos locais

O CSV consolidado do baseline está em:

`tspn-comparison/results/german-gurobi13-20260918/final.csv`

O manifesto da execução está em:

`tspn-comparison/results/german-gurobi13-20260918/manifest.json`

Esses artefatos ficam no checkout privado `tspn-comparison`, ignorado pelo
repositório principal. O CSV contém os limites inferior/superior e gaps do
Gurobi exclusivamente para a comparação técnica; eles não foram reintroduzidos
no site público.

Hashes:

| Artefato | SHA-256 |
|---|---|
| Corpus | `80b627e8db311e2deb6d52abba476ad41bcfc191ff37ba9944cd2a0975755e31` |
| Binding Gurobi 13 | `52a2d81349744778cd80924400fe077b788a50d6b5ed77687db33cd375cf0e07` |

