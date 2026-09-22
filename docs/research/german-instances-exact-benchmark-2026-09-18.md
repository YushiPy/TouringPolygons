# Benchmark alemão: solver exato/híbrido

Data da execução: 2026-09-18  
Corpus: `benchmarks/campaigns/german-instances/inputs/german-instances.bin`  
Instâncias: 558

## Resultado

| Métrica | Resultado |
|---|---:|
| Instâncias certificadas | 475/558 (85,1%) |
| Instâncias no limite de tempo | 83/558 (14,9%) |
| Trajetórias independentemente válidas | 558/558 (100%) |
| Mediana do tempo por instância | 0,0772 s |
| Percentil 95 do tempo | 10,0025 s |
| Tempo somado dos processos | 1.236,974 s |
| Chamadas ao oráculo | 1.883.479 |
| Chamadas de fallback racional | 52.518 |
| Maior distância de uma região | 3,53e-11 |
| Maior erro de comprimento reconstituído | 6,98e-10 |

Os 83 casos que atingiram o limite ainda produziram trajetórias válidas; eles
não são contados como certificados. O tempo-limite é cooperativo, portanto o
tempo observado pode exceder marginalmente 10 s enquanto uma chamada nativa é
finalizada (máximo observado: 10,0198 s).

## Protocolo

1. Todas as 558 instâncias foram executadas com limite inicial de 2 s, quatro
   processos concorrentes e um solver por processo.
2. Os 148 casos que atingiram o limite inicial foram reexecutados com limite de
   10 s e oito processos concorrentes.
3. Para cada caso, foi escolhida a execução de 10 s quando ela existia; os
   outros 410 casos mantiveram a execução inicial. Assim, não há duplicatas no
   resultado final.
4. Cada trajetória foi auditada independentemente contra a geometria codificada,
   com tolerância de validação `1e-7`.

Na rodada inicial, 410 casos foram certificados. A extensão para 10 s
certificou mais 65 dos 148 casos inicialmente interrompidos, reduzindo o
número final de interrupções para 83.

O corpus é a adaptação documentada das instâncias alemãs: regiões simplificadas
sem buracos, ordem livre e dois depósitos externos determinísticos. Portanto,
ele representa um problema de caminho com extremos fixos; não é uma reprodução
literal do problema original de ciclo fechado do solver alemão.

## Artefatos locais

Os JSONL completos ficam em:

`benchmarks/results/german-instances-exact-20260918/`

Eles são artefatos de benchmark ignorados pelo Git. O arquivo final é
`final.jsonl`; `final-summary/summary.json` e `final-summary/cases.csv` são a
auditoria independente. O script versionado que combina a rodada inicial com a
refinada é [merge_unordered_runs.py](../benchmarks/scripts/merge_unordered_runs.py).

## Reprodução

```sh
python3 benchmarks/scripts/unordered_benchmark.py \
  --suite benchmarks/campaigns/german-instances/inputs/german-instances.bin \
  --solver .build/unordered/tpp \
  --seconds 2 --workers 4 \
  --output benchmarks/results/german-instances-exact-20260918/initial-2s.jsonl
```

Em seguida, extraia os casos com `termination == "time_limit"` da primeira
rodada e execute-os com `--seconds 10 --workers 8`. Finalmente, combine as duas
rodadas com `merge_unordered_runs.py` e audite o arquivo combinado com
`summarize_unordered_siicusp.py`.

Hashes usados nesta execução:

| Artefato | SHA-256 |
|---|---|
| Corpus | `80b627e8db311e2deb6d52abba476ad41bcfc191ff37ba9944cd2a0975755e31` |
| Solver `.build/unordered/tpp` | `a860c5d158804a161ea92011af201b5f1e8d972c2376a27dc66c8a634e099568` |
| Commit do repositório | `38c3d7c2e6f121e4eb0d77dedeba810c7decaef1` |

## Comparação com o solver alemão

O baseline foi recompilado contra o Gurobi 13.0.3 e executado nas mesmas 558
instâncias adaptadas. Os resultados pareados estão em
[german-instances-gurobi13-comparison-2026-09-18.md](german-instances-gurobi13-comparison-2026-09-18.md).
O CSV bruto do baseline fica no checkout privado `tspn-comparison`.

Essa comparação separa explicitamente prova exata de certificação numérica:
o Gurobi reporta limites inferior/superior e tolerância relativa, enquanto o
solver deste relatório só chama um caso de certificado quando a prova racional
é concluída.
