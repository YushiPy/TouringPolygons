# Benchmark alemão: rodada progressiva dos 83 timeouts

Data da execução: 2026-09-18  
Corpus: `benchmarks/suites/german-instances.bin`  
Escopo: os 83 casos que permaneciam em `time_limit` no resultado histórico de 558 instâncias.

## Resultado

| Faixa por instância | Casos executados | Certificados | `time_limit` |
|---:|---:|---:|---:|
| 60 s | 83 | 49 | 34 |
| 600 s | 34 | 20 | 14 |
| 1.800 s | 14 | 4 | 10 |
| 3.600 s | 10 | 6 | 4 |
| **Seleção final dos 83** | **83** | **79** | **4** |

Os quatro casos que ainda não fecharam a prova foram 129, 215, 492 e 541.
Todos têm uma trajetória, bounds e resultado de término registrados; o status
`time_limit` não é contado como certificação exata.

Ao substituir os 83 registros no resultado completo histórico, o corpus de 558
instâncias passa de 475 para **554/558 certificadas**, com quatro timeouts.

## Configuração e proveniência

- Solver: `/Users/gabriel/Documents/Scripts/TouringPolygons/.build/unordered/tpp`
- Hash SHA-256 do solver: `1978496fac4fcc72b7b3d4ddd47f06fbd2a6d3453570e624108245327bca70f7`
- Commit do repositório: `d0117be3ad26e6f851a56397dfe69d291736a1c9`
- Hash SHA-256 do corpus: `80b627e8db311e2deb6d52abba476ad41bcfc191ff37ba9944cd2a0975755e31`
- Limite de chamadas: `10_000_000`
- Workers: 8 (6 na repetição independente dos seis casos finais de 1.800 s)

As rodadas usaram o mesmo corpus e o mesmo binário. O tempo observado pode
exceder marginalmente o limite cooperativo durante a finalização de uma chamada
nativa; isso ocorreu, por exemplo, no caso 541 (`3602,0676 s` observados).

## Artefatos

Os JSONL e o CSV são artefatos locais ignorados pelo Git:

- `benchmarks/results/german-instances-progressive-20260918/round-60s.jsonl`
- `benchmarks/results/german-instances-progressive-20260918/round-600s.jsonl`
- `benchmarks/results/german-instances-progressive-20260918/round-1800s.jsonl`
- `benchmarks/results/german-instances-progressive-20260918/round-3600s.jsonl`
- `benchmarks/results/german-instances-progressive-20260918/final.jsonl`
- `benchmarks/results/german-instances-progressive-20260918/focus-cases.csv`
- `benchmarks/results/german-instances-progressive-20260918/summary.json`

Hashes das rodadas usadas no merge:

| Artefato | SHA-256 |
|---|---|
| Rodada 60 s | `48b223b6bf92492de52dd43e97b7a84577d8e73da73ab09127cd4890c8c4acf4` |
| Rodada 600 s | `f5b887f0ce153f57ce5013cb4a7789e7a0b18ded4b22e9eda4e2c211d684b852` |
| Rodada 1.800 s | `2efd1b15d03074802825a9d678ba8466d2e8f5372bf56841a5cea51336fc5c11` |
| Rodada 3.600 s | `1e2a44f2d8bd0b53af880431f4605b14f0f935dac7ae5769a3b006671f500278` |
| Resultado completo final | `84c5c411bbb7ca9f4db52010000af161470696d756187c39eb6303d9ea0d54d9` |

O script versionado que faz a seleção progressiva e gera o CSV é
[merge_progressive_unordered.py](../benchmarks/scripts/merge_progressive_unordered.py).

O executável agora emite `schema_version: "free_order_v1"` com métricas de
estados parciais, posições de inserção, branching, podas por motivo, progresso
de incumbente, chamadas de oráculo por tipo, peças convexas geradas e tempos
por fase. Para gerar uma tabela semicolon-delimited a partir do JSONL, use
[export_unordered_metrics.py](../benchmarks/scripts/export_unordered_metrics.py).

As métricas de qualidade de incumbente agora são comparáveis entre os dois
solvers: `initial_length` é a aproximação, `incumbent_length` é o melhor valor
antes da busca, `best_updates` conta apenas melhorias posteriores, e
`final_length` é o resultado final. `first_best_update_length` registra o
primeiro valor após uma dessas melhorias; fica vazio quando não houve update.
No solver de ordem livre, `incumbent_updates` continua disponível como a
contagem total, incluindo a aproximação inicial.

Os JSONL desta campanha foram produzidos antes da introdução desse schema;
eles continuam válidos como resultados históricos, mas precisam ser
reexecutados para preencher as novas colunas.

## Reprodução

O conjunto inicial de 83 IDs pode ser extraído do `baseline-final.jsonl`:

```sh
FOCUS_CASES=$(python3 -c 'import json,sys; print(" ".join(f"--case {r[\"case\"]}" for r in map(json.loads, open(sys.argv[1])) if r["termination"] == "time_limit"))' \
  benchmarks/results/german-instances-progressive-20260918/baseline-final.jsonl)
python3 benchmarks/scripts/unordered_benchmark.py \
  --suite benchmarks/suites/german-instances.bin \
  --solver /Users/gabriel/Documents/Scripts/TouringPolygons/.build/unordered/tpp \
  --seconds 60 --max-calls 10000000 --workers 8 $FOCUS_CASES \
  --output benchmarks/results/german-instances-progressive-20260918/round-60s.jsonl
```

Para cada rodada seguinte, extraia os casos com `exact == false` da rodada
anterior e execute-os com o novo limite (`600`, `1800` e `3600` segundos).
Depois, gere os artefatos finais com:

```sh
python3 benchmarks/scripts/merge_progressive_unordered.py \
  --baseline benchmarks/results/german-instances-progressive-20260918/baseline-final.jsonl \
  --suite benchmarks/suites/german-instances.bin \
  --round 60=benchmarks/results/german-instances-progressive-20260918/round-60s.jsonl \
  --round 600=benchmarks/results/german-instances-progressive-20260918/round-600s.jsonl \
  --round 1800=benchmarks/results/german-instances-progressive-20260918/round-1800s.jsonl \
  --round 3600=benchmarks/results/german-instances-progressive-20260918/round-3600s.jsonl \
  --output benchmarks/results/german-instances-progressive-20260918/final.jsonl \
  --csv benchmarks/results/german-instances-progressive-20260918/focus-cases.csv \
  --summary benchmarks/results/german-instances-progressive-20260918/summary.json
```

## Validação

Foi executada uma checagem estrutural independente nas 558 linhas finais:
cobertura completa, hash de cada entrada contra o corpus, bounds finitos e
consistentes, permutação completa dos polígonos, caminhos finitos e coerência
entre `exact`, `termination` e o gap dos bounds. Essa checagem passou.

A auditoria geométrica existente (`summarize_unordered_siicusp.py`) requer
Shapely. O pacote não está disponível no Python ativo, e a tentativa de usar o
ambiente travou por falta de rede para baixar dependências; portanto a coluna
`valid` permanece não preenchida nos novos resultados e nenhuma afirmação de
validade geométrica independente é feita aqui. O dashboard e seus dados
históricos não foram alterados.
