# Perfil do TPP não convexo com ordem livre

Medição realizada em 5 de setembro de 2026, no MacBook Pro M4 Pro descrito no
manifesto da tarefa 1. A instrumentação não altera tolerâncias, certificado,
regra de poda ou política de busca.

## Semântica das métricas

`preprocessing_seconds`, `initial_heuristic_seconds`, `search_seconds` e
`finalization_seconds` são fases superiores disjuntas. Dentro da busca,
`convex_oracle_seconds`, `decomposition_seconds`, `search_visit_check_seconds` e
`search_maintenance_seconds` formam a decomposição do tempo; manutenção é o
resíduo exclusivo. O oráculo é inclusivo e contém solver geométrico, verificação
inicial do certificado e fallback. O fallback também é inclusivo e contém as
fases em `long double` e precisão ampliada. `visit_check_seconds` soma verificações
nas três fases e, portanto, sobrepõe-se às fases superiores. Essa mesma explicação
fica em `profile.timing_semantics` de cada JSON.

Os motivos de fallback são mutuamente exclusivos: caminho geométrico sem contatos
recuperáveis, ou certificado cujo gap inicial não fechou. `extended_precision_calls`
conta chamadas que chegaram a `boost::multiprecision::cpp_bin_float_quad`.

## Casos representativos

Foram usados os casos 2, 9, 12, 37 e 55 de `algorithm-dev-v1.bin`, sempre com
uma thread, limite de 2 s e no máximo 10 milhões de chamadas. A tabela mostra a
execução instrumentada; percentuais usam `seconds` como denominador. Tempos
inclusivos internos não devem ser somados ao tempo do oráculo.

| Caso | Classe observada | Estado | Chamadas | Fallbacks | Motivo caminho | Motivo gap | Quad | Oráculo | Fallback | Visitas | Manutenção |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | rápido | ótimo | 3 | 0 (0,0%) | 0 | 0 | 0 | 11,7% | 0,0% | 4,5% | 4,7% |
| 9 | timeout | limite | 19.446 | 8.477 (43,6%) | 8.477 | 0 | 0 | 92,0% | 88,7% | 7,2% | 0,7% |
| 12 | fallback intenso, resolvido | ótimo | 1.878 | 1.644 (87,5%) | 1.422 | 222 | 3 | 96,9% | 89,5% | 2,5% | 0,3% |
| 37 | lento resolvido | ótimo | 13.424 | 9.617 (71,6%) | 9.613 | 4 | 0 | 95,9% | 93,0% | 3,5% | 0,6% |
| 55 | timeout e fallback intenso | limite | 6.882 | 6.032 (87,6%) | 5.783 | 249 | 1 | 98,7% | 94,3% | 0,9% | 0,3% |

No caso 12, a precisão ampliada consumiu 0,066 s; no 55, 0,035 s. Em uma medição
adicional do caso 59, uma única chamada ampliada consumiu 0,045 s de 0,049 s.
Logo, precisão ampliada é rara, mas pode dominar casos pequenos específicos. Nos
casos 9, 37 e 55, o custo dominante é o fallback em `long double`, e o motivo mais
frequente é não recuperar um caminho geométrico válido, não falha do dual depois
de um caminho válido.

## Overhead da instrumentação

O binário histórico `a378f45b...034a` foi executado antes do novo binário. Nos
casos resolvidos com a mesma quantidade de chamadas, as razões de tempo
instrumentado/referência foram 0,906 (caso 2, dominado por ruído), 0,995 (caso 12)
e 1,009 (caso 37). Nos timeouts, o tempo permaneceu em 2,000 s; o throughput caiu
1,3% no caso 9 e 0,3% no caso 55. Uma única execução não separa ruído térmico e de
escalonamento, mas não revelou overhead relevante acima de aproximadamente 1,3%.

Os objetivos, bounds e contagens dos três casos concluídos permaneceram idênticos.
Todos os cinco caminhos passaram no validador independente. Os timeouts podem
explorar quantidades ligeiramente diferentes de nós porque o limite é temporal.

## Próxima melhoria sustentada pelo perfil

A intervenção prioritária é separar e corrigir a recuperação do caminho geométrico
que dispara o fallback, mantendo a verificação primal/dual e o método auxiliar.
Ela mira 8.477 de 8.477 fallbacks no caso 9, 9.613 de 9.617 no caso 37 e 5.783 de
6.032 no caso 55. Otimizar fila ou decomposição primeiro teria teto pequeno nesses
casos. A precisão ampliada merece uma investigação isolada nos casos 3, 12, 55 e
59, sem reduzir precisão ou relaxar tolerâncias.

## Comandos e artefatos

```bash
cmake -S packages/nonconvex-tpp/cpp -B .build/unordered-instrumented -DTARGET=main-unordered
cmake --build .build/unordered-instrumented --target tpp tpp-unordered-tests -j 8
.build/unordered-instrumented/tpp-unordered-tests
apps/benchmark-dashboard/.venv/bin/python benchmarks/scripts/unordered_benchmark.py \
	--suite benchmarks/suites/algorithm-dev-v1.bin --solver SOLVER --seconds 2 \
	--max-calls 10000000 --case 2 --case 9 --case 12 --case 37 --case 55 --output OUTPUT
```

Resultados novos, separados da referência histórica:

- `benchmarks/results/unordered/task-2-profile-20260905/reference-validated.jsonl`
- `benchmarks/results/unordered/task-2-profile-20260905/instrumented.jsonl`

