# Consolidação e modularização do TPP sem ordem

Data: 2026-09-05.

## Escopo

Esta etapa reorganizou o código produzido nas tarefas 1, 2, 3 e 5 sem alterar o
algoritmo, suas tolerâncias, seus certificados ou sua política de busca. O binário
pré-cleanup usado como referência foi
`.build/unordered-instrumented/tpp`, SHA-256
`8025b959ae429bc245aaf2d17909e035d9a156c9d73d101bc3685f92d8b40df7`.
O build canônico modularizado é `.build/unordered/tpp`, SHA-256
`bbef5c6fabcf6899785c9f18321da492a5e1274a837b40a86f2852a8e71e7fff`.

No início desta etapa, `.build/unordered/tpp` já continha o código da tarefa 5, e não
o binário histórico `a378f45b...034a`. O hash e os resultados históricos continuam
documentados, mas não se declara que esse executável histórico ainda está presente.
O baseline instrumentado da tarefa 2 foi preservado em
`.build/unordered-profiled-baseline/tpp`, hash `48cdd429...896c9`.

## Estrutura resultante

O oráculo convexo foi dividido em:

- `certified.cpp`, fluxo e tempos inclusivos;
- `certified_geometry.cpp`, contatos, reparação e dual;
- `certified_refinement.cpp`, fallback `long double` e precisão ampliada;
- `certified_internal.h`, contrato exclusivamente interno.

Na busca sem ordem, as operações geométricas puras foram movidas para
`unordered_geometry.{h,cpp}`. O branch-and-bound, a manutenção dos bounds e as
métricas permanecem juntos em `unordered.cpp`, reduzido de 335 para 268 linhas.
Os runners Python agora compartilham `unordered_runner.py`, eliminando a duplicação
do formato de entrada, chamada do subprocesso, timeout e leitura do JSON. O entry
point ausente de `free_order_campaign.py` também foi restaurado; sua execução direta
com `--dry-run` voltou a produzir a configuração da campanha.

Os dois `CMakeLists.txt` usam `GLOB_RECURSE CONFIGURE_DEPENDS`, de modo que novos
módulos são incorporados por builds existentes sem exigir uma reconfiguração manual.

## Verificação comportamental e desempenho

Foram executadas três repetições alternadas da referência e do cleanup nos casos
2, 9, 12, 37 e 55, com 2 s e 10.000.000 chamadas. Os dados estão em
`benchmarks/results/unordered/task-cleanup-20260905/`.

| Caso | Referência | Cleanup | Diferença | Estrutura da busca |
| ---: | ---: | ---: | ---: | --- |
| 2 | 0,092 ms | 0,080 ms | ruído em escala de microssegundos | idêntica |
| 9 | 1,560 s | 1,565 s | +0,3% | idêntica |
| 12 | 0,364 s | 0,360 s | -1,3% | idêntica |
| 37 | 0,590 s | 0,599 s | +1,6% | idêntica |
| 55 | 7.563 chamadas/2 s | 7.500 chamadas/2 s | -0,8% throughput | timeout em ambos |

Nos quatro casos concluídos, objetivo, lower bound, número de chamadas, nós,
fallbacks, reparações e fila máxima foram idênticos. O upper bound do caso 55 também
foi idêntico; a pequena diferença de lower bound decorre do número de nós processados
antes do mesmo limite de tempo. Todos os 30 caminhos foram validados.

A suíte recompilada passou 86 casos de enumeração exaustiva e 344 verificações com
busca interrompida. `git diff --check` e o parser de sintaxe dos scripts Python também
passaram.

## Artefatos

Foram removidos sete builds regeneráveis:

- `.build/convex-generated-tests`;
- `.build/nonconvex-release`;
- `.build/convex-generate-tests`;
- `.build/unordered-tests`;
- `.build/convex-intersection-regression`;
- `.build/convex-release`;
- `.build/unordered-cleanup`, usado apenas durante esta refatoração.

Também foram removidos metadados `.DS_Store` fora de `.git` e ambientes virtuais,
além dos `__pycache__` encontrados junto às fontes. Os diretórios de resultados
históricos, ambientes virtuais, o build canônico e os dois baselines documentados
não foram removidos. Os builds apagados não são
recuperáveis byte a byte, mas são regeneráveis a partir das fontes; nenhum resultado
experimental foi apagado. O macOS pode recriar `.DS_Store`; o padrão permanece no
`.gitignore`, portanto esses arquivos não contaminam o repositório.

## Comandos principais

```bash
cmake -S packages/nonconvex-tpp/cpp -B .build/unordered -DTARGET=main-unordered
cmake --build .build/unordered --target tpp tpp-unordered-tests -j 8
.build/unordered/tpp-unordered-tests

.venv/bin/python benchmarks/scripts/unordered_benchmark.py \
  --solver .build/unordered/tpp --seconds 2 --max-calls 10000000 \
  --case 2 --case 9 --case 12 --case 37 --case 55 \
  --output benchmarks/results/unordered/task-cleanup-20260905/cleanup-1.jsonl
```

Próximo passo: realizar a tarefa 6 sobre esta estrutura consolidada. Nenhuma política
de busca foi modificada nesta etapa.
