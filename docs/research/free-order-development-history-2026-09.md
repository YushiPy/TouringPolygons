# Histórico de desenvolvimento do TPP não convexo com ordem livre

Este documento consolida os relatórios de perfil, reparação, cleanup e melhorias
produzidos entre 5 e 10 de setembro de 2026. Ele preserva a evidência histórica
das decisões de implementação sem substituir a especificação atual em
`docs/algorithms/`.

## Escopo e contrato experimental

Os experimentos tratam o TPP não convexo com extremos fixos e ordem livre. Os
resultados chamados de “fechados” ou “ótimos” significam que o gap numérico
configurado foi fechado; não são certificados de aritmética exata ou intervalar.
Os caminhos foram validados independentemente, incluindo extremos, contatos e
comprimento recomputado.

Os resultados locais, binários e suítes geradas não foram adicionados ao Git.
Os caminhos de artefatos abaixo servem como referência para reproduções locais,
mas podem não existir em um clone novo.

## 1. Perfil do gargalo — 5 de setembro

O perfil usou `algorithm-dev-v1.bin`, uma thread, limite de dois segundos e no
máximo 10 milhões de chamadas. As fases superiores são disjuntas; dentro da
busca, os tempos do oráculo e do fallback são inclusivos. Portanto, esses
percentuais não devem ser somados.

| Caso | Estado | Chamadas | Fallbacks | Principal motivo | Oráculo | Fallback |
| ---: | --- | ---: | ---: | --- | ---: | ---: |
| 2 | ótimo | 3 | 0 | — | 11,7% | 0,0% |
| 9 | timeout | 19.446 | 8.477 | caminho não recuperável | 92,0% | 88,7% |
| 12 | ótimo | 1.878 | 1.644 | caminho não recuperável | 96,9% | 89,5% |
| 37 | ótimo | 13.424 | 9.617 | caminho não recuperável | 95,9% | 93,0% |
| 55 | timeout | 6.882 | 6.032 | caminho não recuperável | 98,7% | 94,3% |

O fallback consumia a maior parte do tempo nos casos não triviais, e o motivo
dominante era a recuperação de contatos a partir do caminho geométrico, não a
falha do certificado dual depois de um caminho válido. A instrumentação não
alterou objetivos, bounds ou contagens nos casos concluídos; o overhead observado
ficou abaixo de aproximadamente 1,3% em uma execução.

## 2. Reparação de contatos — 5 de setembro

Foi adicionada uma tentativa conservadora de reparar o caminho quando havia
exatamente um ponto por região. Cada ponto era combinado com o centróide da
região usando deslocamentos `1e-12`, `1e-10` e `1e-8`. Cada candidato passava
novamente pela recuperação ordenada de contatos, pelo cálculo primal e pelo mesmo
limite dual. Se viabilidade ou gap não fechassem, o fallback original permanecia
obrigatório. Nenhuma tolerância, margem, bound ou regra de poda foi relaxada.

| Caso | Referência | Com reparação | Resultado |
| ---: | ---: | ---: | --- |
| 2 | 0,083 ms | 0,086 ms | Mesmo ótimo; diferença dominada por ruído |
| 9 | timeout, 19.075 chamadas | ótimo em 1,613 s | A prova passou a fechar no limite |
| 12 | 0,405 s | 0,381 s | 5,9% mais rápido |
| 37 | 1,757 s | 0,623 s | 64,5% mais rápido |
| 55 | timeout, 6.822 chamadas | timeout, 7.321 chamadas | Mesmo bound; mais chamadas no limite |

Foram feitas três repetições alternadas por variante nos cinco casos. Os 30
caminhos foram validados independentemente. Um warm start que misturava o caminho
geométrico com centróides foi rejeitado: ficou 7,5% mais lento no caso 37 e
2,4% mais lento no caso 12. Seus resultados permanecem apenas nos artefatos
locais ignorados.

## 3. Modularização e cleanup — 5 de setembro

O código foi reorganizado sem alterar algoritmo, tolerâncias, certificados ou
política de busca:

- o oráculo certificado foi dividido em `certified.cpp`,
  `certified_geometry.cpp`, `certified_refinement.cpp` e
  `certified_internal.h`;
- as operações geométricas puras da busca foram movidas para
  `unordered_geometry.{h,cpp}`;
- os runners Python passaram a compartilhar `unordered_runner.py`;
- o entry point de `free_order_campaign.py` foi restaurado;
- os `CMakeLists.txt` passaram a incorporar novos módulos com
  `GLOB_RECURSE CONFIGURE_DEPENDS`.

Três repetições alternadas nos casos 2, 9, 12, 37 e 55 mantiveram objetivos,
bounds, chamadas, nós, fallbacks, reparações e fila máxima nos casos concluídos.
A variação de tempo ficou entre `-1,3%` e `+1,6%`; no timeout do caso 55, a
diferença de throughput foi de `-0,8%`. A suíte recompilada passou 86 casos por
enumeração exaustiva e 344 verificações de busca interrompida.

## 4. Melhorias acumuladas — 10 de setembro

O baseline era o commit `d29b4df3943f8489c9e51037ba794c903457c332`. A sequência de
ablações foi executada com os mesmos extremos, limite de 10 milhões de chamadas,
gap final `1e-7 + 1e-9 * upper_bound` e limite de tempo flexível.

| Variante | Fechados / 60 | Tempo total | Decisão |
| --- | ---: | ---: | --- |
| Original | 43 | 62,78 s | Baseline |
| Reparação de caminhos de tamanho variável | 46 | 54,92 s | Mantida |
| Parar refinamento quando o bound permite poda | 50 | 43,06 s | Mantida |
| Precisão interna adaptativa e refinamento estrito das folhas | 53 | 28,46 s | Mantida |
| Otimização analítica de contatos e 2-opt repetido | 53 | 28,41 s | Mantida |
| Screening support-dual por filho | 53 | 26,54 s | Substituída pela versão incremental |
| Screening incremental de inserção | 55 | 25,19 s | Mantida |
| Rejeição por bounding box nas visitas | 55 | 22,26 s | Mantida |
| Escolha alternativa da raiz | 55 | 22,18 s | Rejeitada: ganho desprezível |
| Cache de cobertura de nós | 55 | 22,12 s | Rejeitado: ganho desprezível |
| Comparações de distância ao quadrado | 55 | 21,36 s | Mantida como candidata final |

O resultado final foi comparado também em conjuntos mantidos fora do
desenvolvimento:

| Carga | Limite | Baseline fechou | Melhorado fechou | Baseline | Melhorado |
| --- | ---: | ---: | ---: | ---: | ---: |
| Desenvolvimento, 60 casos | 3 s | 43 | 55 | 62,78 s | 21,36 s |
| Canonical holdout, 60 casos | 1 s | 32 | 53 | 33,76 s | 13,41 s |
| Reference holdout, 49 casos | 1 s | 29 | 43 | 24,11 s | 8,83 s |
| Todos os holdouts, 109 casos | 1 s | 61 | 96 | 57,87 s | 22,24 s |

Todos os caminhos passaram validação independente. O holdout é separação por
hash de instância dentro de corpora relacionados; não demonstra generalização
para qualquer distribuição. As razões de tempo incluem execuções limitadas e
não são estimativas de speedup sem censura.

O bound incremental de inserção reutiliza termos dual dos segmentos que não
mudaram. A implementação usa `long double`, direções alternativas para segmentos
de comprimento zero e uma margem dependente da escala. Folhas factíveis ainda
passam pelo refinamento da tolerância final. Isso mantém as convenções de
segurança de ponto flutuante, mas não constitui prova intervalar para magnitudes
arbitrárias.

## 5. Validação e comparação externa

As verificações registradas incluíram:

- 86 casos exaustivos e 344 casos com busca interrompida;
- 200 caminhos de referência, incluindo contatos coincidentes e inviáveis,
  comparando os três bounds de inserção com solves de ordem fixa;
- 24 comparações independentes com Gurobi, oito delas usando decomposição não
  convexa explícita;
- validação independente de todos os caminhos retornados nos benchmarks.

No solver externo `tspn_bnb2` 0.2.1, executado nos mesmos 60 casos de
desenvolvimento com três segundos por caso, apenas cinco casos declararam
otimalidade na tolerância solicitada. A validação independente aceitou 13
caminhos brutos, ou 28 após encaixe dos extremos. Desvios máximos observados
foram aproximadamente `8,15e-5` nos extremos e `3,42e-5` nas regiões.

Esses números não sustentam uma alegação de superioridade universal: as
dificuldades numéricas do solver externo impedem uma comparação justa em
tolerância apertada. A conclusão sustentada é a melhoria do solver próprio sobre
seu baseline; uma comparação com tolerância comum e tempos repetidos permanece
trabalho futuro.

## Reprodução local

Os comandos abaixo são representativos e requerem as suítes e ambientes locais:

```bash
cmake -S packages/nonconvex-tpp/cpp -B .build/unordered-instrumented \
  -DTARGET=main-unordered
cmake --build .build/unordered-instrumented \
  --target tpp tpp-unordered-tests -j 8
.build/unordered-instrumented/tpp-unordered-tests

python3 benchmarks/scripts/unordered_benchmark.py \
  --solver .build/unordered-instrumented/tpp \
  --seconds 2 --max-calls 10000000 \
  --case 2 --case 9 --case 12 --case 37 --case 55 \
  --output benchmarks/results/unordered/local-run.jsonl
```

Para a comparação de ablações, os resultados locais ficam em
`benchmarks/results/free-order-improvements/`; os binários baseline e melhorado
ficam em `.build/unordered-improvement-baseline/` e
`.build/unordered-improved/`. Esses artefatos são regeneráveis ou locais e não
devem ser adicionados ao Git.

## Limitações

- Os benchmarks de desenvolvimento são pequenos e várias medições são sweeps
  únicos, não estudos estatísticos repetidos.
- Timeouts não podem ser tratados como falhas ou sucessos de otimalidade sem
  registrar explicitamente o status.
- O caminho geométrico ainda exige certificação primal/dual; nunca o aceite
  apenas por parecer válido.
- A comparação externa usa formulações e tolerâncias diferentes.
- As podas geométricas da busca sem ordem continuam sujeitas às provas e
  contraexemplos documentados nos relatórios de branching e pruning.
