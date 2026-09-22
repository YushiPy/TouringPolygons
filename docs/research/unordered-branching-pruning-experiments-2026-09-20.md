# Experimentos de branching e pruning no TPP não convexo de ordem livre

Data: 20 de setembro de 2026.

## Conclusão

Nenhuma das três sugestões deve substituir o comportamento atual do solver.

Os dois contraexemplos, incluindo os nós exatos e entradas executáveis, estão em
[`unordered-pruning-counterexamples-2026-09-20.md`](unordered-pruning-counterexamples-2026-09-20.md).

- Escolher o polígono não visitado mais próximo foi uma regressão grande.
- Podar uma relaxação que entra, sai e volta ao mesmo polígono não é correto para
  caminhos com extremos fixos. O benchmark encontrou certificados incorretos.
- Podar toda relaxação parcial cujo caminho se cruza também não é correto. Uma
  inserção futura pode substituir os segmentos que se cruzam. A variante parecia
  ligeiramente mais rápida nos corpora, mas um teste aleatório pequeno encontrou
  um contraexemplo.
- A versão correta da última regra, aplicada somente quando o caminho relaxado já
  visita todos os polígonos, foi redundante com a poda por bound/incumbente: fez
  zero checagens e zero podas na suíte canônica.

Os protótipos rejeitados foram removidos do solver. O benchmarker passou a aceitar
argumentos específicos por variante com `--solver-argument LABEL=ARGUMENT`, o que
permite repetir ablações isoladas usando o mesmo executável.

## Protocolo

- Solver: `packages/nonconvex-tpp`, exclusivamente a entrada de ordem livre.
- Um processo por vez, `OMP_NUM_THREADS=1`.
- Limite de 1 segundo por instância.
- Limite de calls: `18446744073709551615` (`size_t::max` em 64 bits).
- Sem limite de branching (o solver não possui cap adicional nessa entrada).
- Uma repetição, com ordem dos solvers rotacionada por caso.
- Tempo medido internamente pelo solver; inicialização do processo e validação não
  entram em `seconds`.
- Todos os caminhos foram validados independentemente com Shapely a `1e-7`.

Suítes:

| Suíte | Casos | SHA-256 |
|---|---:|---|
| `benchmarks/suites/canonical-v1.bin` | 300 | `39d43a1cc727378b708443613fefac3fb361384f35df4c31971b1b4596445974` |
| `benchmarks/suites/german-instances.bin` | 558 | `aa442e0546567461621b7fcdb9596ba7b3cc4094929d23fb9bb38d1093c88737` |

## Suíte canônica: três ablações independentes

| Variante | Provas / 300 | Tempo total | Nós totais | Calls totais |
|---|---:|---:|---:|---:|
| Baseline, polígono mais distante | 200 | 123,365 s | 46.027 | 162.440 |
| Polígono mais próximo | 54 | 249,916 s | 26.929 | 144.406 |
| Poda por reentrada | 197 | 123,323 s | 47.909 | 179.027 |
| Poda agressiva por cruzamento | 203 | 122,180 s | 42.881 | 145.923 |

Os totais incluem execuções censuradas em 1 segundo. Por isso, menos nós totais
não significa busca melhor quando uma variante conclui muito menos instâncias.

### Polígono mais próximo

A variante não ganhou nenhuma prova e perdeu 146 provas do baseline. Nos 54 casos
concluídos por ambos, ela foi mais rápida em apenas 7, explorou menos nós em zero,
e teve mediana de 3 vezes mais nós. A soma nesses casos foi 2,657 s e 2.942 nós,
contra 0,070 s e 226 nós no baseline. A regressão é conclusiva neste corpus.

### Reentrada no mesmo polígono

A checagem fez 47.865 decisões e podou 1.619 nós. Seu custo foi 0,838 s, ou 0,679%
do tempo total da variante: 17,50 microssegundos por decisão e taxa de poda de
3,38%.

Apesar do overhead moderado, a regra é incorreta. Em seis casos concluídos por
ambos (`28`, `39`, `202`, `211`, `217` e `226`), a variante certificou um objetivo
pior. Exemplos: no caso 28, baseline `190,0004629646404` contra
`191,3412595263247`; no caso 211, `147,3250026605025` contra
`153,84861328702405`.

Geometricamente, um caminho ótimo entre extremos fixos pode atravessar uma região,
sair para visitar outras regiões e atravessá-la novamente no retorno. As fronteiras
dos polígonos não são obstáculos, portanto a repetição não implica um desvio
removível.

### Cruzamento em relaxações parciais

Na suíte canônica, a versão agressiva fez 42.828 decisões, podou 4.056 nós e custou
0,0159 s: 0,0130% do tempo, 0,371 microssegundo por decisão e taxa de poda de 9,47%.
Nos 199 casos concluídos por ambos, reduziu os nós de 14.992 para 12.807 e o tempo
de 21,937 s para 21,422 s. Não houve divergência de objetivo no corpus canônico.

Como esse resultado era pequeno, a comparação baseline/cruzamentos foi ampliada
para `german-instances.bin`:

| Variante | Provas / 558 | Tempo total | Nós totais | Calls totais |
|---|---:|---:|---:|---:|
| Baseline | 379 | 224,420 s | 86.837 | 308.540 |
| Poda agressiva por cruzamento | 382 | 222,134 s | 81.404 | 276.513 |

Nos 374 casos alemães concluídos por ambos, a variante reduziu o tempo de 40,139 s
para 38,651 s e os nós de 25.933 para 22.416. Fez 81.301 decisões, podou 7.717 nós
e acrescentou 0,0282 s de checagem: 0,0127% do tempo, 0,347 microssegundo por
decisão e taxa de poda de 9,49%. Também não houve divergência nos dois corpora.

Esse ganho, porém, não é válido. Em 1.000 instâncias aleatórias com seis pequenos
retângulos, a instância 496 forneceu um contraexemplo. Com os centros abaixo,
o baseline certificou `32,69461492061567`, enquanto a poda agressiva certificou
incorretamente `32,816118329243004` após uma poda por cruzamento:

```text
(10.77793634693957, -4.218823805715688)
(-1.5483232707868653, 1.0773994081299154)
(0.38844330970246865, 4.92275689710519)
(2.892892972001312, 5.2550538251277406)
(1.809389897888459, 2.8131677980259013)
(0.23829225192562786, -3.735484901397731)
```

Cada polígono era um quadrado de semilado `0,015`, com início `(0, 0)` e destino
`(10, 0)`. O problema é que os descendentes inserem novos polígonos entre elementos
da sequência parcial e substituem os segmentos cruzados; portanto o cruzamento da
relaxação não é herdado por toda extensão do nó.

## Versão segura da poda por cruzamento

A checagem foi restringida a um caminho que já visitava todos os polígonos. Nesse
caso, desfazer um cruzamento próprio com 2-opt mantém todas as visitas e produz um
caminho estritamente menor. A formulação passou os 86 casos de enumeração exaustiva,
344 buscas interrompidas e as 1.000 instâncias aleatórias.

Na suíte canônica, porém, ela realizou **zero checagens e zero podas**. O teste
existente de bound/incumbente já elimina esses nós antes desse ponto. Nos 197 casos
concluídos por ambos, as duas execuções exploraram exatamente 12.813 nós; os tempos
foram 18,661 s (baseline) e 18,732 s (checagem segura). A diferença de uma prova
no total (197 contra 198) decorreu apenas da censura em 1 segundo, sem ativação da
regra.

## Artefatos locais

Os arquivos brutos, ignorados pelo Git, estão em:

- `benchmarks/results/unordered/advisor-pruning-20260920/canonical.jsonl`
- `benchmarks/results/unordered/advisor-pruning-20260920/german-crossing.jsonl`
- `benchmarks/results/unordered/advisor-pruning-20260920/canonical-crossing-safe.jsonl`
- os respectivos arquivos `.meta.json`, com hashes do executável e das entradas.

Hashes dos JSONL, na mesma ordem: `92b97e81494217348a59c4f49a3db6b27bb774d4c8ab2120965cf366df3d5d40`,
`8a8fa3bf3d113e8eadad19af461f00209cd305ec5ce33c6df1d7d685eb9af94b` e
`f78e5a5ff0d1bd3962854ad74e027feaf62c889583d40499be8850660415be52`.

## Decisão

Manter a escolha pelo polígono mais distante e a poda atual por bound/incumbente.
Não integrar as duas podas geométricas na busca. O pequeno ganho aparente da poda
agressiva por cruzamento é consequência de remover ramos que ainda podem conter o
ótimo, e a versão demonstravelmente segura não reduz trabalho na ordem atual das
decisões.
