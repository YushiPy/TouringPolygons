# Estratégias de performance do TPP de ordem livre

Registro de **todas** as estratégias de performance já tentadas no solver de
ordem livre com extremos fixos (`tpp_nonconvex_unordered_solve`), com o
resultado e o status. Consulte esta tabela antes de propor uma otimização;
acrescente uma linha a cada tentativa, inclusive as rejeitadas. Contratos de
correção ficam em [`unordered-tpp.md`](unordered-tpp.md) e
[`certified-convex-oracle.md`](certified-convex-oracle.md); dados brutos ficam
nas campanhas locais (`benchmarks/workspace/`).

## Onde está o tempo (outubro de 2026)

- Por chamada, o oráculo convexo já custa ~20 µs nos OSM de 60 polígonos
  (85–97% do tempo). As otimizações de 29/09–03/10 atacaram esse custo.
- A árvore é profunda e estreita: ~1,1 filho sobrevivente por nó expandido,
  chamadas ≈ nós. Os limites duais analíticos de inserção já descartam ~92%
  dos filhos gerados; só ~12% das chamadas terminam acima do corte.
- Com gap de 0,1%, o UB também limita: o incumbente inicial fica em mediana
  7,5% (p90: 20%) acima do final, e em Dubai (índice 129) a busca estaciona
  0,7% acima do ótimo após 10⁶ chamadas — sem UB ótimo o gap nunca fecha.
- Teto medido do lado primal: partir do caminho ótimo deu 1,9× (mediana) nos
  casos resolvidos em 1 s e certificou 30 casos a mais
  (`runs/german-initial-bound-1s`, 2026-09-24, gap 1e-9).
- O custo do oráculo tem cauda pesada (05/10, caso 417, `5adad03`+limites de
  visita): 5% das chamadas passam de 1 ms e somam 84% do tempo do oráculo.
  São chamadas cuja candidata `double` intersectante é rejeitada e que a
  recuperação filtrada certifica (13.642 de 282.459, todas certificadas). Nas
  entradas dessas chamadas há quase sempre polígonos com vértices exatamente
  compartilhados (59 de 60 amostras); o traço `double` nunca coincidiu com o
  filtrado. Depois das mudanças de 05/10, só o mapa filtrado ainda soma
  23–29% das amostras em 417/419; a recuperação inteira (mapa, replay e
  certificado) é ~50% do oráculo no 417.

## Ganho acumulado por revisão (21/09 → 06/10)

Medição limpa (`tpp.py free-order-history`): mesma máquina e compilador para as
12 revisões, 36 casos estratificados e 3 repetições. Resumo em
[`free-order-history-2026-10-07`](../../benchmarks/results-saved/README.md#free-order-history-2026-10-07).
De `bb1c44a` a `8f8241a`: 25,6× (1,33× menos chamadas, 19,3× por chamada).
Parcelas do ganho em escala log:

- **Oráculo, 03/10 (`d004d64`), 42%:** limites intervalares, recuperação
  filtrada e cache de pares.
- **GMP, 30/09, 16%:** substituiu o `cpp_rational` do Boost.
- **Geometria, 03/10, 14%.**
- **Busca, 23/09, 9%:** mergulho a cada expansão e poda de irmãos. É a única
  etapa que reduz chamadas.
- **Aritmética homogênea e arena, 05/10, 7%.**
- **Demais etapas:** ≤ 3% cada. Os certificados intervalares rigorosos
  (`e759fca`) não mudam o tempo.

## Protocolo padrão

`benchmarks/tpp.py free-order-ablation`, corpus
`results-saved/fekete-comparison/instances.bin`, gap absoluto 0 e relativo
`0.0009990009990009992` (UB ≤ 1,001·LB), tolerância de visita 1e-8, validação
Shapely 1e-7, uma thread, um processo por vez, variantes intercaladas. Tempo
sob teto de chamadas é custo de um trecho de busca, não tempo de solução;
`exact=true` significa gap configurado fechado, não ótimo algébrico. Conjuntos:

- desenvolvimento: diagnósticos 129/451/492/541 + holdout de 18 casos de
  03/10 (fáceis: milissegundos);
- difícil (2026-10-05, seed 20261005, sorteado antes de medir): 8 casos de
  1–10 s, 8 de 10–100 s e 3 de >100 s na campanha de 1 h —
  `213 406 214 262 114 96 408 405 63 95 97 230 476 417 246 156 557 419 64`.

## Tabela

| Estratégia | Alvo | Resultado | Status | Evidência |
|---|---|---|---|---|
| Limites intervalares antes do replay racional | custo/chamada | 1,71× (86 pares) | **ativo** | [`tpp-interval-bounds-2026-10-02`](../../benchmarks/results-saved/README.md#tpp-interval-bounds-2026-10-02) |
| Recuperação racional filtrada sob demanda | custo/chamada | 1,21× | **ativo** | [`tpp-certified-oracle-2026-10-02`](../../benchmarks/results-saved/README.md#tpp-certified-oracle-2026-10-02) |
| Construção filtrada em todas as chamadas | custo/chamada | piorou casos já certificados | rejeitado | idem |
| Contração 2⁻²⁰ para fronteiras compartilhadas | custo/chamada | 1,015× (1,23× Voronoi 11+) | **ativo** | [`tpp-boundary-disjoint-2026-10-02`](../../benchmarks/results-saved/README.md#tpp-boundary-disjoint-2026-10-02) |
| Recorrência disjunta antes da construção | custo/chamada | custo extra em casos fáceis | rejeitado | idem |
| Cache de pares do despacho + geometria intervalar | custo/chamada | 1,26–4,69× (teto de chamadas) | **ativo** | experimento local `tpp-runtime-20261003` |
| Geometria racional emprestada + cache por segmento | custo/chamada | 1,04–2,17× | **ativo** | `tpp-runtime-next-20261003` |
| Arredondamento por bits IEEE, orientação inteira, memo de pertencimento | custo/chamada | 1,05–1,54× | **ativo** | `tpp-unlikely-runtime-20261003` |
| Tabela densa de pares + subconjuntos disjuntos | custo/chamada | 1,00–1,07× | **ativo** | `tpp-certificate-next-20261003` |
| Duais homogêneos inteiros (elos nulos) | custo/chamada | 1,14× Voronoi, neutro OSM | **ativo** | `tpp-voronoi-profile-20261003` |
| KKT `straight-first` | custo/chamada | 0,996× | rejeitado (flag OFF) | `tpp-unlikely-runtime-20261003` |
| `--lazy-oracles` | árvore | fila 8×, gap pior | rejeitado | `tpp-runtime-next-20261003` |
| `--path-strong-branching` (3 mais distantes, max-min dual) | árvore | piorou Voronoi; 15/18 vs 16/18 | rejeitado | idem |
| `--path-dual-reuse` (dual do caminho do pai) | árvore | 0,20–0,71× | rejeitado | idem |
| `--bound-first` | custo/chamada | 0,99–1,00× | rejeitado | idem |
| `--path-certificate-dual` (dual seletivo do pai) | árvore | 1,007× (18 casos) | rejeitado (OFF) | `tpp-parent-dual-20261003` |
| `--relocate-initial`, bidirecional, perímetro amostrado, refinamento convexo inicial | UB inicial | sem ganho consistente | rejeitado (OFF) | `tpp-runtime-20261003` |
| Armazenamento `packed`/`deltas` da fronteira | memória | mesma busca; `packed` padrão | **ativo** (`packed`) | `tpp-memory-20261002` |
| Poda por cruzamento em relaxação parcial | árvore | **inválida** | proibida | [`unordered-pruning-counterexamples.md`](unordered-pruning-counterexamples.md) |
| Poda de reentrada no mesmo polígono | árvore | **inválida** | proibida | idem |
| One-tree e branching aprendido (TSPN) | árvore (ciclo) | sem ganho | rejeitado (OFF) | [`tspn-held-karp-learning-2026-09-30`](../../benchmarks/results-saved/README.md#tspn-held-karp-learning-2026-09-30) |
| Lookahead *first-fail* (ramificar no polígono mais restrito) | árvore | 2–8× mais lento; árvore cresce | rejeitado (removido) | este documento, 2026-10-05 |
| Lookahead só de poda, avaliação completa | árvore | −5…−20% chamadas, tempo neutro | substituído | idem |
| Lookahead só de poda com saída antecipada (`--insertion-lookahead K`) | árvore | 1,04× sozinho; 0,92× sobre os limites de visita | rejeitado (OFF) | idem |
| Polimento local do dual dos limites de inserção | árvore | não implementado: teto ≤12% das chamadas | descartado por análise | idem |
| Limites superiores de visita por âncora (`visit_upper_bounds`) | custo/nó | busca idêntica; 1,289× (difícil) e 1,218× (validação), nunca mais lento | **ativo** | idem |
| Rodadas paralelas com um único mergulho | paralelismo | 4–5× mais chamadas, 2,5× mais lento | rejeitado (corrigido) | idem |
| LNS exata por janelas (`--window-lns`) | UB | sobre os limites de visita: +9% (difícil), +3% (validação), +1% (Dubai); pior caso −11% em caso de 0,2 s | opcional (OFF) | idem |
| Rodadas paralelas de nós com mergulhos múltiplos (`--parallel-nodes`) | paralelismo | 8 threads: até 2,06× (419), ~1× em casos de visita; +7–24% sobre `--threads` de irmãos | opcional (OFF) | idem |
| Arena por thread para os nós de `FilteredRational` (sem `shared_ptr`) | custo/chamada | busca idêntica; recuperação filtrada 1,69× (156) | **ativo** | este documento, oráculo 2026-10-05 |
| Sinais de predicados do mapa filtrado só por intervalos, DAG sob demanda | custo/chamada | busca idêntica; filtrado +12% sobre a arena; com ela 1,158× (difícil, 1 rep.) | **ativo** | idem |
| KKT do certificado em inteiros homogêneos, sem MDC | custo/chamada | busca e contagem de predicados idênticas; oráculo 1,10–1,19× | **ativo** | idem |
| Sinais homogêneos na materialização (`logarithmic_clip`, `support_max`, `feature_on_edge`) | custo/chamada | busca idêntica; oráculo 1,04–1,08× | **ativo** | idem |
| `set_exact_bounds` com pisos e soma inteiros | custo/chamada | mesmo racional; oráculo 1,02–1,04× | **ativo** | idem |
| Caixa por polígono antes do teste aresta a aresta do mapa direcional | custo/chamada | mesmos pares; oráculo 1,11× (156), neutro (213) | **ativo** | idem |
| Estado thread-local único no filtrado | custo/chamada | neutro (0,99×) | rejeitado | idem |
| Reter 64 blocos da arena em vez de 16 | memória/tempo | neutro (0,997×) | rejeitado | idem |
| Reuso de prefixo dos mapas filtrados entre chamadas | custo/chamada | só 22% dos níveis compartilhados; 55% das chamadas sem prefixo | descartado por medição | idem |
| Memo de resultados por entrada idêntica do oráculo | chamadas | 0 repetições em 75.746 chamadas (156, 213) | descartado por medição | idem |
| Pular o replay exato da candidata `double` rejeitada | custo/chamada | muda decisões: perde o corte dual da candidata e traços distintos podem ambos certificar | descartado por análise | idem |
| Proposta `double` sobre polígonos contraídos 2⁻²⁰ antes do filtrado (intersectantes) | custo/chamada | aceita 22% (368/1641); oráculo 0,92× | rejeitado | idem |
| Busca local iterada inicial (`--primal-ils 0.5 --primal-ils-stagnation 200`, padrões de 2026-10-07) | UB inicial | UB inicial melhor, mas 0,07–0,82× nos casos que fecham em até 3 s e 0,91–0,99× em 417/419 (11 casos difíceis, uma execução) | opcional (OFF); útil só para UB no TSPN da Paula | [`tspn-paula-cycle-2026-10-06`](../../benchmarks/results-saved/README.md#tspn-paula-cycle-2026-10-06) |
| Sinais da materialização primeiro por intervalos `double`, inteiros homogêneos só se o sinal fica aberto | custo/chamada | busca idêntica; materialização 6,9 → 5,0 s (417); 1,048× (difícil, 1 rep.) | **ativo** | [`tpp-oracle-allocation-2026-10-06`](../../benchmarks/results-saved/README.md#tpp-oracle-allocation-2026-10-06) |
| Buffer de divisões reaproveitado e vetores reservados no mapa direcional | custo/chamada | busca idêntica; construção 10,2 → 8,8 s (417); +3,5% (difícil, 1 rep.) | **ativo** | idem |
| Valores exatos do DAG filtrado em posições retidas da arena (atribuição no lugar) | custo/chamada | busca idêntica; filtrado 10,7 → 9,2 s (417); +2,5% (difícil, 1 rep.) | **ativo** | idem |
| Conversão racional → `double` por truncamento + comparação com o ponto médio | custo/chamada | idêntica ao Boost em 10⁷ casos; +2,9% (difícil, 1 rep.) | **ativo** | idem |
| Consulta de visita interrompida quando a distância parcial já é menor que a mais distante | custo/nó | busca idêntica; só −6% de consultas por segmento; neutra no 417 | rejeitado | idem |
| Limite de inserções múltiplas (`--multi-insertion-bound`), caminho e ciclo | LB (TSPN Paula) | validação (25 abertas, 120 s, 3 rep.): +10,7% do gap (24/24 melhoram; +52,5% com `lazy` sem mergulhos); 197 fechadas: mesmos ótimos, 1,03× | opcional (OFF) | [`tspn-paula-lower-bound-2026-10-07`](../../benchmarks/results-saved/README.md#tspn-paula-lower-bound-2026-10-07) |
| Inserções múltiplas só com alternância (`π = 1/2`) | LB | 50rat99, 30 s: 0,840 → 0,855 (LB/UB) | substituído pelos cortes por largura | este documento, 2026-10-07 |
| Inserções múltiplas com todas as lacunas e cortes por largura | LB | 50rat99, 30 s: 0,855 → 0,866 | **na opção** | idem |
| Preços também em ordem decrescente (melhor das duas) | LB | 4.000 chamadas: 0,8089 → 0,8105 (50rat99), 0,8079 → 0,8087 (100i1500); dobra a precificação | rejeitado | idem |
| Árvores de segmento na precificação | custo do limite | mesma busca (teto de chamadas); custo igual nestes tamanhos | **na opção** (`O(m·k·log m)`) | idem |
| Ganhos do ciclo com contato barato (vértice mais próximo do ponto médio) e `best_contact` só nas 3 lacunas mais baratas; lacunas que não podem baixar o preço puladas | custo do limite | 100i1500 (`lazy`, 4.000 chamadas): limite 0,75 → 0,68 s em 3,0 s (~2,5% do total), LB praticamente igual | não adotado (ganho pequeno; manteve o binário validado) | este documento, 2026-10-07 |
| `lazy` + sem mergulhos com incumbente do ILS | LB (TSPN Paula) | validação: +47,0% do gap (24/24); 197 fechadas 1,19× (com inserções múltiplas 1,23×, sem regressão > 1,5×) | opcional (OFF) | [`tspn-paula-lower-bound-2026-10-07`](../../benchmarks/results-saved/README.md#tspn-paula-lower-bound-2026-10-07) |
| `--insertion-lookahead 8` sobre inserções múltiplas + `lazy` sem mergulhos | LB | igual sem ele (13 abertas) | rejeitado | idem |
| Oráculo só em ponto flutuante antes da prova intervalar: traço `double` + limites intervalares + polimento de barreira | custo/chamada | 0 fallbacks em 6,6 M chamadas; difícil 1,32× (0,76–3,37×), validação 1,54× (0,81–4,24×); 15–25% mais lento onde só há chamadas baratas (sem caches) | dominado; removido da busca (API mantida) | [`tpp-float-oracle-2026-10-08`](../../benchmarks/results-saved/README.md#tpp-float-oracle-2026-10-08) |
| Prova intervalar do híbrido + polimento no lugar do replay exato e das recuperações (`float_recovery`) | custo/chamada | difícil 1,47× (0,97–3,23×), validação 1,68× (0,99–4,00×); 222/222 fecharam e validaram | **ativo** (padrão; `--no-float-recovery`) | idem |
| Polimento a partir dos contatos da prova intervalar (`interval_seed`) em vez de recalcular traço, despacho e reparo | custo/chamada | sobre o recovery anterior: difícil 1,466 → 1,495×, validação 1,676 → 1,716× | **ativo** | idem |
| Confiar no traço `double` sem certificado (`--trust-double`, diagnóstico) | custo/chamada | 1,26× (≥ 1 ms), 1,18× (≥ 0,1 s); 46/558 casos com LB declarado acima de um caminho viável e 44/558 com o caminho final > 0,1% acima do ótimo (até 3,3%) | **inválido**; só diagnóstico | [`trust-double-2026-10-08`](../../benchmarks/results-saved/README.md#trust-double-2026-10-08) |
| Ponto de partida do polimento: μ inicial 10²–10⁴ × o final; fração para o interior 2⁻⁷ ou 2⁻¹² | custo/chamada | replay (156/417/419): 4,1–4,3×, 5,4–5,7×, 4,8–4,9× em todas as combinações | neutro; mantido 10⁴ e 2⁻⁷ | este documento, 2026-10-08 |
| One-tree, strong branching, `bound-first`, `dual`+`dual-screen` nas abertas da Paula | LB | +0,5%, −7%, +0,5%, +0,4% do gap | rejeitados | idem |

## Detalhes das tentativas de 2026-10-05

Campanha local `experiments/free-order-perf-20261005` (comandos, seleções e
hashes); resumo preservado em
[`results-saved/tpp-visit-bounds-lns-2026-10-05`](../../benchmarks/results-saved/README.md#tpp-visit-bounds-lns-2026-10-05).
Baseline: commit `5adad03`.

### Limites superiores de visita — ativo

O perfil (`sample`) do caso 97 mostrou 55% das amostras em
`SegmentContactCache::query`: cada nó calculava contatos exatos do caminho com
todas as ~60 regiões só para achar a mais distante. Com âncoras por região e
varredura por limite decrescente, os contatos exatos por nó caíram de ~62 para
~6, sem mudar nenhuma decisão. Confirmado em 3 repetições: 1,289× (difícil) e
1,218× (validação), mínimo 1,004×, busca idêntica em 37/37 casos; Dubai
507 → 333 s. Ganha pouco quando o oráculo domina (419: 1,03×).

### LNS exata por janelas — opcional

Motivação: com gap de 0,1%, o UB parado acima do ótimo impede o fechamento, e
um caminho ótimo dado de início rendeu 1,9× no passado. Evolução:

1. Orçamento de 10% de `max_seconds` à frente: achou o ótimo de Dubai em
   2,4 s, mas custou até 3× em casos de segundos (213: 1,1 → 3,2 s).
2. Orçamento proporcional ao tempo decorrido: sem regressões, mas só rodava
   com incumbente novo; em Dubai estagnou em 1068,9.
3. Retomável a cada fatia: voltou a gastar em casos já ótimos (+5–10%).
4. Rajadas de metade do orçamento e recuo exponencial após vizinhança
   esgotada: versão mantida. Ver números na tabela.

Conclusão: quando a própria busca acha o UB ótimo cedo, a LNS é custo puro
(limitado a ~10%); ela vale quando o incumbente estagna. Janelas de 24 contatos
encontram o ótimo de Dubai, mas uma janela grande custa milhares de chamadas.

### Lookahead de inserção — rejeitado

- *First-fail* (ramificar no polígono com menos posições admissíveis): 2–8×
  mais lento; a árvore cresce porque o polígono mais restrito costuma estar
  perto do caminho e eleva pouco o limite. O critério do mais distante é bom
  justamente por maximizar o crescimento do LB.
- Só poda (nó morto se algum dos K candidatos não tem posição admissível):
  muitas podas, mas quase sempre em nós que já não gerariam filhos;
  −5…−20% chamadas. Com saída antecipada, 1,04× sozinho; sobre os limites de
  visita passou a custar (exige contatos exatos de K candidatos): 0,92×.

### Polimento dual dos limites de inserção — descartado por análise

Os limites analíticos já descartam ~92% dos filhos e só ~12% das chamadas ao
oráculo terminam acima do corte: mesmo um limite perfeito economizaria no
máximo ~12% das chamadas. Não implementado.

### Rodadas paralelas de nós — opcional

Primeira versão (um único mergulho por rodada): 4–5× mais chamadas e 2,5× mais
lenta, porque a busca serial depende de mergulhos para achar cedo o UB que
torna a triagem analítica eficaz. Com um mergulho por nó da rodada, as chamadas
voltam ao nível serial; com 8 threads: 419 2,06×, 557 1,54×, 417 1,36×, mas
~1× em casos dominados por visitas. Sobre o paralelismo existente entre irmãos
(`--threads 8`), +7–24%. A eficiência por núcleo é baixa (contenção de alocação
GMP, caches por thread frios); para campanhas, instâncias paralelas com uma
thread continuam mais eficientes.

## Oráculo convexo — tentativas de 2026-10-05

Mesma campanha local (`experiments/free-order-perf-20261005`, arquivos
`oracle-*`); resumo em
[`results-saved/tpp-oracle-exact-arithmetic-2026-10-05`](../../benchmarks/results-saved/README.md#tpp-oracle-exact-arithmetic-2026-10-05).
Referência: binário `final` da rodada anterior (limites de visita ativos).
Todas as mudanças ativas preservam a busca: caminho, ordem, limites, chamadas,
nós e contadores coincidem em todas as 222 execuções (37 casos, 3 repetições).
Confirmação: difícil 1,302× (média geométrica; 0,990–2,056×; 466 → 295 s);
validação 1,339× (0,983–2,069×; 234 → 176 s).

### Diagnóstico

Perfis `sample` (419, 557, 417) e contadores novos do agregado do oráculo
(`TPP_HYBRID_AGGREGATE=1`). No 419, metade das amostras estava na recuperação
filtrada (`solve_intersecting_map_trace_filtered`); dentro dela, ~70% eram
`malloc`/`free`/contagem de referências dos nós `shared_ptr` do DAG e só ~6%
GMP. No 417 ela somava 30,0 dos 43,4 s do oráculo. No 156 cada recuperação
criava ~22 mil nós e ~800 avaliações racionais exatas. Os motivos de rejeição
da candidata `double` eram localizador (370), construção (87), otimalidade
local (586) e contatos coincidentes (598) em 1641 chamadas, sempre com
divergência do traço filtrado, em geral a 7 ou mais níveis do topo.

Depois da arena, o racional exato passou a dominar (~42% do tempo no 417), e
nele a normalização por MDC (`hgcd2`, `div2`, `gcd_22`). Os testes do KKT e da
materialização são sinais invariantes a escalas positivas; por isso foram
levados a inteiros homogêneos sem mudar nenhuma decisão.

### Rejeitadas ou descartadas

- **Estado thread-local único** e **64 blocos retidos**: neutros.
- **Reuso de prefixo dos mapas filtrados**: o mapa do nível *i* depende só de
  `start` e dos polígonos 0..*i*, mas chamadas filtradas consecutivas
  compartilham só 22% dos níveis.
- **Memo de entradas idênticas**: nenhuma sequência se repete.
- **Pular o replay da candidata rejeitada**: o replay também alimenta o corte
  dual da candidata; além disso, traços distintos podem ser ambos ótimos.
- **Proposta contraída para intersectantes**: separa os vértices
  compartilhados, mas o traço contraído raramente é o traço ótimo do original
  (22%), e a tentativa extra custa mais do que economiza.

### Paralelismo

Com `--threads 8 --parallel-nodes` (156, 417; bateria), os lotes têm em média
6,4 chamadas, mas a razão CPU/parede do oráculo é só 1,7–2,3: a chamada mais
lenta do lote (cauda acima de 1 ms) dita o tempo de parede. A soma do tempo
de CPU do oráculo também sobe com oito threads (417: +18% na referência, +33%
no candidato; 156: +40% e +58%). Oito threads aceleram 1,32–1,38× a
referência e 1,25–1,30× o candidato; contra a referência com oito threads, o
candidato com oito threads é 1,37× (156) e 1,68× (417) mais rápido. O gargalo principal é
desequilíbrio de carga entre chamadas, não contenção de alocação; rodadas que
não esperem o oráculo mais lento mudariam a busca e não foram tentadas.

## Oráculo convexo — alocação e conversões (2026-10-06)

Branch `oracle-shared-vertex-speedup`, campanha local
`experiments/free-order-perf-20261005` (arquivos `mem-*`; binários
`sign-filter`, `map-buffers`, `exact-slots`, `nearest-double`, `visit-abort`).
Referência: `main` (`7f4c8dd`). Resumo:
[`results-saved/tpp-oracle-allocation-2026-10-06`](../../benchmarks/results-saved/README.md#tpp-oracle-allocation-2026-10-06).

- **Perfil do 417 no `main`**: `malloc`/`free` somavam ~25% das amostras;
  `homogeneous`/`scaled_difference` da materialização alocavam inteiros GMP em
  todo teste de sinal, o construtor do mapa `double` alocava um vetor por
  aresta (no macOS o `free` de blocos grandes aparecia como
  `mach_absolute_time`), e o `reset` da arena liberava cada racional exato.
- **Ativas** (busca idêntica em 222/222 execuções; confirmação com três
  repetições): difícil **1,142×**, validação **1,164×**. Etapas na triagem de
  uma repetição: intervalos na materialização 1,048×, buffers do mapa 1,085×,
  posições retidas 1,112×, conversão +2,9% sobre esta.
- **Rejeitada**: interromper a consulta de visita assim que a distância
  parcial fica abaixo da mais distante. É segura (a distância final só pode
  diminuir e a comparação é estrita), mas a varredura ordenada por limite
  superior já pula quase tudo; economizou 6% das consultas e nada de tempo.
- **Onde está o tempo agora (417)**: recuperação filtrada ~31% do oráculo,
  materialização ~16% (sobretudo aritmética racional de pontos), limite
  intervalar ~14% (laços sobre todos os vértices), certificado ~10%, mapa
  `double` ~9%. Fora do oráculo, consultas de visita ~8%.

## Limite inferior nas abertas da Paula (2026-10-07)

Branch `tspn-lower-bound`, campanha local
`campaigns/paula-tspn-cycle-20261006/screen/lb`; resumo em
[`results-saved/tspn-paula-lower-bound-2026-10-07`](../../benchmarks/results-saved/README.md#tspn-paula-lower-bound-2026-10-07).
Todas as variantes partem do mesmo incumbente (melhor tour do ILS), então só
o LB difere.

- **Diagnóstico.** Os fechos convexos quase não se sobrepõem, então a
  decomposição não é o gargalo: o gargalo é a ordem. A fronteira tem
  profundidade média ~15. Dobrar o tempo fecha ~1/3 do gap restante. Num nó de
  11 regiões, a soma das inserções mais baratas das ausentes cobria quase todo
  o gap até o UB, mas o branching só vê uma região por vez.
- **Inserções múltiplas.** A primeira versão, só com lacunas alternadas
  (`π = 1/2`), perdia metade da soma. Na segunda, a perda por adjacência é a
  largura da região compartilhada, e os cortes por largura deixaram o termo
  extra em ~0,7 da soma de preços. A ordem gulosa decrescente quase não muda
  nada. No ciclo de 100 polígonos, o limite custa ~5% do tempo com mergulhos
  e ~20% com `lazy`, sobretudo em `best_contact` e na precificação (no caminho
  com pontos, ~1–2% e ~12%).
- **`lazy` e mergulhos.** Antes eram rejeitados porque atrasavam o
  incumbente. Com o incumbente vindo do ILS, viraram os maiores ganhos de LB, e
  mesmo sem ILS aceleram as 197 fechadas (1,19×).
- **Rejeitados:** one-tree, strong branching, `bound-first`,
  `dual`+`dual-screen` e `--insertion-lookahead 8` (este fechou o 49lin318 na
  triagem, mas não acrescenta nada sobre `mln`), além dos preços em ordem
  decrescente e dos ganhos com contato barato (ver a tabela).
- **Ainda não tentado:** limites por filho, com a região escolhida fixada na
  sua lacuna dentro da mesma precificação (k precificações por nó), e um
  oráculo `double` com certificado para pontos e segmentos no caminho (hoje
  toda chamada é racional).

## Oráculo sem aritmética racional (2026-10-08)

Branch `oracle-float-polish`; contrato em
[`certified-convex-oracle.md`](certified-convex-oracle.md#oráculo-em-ponto-flutuante-experimental-2026-10-08);
campanha local `experiments/float-oracle`. Motivação: o B&B pede a cada
chamada uma tolerância de `0,25 × gap do nó` (~2,5×10⁻⁴·UB com o gap de 0,1%).
A aritmética racional só é necessária para provar a estrutura combinatória
exata (gap zero). Degenerescências como vértices compartilhados mudam a
estrutura, não o valor ótimo.

- **Replay de chamadas capturadas** (`--oracle-capture-every`, todas as
  chamadas acima de 0,5 ms mais uma amostra uniforme; 156 completo, 417
  completo, 419 com 60 s). 55.454 chamadas: todas fecharam só com ponto
  flutuante, sem par de limites incompatível com o oráculo híbrido. Nas que o
  híbrido resolve pela recuperação filtrada, mediana de 597 → 102 µs
  (24 iterações de Newton). Oráculo estimado na execução inteira: 1,6× (156),
  2,2× (417), 3,0× (419).
- **Critério de parada do Newton.** Com μ pequeno, a Hessiana da barreira
  esconde resíduos grandes no decremento de Newton, e o dual sai dele. O limite
  `−slope < 10⁻³μ` deixava o dual 10⁻⁴ abaixo do ótimo; `10⁻¹⁰μ`, mais aceitar
  o passo de Newton quando a melhora fica abaixo da resolução da função
  objetivo, deixa 5×10⁻⁷ e acelerou o replay (417: 7,6 → 1,3 s).
- **Ponta a ponta** (binário `d854b27`, protocolo padrão, 2 repetições,
  variantes intercaladas): ver a tabela. O modo só em ponto flutuante perde
  apenas onde o híbrido já é barato, porque refaz por chamada a normalização, o
  despacho e o pertencimento que o híbrido guarda no workspace (perfil do 476:
  subtrações e produtos intervalares, despacho par a par, `malloc`). Ele seria
  uma segunda implementação da prova intervalar sem cache. Por isso a busca usa
  a prova do híbrido e só troca o que vem depois dela.
- **Padrão** (`float_recovery`, polimento a partir de `interval_seed`; uma
  repetição contra `--no-float-recovery`): difícil 1,495× (0,99–3,57×;
  149 → 94 s), validação 1,716× (0,98–4,46×; 90 → 52 s), 74/74 fecharam e
  validaram; 279.509 chamadas fechadas pelo polimento, 0 fallbacks.
- **Corpus completo na dantzig** (558 casos, `ded35f2`, máquina compartilhada;
  [`float-recovery-dantzig-2026-10-08`](../../benchmarks/results-saved/README.md#float-recovery-dantzig-2026-10-08)):
  1116/1116 fecharam e validaram; 0 chamadas racionais em 27,5 M (379.405 pelo
  polimento) contra 470.841 na referência. Com a mesma carga nas duas variantes,
  1,51× nos casos ≥ 0,1 s (mín. 0,97×) e 1,65× na soma dos tempos.
