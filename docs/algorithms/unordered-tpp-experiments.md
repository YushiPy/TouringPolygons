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
| Limites intervalares antes do replay racional | custo/chamada | 1,71× (86 pares) | **ativo** | `results-saved/tpp-interval-bounds-2026-10-02` |
| Recuperação racional filtrada sob demanda | custo/chamada | 1,21× | **ativo** | `results-saved/tpp-certified-oracle-2026-10-02` |
| Construção filtrada em todas as chamadas | custo/chamada | piorou casos já certificados | rejeitado | idem |
| Contração 2⁻²⁰ para fronteiras compartilhadas | custo/chamada | 1,015× (1,23× Voronoi 11+) | **ativo** | `results-saved/tpp-boundary-disjoint-2026-10-02` |
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
| One-tree e branching aprendido (TSPN) | árvore (ciclo) | sem ganho | rejeitado (OFF) | `results-saved/tspn-held-karp-learning-2026-09-30` |
| Lookahead *first-fail* (ramificar no polígono mais restrito) | árvore | 2–8× mais lento; árvore cresce | rejeitado (removido) | este documento, 2026-10-05 |
| Lookahead só de poda, avaliação completa | árvore | −5…−20% chamadas, tempo neutro | substituído | idem |
| Lookahead só de poda com saída antecipada (`--insertion-lookahead K`) | árvore | 1,04× sozinho; 0,92× sobre os limites de visita | rejeitado (OFF) | idem |
| Polimento local do dual dos limites de inserção | árvore | não implementado: teto ≤12% das chamadas | descartado por análise | idem |
| Limites superiores de visita por âncora (`visit_upper_bounds`) | custo/nó | busca idêntica; 1,289× (difícil) e 1,218× (validação), nunca mais lento | **ativo** | idem |
| Rodadas paralelas com um único mergulho | paralelismo | 4–5× mais chamadas, 2,5× mais lento | rejeitado (corrigido) | idem |
| LNS exata por janelas (`--window-lns`) | UB | sobre os limites de visita: +9% (difícil), +3% (validação), +1% (Dubai); pior caso −11% em caso de 0,2 s | opcional (OFF) | idem |
| Rodadas paralelas de nós com mergulhos múltiplos (`--parallel-nodes`) | paralelismo | 8 threads: até 2,06× (419), ~1× em casos de visita; +7–24% sobre `--threads` de irmãos | opcional (OFF) | idem |

## Detalhes das tentativas de 2026-10-05

Campanha local `experiments/free-order-perf-20261005` (comandos, seleções e
hashes); resumo preservado em
[`results-saved/tpp-visit-bounds-lns-2026-10-05`](../../benchmarks/results-saved/tpp-visit-bounds-lns-2026-10-05/README.md).
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
