# Limites de visita e LNS exata — 2026-10-05

Formulação: TPP euclidiano de ordem livre com extremos fixos e polígonos
simples (`tpp_nonconvex_unordered_solve`). Comparação **nosso solver no commit
`5adad03` × nosso solver modificado**, sem nova execução de Fekete. Registro de
todas as estratégias tentadas e rejeitadas:
[`docs/algorithms/unordered-tpp-experiments.md`](../../../docs/algorithms/unordered-tpp-experiments.md).

## Mudanças

- **Limites superiores de visita (padrão ativo).** Cada região guarda um ponto
  próprio (âncora) do seu último contato exato; a distância do caminho à âncora
  limita superiormente a distância à região. A escolha do polígono mais
  distante percorre as regiões por limite decrescente e só calcula contatos
  exatos que ainda podem alcançar o máximo, reproduzindo o desempate pelo menor
  índice. Não muda nenhuma decisão da busca. Ablação: `--no-visit-upper-bounds`.
- **LNS exata por janelas (`--window-lns`).** Reotimiza janelas de contatos
  consecutivos do incumbente com o próprio B&B e extremos fixos; só altera o
  UB. Orçamento proporcional (≤10% do tempo decorrido), rajadas e recuo
  exponencial. **Opcional (OFF)**: ganho médio pequeno e positivo, mas muda
  a trajetória da busca e acrescenta chamadas próprias; ativação por padrão
  exige uma campanha maior.

## Protocolo

- Corpus `fekete-comparison/instances.bin` (SHA-256 `aa442e05…88737`).
- Gap absoluto 0 e relativo `0.0009990009990009992` (UB ≤ 1,001·LB);
  tolerância de visita 1e-8; validação independente Shapely 1e-7; teto de
  600 s; uma thread; um processo por vez; variantes intercaladas; três
  repetições. Release C++23, AppleClang, GMP, macOS arm64 (M4 Pro), sem
  isolamento térmico. Exploratório: não é uma campanha dedicada para o paper.
- **Difícil** (desenvolvimento; seed 20261005, sorteio antes de medir): 8/8/3
  casos com 1–10 s, 10–100 s e >100 s na campanha local de 1 h:
  `213 406 214 262 114 96 408 405 63 95 97 230 476 417 246 156 557 419 64`.
- **Validação** (seed 20261006, sorteada antes de medir, excluindo os casos
  anteriores; não usada para nenhuma escolha):
  `414 80 477 244 515 415 202 420 173 421 493 540 461 77 411 261 65 215`.
- Speedup = mediana baseline / mediana variante, por caso; média geométrica
  sobre os casos com gap fechado em todas as execuções (todas fecharam).
  Binários: baseline `c4bef03a…`, final `ced8d206…`.

## Resultados

| Conjunto | Variante | Casos | Média geom. | Mediana | Mín. | Máx. | Soma das medianas | Busca idêntica |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Difícil | limites de visita | 19 | **1,289×** | 1,198× | 1,006× | 1,778× | 332 → 284 s | 19/19 |
| Difícil | + LNS | 19 | 1,410× | 1,383× | 0,969× | 4,448× | 332 → 277 s | — |
| Validação | limites de visita | 18 | **1,218×** | 1,196× | 1,004× | 1,676× | 171 → 146 s | 18/18 |
| Validação | + LNS | 18 | 1,252× | 1,232× | 1,000× | 1,667× | 171 → 145 s | — |

Todas as 333 execuções fecharam o gap e validaram o caminho. Com os limites,
caminhos, LB, UB, chamadas e nós coincidem bit a bit com o baseline em todos
os 37 casos. No caso 97, contatos exatos caíram de 9,6 M para 0,93 M e as
consultas de visita de 4,28 para 1,27 s. O ganho é menor quando o oráculo
domina (419: 93% do tempo no oráculo, 1,03×).

Dubai (índice 129), uma execução de até 1200 s por variante:

| Variante | Status | Tempo | Chamadas |
|---|---|---:|---:|
| baseline | gap fechado | 507 s | 9.585.541 |
| limites de visita | gap fechado | **333 s (1,52×)** | 9.585.541 (busca idêntica) |
| + LNS | gap fechado | 330 s (1,54×) | 9.533.044 |

Na campanha local de 1 h com oito processos simultâneos e binário anterior,
esse caso não havia fechado o gap; isolado e com o binário atual, o baseline já
fecha em 8,5 min. A LNS encontrou o UB ótimo cedo, mas a busca padrão também o
alcança antes do fim, portanto o ganho adicional foi de ~1%.

## Limitações

- Uma máquina de trabalho, sem afinidade/isolamento; as medianas de três
  repetições reduzem, mas não eliminam, o ruído. Casos abaixo de 0,5 s têm
  ruído relativo alto.
- O conjunto difícil orientou o desenvolvimento; a validação foi medida uma
  única vez, no fim. Ambos vêm do mesmo corpus de 558 casos.
- `exact=true` é fechamento do gap configurado, não otimalidade algébrica.
- Não substitui a comparação com Fekete nem multiplica os speedups históricos.
