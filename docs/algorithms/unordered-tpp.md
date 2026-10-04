# TPP não convexo com ordem livre

Implementação: `packages/nonconvex-tpp/cpp/src/solvers/unordered.cpp`.
API: `tpp/nonconvex/unordered.h`.

O modo nativo opcional `--portfolio` executa duas buscas independentes com
incumbente compartilhado, orçamento global de chamadas e encerramento por prova
do gap. A implementação e o protocolo são comuns ao TPP e ao TSPN; veja
[portfólio cooperativo](tspn.md#cooperative-search-portfolio). `--threads` deve
permanecer em 1 nesse modo: o portfólio cria dois workers, um por estratégia.

A mesma busca também oferece ciclos sem extremos fixos pela API
`tpp_nonconvex_tspn_solve`. Formulação, limites cíclicos e contrato de exatidão
estão documentados em [TSPN](tspn.md).

## Escopo

Menor caminho euclidiano de um ponto fixo `start` a um ponto fixo `target`, visitando
cada polígono simples, sem ordem prescrita. São permitidos polígonos não convexos,
interseções, contatos de borda, extremos no interior dos polígonos e `start == target`.
Polígonos com buracos não são representáveis nesta API. As fronteiras não são obstáculos.
O pior caso continua exponencial.

A implementação adapta as ideias de inserção do polígono mais distante e refinamento
preguiçoso do trabalho de Fekete, Kniep, Krupke e Perk, estudadas no checkout local
`third_party/tspn-socg`, principalmente `strategies/branching_strategy.cpp`.
O código foi implementado no nosso pacote, sem copiar o solver externo. Os extremos
permanecem fixos; uma sequência com `m` polígonos tem `m + 1` posições de inserção.
Não se usa simetria de reversão de tours para eliminar caminhos com extremos distintos.

## Árvore de busca

Um nó contém uma sequência parcial de identificadores. Cada identificador representa
seu fecho convexo ou uma peça específica da decomposição convexa existente no projeto.
A raiz tem sequência vazia e caminho reto entre os extremos.

1. Resolver a sequência convexa parcial e obter limites primal e dual.
2. Podar se o limite inferior não puder melhorar o incumbente além da tolerância.
3. Verificar o caminho contra **todos os polígonos originais**. Uma região pode ser
   visitada incidentalmente por um segmento, sem aparecer na sequência.
4. Escolher o polígono não visitado mais distante do caminho.
5. Se ele não estiver na sequência, inserir seu fecho em cada uma das `m + 1` posições.
6. Se estiver representado pelo fecho, substituí-lo por cada peça de sua decomposição.

A busca prioriza o menor limite inferior e faz uma descida pelo melhor filho em
cada expansão (`dive_interval = 1` por padrão), para obter incumbentes cedo.
`dive_interval = 0` desativa as descidas; a CLI aceita `--dive-interval N`.
O padrão usa uma thread. `--threads N` avalia até `N` filhos irmãos em paralelo
no oráculo convexo, dentro de uma única instância. Cada worker tem seu próprio
workspace; limites, incumbentes, fila e traço são atualizados depois que o lote
termina. O lote usa o incumbente disponível no início, então pode executar
chamadas que a execução serial dispensaria após uma melhora intermediária; isso
afeta trabalho e runtime, não a validade dos limites. `parallel_oracle_calls` e
`parallel_oracle_batches` mostram quando houve paralelismo efetivo. O contador
`calls` inclui todas as chamadas já lançadas, inclusive as que terminam após o
limite cooperativo de tempo.
A região OpenMP fica em uma função separada que não é expandida no chamador,
para que o caminho serial não inicialize o runtime de threads. Essa função só
é chamada quando há mais de um oráculo no lote; avaliação e propagação de
exceções mantêm o mesmo contrato.
Opcionalmente, `--detour-root` escolhe a primeira região pelo maior desvio
mínimo do caminho reto ao visitar seu fecho convexo; empates favorecem a maior
distância do caminho ao polígono original. A opção só muda a ordem da busca,
permanece desativada por padrão e não acrescenta uma condição de poda.
Depois que um filho melhora o incumbente, a busca compara novamente o limite
inferior dos irmãos ainda não avaliados com o novo corte e dispensa o oráculo
quando já é impossível melhorar. Isso é seguro porque cada limite inferior
permanece válido e o incumbente só diminui.
Se a candidata geométrica do oráculo convexo não fecha sua certificação, um
dual racional pode ainda provar que seu nó supera o corte atual. Nesse caso o
oráculo retorna sem executar o fallback racional completo; o contador
`oracle_dual_cutoff_prunes` registra essas ocorrências.
A solução inicial padrão usa vértices próximos, 2-opt e otimização dos contatos nas arestas.
Essas heurísticas só fornecem limites superiores. A decomposição é calculada sob demanda.
As opções experimentais `--sampled-perimeter-initial`, `--convex-initial-refinement`
e `--bidirectional-initial` ativam, respectivamente, candidatos adicionais espaçados
no perímetro, um solve convexo certificado para a ordem e as peças visitadas pela melhor
rota inicial, e uma segunda construção começando do alvo. A amostragem conserva os
vértices originais, aloca pontos pela razão entre perímetros e usa o mesmo orçamento de
trabalho do amostrador de ordem fixa (`TPP_APPROX_WORK_BUDGET`, padrão 1.000.000;
`TPP_APPROX_BUDGET_MODE=adaptive` ativa o fator adaptativo). Para ordem livre, o modelo
estima o trabalho por todos os pares de regiões. A rota sem amostragem também é mantida
como candidata, então a amostragem não piora o limite superior inicial.

O refinamento convexo atribui a cada contato da melhor rota uma peça fechada da
decomposição que contém esse contato e resolve a sequência completa com o oráculo
certificado. O resultado só substitui a rota se for mais curto e continuar cobrindo os
polígonos originais. Essa chamada consome o orçamento de chamadas e tem teto de tempo
igual ao menor entre 1 s, 10% do limite da instância e o tempo restante. A decomposição
feita nessa etapa fica em cache para a busca. As três opções são desativadas por padrão.

`--relocate-initial` / `relocate_initial_heuristic` acrescenta até quatro
varreduras de realocação à melhor candidata inicial. Remove um contato,
considera todas as posições de reinserção e minimiza o contato da mesma região
na nova posição com `best_contact`. A redução local seleciona a proposta;
o comprimento completo precisa diminuir antes de aceitar o movimento.
Uma varredura de contatos segue cada passe. O caminho final só substitui o
incumbente se for mais curto e sua cobertura for validada. Extremos do TPP e
fechamento do TSPN são preservados. O orçamento é o menor entre 100 ms e 5%
do tempo restante, verificado entre candidatos; não usa chamadas do oráculo.
A opção é experimental e
desativada por padrão. `initial_relocation_moves` conta os movimentos da
candidata aceita, e `initial_relocation_seconds` está incluído na heurística.

O refinamento geométrico padrão otimiza um contato por vez; ele não equivale a resolver
conjuntamente a sequência completa com o TPP convexo de ordem fixa.
Opcionalmente, `options.initial_path` fornece um caminho completo de `start` a
`target`, incluindo ambos os extremos. O solver valida a visita a todas as regiões,
substitui a heurística inicial e usa somente seu comprimento como limite superior.
Ele não recebe ordem, limite inferior ou informação de otimalidade desse caminho;
a árvore e o certificado continuam iguais. Um caminho inválido causa erro.

A árvore integra ordem e peças. Não chama um B&B não convexo completo para cada
permutação: reutiliza o solver convexo e a interface `decompose_polygon` existentes,
e combina as duas ramificações na mesma busca.

### Armazenamento das sequências da fronteira

`options.sequence_storage` / `--sequence-storage native|packed|deltas` altera
somente a representação das sequências retidas. Os oráculos e os irmãos em
avaliação usam vetores completos de índices `size_t`; a ordem de expansão,
os cortes, tolerâncias e certificados são os mesmos nos três modos. O padrão
é `packed`; `deltas` é experimental até uma comparação de tempo e RSS.

`packed` codifica cada par `(polígono, peça)` com índices de 8, 16, 32 ou
64 bits. A largura é escolhida uma vez por busca pelo número de polígonos e
pelo limite conservador de `vértices - 2` peças em uma partição por diagonais
de um polígono simples. Isso mantém a decomposição sob demanda. O maior valor
da largura é reservado para indicar o fecho convexo: índices reais nunca são
truncados e toda codificação verifica overflow. Até 255 polígonos e esse
limite de até 255 peças usam um byte por índice. Uma sequência com 30 entradas
ocupa 60 bytes de dados nesse caso, contra 480 em uma plataforma com `size_t`
de 64 bits; a redução de oito vezes vale para os dados da sequência.
Cabeçalhos dos nós, caminhos e caches continuam consumindo memória.
Na implementação arm64 atual, o wrapper das representações acrescenta oito
bytes ao cabeçalho do nó (208 para 216); os três modos usam esse mesmo wrapper.

`deltas` guarda uma referência à sequência ancestral e uma operação:
inserir um polígono em uma posição, ou atribuir uma peça a um polígono já
presente. Cada registro tem IDs e contagens de referências de 32 bits,
independentemente da largura dos índices geométricos. Com índices de oito
bits um registro ocupa 12 bytes. O registro conserva apenas a sequência,
sem reter o caminho ou outros campos de um nó ancestral. Ele é criado somente
para um filho que sobrevive aos cortes e será retido. Uma arena libera cadeias
sem donos de forma iterativa e reutiliza seus slots; sua capacidade alocada
permanece disponível até o fim da busca. Overflow dos IDs/contagens causa erro,
sem wraparound; a arena admite menos de `2^32` registros simultaneamente alocados.

A sequência é reconstruída uma vez ao retirar o nó, antes de chamar o
oráculo ou gerar irmãos. As operações são lidas da mais recente para a mais
antiga; cada inserção ocupa a posição correspondente entre os lugares ainda
vagos, e a atribuição de peça mais recente prevalece. Até 64 entradas, a
seleção usa uma máscara de bits; sequências maiores usam uma árvore Fenwick.
A raiz vazia do TPP ou a sequência inicial do TSPN preenche os lugares
restantes. A reconstrução custa `O(d + k log k + n)` para `d` operações,
`k` entradas e `n` polígonos, com scratch reutilizado. Em cada ramo do solver
há no máximo uma inserção e uma atribuição por polígono. O pai reconstruído
permanece vivo durante a geração dos irmãos, inclusive se uma chamada for
interrompida ou a busca devolver o nó à fila para refinamento.

Os modos compartilham a mesma implementação do B&B, incluindo DFS/BFS,
diving, avaliação paralela de irmãos e portfólio. Cada busca do portfólio
possui sua própria arena. A retirada da fila move os buffers, e uma posição
de inserção já podada não constrói sequência, salvo se o trace solicitar sua
descrição. A construção dos filhos reserva o tamanho final para evitar a
realocação após copiar o vetor do pai.

O JSON informa `sequence_storage`, `node_index_bits`,
`peak_sequence_storage_bytes`, `peak_frontier_node_bytes`,
`sequence_history_record_bytes`, `peak_sequence_records` e
`sequence_reconstructions`. Os bytes de sequência contabilizam buffers/arena
reservados, incluindo a capacidade livre para reutilização, sem cabeçalhos
inline, metadados do alocador, vetores temporários de irmãos, caminhos ou
caches dos oráculos. Os bytes de nós contabilizam a capacidade do vetor da
fila, sem o índice de limites da DFS. No portfólio, os picos são somados e
constituem uma estimativa superior à ocupação simultânea dos dois workers.
`process_peak_rss_bytes` mede o pico do processo CLI em macOS/Linux, incluindo
caminhos, caches e demais alocações; zero indica métrica indisponível em outras
plataformas. Esses campos complementam, mas não substituem, a validação dos
limites e da trajetória retornada.

Deltas podem perder em tempo ou mesmo em memória quando muitos ancestrais
distintos permanecem vivos. Compare os três modos com o mesmo binário,
gap, orçamento e política, preferindo um processo por vez para diagnosticar
RSS e runtime. Não atribua aos deltas o ganho já obtido pela compactação.

Duas podas geométricas rejeitadas e seus fixtures de regressão estão documentados
em [`unordered-pruning-counterexamples.md`](unordered-pruning-counterexamples.md).

### Por que a ramificação é completa

Considere uma solução de um nó e escolha uma ocorrência de visita de cada região
já representada. A visita de um novo polígono ocorre em alguma posição entre essas
ocorrências, incluindo antes da primeira e depois da última. A união dos filhos por
inserção contém todas essas possibilidades. Quando um fecho é refinado, qualquer
visita ao polígono original pertence a pelo menos uma peça da decomposição.

Substituir uma região por seu fecho e omitir regiões relaxa o problema. Portanto,
o ótimo da sequência parcial limita inferiormente qualquer extensão do nó. Se o
caminho parcial já visita todos os polígonos originais, ele também é um incumbente
global. Em cada ramo há no máximo uma inserção e um refinamento por polígono.

## Consultas de visita preparadas

### Ablações adicionais de runtime

Geometria por referência e cache por segmento são ativados por padrão. As
demais opções abaixo são experimentais e desativadas. Não mudam os gaps, a
geometria de entrada ou o critério de factibilidade. As flags
`--no-borrow-oracle-geometry` e `--no-segment-visit-cache` isolam os dois ganhos.

- `--borrow-oracle-geometry` / `oracle_borrow_geometry` conserva handles
  imutáveis durante cada chamada e usa referências aos polígonos racionais
  preparados, dispensando a cópia de seus vértices. A limpeza do cache não
  invalida esses handles. As versões binárias ainda usam o buffer existente.
- `--segment-visit-cache` / `segment_visit_cache` reutiliza consultas de
  segmentos exatamente iguais contra polígonos originais preparados. A chave
  contém todos os bits dos quatro valores dos extremos, mantendo sua direção.
  A redução mantém a ordem dos segmentos e compara distâncias quadráticas,
  preservando empates e a primeira posição de visita. O cache é local à busca,
  exige geometria preparada imutável e é invalidado quando muda a identidade
  do polígono ou a tolerância. Retém no máximo 1024 segmentos; os vetores de
  resultados têm orçamento de 2 MiB, exceto a entrada mínima quando ela sozinha
  excede esse orçamento. Metadados da tabela ficam fora dessa contagem.
  `--no-prepared-visits` também desativa esse cache. `segment_visit_queries`
  e `segment_visit_hits` contam consultas até o término da fase de busca.
- `--lazy-oracles` / `lazy_oracles` conserva filhos com seus limites de
  inserção/pai e resolve sua sequência apenas quando o filho é retirado da
  fronteira. Filhos sem caminho continuam participando do lower bound global;
  interrupções preservam seus limites. A política pode economizar chamadas e
  caminhos retidos, mas também ampliar a fronteira e enfraquecer sua prioridade.
- `--bound-first` / `oracle_bound_first` tenta o corte dual racional de uma
  candidata já materializada antes do certificado KKT completo. O filtro
  flutuante só seleciona a tentativa; apenas o dual factível racional causa
  poda. O retorno conserva os contatos factíveis e os limites dirigidos.
- `--path-dual-reuse` / `path_dual_reuse` constrói vetores factíveis no disco
  a partir do caminho do pai e reutiliza-os nos suportes dos filhos. Elos zero
  podem conservar a direção herdada. Normas e suportes são avaliados em
  racional e o limite é arredondado para baixo. Esta proposta não retém o
  testemunho ótimo da propagação de discos/cones: seu custo e sua força são
  hipóteses a medir, não uma garantia de ganho.
- `--path-certificate-dual` / `path_certificate_dual` conserva vetores binários
  factíveis já propostos pelo certificado intervalar, somente quando existem
  elos nulos ou curtos. A proximidade seleciona uma proposta; não identifica
  contatos nem autoriza uma poda. Cada vetor exportado passa por uma prova
  intervalar de norma no máximo um. O limite de cada inserção reutiliza os
  suportes inalterados e avalia os termos novos com intervalos arredondados
  para fora, usando os polígonos originais. Uma aritmética não suportada ou
  uma norma não comprovada dispensa essa tentativa. O cache local usa o serial
  imutável do nó, no máximo 4096 entradas e 2 MiB de capacidade dos vetores;
  metadados da tabela ficam fora desse orçamento. Limpar o cache só perde uma
  oportunidade de poda. Nenhum vetor adicional é armazenado no próprio nó.
  Esta opção não extrai o testemunho ótimo do certificado KKT de discos/cones;
  seu dual pode ser mais fraco que o limite retornado pelo oráculo. Os
  contadores `path_dual_screen_children` e `path_dual_screen_prunes` registram
  avaliações e podas adicionais no corte corrente; não demonstram ganho de
  tempo, que exige medir o encerramento com o mesmo gap.
- `--path-strong-branching` / `path_strong_branching` compara os limites de
  todas as inserções dos três polígonos ausentes mais distantes e escolhe o
  candidato com maior mínimo desses limites. Isso só escolhe o ramo completo
  a expandir; não elimina regiões ou posições sem o teste de bound existente.
  A seleção pode melhorar ou piorar a trajetória e acrescenta trabalho por nó.

Geometria por referência e cache por segmento devem preservar caminhos,
limites, contagens e trace sob orçamento de chamadas igual, sem interferência
do limite de tempo. As demais opções podem mudar a busca: valide cobertura,
preservação dos limites e o esforço para fechar o mesmo gap. Tempo até um
teto de chamadas é custo de um trecho de busca, não tempo de solução.

`prepared_visit_queries` (padrão true) prepara uma vez caixas e arestas dos
polígonos originais, e prepara os segmentos de cada caminho consultado. A
sobrecarga preparada e a consulta sem preparação compartilham o mesmo
algoritmo em `unordered_geometry.cpp`, incluindo pertencimento, projeções,
distâncias, posição de primeira visita e critérios de desempate.
Além disso, a busca retém os contatos calculados para o último caminho,
conferindo todos os bits das coordenadas antes de reutilizá-los. Isso evita
repetir consultas entre a verificação de um incumbente e a escolha do ramo.
Polígonos e tolerância permanecem constantes durante essa retenção.
Cada busca/worker do portfólio tem sua própria preparação; ela não é guardada
nos nós. A memória adicional é linear nos vértices, tamanho do caminho e
número de polígonos. `--no-prepared-visits` desativa ambos os reaproveitamentos.
`visit_query_evaluations` e `visit_query_cache_hits` contam consultas executadas
e reaproveitadas nas fases de cobertura, ramificação e finalização; consultas
específicas da raiz e decomposição não fazem parte desses contadores.

## Certificado convexo e interseções

A API `tpp_convex_solve_certified` delega ao oráculo híbrido seguro descrito em
[`certified-convex-oracle.md`](certified-convex-oracle.md). O `workspace` reutiliza
polígonos já normalizados em aritmética racional entre chamadas da mesma busca;
as entradas do cache são identificadas e conferidas pelas coordenadas binárias
completas e seu tamanho é limitado. O oráculo tenta o solver geométrico
em `double`, reconstrói os contatos e verifica sua proveniência e otimalidade com
predicados racionais. Se não conseguir certificá-los, resolve a sequência pelo
fallback racional: recorrência para polígonos disjuntos e mapas direcionais para
polígonos com interseção. Um limite dual racional suficiente também pode encerrar
a chamada quando existe um corte finito. A proteção vale para essa API; as APIs
antigas de ordem fixa não foram redirecionadas.

O despacho entre disjuntos/intersectantes conserva em cache a classificação
racional de pares já preparados, com as condições e limites de retenção
descritos no contrato do oráculo. `--no-oracle-dispatch-cache` repete esse
despacho para comparar runtime no mesmo binário. Isso não modifica a agenda,
os contatos ou os limites da busca. `oracle_dispatch_pair_queries`,
`oracle_dispatch_pair_cache_hits` e `oracle_dispatch_pair_exact_checks` ajudam
a separar reaproveitamento da geometria de mudanças na árvore; o `profile`
também exporta `convex_dispatch_seconds` e `convex_bound_evaluation_seconds`.
`--no-oracle-interval-geometry-cache` isola o reaproveitamento dos vértices
binários normalizados usados na proposta intervalar. Sua preparação aparece
em `convex_proposal_preparation_seconds`; ambos os caches mantêm o mesmo
replay, pertencimento e certificado do oráculo seguro.

`--interpolated-zero-dual` ativa uma proposta dual intervalar adicional para
elos curtos/coincidentes, documentada no contrato do oráculo. Essa opção
experimental não assume que contatos próximos são iguais, não perturba os
polígonos e não altera os gaps solicitados. Permanece desativada por padrão.

Para regiões convexas $C_1,\ldots,C_m$ e vetores $u_0,\ldots,u_m$ com norma no máximo 1,
um limite inferior é

$$
D(u) = (t-s)\cdot u_m + \sum_{i=1}^{m}\min_{v\in C_i}(v-s)\cdot(u_{i-1}-u_i).
$$

Basta minimizar o produto escalar sobre os vértices de cada região. A desigualdade
$u\cdot d\leq\|d\|$ e o cancelamento dos termos intermediários provam a validade.
Assim, o comprimento de um caminho retornado pelo solver não é usado automaticamente
como limite inferior. Atribuições diferentes de direções em segmentos de comprimento
zero também são testadas, sempre dentro da bola unitária.

O caminho de pontos interiores em `certified_refinement.cpp` permanece no código,
mas não é chamado por essa API. O fallback ativo usa `boost::multiprecision::cpp_rational`.
O resultado global ainda emprega tolerâncias de visita e de gap; `exact` significa que

```
upper_bound - lower_bound <= absolute_gap + relative_gap * abs(upper_bound)
```

Os padrões são `absolute_gap = 1e-7`, `relative_gap = 1e-9` e tolerância de visita
`1e-8`, nas unidades das coordenadas. Há margem de arredondamento no dual.
Vértices consecutivos separados por até `1e-4` da tolerância de visita são unidos;
o limite inferior final desconta duas vezes a soma dos deslocamentos removidos.
Isso evita peças espúrias quase degeneradas, como as produzidas por dois vértices
que diferem por aproximadamente `3e-15` na instância 57 da suíte de desenvolvimento.

Uma falha em fechar o certificado numérico não autoriza declarar otimalidade.
O resultado recebe `numerical_limit`, preservando caminho e limites. Nos limites
de tempo/chamadas, filhos ainda não avaliados conservam o limite do pai. A fronteira,
o nó da descida e as regiões já podadas participam do certificado global.
Isso descreve o gap não fechado de um resultado utilizável. Falhas geométricas
irrecuperáveis, como caminho não finito, peça atribuída não visitada ou falha do
ponto interior, podem lançar exceção sem devolver caminho e limites.

## Compilação e uso

Na raiz do projeto:

```bash
brew install eigen boost
cmake -S packages/nonconvex-tpp/cpp -B .build/unordered -DTARGET=main-unordered
cmake --build .build/unordered --target tpp tpp-unordered-tests -j 8
.build/unordered/tpp < packages/nonconvex-tpp/cpp/tests/unordered-example.txt
```

São mantidos os requisitos de compilador e OpenMP do projeto. Eigen e Boost são
dependências de headers do certificado. Também existe o target `tpp-unordered`,
que permite construir o executável em uma configuração usada por outros targets.

Formato de entrada por espaços/brancos:

```
sx sy tx ty polygon_count max_calls max_seconds
vertex_count x0 y0 x1 y1 ...
... uma linha por polígono ...
```

A saída JSON inclui `path`, `order` (índices a partir de zero, pela primeira visita),
`lower_bound`, `upper_bound`, `exact`, `termination`, `calls`, `fallback_calls`,
motivos de fallback e contadores de diagnóstico,
`nodes`, `sibling_bound_prunes`, `oracle_dual_cutoff_prunes`, os dois contadores de
ramificação, `peak_queue`, `seconds` e `profile`.
A API C++ não tem limite de busca por padrão. A CLI exige limites explícitos.
Com `--initial-path`, a CLI lê após os polígonos a contagem de pontos do caminho
e seus pares `x y`, incluindo os extremos. Sem essa opção, a entrada antiga e a
heurística padrão permanecem iguais.
O limite de tempo é cooperativo: uma chamada geométrica/decomposição já iniciada
pode excedê-lo; o pré-processamento e partes da heurística inicial também não
consultam o limite a cada operação. `calls` conta invocações do oráculo convexo certificado,
incluindo o refinamento inicial opcional; `initial_convex_refinement_calls` separa essa
chamada das chamadas da busca.
`termination` distingue `optimal`, `call_limit`, `time_limit`,
`numerical_limit` e `interrupted`. `Ctrl+C` no executável nativo solicita uma
parada cooperativa: a chamada convexa/decomposição em andamento termina, a
fronteira restante mantém seu limite inferior e o resultado inclui o incumbente
viável atual e sua trajetória.

### Organização da implementação

- `certified.cpp` adapta o resultado do oráculo híbrido para a API usada pela busca.
- `hybrid.cpp` orquestra o solver geométrico, a certificação e o fallback ativo.
- `certified_refinement.cpp` contém um método de pontos interiores que não é usado nessa API.
- `unordered.cpp` contém heurística, branch-and-bound, bounds e instrumentação.
- `unordered_geometry.cpp` contém operações puras de fecho convexo e contato.
- `unordered_runner.py` centraliza o protocolo de processo usado pelos benchmarks e
  campanhas Python.

Os headers `certified_internal.h` e `unordered_geometry.h` são internos aos seus
respectivos módulos e não ampliam a API pública.

Em `profile`, pré-processamento, heurística inicial, busca e finalização são fases
superiores disjuntas. O tempo da busca contém oráculo convexo, decomposição,
verificação de visitas e manutenção exclusiva da busca. O tempo do oráculo é
inclusivo e contém solver geométrico, verificação inicial do certificado e fallback;
o fallback ativo usa aritmética racional. O total de
verificação de visitas soma medições nas fases superiores e se sobrepõe a elas,
portanto não deve ser somado novamente. A semântica também acompanha cada resultado
em `profile.timing_semantics`.
Com múltiplas threads, `convex_oracle_seconds` soma a duração individual das
chamadas e pode exceder o tempo de parede. `convex_oracle_wall_seconds` conta o
tempo de parede de cada lote uma vez; `search_maintenance_seconds` usa essa medida.

```cpp
#include "tpp/nonconvex/unordered.h"

tpp::UnorderedTppSolveOptions options;
options.max_seconds = 30;
options.threads = 8; // Até oito filhos do mesmo nó, na mesma instância.
// Opcional: options.initial_path = caminho_factivel;
auto result = tpp::tpp_nonconvex_unordered_solve(start, target, polygons, options);
```

## Validação reproduzível

```bash
scripts/verify_unordered.sh
python3 benchmarks/tpp.py free-order-run \
  --seconds 2 --output benchmarks/results/unordered/dev.jsonl
```

O teste C++ enumera todas as ordens e todas as combinações de peças de 86 instâncias,
comparando o melhor objetivo obtido com o resultado do B&B. Essa enumeração
reutiliza o mesmo oráculo convexo e a mesma decomposição, portanto não é uma
verificação independente desses componentes. Executa ainda 344 buscas
interrompidas com limites de chamadas 0, 1, 3 e 10, verificando a preservação dos limites.
Inclui caminhos fechados, regiões sobrepostas, regiões em L e U e orientações invertidas.

A validação independente abaixo usa SOCP/Gurobi por enumeração de ordens e de duas
regiões retangulares cuja união é um L. Não usa nossa decomposição nem nosso oráculo:

```bash
python3 packages/nonconvex-tpp/cpp/tests/validate_unordered_gurobi.py
```

Requer `gurobipy`, licença e Shapely somente para validação. São 24 instâncias, incluindo
8 que forçam a ramificação não convexa. O benchmark pode usar Shapely para verificar
independentemente a distância do caminho a cada polígono e a igualdade dos extremos;
sem Shapely, registra `valid: null`, não uma validação fictícia.

Para comparar com o checkout privado instalado, em modo caminho e uma thread:

```bash
third_party/tspn-socg/.venv/bin/python \
  benchmarks/tpp.py compare-external \
  --mode path --threads 1 --time-limit 2 --eps 0.000001
python3 benchmarks/tpp.py summarize-external \
  benchmarks/results/unordered/dev.jsonl EXTERNAL_RESULTS.csv \
  --output benchmarks/results/unordered/comparison
```

O adaptador externo exporta a trajetória bruta e uma candidata diagnóstica com
extremos encaixados, além de aplicar `benchmarks/_internal/unordered_validation.py`,
o mesmo validador independente usado nos caminhos próprios. Otimalidade declarada,
viabilidade bruta e viabilidade após encaixe são campos separados. O resumo exige
hashes iguais e modo `path`. A dificuldade original da suíte se refere
a ordem fixa, não necessariamente à dificuldade com ordem livre. Não se deve comparar
o antigo benchmark de ordem fixa com o externo de ordem livre como se fossem o mesmo
problema. Os resultados gerados ficam em `benchmarks/results/` e não entram no Git.

## Benchmark preservado

A comparação canônica atual com o solver de Fekete et al. mantém corpus, saídas,
análise e instruções em
[`benchmarks/results-saved/fekete-comparison`](../../benchmarks/results-saved/fekete-comparison/README.md).
Resultados temporais anteriores não fazem parte deste contrato de algoritmo.
