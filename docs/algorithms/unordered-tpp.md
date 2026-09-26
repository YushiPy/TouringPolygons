# TPP não convexo com ordem livre

Implementação: `packages/nonconvex-tpp/cpp/src/solvers/unordered.cpp`.
API: `tpp/nonconvex/unordered.h`.

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
`termination` distingue `optimal`, `call_limit`, `time_limit`
e `numerical_limit`.

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
[`benchmarks/results-saved/german-comparison`](../../benchmarks/results-saved/german-comparison/README.md).
Resultados temporais anteriores não fazem parte deste contrato de algoritmo.
