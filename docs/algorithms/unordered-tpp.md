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

A busca prioriza o menor limite inferior e faz descidas periódicas pelo melhor filho,
para obter soluções completas cedo. `dive_interval = 0` desativa essas descidas.
A solução inicial usa vértices próximos, 2-opt e otimização dos contatos nas arestas.
Essas heurísticas só fornecem limites superiores. A decomposição é calculada sob demanda.
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

A API nova `tpp_convex_solve_certified` primeiro chama o solver geométrico com seu
workspace reutilizável. Recupera contatos na ordem exigida e verifica um limite dual.
Isso é necessário porque foram encontrados caminhos inviáveis e valores incorretos
na implementação antiga para algumas sequências com interseções. A proteção vale
para a nova API; as APIs antigas de ordem fixa não foram redirecionadas.

Quando o caminho geométrico tem um ponto por região, mas a recuperação falha por um
contato numericamente exterior à fronteira, o certificado tenta deslocar cada ponto
para o centroide de sua região por fatores `1e-12`, `1e-10` e `1e-8`. O caminho
reparado ainda passa por `contacts`, pela avaliação primal e pelo limite dual. Se a
viabilidade ou o gap original não fechar, o fallback permanece obrigatório.

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

Se o certificado não fechar, um método de pontos interiores resolve a sequência:
suavização das normas, barreira logarítmica das faces e Newton com Hessiana em blocos
tridiagonais de dimensão 2. Usa Eigen e, quando necessário, precisão quádrupla em
software do Boost.Multiprecision. Preserva o melhor primal e o melhor dual entre
iterações. Não requer Gurobi, CGAL ou outro otimizador comercial.

Os cálculos e certificados são **numéricos**, não provas em aritmética racional ou
intervalar. `exact` significa que

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
motivos de fallback, reparações do caminho geométrico, uso de precisão ampliada,
`nodes`, os dois contadores de
ramificação, `peak_queue`, `seconds` e `profile`.
A API C++ não tem limite de busca por padrão. A CLI exige limites explícitos.
Com `--initial-path`, a CLI lê após os polígonos a contagem de pontos do caminho
e seus pares `x y`, incluindo os extremos. Sem essa opção, a entrada antiga e a
heurística padrão permanecem iguais.
O limite de tempo é cooperativo: uma chamada geométrica/decomposição já iniciada
pode excedê-lo; o pré-processamento e partes da heurística inicial também não
consultam o limite a cada operação. `calls` conta invocações do oráculo convexo certificado, não passos
internos de Newton. `termination` distingue `optimal`, `call_limit`, `time_limit`
e `numerical_limit`.

### Organização da implementação

- `certified.cpp` orquestra o solver geométrico, a certificação e o fallback.
- `certified_geometry.cpp` recupera/repara contatos e calcula o limite dual.
- `certified_refinement.cpp` contém o método de pontos interiores nas duas precisões.
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
o fallback contém suas fases em `long double` e precisão ampliada. O total de
verificação de visitas soma medições nas fases superiores e se sobrepõe a elas,
portanto não deve ser somado novamente. A semântica também acompanha cada resultado
em `profile.timing_semantics`.

```cpp
#include "tpp/nonconvex/unordered.h"

tpp::UnorderedTppSolveOptions options;
options.max_seconds = 30;
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
