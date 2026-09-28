# Resultado e diagnóstico

A versão final foi mais rápida em **23 de 33 instâncias**. Nas 25 instâncias
em que ambos fecharam o gap solicitado em todas as repetições, foi mais rápida
em **17**, com média geométrica de `tempo Fekete / nosso tempo = 1,70`.
Todas as 165 execuções de cada solver produziram ciclos válidos. Nosso B&B
fechou o gap em 33/33 instâncias; Fekete, em 25/33. Os intervalos reportados
se sobrepõem em todos os casos. Isso não estabelece dominância universal.

## Por que a comparação anterior perdia

Vencer Gurobi em ciclos sintéticos de ordem fixa não implica vencer todo o
B&B TSPN. As relaxações reais incluem polígonos inclinados quase paralelos,
interseções e ordens parciais que não aparecem naquele conjunto pequeno.
No caso `german_3_n10`, apenas nove chamadas de oráculo consumiam quase todo
o tempo de aproximadamente 19 segundos: a árvore grande não era a causa.

O construtor confundia um contato retilíneo na fronteira com uma reflexão,
abandonava uma combinação de arestas quando a interseção caía além de um
extremo e descartava a cadeia construída antes da próxima atualização.
Isso acionava a redução geral por âncoras, com centenas de candidatas e
crescimento dos inteiros na aritmética racional.

A correção mantém contatos retilíneos inativos, fixa uma aresta bloqueada em
seu extremo e reutiliza a cadeia construída na próxima atualização. Racionais
e doubles usam o mesmo template. Cada pivô troca uma aresta por um vértice;
há no máximo `k` pivôs por atualização e `k+1` atualizações. Só o certificado
independente aceita uma solução; a redução geral completa permanece como
fallback. A prova de correção não depende de a proposta acertar os contatos.
O custo adicional dos pivôs é `O(k²)` por atualização, coberto pelo limite
anterior `O(k(N²+V))` da fase de propostas, com `V` sendo o custo de verificar
o certificado. O pior caso do B&B continua exponencial.

Foi encontrado também um custo independente de aproximadamente 0,35 ms:
Clang emitia `__kmpc_global_thread_num` na entrada da função da busca, antes
até do relógio das fases internas, mesmo com uma thread e zero oráculos.
A região OpenMP agora fica numa função não expandida no chamador, chamada
somente para lotes paralelos. A inspeção do assembly confirmou a remoção
da inicialização da entrada serial. Avaliação dos filhos, limites e exceções
mantêm o mesmo comportamento. O caso `common_point` caiu de 0,390 ms na etapa
`tspn-after` para 0,021 ms na etapa final, sem chamada ao oráculo em ambas.

## Tempos do TSPN completo

Medianas em milissegundos. A coluna anterior vem da campanha preservada de
27/09, com três repetições; final e Fekete foram medidos juntos nesta campanha,
com cinco repetições. Diferenças de máquina/carga entre campanhas impedem
atribuir pequenas variações ao algoritmo. Os ganhos grandes também aparecem
no diagnóstico direto das mesmas relaxações.

| Caso | Nosso anterior | Nosso final | Fekete final |
|---|---:|---:|---:|
| german_2_n5 | 24,015 | 2,292 | 0,287 |
| german_3_n10 | 19025,976 | 15,679 | 9,649 |
| german_6_n15 | 4619, timeout | 107,746 | 47,326 |

O resultado anterior de `german_6_n15` terminava com gap de cerca de 10,07%;
o atual fecha o gap e retorna objetivo `225.82948198897853`. Portanto, a linha
do timeout não é uma comparação de tempos até a mesma garantia.

| Conjunto | Casos | Nosso mais rápido | Razão geométrica Fekete/nosso | Nosso gap fechado | Fekete gap fechado |
|---|---:|---:|---:|---:|---:|
| Casos da comparação anterior | 25 | 18 | 1,710 | 25 | 18 |
| Oito casos adicionais | 8 | 5 | 1,355 | 8 | 7 |
| Todos | 33 | 23 | 1,617 | 33 | 25 |
| Ambos fecharam o gap | 25 | 17 | 1,700 | 25 | 25 |

As primeiras três razões comparam tempo de retorno; a última restringe a
comparação a conclusões com o mesmo critério de gap. As tabelas completas e
os dados por repetição estão em `tspn-final/`.

## Oráculos e perdas restantes

Nas 16 relaxações diagnosticadas, todas as soluções racionais foram
certificadas ótimas; os contatos double foram factíveis e seus intervalos
compatíveis com os racionais e os certificados independentes dos contatos
Gurobi. Com ordem fixa `[0,4,3,9,8,5,2]` de `german_3_n10`, o racional caiu
de 17781,78 para 14,37 ms e o double de 17570,59 para 3,27 ms.
No subproblema `[0,10,7,8,13]` de `german_6_n15`, caíram de 4399,95 para
4,85 ms e de 4505,74 para 2,40 ms, respectivamente. O baseline tem só uma
repetição medida; esses números servem para diagnosticar os atrasos enormes.

Não houve melhora em cada chamada: por exemplo, o double de
`german_3_n10_[0,3,4,9,8,5,2]` passou de 1,91 para 4,04 ms, enquanto seu
racional caiu de 67,52 para 19,79 ms. Vários oráculos ainda perdem para Gurobi.
Reconstrução racional e certificação têm custo real; a recuperação local
permanece habilitada e explicitamente contada. O double padrão não é uma
execução exclusivamente binary64.

No B&B final ainda perdemos em `nonconvex_l`, `seeded_concave_8`,
`german_1_n5`, `german_2_n5`, `german_3_n10`, `german_5_n15`,
`german_6_n15`, `german_17_n5`, `german_18_n5` e `german_20_n10`.
Há duas fontes distintas de custo:

- No `german_6_n15`, cerca de 94% do tempo da primeira repetição está nos
  oráculos, e todas as 31 chamadas registram recuperação racional. São 34
  chamadas no solver externo: a diferença principal aqui é custo por chamada.
- Fekete começa por um triângulo escolhido por distância, enquanto nossa raiz
  representa a região 0. Em `nonconvex_l`, são quatro oráculos nossos contra
  um externo; em `german_2_n5`, dois contra um. A raiz pode reduzir trabalho,
  mas não explica todos os casos. Não foi alterada nesta otimização.

O SOCP isolado do benchmark de ciclos e o SOCP interno de Fekete usam
formulações Gurobi distintas; seus tempos não são intercambiáveis.

## Formulação, exatidão e validação

TSPN fechado, ordem livre, sem depósito fixo, incluindo interseções e polígonos
não convexos. Uma thread em ambos. Gap externo `1e-6`; nosso gap relativo
`1e-6/(1+1e-6)` e absoluto zero, equivalentes a `UB <= (1+1e-6) LB`.
Factibilidade do B&B `1e-8`, validação independente `1e-7`. Gurobi:
`FeasibilityTol=OptimalityTol=1e-9`, `BarConvTol=BarQCPConvTol=1e-10`;
teste externo de visita `1e-9`. Os detalhes completos estão em `config.json`.

Nenhum epsilon de otimização ou discretização foi introduzido no oráculo
convexo. Toda candidata completa continua passando por
`tpp_convex_verify_cycle_certificate`. A API racional é exata para suas
entradas; o objetivo é uma soma de raízes de racionais. O B&B completo
preserva normalização, geometria numérica e contrato de gap existentes:
`exact` significa gap solicitado fechado, não TSPN racional com erro zero.
Os limites numéricos de Fekete não são certificados racionais exatos.

Passaram os testes afetados:

- Ciclos: referências Gurobi, 67 comparações double, 96 casos com interseções
  e quatro novas regressões de contatos ativos com verificação independente.
- TSPN: 19 casos por enumeração com uma e duas threads, 152 buscas
  interrompidas, decomposição, dois lotes concorrentes e 240 testes de dual.
- TPP de ordem livre: 86 casos por enumeração, 344 buscas interrompidas e
  regressões de avaliação paralela.

As saídas estão em `tests/`. `git diff --check` passou. A validação ampla do
dashboard, WASM e `sanity_check.sh` não foi repetida nesta alteração localizada.
O limite de tempo continua cooperativo, sem interromper um oráculo já iniciado.
O conjunto de 33 instâncias, embora inclua oito casos não usados no diagnóstico,
não substitui uma campanha maior nem demonstra vantagem para todo tamanho.
