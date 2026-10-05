# Oráculo convexo híbrido certificado

Implementação principal:
`packages/convex-tpp/cpp/src/solvers/hybrid.cpp`.
API pública: `tpp/convex/hybrid.h`.

## Contrato

O oráculo híbrido constrói primeiro uma solução em `double`, mas o modo seguro
com `max_gap=0` e sem cutoff somente aceita essa candidata depois de reproduzir
exatamente sua proveniência combinatória em aritmética racional binária.
Um gap positivo ou corte finito também permite o certificado primal-dual
descrito abaixo, sem afirmar otimalidade algébrica. O resultado contém exatamente um
contato ordenado por polígono, sem incluir os extremos `start` e `target`, e
preserva contatos duplicados.

A validação segura:

1. reconstrói exatamente vértices, interseções com arestas e regiões da última
   etapa do mapa;
2. materializa um contato pertencente a cada polígono na ordem exigida;
3. verifica as condições locais de otimalidade convexa, inclusive cones
   factíveis de arestas e vértices e sequências de contatos coincidentes;
4. encerra o objetivo radical por limites díadicos dirigidos;
5. tenta recuperar uma candidata rejeitada de polígonos intersectantes com
   predicados filtrados do mesmo mapa direcional;
6. usa o solver racional correspondente quando a recuperação também não pode
   ser certificada.

`TPP_HOMOGENEOUS_ZERO_DUAL`, ativada por padrão, representa direções e normais
da propagação dual de contatos coincidentes com inteiros arbitrários, no mesmo
algoritmo de disco/cones usado pelo certificado de ciclos. Eliminar
denominadores positivos e fatores comuns preserva as direções unitárias e
todos os sinais exatos. Reflexões também são calculadas até um fator positivo,
dispensando divisões racionais intermediárias. Coordenadas de polígonos e
contatos continuam racionais; pertencimento, limites, ordem dos predicados e
critérios de aceitação não mudam. `OFF` conserva a representação racional para ablação.
A redução por MDC é acionada por magnitudes a partir de `2^512`, sem limitar
inteiros coprimos ou precisão. A prova da equivalência e a cobertura dos testes
estão em [convex-cycle-certificate.md](convex-cycle-certificate.md).

Polígonos dois a dois disjuntos usam como fallback a recorrência estabelecida
em aritmética racional. Casos com interseção usam mapas direcionais racionais.
Uma falha de certificação significa apenas que a candidata rápida não foi
provada; ela nunca autoriza usar seu comprimento como limite inferior.
Quando o chamador fornece um corte finito, uma candidata materializada mas
não certificada pode dispensar o fallback se seu **dual factível em aritmética
racional** já alcançar o corte. Um dual em `double` serve apenas como filtro
para decidir se vale calcular o dual racional; ele nunca causa poda sozinho.
Nesse retorno antecipado, os contatos fornecem um caminho factível para a
sequência, o limite superior é o comprimento arredondado para cima desse
caminho e o limite inferior é o dual racional, sem afirmar que a sequência
foi resolvida até fechar o gap.

As funções `tpp_convex_solve_hybrid_safe` e
`tpp_convex_solve_length_hybrid_safe` expõem esse contrato. As variantes
`hybrid_unchecked` omitem reconstrução, certificação e fallback exato: existem
somente para diagnóstico de desempenho e não fornecem um limite seguro para
branch-and-bound.

`retain_binary_dual`, desativada por padrão, exporta uma proposta dual do
certificado intervalar quando há elos nulos ou curtos. A seleção converte cada
intervalo em um vetor binário específico e prova, por um limite superior
intervalar de sua norma quadrática, que ele pertence ao disco unitário.
Quando necessário, contrai a proposta; falhar nessa prova exporta zero.
Esses vetores não precisam atingir o limite do certificado KKT, cujo teste
completo prova existência e não devolve o testemunho. Retê-los não modifica
o caminho, os limites ou a decisão de aceitação da chamada original.

`tpp_convex_binary_dual_insertion_bounds` avalia esses duais em todas as
inserções de uma região, com intervalos arredondados para fora. Escrevendo
`u_0,...,u_n` para vetores de norma no máximo um, o limite é
`(t-s)·u_n + sum_i min_{x em P_i} (x-s)·(u_i-u_{i+1})`.
Inserir uma região altera apenas seus dois vetores incidentes e os suportes
dos vizinhos. Os demais termos são reutilizados, sem assumir que os contatos
ótimos dos filhos ficam parados. Os contatos fornecidos só propõem direções
novas; podem ser inviáveis. Normas e suportes precisam ser comprovados nos
dados binários originais; não se usa uma tolerância para o disco. Um vetor
cuja norma não seja comprovada, ou um ambiente sem arredondamento suportado,
retorna uma tentativa vazia. Assim, o valor binário sem certificado continua
sem autorização para poda; um limite intervalar certificado tem seu próprio
contrato de validade.

A sobrecarga de `tpp_convex_solve_hybrid` que recebe
`DynamicConvexTppWorkspace` reutiliza polígonos já convertidos e normalizados
em aritmética racional entre chamadas. O cache confere todas as coordenadas
binárias antes de reutilizar uma entrada, limita a retenção a 8192 vértices e
não altera os predicados, os limites ou a escolha do fallback.

O workspace também reutiliza caixas racionais e a classificação exata de
cada par de polígonos. A primeira consulta usa os mesmos testes racionais de
caixas, pertencimento e corte de segmentos; contatos de aresta/vértice contam
como interseção. Os pares usam identidades atribuídas somente após conferir
integralmente as coordenadas binárias. A classificação independe dos extremos,
da ordem da sequência e do cutoff; nenhum caminho, dual ou ótimo é reutilizado.
Colisões de hash não autorizam reutilização e identidades nunca são recicladas.

A geometria retida continua limitada a 8192 vértices de entrada, e a tabela
de pares tem no máximo 65536 registros. Limpar a geometria limpa a tabela de
pares; handles locais preservam entradas já selecionadas quando a preparação
do restante da sequência causa uma limpeza. Entradas grandes demais podem
participar da chamada, mas não ficam retidas. Copiar um workspace desacopla seu
cache mutável na chamada seguinte. Cada worker mantém seu próprio workspace.

`TPP_DENSE_PAIR_CACHE`, ativada por padrão, guarda as respostas dos pares cujas
duas identidades são menores que 256 em uma tabela de bits de 16 KiB por workspace.
Cada posição tem um bit de presença e um bit de disjunção: uma interseção
comprovada é distinta de uma consulta ainda não realizada. Identidades maiores
continuam na tabela de hash; não há limite adicional de polígonos ou peças.
As chaves continuam ordenadas e as identidades nunca são recicladas. A soma
dos registros nas duas representações obedece ao mesmo limite de 65536; tanto
a limpeza por capacidade quanto a limpeza da geometria apagam ambas. Nenhum
teste geométrico é substituído: mudam somente o armazenamento e a consulta de
respostas já comprovadas. Consultas, acertos e decisões de despacho devem
coincidir com a versão que usa somente hash.

`TPP_DENSE_DISJOINT_SUBSETS`, ativada por padrão, usa os bits positivos dessa
tabela para despachar sequências com pelo menos 12 entradas sem percorrer os
pares individualmente.
Uma máscara reúne as identidades da sequência; cada linha precisa conter provas
de disjunção com todas as outras identidades. Bits positivos são simétricos,
mas cada par continua contando como um único registro no orçamento do cache.
Identidades repetidas, maiores que 255, pares desconhecidos ou intersectantes
fazem a consulta em bloco declinar e mantêm o laço original. O atalho só retorna
disjunção quando todas as provas já existem. Nesse caso, as estatísticas contam
as mesmas `n(n-1)/2` consultas lógicas e acertos que o laço teria realizado;
os testes exatos continuam com a mesma contagem. A opção depende de
`TPP_DENSE_PAIR_CACHE`; desligá-la preserva somente o armazenamento compacto.

`DynamicConvexTppWorkspace::cache_disjoint_dispatch=false` repete o despacho
original mantendo o cache de conversão racional. A CLI de ordem livre expõe
essa ablação como `--no-oracle-dispatch-cache`. Ambos os caminhos precisam
produzir os mesmos contatos, limites, backend e decisões de busca quando o
orçamento de chamadas é igual e o limite de tempo não interfere.
`dispatch_pair_queries`, `dispatch_pair_cache_hits` e
`dispatch_pair_exact_checks` contam consultas, acertos e consultas que
precisaram de testes geométricos após as caixas; essas métricas não são os
predicados do certificado de otimalidade.

O cache guarda também os vértices binários da normalização racional. A etapa
intervalar reutiliza essa mesma representação, evitando converter cada
coordenada racional de volta para double duas vezes por chamada. O replay,
reparador, teste de pertencimento e dual são os mesmos. A opção
`cache_interval_geometry=false` / `--no-oracle-interval-geometry-cache` repete
essas conversões para uma ablação independente do cache de pares. Os dois
caches são ativados por padrão em workspaces; a API sem workspace continua
preparando a geometria a cada chamada.

Cada entrada também conserva o índice de rotação angular das arestas da
fronteira normalizada, calculado pelos mesmos predicados racionais usados na
materialização. Esse índice depende somente do polígono. A reconstrução de
contatos reutiliza-o nas propostas ordinárias, de fronteira e filtradas;
a API sem workspace continua calculando-o diretamente. Handles imutáveis
mantêm os índices associados à geometria correta mesmo durante uma limpeza
do cache. O caminho que encerra pelo certificado intervalar não prepara um
vetor adicional de índices por chamada. A retenção acrescenta um `size_t` por
entrada e preserva o limite de 8192 vértices; nenhum contato ou ótimo é retido.

`DynamicConvexTppWorkspace::borrow_hybrid_geometry` permite usar referências
imutáveis aos polígonos racionais preparados. Handles locais conservam os
objetos durante toda a chamada, inclusive se o cache for limpo ao preparar
outro polígono. O buffer por valor continua disponível para ablação; coordenadas,
normalização, predicados e certificados são iguais. A API sem workspace conserva
seus vetores próprios. A opção é true por padrão; false repete as cópias para ablação.

`bound_before_optimality` permite tentar o corte dual racional antes do KKT
completo, somente após materializar contatos factíveis. O filtro flutuante não
é usado para poda. Um corte certificado dispensa apenas a prova de otimalidade
local; os limites e a factibilidade mantêm o contrato existente. O modo shadow
conserva seu caminho de validação completo. Essa opção permanece false por padrão.

`dispatch_seconds` inclui preparação/cópias de geometria e classificação de
disjunção. `proposal_preparation_seconds` separa a preparação dos vértices
binários e objetos de replay da proposta intervalar. `bound_evaluation_seconds`
mede a avaliação explícita de limites
finais e duais usados nos retornos fora das fases de materialização/certificado.
Limites intervalares construídos dentro do certificado continuam no tempo
desse certificado. O perfil exportado pelo B&B usa
`convex_dispatch_seconds`, `convex_proposal_preparation_seconds` e
`convex_bound_evaluation_seconds`, somando trabalho
por chamada; os tempos dos workers podem exceder o tempo de parede do lote.

`ConvexHybridAggregate` também conta cortes duais, tentativas e sucessos das
recuperações de fronteira disjunta e filtrada, e mede o tempo inclusivo de
cada recuperação (`touching_disjoint_seconds`, `filtered_seconds`) e o do
replay exato, materialização e certificado de uma candidata intersectante
rejeitada (`rejected_replay_seconds`, já contido nos totais por fase). Com
`TPP_HYBRID_AGGREGATE=1`, `tpp-unordered` imprime esses agregados numa linha
JSON em stderr ao terminar; stdout não muda. São diagnósticos de perfil, sem
efeito no oráculo.

## Encerramento por limites primal-dual intervalares

### Aritmética e reutilização de pertencimento

As opções CMake `TPP_FAST_INTERVAL_ROUNDING`, `TPP_DYADIC_MEMBERSHIP` e
`TPP_MEMO_BINARY_MEMBERSHIP` permitem ablar três otimizações internas do mesmo
certificado, ativadas por padrão. `OFF` em cada opção conserva sua implementação
anterior. Não mudam os gaps, contatos, cutoffs ou condições de aceitação.

`TPP_FAST_INTERVAL_ROUNDING` expande um double armazenado para seu vizinho
IEEE binary64 por manipulação da representação inteira. Magnitudes positivas
são ordenadas por seus bits; para negativos, a ordem se inverte. Zeros,
subnormais, infinitos e NaNs recebem o mesmo extremo de `CycleInterval::down/up`
anterior. A comparação de regressão verifica igualdade bit a bit com
`std::nextafter` ao redor de todas as transições de expoente e em amostras de
representações. São preservados os valores dos intervalos, não os efeitos de
libm em `errno` e flags de exceção flutuante; o certificado não os consulta.
As operações continuam armazenadas antes da expansão, e o teste de ambiente
de arredondamento/subnormais continua exigido para a prova intervalar.

`TPP_DYADIC_MEMBERSHIP` trata orientações ambíguas de contatos binários como
determinantes de inteiros escalados. Cada coordenada binary64 finita é um
inteiro assinado vezes uma potência de dois. Escalar cada eixo por sua menor
potência conserva o sinal do determinante. O caminho curto só opera quando
as três coordenadas de cada eixo têm no máximo 61 bits de magnitude: as
diferenças têm no máximo 62 bits e o determinante tem magnitude menor que
`2^125`, cabendo em `__int128` assinado. O guard é verificado antes de shifts
ou produtos. Exponentes mais distantes e compiladores sem esse tipo conservam
a comparação racional original. Orientações zero continuam zero, sem epsilon.

`TPP_MEMO_BINARY_MEMBERSHIP` guarda até dois resultados de pertencimento por
entrada de geometria preparada, incluindo recusas e reparos aceitos. A chave
contém os bits completos das duas coordenadas e a identidade da geometria
racional imutável. Handles locais conservam as entradas durante a chamada;
evicção destrói também seus resultados e a cópia de workspace continua
desacoplando o cache mutável. Cada worker tem seu workspace. A API sem workspace
ou com `borrow_hybrid_geometry=false` repete a prova. Geometria binária e racional
precisam representar a mesma fronteira imutável. Há armazenamento constante por
entrada, sem tabela de caminhos e sem estado adicional nos nós de B&B. O dual e
o objetivo de cada sequência continuam calculados a cada chamada. A contagem de
predicados exatos passa a refletir os testes efetivamente executados em misses do memo.

As três otimizações devem preservar bit a bit contatos e limites e, sob o
mesmo orçamento de chamadas sem interferência temporal, decisões da busca.
Contagens de trabalho físico podem diminuir por reutilização de provas.

`TPP_KKT_STRAIGHT_FIRST` é experimental e permanece `OFF` por padrão; muda
somente a ordem dos testes exatos no KKT local.
Um vetor racional tem norma quadrática zero exatamente quando ambas as
coordenadas são zero. Dois vetores não nulos com produto vetorial zero e
produto escalar positivo têm a mesma direção normalizada; sua diferença de
suporte é zero em qualquer polígono. Essas decisões podem ocorrer antes de
calcular as normas quadráticas racionais. Os outros contatos e os blocos de
elos zero conservam o certificado completo existente.

`TPP_INTERVAL_PRIMAL_DUAL=ON` (padrão CMake) permite tentar um certificado
barato antes do replay racional. `ConvexHybridOptions::max_gap` é zero por
padrão: a API híbrida sem opções preserva seu contrato de otimalidade exata.
A API `tpp_convex_solve_certified` encaminha a tolerância que já recebia do
B&B, antes ignorada. Não há mudança dos gaps globais, do cutoff nem do orçamento.
`OFF` desativa esse encerramento para ablações, mantendo a API e o fluxo exato.

O mesmo replay do traço é instanciado em `double` para propor um caminho.
O reparador geométrico existente propõe um contato por polígono. Cada contato
binário exportado precisa pertencer **exatamente** ao polígono original:
orientações intervalares decidem sinais claros e orientações racionais binárias
decidem os ambíguos. Um contato arredondado para fora pode ser movido uma pequena
fração em direção à média dos vértices, mas sua nova posição também exige esse
teste. As margens do reparador e as frações não autorizam factibilidade.
O teste de pertencimento é compartilhado com o certificado intervalar de ciclo
em `binary_certificate.h`. O comprimento dessa cadeia factível é encerrado por
`CycleInterval`, produzindo um limite superior rigoroso `U`.

Para a sequência `q_0=start, q_1, ..., q_m, q_{m+1}=target`, quaisquer vetores
`u_0,...,u_m` no disco unitário fornecem o limite dual

`D = (target-start).u_m + sum_i min_{v em P_i} (v-start).(u_{i-1}-u_i)`.

Isso decorre de `|q_{i+1}-q_i| >= u_i.(q_{i+1}-q_i)` e da soma telescópica.
As direções propostas são diferenças **exatas** de coordenadas binárias,
divididas por um limite superior intervalar de sua norma. Os intervalos encerram
esses vetores factíveis; seus extremos não são escolhidos como vetores duais.
Os suportes mínimos, a soma e o comprimento direto são arredondados para fora,
produzindo `L <= OPT <= U` na entrada original. As direções podem vir de contatos
ainda não reparados: factibilidade do dual depende apenas de suas normas.

Elos curtos podem receber a direção do elo longo anterior, posterior ou dos
extremos. A escolha usa o gap solicitado e a escala numérica somente como
heurísticas para propor vetores do disco. Ela não declara coincidência e não
elimina restrições. Cada política gera outro dual factível; usa-se o maior LB.
Sem elos curtos, as quatro políticas originais produzem o mesmo vetor e são
avaliadas uma única vez, preservando o valor e os arredondamentos do limite.
`ConvexHybridOptions::interpolated_zero_dual` (padrão false), encaminhado pelo
workspace e pela CLI `--interpolated-zero-dual`, acrescenta uma interpolação
entre as direções longas que delimitam cada bloco curto. Cada direção proposta
é uma combinação convexa de vetores factíveis no disco unitário; portanto
também é factível. A combinação é encerrada por intervalos antes da avaliação
dos suportes mínimos. O tamanho do elo apenas seleciona uma candidata: não
declara coincidência, não muda os contatos e não remove restrições.
Quando nenhum desses duais basta, o certificado completo de elos zero e os
demais fallbacks continuam disponíveis.

Uma chamada encerra antecipadamente somente se `L >= cutoff` ou se a subtração
dirigida provar `U-L <= max_gap`, com gap positivo e finito. O retorno marca
`interval_bounds_certified`, separado de `double_certified` (KKT exato).
O modo shadow conserva o replay/certificado exatos. Ambiente incompatível,
overflow, pertencimento não provado ou limites insuficientes retêm o fluxo
original. `CycleInterval` exige o mesmo ambiente IEEE binary64 já documentado
para os predicados filtrados.

Se o primeiro traço não bastar e a heurística sugerir contato de fronteiras,
a mesma contração `2^-20` da recuperação disjunta fornece uma segunda proposta.
Seus vértices e arestas são remapeados para a geometria original antes do replay
em double. O certificado primal-dual continua sendo calculado nos **polígonos
originais**, sem assumir disjunção da proposta ou estabilidade combinatória.
Assim, esta implementação não precisa aceitar `OPT(P')` nem aplicar a correção
de continuidade `2 eta sum R_i`: ela mede diretamente um intervalo do problema
original. `interval_bounds_contracted` identifica esse caminho; no B&B,
`oracle_interval_bound_calls` e `oracle_contracted_bound_calls` contam os retornos.
Tempos de materialização/certificado incluem as tentativas fracassadas.

## Recuperação com predicados filtrados

`TPP_FILTERED_DIRECTIONAL=ON` (padrão) compila a mesma construção em
`intersecting_maps.cpp` com o escalar interno `FilteredRational`. A tentativa
em `double`, sua certificação e o teste barato de cutoff continuam primeiro.
Somente uma candidata intersectante rejeitada que ainda exige refinamento
aciona a recuperação. `OFF` mantém o fluxo anterior para ablações; não altera
o contrato seguro. O modo `Unchecked` continua usando a construção não
certificada em `double`.

Cada expressão guarda um intervalo binário e um DAG das operações racionais
originais. Intervalos disjuntos decidem uma comparação; quando se sobrepõem,
o DAG é avaliado exatamente, com cache por nó. As primitivas de intervalos
compartilham `CycleInterval`: operações armazenadas separadamente e expansão
com `nextafter`, sem epsilon. Divisão por um intervalo contendo zero e overflow
deixam a comparação inconclusiva, exigindo avaliação racional. A filtragem só
opera em ambiente IEEE binary64 com subnormais e arredondamento suportados;
outros ambientes usam comparações racionais. Expressões construídas sem esse
suporte continuam inconclusivas mesmo após uma mudança de ambiente.

Assim, a construção toma as mesmas decisões combinatórias que a variante
racional, avaliando exatamente apenas expressões necessárias a predicados
ambíguos. As coordenadas exportadas continuam sendo propostas: o traço precisa
ser reproduzido em racional e passar os mesmos testes de pertencimento e
suporte. Nenhuma comparação aproximada ou comprimento da candidata autoriza
uma poda. Uma falha em qualquer etapa mantém a recuperação racional completa.
Essa mudança não acrescenta tolerância de otimização nem muda o critério de
encerramento do B&B.

Os nós do DAG são imutáveis e ficam numa arena por thread, liberada quando
termina o `FilteredRational::Scope` mais externo; não há alocação nem contagem
de referências por operação. Valores não podem sobreviver a esse escopo; a
única entrada, `solve_intersecting_map_trace_filtered`, destrói o mapa antes.
A arena retém até 16 blocos de 4096 nós para a chamada seguinte.

Predicados de sinal do mapa (`cross_sign`, `dot_sign`, `same_direction`, cones
e pseudo-arestas) avaliam primeiro só os intervalos da expressão, sem criar
nós (`FilteredRational::Virtual`). Cada operador repete os atalhos e as
expansões de `operation()`, e um intermediário nunca compartilha nó com outro
valor; assim o intervalo é bit a bit o que o DAG teria e decide exatamente as
mesmas comparações. Se ele não decide, o DAG é construído como antes e avaliado
exatamente. Decisões, avaliações exatas e traço exportado não mudam.

`filtered_attempted` e `filtered_certified` identificam essa recuperação nas
estatísticas do híbrido. O backend `DoubleIntersection` continua indicando uma
candidata reconstruída e certificada, inclusive quando recuperada pelo filtro.
Os tempos de construção, materialização e certificado somam ambas as tentativas;
`rational_fallback_seconds` continua medindo somente a recuperação completa.
O DAG é local à chamada e descartado ao retornar. Seu custo e memória dependem
da quantidade de operações e do tamanho dos operandos racionais; não há uma
garantia de speedup para toda instância.

### Predicados racionais sem normalização

O certificado KKT, a materialização dos contatos (`logarithmic_clip`,
`support_max`, `feature_on_edge`) e `set_exact_bounds` usam pontos racionais
em forma homogênea inteira `(x/w, y/w)`, `w > 0`, e diferenças como múltiplos
inteiros positivos do vetor racional, sem MDC por operação. Os testes são
sinais de produtos vetoriais e escalares, testes de vetor nulo e a comparação
`u.f/|u|` contra `v.f/|v|` (`convex_normalized_difference_sign_integer`), todos
invariantes a escalas positivas separadas de cada direção e do vetor factível.
O parâmetro `u` de uma aresta pertence a `[0,1]` exatamente quando
`(q-a).e >= 0` e `(q-b).e <= 0`. Em `set_exact_bounds`, cada termo
`floor(sqrt(floor(|d|^2 2^192)))` é calculado com `|D|^2/w^2` para `D = w d`,
e as somas de termos com denominador `2^96` são formadas uma vez: o racional
final é o mesmo. A propagação pelo disco nos blocos de elos nulos continua
recebendo as direções racionais originais. Contatos, cortes e pontos
construídos continuam racionais. Decisões, contagens de predicados e limites
são idênticos aos da versão racional.

No mapa direcional, a caixa de cada polígono (união das caixas das arestas)
descarta de uma vez as arestas cuja caixa é disjunta dela; são exatamente os
pares que o teste aresta a aresta já descartava.

## Recuperação de contatos de fronteira com a recorrência disjunta

`TPP_TOUCHING_DISJOINT=ON` (padrão CMake) tenta recuperar uma candidata
intersectante rejeitada antes de construir o mapa filtrado. A construção em
double, o certificado e o cutoff dual barato continuam primeiro. Casos já
certificados não pagam pelas propostas adicionais. `OFF` desativa apenas essa
recuperação para ablações; o contrato seguro continua igual.

Um teste de caixas e eixos separadores em ponto flutuante sugere sequências
com interiores disjuntos e fronteiras compartilhadas. Esse teste é **somente
uma heurística para escolher candidatas**: sua margem de arredondamento pode
aceitar pequenas sobreposições de área. Ele não prova disjunção, pertencimento,
otimalidade nem autoriza poda.

Primeiro, cada polígono normalizado é contraído em direção à média de seus
vértices por uma fração `2^-20`. A recorrência existente de busca binária para
disjuntos fornece um traço de cruzamentos, reflexões e vértices. As identidades
de arestas são preservadas e os vértices contraídos são substituídos pelos
vértices originais. O traço inteiro é então reproduzido **na geometria original
em racional**, incluindo refolding, ordem e pertencimento. O mesmo certificado
KKT, inclusive o disco/cone de elos zero, precisa provar sua otimalidade. Uma
segunda proposta usa diretamente as coordenadas originais se a primeira falha.
Falhar em ambas mantém a recuperação por mapas filtrados/racionais.

Assim, não há um segundo solver geométrico, nem retorno de comprimento ou
contatos perturbados como limite do problema original. A fração `2^-20` é um
parâmetro da proposta, não uma tolerância de factibilidade ou otimização.
Contração arredondada pode manter contatos ou criar degenerescências; isso
somente reduz a taxa de certificação. As estatísticas por chamada
`touching_disjoint_attempted`, `touching_disjoint_certified` e
`touching_disjoint_perturbed` distinguem tentativa, sucesso e uso da contração.
Uma recuperação certificada usa backend `DoubleDisjoint`; `stats.disjoint`
continua significando disjunção **estrita** provada pelo despacho original.
Os tempos por fase incluem as tentativas fracassadas dessa recuperação.

### Continuidade e seus limites

Uma translação comum preserva todas as interseções. Já a contração ideal
`P_i' = c_i + (1-eta)(P_i-c_i)`, com centro interior e `0 < eta < 1`, torna
estritamente disjuntos polígonos convexos de área positiva com interiores
disjuntos. Para extremos fixos e ordem fixa, escrevendo
`R_i = max_{x em P_i} |x-c_i|`, vale

`0 <= OPT(P') - OPT(P) <= 2 eta sum_i R_i`.

A desigualdade inferior decorre de `P_i'` ser subconjunto de `P_i`. Para a
superior, contraia cada contato de um caminho ótimo original: seu deslocamento
é no máximo `eta R_i`, e a desigualdade triangular conta cada deslocamento
em no máximo dois segmentos. Esse argumento vale para a contração ideal;
não certifica a implementação arredondada nem estabilidade do traço
combinatório. Continuidade do objetivo não permite subtrair um shift de
cada contato e declará-lo ótimo. O replay e o certificado originais eliminam
essa dependência de um argumento aproximado.

## Casos de fronteira

O localizador de contato trata tangência, passagem por vértice, fechamento
circular, retas paralelas e sobreposição colinear com uma aresta de suporte.
Contatos coincidentes com direções alinhadas usam um testemunho constante.
Em blocos internos com mudança de direção, o certificado completo propaga
subgradientes no **disco unitário fechado**, por somas com os cones normais
dos polígonos. A implementação compartilhada em `zero_contact_certificate.h`
é a mesma usada pelo certificado de ciclo; seu argumento de correção está em
[certificado de ciclo](convex-cycle-certificate.md#complete-zero-block-test-by-diskcone-reachability).
Isso inclui vetores de norma menor que um, necessários em elos de comprimento
zero. Enumerar somente direções unitárias e reflexões pode rejeitar uma
candidata ótima e provocar um fallback desnecessário.

Cada bloco inclui os dois contatos adjacentes aos elos não nulos. Uma vez
provado o alcance da direção de saída, todas as condições de suporte desse
bloco estão certificadas; os outros contatos continuam usando os predicados
locais exatos. Blocos que alcançam um extremo fixo usam a direção do elo
não nulo adjacente (ou zero para um caminho estacionário), um testemunho válido
porque o extremo fixo não impõe condição de suporte. Não há amostragem nem
limite de quantidade de testemunhos. Uma candidata que falhe nesse teste, na
reconstrução ou em qualquer outro predicado ainda exige o fallback racional.

Antes da propagação pelo disco, dois testemunhos suficientes são testados:
atribuir a direção unitária de entrada a todos os elos zero e certificar a
mudança de direção no último contato; ou atribuir a direção de saída e
certificar a mudança no primeiro contato. Em cada caso, todos os outros
contatos do bloco têm diferença de suporte zero, válida para qualquer cone
normal. Isso custa somente os testes KKT exatos dos contatos de fronteira e
é útil em arestas compartilhadas. Falhar nesses dois testes não rejeita a
candidata: a propagação completa pelo disco continua em seguida, cobrindo
os testemunhos interiores e as mudanças distribuídas por vários contatos.

As suítes focadas devem continuar cobrindo cardinalidade dos contatos, contatos
duplicados, orientação invertida, polígonos repetidos, extremos estacionários,
tangência, sobreposição colinear, polígonos finos ou quase colineares e escalas
de coordenadas muito pequenas e muito grandes.

## Validação

```bash
cmake --preset convex-release -DTARGET=main-directional_tests
cmake --build --preset convex-release -j 4
.build/convex-release/packages/convex-tpp/cpp/tpp-convex \
  --random-boxes 1000 --random-convex 200

cmake --preset convex-release -DTARGET=main-intersection_tests
cmake --build --preset convex-release -j 4
.build/convex-release/packages/convex-tpp/cpp/tpp-convex

cmake --preset nonconvex-release -DTARGET=main-unordered_tests
cmake --build --preset nonconvex-release -j 4
.build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp
```

Resultados temporais e comparações entre versões não pertencem a este contrato.
`benchmarks/results-saved/` mantém resumos agregados e fixtures mínimos; dados
brutos, entradas repetidas e procedimentos de campanha completos ficam em
execuções locais ignoradas. A comparação alemã é a exceção por ser consumida pelo
material SIICUSP.
