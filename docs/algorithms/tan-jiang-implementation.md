# A construção de Tan e Jiang no código: correspondência, correção e custo

Data: 18 de novembro de 2026.

Esta nota compara as duas implementações da construção de Tan e Jiang com o
algoritmo do artigo deles (`docs/bibliography/TPP-Tan&Jiang/main.tex`, Lemas 1
e 2, §4 e Teoremas 1 e 2). As implementações são:

- `SolutionTAMC`, em
  [tan_jiang.cpp](../../packages/convex-tpp/cpp/src/solvers/tan_jiang.cpp),
  para polígonos disjuntos;
- `build_by_scans`, em
  [intersecting_maps.cpp](../../packages/convex-tpp/cpp/src/solvers/intersecting_maps.cpp),
  para polígonos que podem se intersectar.

A nota prova três coisas:
1. as duas seguem o algoritmo deles, com as diferenças listadas na Seção 1;
2. `build_by_scans` localiza cada consulta exatamente onde a busca binária dos
   mapas memoizados a localiza;
3. o custo é o deles, O(kn) e O(k²n) em operações aritméticas, sob o Lema 1
   deles, que é a mesma hipótese de que depende o limite deles.

A nota prova o Lema 1 para o mapa imediatamente anterior. Para os mapas mais
antigos, ele fica como hipótese, com a medição da Seção 4. As definições e os
lemas citados como "prova, Seção j" são os de
[intersecting-tpp-proof.md](intersecting-tpp-proof.md): funções valor F_i,
lema do raio, células exteriores e ação válida na aderência de uma célula.

Notação. Os polígonos são P_1, …, P_k, com n vértices ao todo. V_i é o
conjunto dos vértices divididos de P_i (originais, pseudo-vértices e o início,
se incidente). M = Σ|V_i| = O(kn). O mapa de P_h tem c_h = 2|V_h| células,
em ordem circular: 2j é o cone do vértice j, e 2j+1 é a região da
pseudo-aresta que vai de j a j+1 (prova, Seções 4–6).

## 1. Correspondência com o algoritmo deles

| Tan e Jiang | Código |
| --- | --- |
| Mapas M_1, …, M_k, cada um com regiões em ordem circular (§3) | Mapas divididos com as células 2j e 2j+1 (prova, Seções 3–5). Para disjuntos, os cones de `Solution` |
| Pseudo-vértices: as interseções de P_i com P_h, h<i, calculadas antes, O(kn) delas (§4) | `split_boundaries`, com os pares de [convex_contacts.h](../../packages/convex-tpp/cpp/include/tpp/convex/detail/convex_contacts.h): O(kn) |
| Lema 2: os vértices de P_{i+1}, inclusive os pseudo-vértices, são localizados juntos em M_i, …, M_1, por varreduras monótonas a partir da região do primeiro | `locate_together`: os dois lados de cada vértice dividido, na ordem da fronteira, nível a nível. O primeiro de cada lote é localizado por busca binária; os seguintes por varredura a partir da célula do anterior |
| Cruzamento ou reflexão: passa ao mapa anterior o ponto de contato q_{h+1,j}, ou seu refletido, "na mesma região de M_h" | `batch_sources` passa a consulta desdobrada. Pelo Lema A, ela está na mesma célula do contato |
| Pertinência a P_h, só para os vértices de P_{i+1}, pelas interseções já calculadas (§4) | A âncora de cada consulta, inclusive as refletidas (Lemas B e C) |
| Relatório do caminho até t: O(n), ou O(k log(n/k)) por busca binária | `query_path` por busca binária, O(k log M) |

Há quatro diferenças.

1. **Cones nos pseudo-vértices.** O código usa os raios limite incidentes de
   Dror et al. A regra local do §4 deles erra num exemplo sem degenerescência
   ([intersecting-tpp-audit.md](intersecting-tpp-audit.md)). A correção muda os
   cones, não o procedimento nem o número de células.
2. **Pertinência das imagens refletidas.** O argumento deles cobre só os
   vértices de P_{i+1}. Nos níveis abaixo, a maioria das consultas é refletida.
   No caso disjunto ela nunca está dentro de P_h, mas com interseções pode
   estar. A Seção 2 completa o argumento.
3. **Varredura.** Em vez de até três cadeias x-monótonas, o código procura a
   célula alargando a partir da anterior, nos dois sentidos. O custo é o dobro
   da distância circular percorrida, o mesmo que o deles a menos de constante.
   `SolutionTAMC` faz rodadas de ida e volta, como eles.
4. **Salvaguardas.** Elas não alteram o resultado.
   - Um orçamento de 4(c_h+b)+16 testes por lote, com b consultas.
   - A busca binária para o primeiro do lote, para as âncoras degeneradas e
     para uma varredura que falha.
   - Em `SolutionTAMC`, no máximo quatro rodadas e depois a busca binária.

   Em aritmética exata, nenhuma varredura falhou e nenhum orçamento se esgotou
   nas medições (Seção 4).

## 2. Correção: as varreduras localizam como a busca binária

Uma consulta no nível h é um germe q_ε = q + εd, com desempates secundários
(prova, Seção 8). `locate` decide primeiro a pertinência fechada de q_ε a P_h
(`inside`). Fora de P_h, a busca binária devolve a célula exterior que contém
a consulta com os desempates.

**Lema A (último segmento).** Seja y um ponto fora de P_h cujo h-caminho
termina no segmento [w, y], com w a base no bordo de P_h. Então todo ponto p
de (w, y] está na mesma célula de y.

*Prova.* As células são uniões de raios a partir das bases (prova, Seção 5).
Pelo lema do raio, F_h é afim ao longo do raio de w que passa por y, com
gradiente constante (prova, Seção 1). Assim p está no mesmo raio e na mesma
célula. ∎

Cada consulta carrega uma âncora: a pseudo-aresta aberta e que contém o ponto
que ela representa.
- Para um lado de um vértice, e é a pseudo-aresta do próprio vértice naquele
  lado.
- Para uma consulta refletida, e é a pseudo-aresta da última reflexão. Os
  níveis entre aquela reflexão e h foram cruzamentos.

Como as fronteiras são divididas em todos os contatos (prova, Seção 3), e
está inteira dentro de int P_h, inteira fora de P_h ou inteira sobre ∂P_h. O
ponto médio decide, e `compute_statuses` classifica todos os pontos médios de
P_g contra P_h em O(|V_g|+|P_h|). Ele percorre a fronteira convexa de P_g, e a
cunha do ponto médio em torno de um ponto interior de P_h muda de sentido no
máximo duas vezes.

**Lema B (âncora fora).** Suponha que a âncora esteja fora de P_h, que a
varredura ache uma célula X com q_ε em X e que q_ε esteja estritamente fora da
reta de uma aresta de P_h em X, pelo par (termo constante, termo em d). Então
`locate` devolve X.

*Prova.* Estritamente fora de uma reta de aresta, o germe está fora de P_h
fechado, e `inside` o rejeita. Com os desempates, a consulta é um ponto
exterior genérico, e as células abertas são disjuntas (prova, Seções 5 e 8).
Os predicados de uma célula isolada (`in_cone`, `in_edge_plus` com vértices
vizinhos) caracterizam a célula aberta (prova, Seções 4 e 6). Logo a busca
binária acha a mesma X. ∎

Se a célula achada não passa nesse teste, o código usa `locate`.

**Lema C (âncora dentro, contato estrito).** Suponha que e esteja dentro de
int P_h. No caso de uma consulta refletida, suponha também que a consulta
desdobrada anterior não estivesse em nenhuma das retas dos dois raios limite
da célula de reflexão, pelo par (constante, d). Então `locate` devolve um
cruzamento.

*Prova.* Para um lado de um vértice, q_ε pertence à pseudo-aresta aberta e
para ε pequeno, logo a int P_h, e `inside` aceita.

Para uma consulta refletida no nível g>h, fora dos dois raios limite, o
contato c_ε da reflexão está no interior de e para todo ε>0 pequeno, logo em
int P_h. Os níveis g−1, …, h+1 foram cruzamentos, e assim o h-caminho até a
consulta desdobrada y_ε é o (g−1)-caminho até ela. Pelo invariante de sufixo
terminal (prova, Seção 7), c_ε está no último segmento desse caminho.

Se a localização de y_ε em P_h fosse um cone ou uma reflexão com base w, o
último segmento seria [w, y_ε]. Os raios das células que não são cruzamento
deixam P_h (prova, Seção 5), e então esse segmento não encontraria int P_h.
Isso contradiz c_ε ∈ int P_h. Logo, para todo ε pequeno, y_ε está em P_h ou
numa célula exterior de cruzamento.

Resta ver o que `locate` devolve.
- Se `inside` aceita o germe, o resultado é −1.
- Caso contrário, os desempates escolhem uma célula cuja aderência contém o
  germe (prova, Seção 8). Se fosse uma célula vizinha que não é cruzamento,
  sua ação seria válida na aderência. Então haveria um h-caminho ótimo com
  último segmento sobre um raio que não entra em int P_h, outro passando por
  c_ε, e os dois teriam a mesma geometria (prova, Seção 2). É impossível.

Logo `locate` devolve −1. ∎

Sem a condição estrita, o contato pode estar num extremo de e, sobre ∂P_h, e o
argumento não vale. O código manda essas consultas para `locate`
(`Anchored::strict`). O mesmo vale para as âncoras sobre o bordo e as de
polígonos com menos de três vértices.

**Teorema 1.** `build_by_scans` calcula, para cada lado de cada vértice, a
mesma fonte virtual e a mesma trilha que `virtual_source_trail`. Logo constrói
os mesmos raios e a mesma decisão `single_ray` que `build_vertex`, e todo o
resto (`query_path`, `length`, `trace`) devolve o mesmo resultado que os mapas
memoizados.

*Prova.* Por indução no nível. Pelos Lemas B e C, cada localização do lote é
a de `locate`; nos demais casos é a própria `locate`. `batch_sources` aplica as
mesmas três ações de `virtual_source_trail`, com a mesma internação de
trilhas: vértice, reflexão e passagem. `finish_vertex` é o final comum dos
dois caminhos. ∎

Os testes direcionais conferem isso para cada consulta
(`TanJiangScanCounters::verify`). Os números estão na Seção 4.

## 3. Custo em operações aritméticas

Como no artigo deles, suponha posição geral: três arestas não passam por um
mesmo ponto, e nenhum vértice de um polígono está sobre a reta de uma aresta
de outro. Nesse caso, nenhuma âncora fica sobre o bordo, nenhuma consulta
refletida perde a condição estrita e nenhuma célula é rejeitada pelo Lema B.
Fora de posição geral o resultado continua exato, e cada consulta degenerada
custa O(log c_h).

Para um lote (i, h), sejam b_{i,h} ≤ 2|V_i| suas consultas e T_{i,h} a soma
das distâncias circulares entre as células de consultas varridas
consecutivas.

**Teorema 2.**
- A divisão custa O(kn + M log k). Em posição geral, uma aresta corta cada
  fronteira anterior no máximo duas vezes, e a ordenação por aresta, sem
  ordenar candidatos, custa O(M log k).
- Os estados das âncoras custam O(Σ_{h<g}(|V_g|+|P_h|)) = O(kM).
- Um lote custa O(c_h + b_{i,h} + T_{i,h}) mais uma busca binária,
  O(log c_h).

A construção inteira custa O(kn + M log k + kM + Σ T_{i,h} + k² log M).

*Prova.* Uma consulta varrida testa no máximo 2·dist+1 células, onde dist é a
distância circular até a célula da varredura anterior. As internações de
trilhas usam tabela de dispersão, O(1) esperado. O resto é contagem direta. ∎

**Lema 1 (Tan e Jiang), forma usada.** Para todo lote (i, h),
T_{i,h} = O(c_h + b_{i,h}).

**Corolário.** Sob o Lema 1, a construção custa
O(Σ_i Σ_{h<i}(c_h + |V_i|)) = O(kM) = O(k²n), o Teorema 2 deles. Para
polígonos disjuntos, M = n, e `SolutionTAMC` custa O(kn), o Teorema 1 deles.
Sem o Lema 1, o orçamento mantém cada lote em O(c_h + b log c_h), e o total
fica no O(kM log M) dos mapas memoizados.

**Proposição 3 (Lema 1 no mapa anterior, h = i−1).** T_{i,i−1} ≤ 4c_{i−1}.

*Prova.*
1. *As consultas.* As consultas do lote (i, i−1) são os lados dos vértices de
   P_i, na ordem de ∂P_i, uma curva convexa. As varridas são as que estão
   fora de P_{i−1}.
2. *Células e a curva.* Cada célula de M_{i−1} encontra ∂P_i em no máximo três
   arcos.
   - Uma região de pseudo-aresta é a interseção de três semiplanos: o lado
     exterior da aresta e os lados dos dois raios, que não se cruzam e
     apontam para fora.
   - Um cone de ângulo até π é a interseção de dois semiplanos. Um cone maior
     é o complemento de um cone convexo.

   Em cada caso, a interseção com uma curva convexa fechada tem no máximo
   três componentes. Assim, ao percorrer ∂P_i uma vez pelo exterior, a curva
   atravessa no máximo 3c_{i−1} fronteiras de células. Atravessar uma
   fronteira muda o índice em ±1, ou ±2 num cone de largura zero.
3. *Trechos dentro de P_{i−1}.* Os pontos de entrada e de saída estão em
   ∂P_{i−1}. Dois polígonos convexos se cruzam na mesma ordem cíclica ao longo
   das duas fronteiras. Por isso, a soma das distâncias circulares entre as
   células nos pontos de saída e de reentrada é no máximo uma volta, c_{i−1}.
4. *Conclusão.* A distância entre as células de duas consultas varridas
   consecutivas é no máximo o número de fronteiras atravessadas entre elas,
   mais esses saltos. ∎

**Mapas mais antigos (h < i−1).** O Lema A reduz o problema às âncoras. A
célula de uma consulta varrida é a do seu ponto âncora, que fica em ∂P_g, g>h,
fora de P_h. Os segmentos da base em ∂P_h até as âncoras estão nos raios de
F_h, que não se cruzam (prova, Seção 1). Esse é o fato que a prova deles
invoca ("os últimos passos não se cruzam").

Falta passar dessa não interseção para uma ordem circular com poucas voltas.
As âncoras de um lote ficam em curvas convexas diferentes, P_g para vários
g, intercaladas na ordem do lote. O argumento da Proposição 3 vale para cada
curva separadamente, mas a intercalação não fica controlada. A prova do Lema
1 deles não trata disso. Por isso, para h < i−1 o Lema 1 fica como hipótese,
a mesma de que depende o Teorema 2 deles e, no caso disjunto, o Teorema 1.

## 4. Medições

Medições no Mac, em 18/11/2026, com cópias instrumentadas não guardadas.

**Instâncias testadas.**
- Testes direcionais: 3000 caixas inteiras e 200 convexos aleatórios.
- Sondas da família do artigo, da grade fina, da família "line" e de rosetas.
- Um teste de estresse de 1500 instâncias aleatórias por família:
  - aglomerados sobrepostos;
  - agulhas com razão 0,03 em orientações aleatórias;
  - polígonos aninhados;
  - cadeias;
  - leques de agulhas pelo mesmo centro.

  Cada instância tem k de 4 a 17 e m de 6 a 96, em aritmética exata.

**Resultados.**
- **Igualdade.** Todas as localizações foram iguais às da busca binária, em
  dezenas de milhões de consultas. Os comprimentos foram idênticos aos dos
  mapas memoizados.
- **Salvaguardas.** Em aritmética exata, nenhuma varredura falhou e nenhum
  orçamento se esgotou.
- **Pior lote.** O maior número de células testadas num lote, dividido por
  c_h+b_{i,h}, foi 3,23, nas agulhas. Nas outras famílias ficou abaixo de 2,6.
  O estresse somou 24,8 milhões de consultas. Isso é compatível com as "no
  máximo três cadeias" do Lema 1 deles.

Os tempos estão em
[intersecting-tpp-correction.md](intersecting-tpp-correction.md#complexity-and-performance).

## 5. O que se pode afirmar

- As duas implementações executam a construção de Tan e Jiang. As diferenças
  são as da Seção 1, e nenhuma muda o procedimento.
- Com interseções, os cones dos pseudo-vértices são os corrigidos, e as
  imagens refletidas têm uma pertinência que o artigo deles não descreve.
- O resultado é idêntico ao dos mapas memoizados (Teorema 1).
- O custo é O(kn) para polígonos disjuntos e O(k²n) quando podem se
  intersectar, sob o Lema 1 deles (Corolário).
  - O lema está provado aqui para o mapa anterior (Proposição 3).
  - Nas medições, ele valeu em todos os lotes, com constante até 3,23.
  - Se falhasse, as salvaguardas manteriam o custo dos mapas memoizados.
