# Correção dos mapas direcionais para polígonos intersectantes

Data: 4 de novembro de 2026.

Esta nota investiga a hipótese ainda explícita no teorema de terminação de
`papers/artigo1-tpp/main.tex`. A prova abaixo separa a construção matemática
(incluindo contatos simultâneos) da representação usada por
`packages/convex-tpp/cpp/src/solvers/intersecting_maps.cpp`.
O escopo são polígonos convexos fechados **de área positiva**, início e fim
fixos, ordem fixa, sem obstáculos e aritmética racional exata. Admitem-se
contenções, repetições, interseções, lados compartilhados e vértices
colineares redundantes. As rotinas especiais de entrada ponto/segmento,
`use_last_contact=true` e as variantes `unchecked_double` não são abrangidas
pela conclusão sobre a implementação. O solver ancorado do artigo usa os
polígonos de área positiva e a extração dos primeiros contatos.

**Situação:** prova revisada e integrada ao artigo no escopo acima. A Seção 8 fecha
a correspondência com a representação por uma indução conjunta em nível,
incluindo o invariante de sufixo terminal da Seção 7. Os testes da Seção 9
são uma verificação adicional de implementação; não são premissas da prova.

**Teorema.** Dadas coordenadas racionais, polígonos no escopo
acima e quaisquer extremos racionais s,t (inclusive s=t), o mapa direcional
exato com limites incidentes e extração dos primeiros contatos termina e
resolve o TPP de ordem fixa. Consequentemente, para esses mesmos polígonos,
o teorema de terminação do ciclo dispensa a hipótese adicional de
correção dos TPPs ancorados. As demais hipóteses desse teorema, como ausência
de um ponto comum e execução ininterrupta sem cutoff, continuam necessárias
para o caso que ele trata. As cotas são de operações aritméticas, sem uma
nova alegação de complexidade em bits.

## 1. Funções valor e uma identidade de subgradientes

Defina, para todo ponto q do plano,

\[
 F_0(q)=\|q-s\|,\qquad
 F_i(q)=\min_{x\in P_i}\{F_{i-1}(x)+\|q-x\|\}.
\]

As funções são finitas, convexas e 1-Lipschitz. A convexidade segue da
convexidade conjunta e da projeção do epígrafo; a desigualdade de Lipschitz
segue da desigualdade triangular, usando um mesmo contato x. O mínimo existe
porque P_i é compacto. Se q pertence a P_i, então F_i(q)=F_{i-1}(q): uma
inequação usa x=q e a outra usa Lipschitz.

Escreva D para o disco unitário fechado e N_P(x) para o **cone normal
exterior**, isto é, os vetores n tais que n·(z-x)≤0 para todo z em P. Para
x em P_i, vale a identidade

\[
 \boxed{\quad \partial F_i(x)
  =\bigl(\partial F_{i-1}(x)+N_{P_i}(x)\bigr)\cap D.\quad}       \tag{1}
\]

De fato, um subgradiente de F_i em x tem norma ≤1 e suporta F_{i-1}
restrita a P_i. A regra de soma aplica-se a F_{i-1}+δ_{P_i}, pois a primeira
função é finita e contínua. Reciprocamente, se w pertence ao lado direito,
para z em P_i e qualquer q temos

\[
 F_{i-1}(z)+\|q-z\|
 \ge F_{i-1}(x)+w\cdot(z-x)+w\cdot(q-z).
\]

Tomar o mínimo em z prova que w suporta F_i em x. A fórmula inclui todos os
contatos de comprimento zero; não pressupõe que somente duas fronteiras
passem por x.

**Lema do raio.** Se F é convexa e 1-Lipschitz, u pertence a ∂F(x) e
\|u\|=1, então, para t≥0,

\[
 F(x+tu)=F(x)+t.
\]

Para t>0, ∂F(x+tu)={u}. A igualdade resulta do suporte e de Lipschitz.
Se v é outro subgradiente em x+tu, a monotonicidade dos subgradientes dá
(v-u)·tu≥0. Como \|v\|≤1, segue v·u≥1, logo v=u.
A recíproca usada no invariante de sufixo também vale: se q≠x e
F(q)=F(x)+\|q-x\|, qualquer v∈∂F(q) satisfaz
v·(q-x)≥\|q-x\|. A norma ≤1 força v=(q-x)/\|q-x\|; seu suporte tem
igualdade em x e portanto também suporta F em x. Assim a igualdade terminal
já dá a hipótese de subgradiente unitário na base.

Em particular, dois raios unitários válidos não se cruzam fora de suas bases,
salvo se forem colineares e tiverem o mesmo sentido.

## 2. Unicidade da geometria orientada

Os contatos ótimos podem deslizar sobre trechos retos, mas duas soluções
ótimas com os mesmos extremos têm a mesma curva poligonal orientada, após a
remoção de contatos retos redundantes.

Para provar isso, tomem-se duas sequências ótimas A e B e sua média M.
A convexidade torna M ótima. A igualdade na desigualdade triangular implica
que, em cada ligação não nula de M, as ligações correspondentes de A e B
são múltiplos não negativos do mesmo vetor unitário u_j. Uma ligação nula
de M é nula nas duas sequências. Escolham-se os vetores KKT de M, com
norma ≤1 nas ligações nulas. Cada forma linear de suporte no contato M_i
atinge o mesmo mínimo em A_i e B_i; portanto, escrevendo Δ_i=A_i-B_i,

\[
 (u_{i-1}-u_i)\cdot\Delta_i=0.                         \tag{2}
\]

Enquanto os vetores KKT são um mesmo vetor unitário u, Δ_i é um múltiplo γu,
pois Δ_0=0 e as diferenças das ligações são paralelas a u. Na primeira
mudança para v≠u, (2) dá γ(1-u·v)=0. Como \|v\|≤1, o fator é positivo e
Δ_i=0. Ligações nulas em ambas as soluções não alteram Δ. Repetindo o
argumento, os extremos de cada trecho de direção constante coincidem. Os
contatos intermediários apenas subdividem esse mesmo trecho. O extremo final
fixo trata o último trecho. A existência dos vetores KKT usa a regra de soma
para a função comprimento, finita, e o produto dos polígonos.

Esta é uma unicidade geométrica; não é unicidade da lista de contatos.
Ela permite deduzir que qualquer solução ótima de prefixo tem todas as visitas
necessárias antes de uma base x quando existe uma solução ótima que já as
completou em x e depois segue reta até q.

## 3. Eventos completos e tipo constante em uma pseudo-aresta

Divida cada lado de P_i em seus vértices originais, em s se incidente, e nas
interseções com todas as fronteiras anteriores. Nas sobreposições colineares,
inclua os dois extremos da sobreposição. Remova repetições. A pertinência de
um ponto de uma pseudo-aresta aberta E a cada P_h, h<i, é constante.
Isso inclui a pertinência **fechada** quando E é compartilhada.

Remova o sufixo máximo de polígonos anteriores que contêm E e denote por r
seu predecessor (r=0 se nenhum restar). Em toda E,

\[
 F_{i-1}|_E=F_r|_E.
\]

Se r>0, todo q de E está fora de P_r. Fixar os contatos de uma solução
ótima até o último contato a em P_r fornece a majorante suave
F_{r-1}(a)+\|q-a\|, com igualdade em q. Uma função convexa finita que tem
uma majorante diferenciável com igualdade é diferenciável nesse ponto.
Logo F_r tem um único gradiente u(q), unitário. Para r=0, usa-se
u(q)=(q-s)/\|q-s\|; o corte em s exclui a singularidade. O gradiente é
contínuo em E, e u(q) pertence a ∂F_{i-1}(q), pois F_{i-1}≥F_r e há
igualdade em E.

Se u é tangente a E em um ponto interior q, o último contato a (ou s)
está na reta de E. Como E não encontra P_r nem s, toda E está no mesmo raio
que sai de a e passa por q. Pelo lema do raio, F_r é afim com esse mesmo
gradiente em toda E. Assim há exatamente três possibilidades: u é sempre
tangente, sempre aponta para o interior de P_i, ou sempre aponta para seu
exterior. Os raios exteriores são respectivamente u, R_Eu e u, onde R_E
é a reflexão na reta de E. Todo subgradiente de F_{i-1} em E tem a mesma
componente tangencial de u, pois a restrição é diferenciável nos dois
sentidos tangenciais. Somar o cone normal exterior e intersectar o disco
em (1) deixa exatamente uma raiz unitária exterior. No caso tangente,
passagem e reflexão coincidem; o código usa passagem.

### Limites nos extremos e a ausência de uma exceção tangencial isolada

É necessário ainda mostrar que uma aresta não tangente não tem um raio
limite tangente em um extremo. A continuidade no intervalo aberto, sozinha,
não prova isso.

Há finitos tipos combinatórios de soluções ótimas: agrupe contatos coincidentes
em blocos máximos; cada bloco pertence à interseção dos polígonos que visita.
Em dimensão dois pode passar reto; em uma face de dimensão um passa reto ou
reflete na reta da face; em dimensão zero ou em um vértice fica preso em um
ponto fixo. Os pontos presos e as retas são racionais. Essas alternativas
seguem da estacionariedade da variação conjunta do bloco. Os dois segmentos
incidentes de um bloco interno são não nulos, por maximalidade. No último
bloco, q fora de P_r garante também um segmento terminal não nulo.

A viabilidade e o KKT completo de um tipo, com vetores no disco nas ligações
nulas, são condições semialgébricas após introduzir comprimentos e direções.
Sua projeção no parâmetro de E é semialgébrica. Portanto, numa vizinhança
unilateral suficientemente pequena de um extremo, um tipo fixo está
disponível e

\[
 F_r(q)=c+\|q-V\|                                      \tag{3}
\]

para uma fonte virtual fixa V: desdobre as reflexões após o último ponto
preso, ou a partir de s se não houver ponto preso. A soma das ligações
colineares desdobradas é sua distância, com o mesmo sentido de percurso.
Se o gradiente limite em (3) fosse tangente a E, V estaria na reta de E;
se V fosse o próprio extremo, a mesma conclusão valeria. Então todo esse
subintervalo seria tangente, e o argumento anterior tornaria toda E tangente.
Consequentemente os dois testes `reflects` dos extremos concordam, e são
ambos falsos no caso inteiramente tangente.

**Correspondência com o código.** Na consulta primária v+εd para dentro de E,
`inside` pula exatamente o sufixo acima, antes de qualquer desempate
secundário. A consulta restante tem gradiente unitário único em E. Logo os
limites calculados por `virtual_source` dão u de cada lado, não uma escolha
normal arbitrária na fronteira compartilhada. Se V=v, o limite de (q-V) é d.
As reflexões devem transportar as quatro componentes da consulta.

## 4. O cone inteiro de um vértice, com quaisquer contatos simultâneos

Sejam v um vértice dividido, C=∂F_{i-1}(v)⊂D, e W o cone tangente a P_i em v.
Ele é gerado pelas direções d_- e d_+ para os vizinhos anterior e posterior.
Seu ângulo interno θ satisfaz 0<θ≤π. Para uma direção unitária w fora do
interior de W, vale

\[
 w\in C+N_{P_i}(v)
 \quad\Longleftrightarrow\quad
 w\cdot d_-\le h_C(d_-),\qquad
 w\cdot d_+\le h_C(d_+),                               \tag{4}
\]

onde h_C é a função suporte.

A necessidade é imediata. Para a suficiência, suponha que w não pertença a
C+N. A separação fornece d em W com w·d>h_C(d). O conjunto
L={d:w·d>h_C(d)} é um cone aberto convexo. Como w é unitário e w não pertence
a C, a compacidade de C⊂D implica w·w>h_C(w): w também está em L. Se w
estiver na fronteira de W, uma das duas desigualdades de (4), aplicada à
própria direção w, já contradiz esse fato. Caso contrário, o segmento em L
de w até d atravessa uma das duas semirretas de fronteira de W. Ele não
atravessa somente a origem, que não está em L. Isso contradiz (4).

A derivada direcional de F_{i-1} em d_± é h_C(d_±). Por convexidade, ela é o
limite da derivada da restrição à pseudo-aresta incidente. Assim cada raio
armazenado tem a projeção tangencial exata imposta por (4), e é a solução
unitária situada no lado exterior da respectiva aresta.

Para verificar qual arco esses dois raios delimitam, normalize d_±. Meça
ângulos sobre o arco exterior de d_- até d_+, no sentido anti-horário, de
comprimento L_0=2π-θ. Escreva a=acos h_C(d_-), b=acos h_C(d_+).
A função d↦acos h_C(d) é 1-Lipschitz na distância angular: para z em D,
acos(z·d) o é, e tomar o mínimo em z preserva essa propriedade. Logo
|a-b|≤θ. As duas desigualdades em (4), restritas ao arco exterior, dão
exatamente o intervalo [a,L_0-b]. Ele é não vazio: tome z em C e some um
múltiplo não negativo da direção oposta à bissetriz interna até a última
interseção com o círculo unitário. O resultado está fora de W e diminui
ambos os produtos com d_±. Esta construção também vale quando θ=π.

Portanto o conjunto admissível é exatamente o arco anti-horário entre
`before_ray` e `after_ray`, inclusive quando ele é maior que um semicírculo
ou se reduz a um raio. É o predicado `in_cone`. Esta prova cobre mais do que
o lema local do artigo sobre uma única interseção de duas fronteiras.

## 5. Cobertura e ordem circular das células exteriores

Para q fora de P_i, escolha um contato ótimo x em P_i. Se o segmento xq
ainda encontra P_i após x, substitua x pelo último ponto y desse encontro.
Lipschitz e a igualdade ótima mostram que y também é um contato ótimo.
A direção unitária w de yq pertence a ∂F_i(y), por (1), e o raio deixa P_i.
Se y está em uma pseudo-aresta aberta, a Seção 3 fornece seu único raio
exterior; se y é um vértice dividido, (4) fornece seu cone. Logo as células
cobrem todo o exterior.

Todos esses raios são globalmente válidos pelo lema do raio. Eles não se
cruzam no exterior, a menos que sejam colineares no mesmo sentido. Duas
bases distintas nesse último caso só podem estar em um mesmo trecho plano
da fronteira: caso contrário, o primeiro raio entraria no interior do
polígono. Esses são os trechos tangentes de área nula.

Para ver a ordem circular, interseccione os raios com um círculo grande que
contém o polígono. Cada raio o encontra exatamente uma vez. Arcos disjuntos
no anel entre o polígono e o círculo preservam a ordem de suas bases; os
cones preenchem os intervalos entre os limites incidentes. A cobertura exclui
lacunas no círculo. Ao contrair os intervalos tangentes de área nula, obtém-se
uma subdivisão circular, sem precisar armazenar ramos interiores herdados.
Dentro de P_i vale F_i=F_{i-1}, portanto a recursão representa esses ramos.

## 6. O predicado da corda e a busca binária

Considere uma cadeia anti-horária da fronteira entre a e b, seu raio inicial
r_A (o raio `after` de a) e seu raio final r_B (o raio `before` de b).
Excluam-se primeiro os cones dos próprios extremos. Para uma consulta fora
do polígono, pertencer às células da cadeia equivale a estar do lado direito
da curva orientada que vem do infinito por r_A, percorre a cadeia de a a b
e volta ao infinito por r_B. Trocar a cadeia pela corda ab altera apenas a
região entre a cadeia e a corda, que está dentro do polígono convexo.
Os raios não atravessam essa região nem se cruzam entre si no exterior.

Normalize a=(0,0), b=(1,0). Ponha

- A: r_A×(q-a)≥0;
- B: r_B×(q-b)≤0;
- H: q_y<0.

Se os componentes verticais dos dois raios são não nulos, a interseção da
curva com uma horizontal fornece a tabela:

| r_A.y | r_B.y | q abaixo da corda | q acima da corda |
| --- | --- | --- | --- |
| <0 | <0 | A e B | falso |
| <0 | >0 | A | B |
| >0 | <0 | B | A |
| >0 | >0 | verdadeiro | A ou B |

Quando os dois raios estão no mesmo semiplano, suas interseções com cada
horizontal mantêm a ordem inicial x_A<x_B: inverter a ordem exigiria um
cruzamento dos raios. A primeira linha é o intervalo entre as interseções;
a última é o exterior desse intervalo. As linhas mistas têm uma única
interseção em cada semiplano. Essa dedução dá exatamente os quatro ramos de
`in_edge_plus`, com o teste adicional da corda na primeira e na última linha.

Um raio horizontal que aponta para fora da corda (r_A para -ab ou r_B para
+ab) satisfaz a mesma tabela por inspeção dos produtos vetoriais, com zero
classificado no lado não negativo como no código. Um raio que aponta **ao
longo da corda para o outro extremo** só é possível se toda a cadeia for
colinear na fronteira do polígono. Se r_A aponta para +ab, a derivada
convexa da restrição já é +1 no início e permanece +1 em toda a cadeia;
se r_B aponta para -ab, o argumento simétrico usa -1 no fim. Todos os seus
raios são tangentes, sem célula exterior de área positiva. É precisamente a
rejeição antecipada por `same_direction`. Consultas colineares devem ser
atribuídas à aderência de uma célula vizinha, conforme a Seção 8.

Assim, para consultas genéricas, o predicado separa exatamente o intervalo
de índices [a,b). Testar primeiro o cone de b e depois escolher uma das duas
cadeias conserva o invariante da busca binária em `locate`. A última
verificação `in_edge_plus` é satisfeita. Isso é uma prova do predicado para
a subdivisão produzida nas Seções 3–5, e não para pares arbitrários de raios.

## 7. Ações recursivas e uma obrigação mais forte para a redobragem

As identidades de valor nas três células são as seguintes.

- No cone de v, F_i(q)=F_{i-1}(v)+\|q-v\|: resolva o prefixo em v e anexe vq.
- Em uma passagem com direção u, o lema do raio dá
  F_{i-1}(q)=F_{i-1}(x)+\|q-x\|=F_i(q). Pela unicidade geométrica, qualquer
  solução ótima anterior a q tem as visitas anteriores completas até x e
  pode visitar P_i em x, sem mudar a geometria.
- Em uma reflexão numa reta e, ponha q'=R_e(q). Com contato x,
  F_{i-1}(q')=F_{i-1}(x)+\|q'-x\|=F_i(q). A solução anterior tem a geometria
  de uma solução ótima até x seguida de xq'. Redobre a cauda em x.

A última frase **não basta** para justificar o código: ele intersecta somente
o último segmento armazenado. A unicidade da curva, sozinha, não exclui um
contato reto redundante armazenado depois de x.

Uma indução apropriada precisa manter o invariante de **sufixo terminal**:

> Se F_i(q)=F_i(x)+\|q-x\|, com x≠q, o último segmento da lista retornada
> por `query_path(q,i)` contém x (ou começa antes de x nessa mesma reta).

Além da correção do valor, o caso base satisfaz o invariante. Na passagem,
F_{i-1}≤F_i e Lipschitz forçam igualdade também na base x, de modo que o
invariante do nível anterior se aplica. No caso de vértice, se o vértice
armazenado v estivesse estritamente depois de x, o lema do raio tornaria
∂F_i(v) um único vetor unitário. Seu cone seria então um raio de área nula,
que não é selecionado por uma consulta com desempates independentes. Assim
v está em x ou antes dele.

Na reflexão, o prefixo satisfaz a igualdade de raio de x até q'; o invariante
anterior garante que o último segmento realmente contém o contato de
redobragem. O denominador é não nulo porque uma reflexão selecionada é não
tangente, inclusive nos extremos da pseudo-aresta (Seção 3). O parâmetro do
contato está no segmento e na pseudo-aresta. Para conservar o invariante no
nível atual, se uma base terminal z estivesse antes do contato x na reta
exterior, o lema do raio tornaria ∂F_i(x) unitário único. Entretanto esse
conjunto contém tanto a direção incidente u quanto a refletida R_eu, que
são distintas: ambas pertencem a ∂F_i(x) por (1). Contradição. Assim z está
na cauda xq. Essa é a razão para nenhum vértice reto redundante invalidar a
interseção com o último segmento.

A extração gulosa dos **primeiros** contatos sobre a curva ótima preserva a
ordem: por indução, o primeiro encontro disponível de P_i não é posterior
ao i-ésimo encontro de qualquer testemunha viável da mesma curva.
Substituir a subcurva entre contatos por suas cordas não aumenta o comprimento;
se o diminuísse estritamente, contrariaria a otimalidade da curva original.
Logo a lista extraída também é ótima e passa pelo KKT completo.

## 8. Indução conjunta para consultas fechadas e simbólicas

O código não executa simplesmente todas as operações sobre
q+εd+ε²e_1+ε³e_2: `inside` usa somente q+εd e pertinência fechada;
os demais testes usam também e_1 e e_2. A prova usa a consulta primária
q_ε=q+εd e desempates **estratificados**, sem identificar o programa com
uma perturbação global dos dados.

Mantenham-se conjuntamente, após cada nível i:

1. a construção matemática das células descrita nas Seções 3–6;
2. a correção da curva ordinária e o invariante de sufixo da Seção 7;
3. para cada germe primário não constante q_ε, a fonte V devolvida é fixa
   para ε>0 suficientemente pequeno, a expressão do valor é
   c+\|q_ε-V\|, e q_ε-V dá a direção do último trecho geométrico não nulo
   de uma solução ótima. Quando V=q no limite, essa direção limite é d.

Os itens 2–3 valem para qualquer escolha de desempates secundários
linearmente independentes, inclusive se a escolha muda entre chamadas.
Reflexões preservam essa independência. Esta quantificação é necessária:
`query_path` reinicia seus desempates ordinários em cada chamada, enquanto
`virtual_source` os transporta.

**Base.** No nível zero, a curva é sq e V=s. Para um germe não constante,
q_ε=s ocorre no máximo num parâmetro isolado, excluído ao reduzir a
vizinhança unilateral.

**Construção no nível i.** Todos os raios de i consultam fontes somente em
níveis menores. O item 3 anterior e a remoção do sufixo fechado da Seção 3
identificam suas direções com os gradientes únicos de F_r em cada
pseudo-aresta. Logo os lemas das Seções 3–6 se aplicam ao mapa atual sem
assumir a correção de consultas no próprio nível i. Isso prova o item 1.

**Seleção de uma célula fechada.** Fixados os raios, os predicados de
orientação são lineares na consulta. Há apenas um número finito de
predicados possíveis no mapa. Se a consulta primária está fora de P_i,
as decisões lexicográficas são as de uma consulta exterior genérica
arbitrariamente próxima dela. Os desempates independentes afastam a consulta
de toda reta de teste; cones de largura zero não são selecionados.
Pelo item 1 e pela Seção 6, o resultado é uma célula exterior cuja aderência
contém a consulta primária. Isso vale também para um germe, para todo ε
positivo suficientemente pequeno: os sinais dos coeficientes são estáveis.
Se a consulta primária pertence a P_i, o retorno de passagem é válido pela
igualdade fechada F_i=F_{i-1}, independentemente dos termos secundários.

**Validade da ação na aderência.** A ação em uma célula aberta é ótima pelas
identidades da Seção 7. Tome consultas genéricas dessa célula tendendo à
consulta primária. Os contatos pertencem a polígonos compactos; uma
subsequência converge, e a continuidade dos valores conserva a otimalidade.
No caso de reflexão, o contato limite x continua na pseudo-aresta fechada,
q' é a reflexão de q e
F_{i-1}(q')=F_{i-1}(x)+\|q'-x\|=F_i(q).
O contato não coincide com q' porque q está fora de P_i. A direção
incidente limite é não tangente mesmo num extremo (Seção 3). Portanto a
identidade de raio e o ângulo de reflexão continuam válidos. No caso de
vértice, a identidade com F_{i-1}(v)+\|q-v\| é direta; no de passagem,
F_i(q)=F_{i-1}(q).

Não se toma aqui o limite da **lista** devolvida pelo programa. As listas
podem mudar, e é por isso que apenas continuidade do valor não bastaria.

**Curva e sufixo no nível i.** Aplique agora o item 2 anterior às chamadas
ordinárias realmente efetuadas. Em particular, na reflexão, a identidade
de raio acima garante que a lista anterior tem seu último segmento contendo
x, mesmo que sua escolha de célula seja diferente da escolha nas consultas
aproximantes. A indução de sufixo da Seção 7 prova simultaneamente que a
redobragem é executável e que o novo sufixo satisfaz o mesmo invariante.
Na passagem, a unicidade geométrica assegura as visitas até a base; no
vértice, o prefixo ótimo é simplesmente anexado. Isso prova o item 2 atual.
A prova não pressupõe a continuidade combinatória da lista armazenada.

**Fonte no nível i.** Em um germe não constante, reduza a vizinhança para
que todas as decisões sejam fixas. Na passagem conserva-se V; num vértice
usa-se v; numa reflexão usa-se R_e V. As identidades de valor e a isometria
dão c+\|q_ε-V\|. O item 3 anterior dá a direção terminal do prefixo.
Nas reflexões, a identidade de raio e a unicidade geométrica fazem esse
trecho passar pelo contato, logo refletir a direção terminal dá a direção
terminal correta da nova curva. No vértice, é q_ε-v, não nula numa
vizinhança unilateral reduzida. Passagens conservam a geometria.
Portanto o vetor para a fonte devolvida tem sempre o sentido de percurso
correto; não se deduz isso apenas de uma igualdade de valores sobre uma
reta. Se V=q, q_ε-V=εd, justificando o fallback exato `incoming=side`.
Isso prova o item 3 atual e encerra a indução.

A construção preguiçosa é equivalente à construção completa: cada par de
raios é imutável após calculado, e suas dependências têm nível estritamente
menor. O cache guarda a consulta completa, inclusive todos os desempates.

**Conclusão.** Para entradas do escopo inicial, a construção racional de
`solve_intersecting_map_contacts_exact` termina, não dispara as exceções de
localização/tipo/redobragem e devolve contatos ótimos na ordem. As
identidades e o invariante acima justificam a retirada da hipótese sobre o
solver ancorado no teorema do artigo, sem alterar o algoritmo nem seus limites
de complexidade aritmética. A conclusão é restrita à implementação exata e
não atribui correção às versões de Tan–Jiang nem ao caminho em double.

## 9. Validação dirigida da investigação

Experimento local ignorado:
`benchmarks/workspace/experiments/intersecting-proof-2026-11-04/`.
O harness inclui uma cópia instrumentalizada do mesmo C++ exato, tornando
acessíveis seus mapas para inspeção; não é um solver de produção alternativo.
Semente 20261104; 200 sequências de 2–6 caixas inteiras com cisalhamentos,
repetições e início por vezes num vértice. Consultas em pontos aleatórios,
vértices, raios e prolongamentos para trás, incluindo fatores racionais 1/3.

- 10.560 consultas: todos os contatos extraídos passaram pelo certificado
  KKT racional independente, sem tolerância de otimalidade. Acrescentaram-se
  os pontos fixos s e q como regiões singleton no certificado cíclico; a
  aresta de fechamento tem comprimento constante e não afeta a otimalidade.
- 100.904 testes do predicado de corda contra a união obtida por varredura
  das células intermediárias: nenhuma divergência. Esse teste verifica a
  coerência do localizador com as células calculadas; não é uma prova
  independente de que os raios sejam corretos.
- Suíte mantida `tpp-convex-directional-tests --random-boxes 300
  --random-convex 50`: 425.984 verificações, zero falhas ou casos não
  resolvidos. Os contadores de fallback são internos à suíte e não constituem
  falhas do mapa racional.

O núcleo C++ e seus contratos públicos permanecem como estavam antes desta
investigação. O artigo e o PDF foram atualizados após a revisão abaixo. Os testes são evidência de correspondência com a implementação, não substituem
a indução da Seção 8. Não se executaram as suítes de dashboard/WASM nem o
sanity check completo, pois esta alteração é apenas documental e não muda o
programa.


## 10. Revisão rigorosa e integração

Na revisão do mesmo dia, o argumento foi conferido novamente contra
`build_vertex`, `locate`, `virtual_source`, `query_path`, a extração dos
primeiros contatos e o `evaluate` ancorado de `cycle.cpp`. Este último mantém
as regiões originais de área positiva; a região singleton do anchor aparece
somente no certificado, portanto não exige provar o ramo de entrada ponto do
mapa. A remoção do prefixo/sufixo contendo o anchor é compatível com a prova.

Foram explicitados a recíproca do lema do raio, a quantificação sobre
quaisquer desempates secundários independentes (inclusive reiniciados entre
chamadas), o fato de a corda tangente rejeitada seguir a cadeia incidente e a
necessidade de não tomar limites das listas armazenadas. Nenhuma lacuna
remanescente foi identificada no escopo declarado. Isso não estende o
resultado aos ramos explicitamente excluídos no início desta nota.

O Apêndice A de `papers/artigo1-tpp/proof-intersecting.tex` contém a prova
completa em inglês; o novo Teorema 5.1 de `main.tex` é usado no Teorema 6.1,
que dispensa a hipótese adicional sobre o solver ancorado. As cotas de
complexidade aritmética e os resultados experimentais não foram alterados.

A execução dirigida ampliada, na mesma campanha ignorada e semente, passou
pelas 10.560 consultas com KKT exato e 100.904 comparações de corda anteriores,
além de **21.180 verificações de sufixo terminal** em raios de vértices e de
interiores de pseudo-arestas, em todos os níveis, e **7.060 comparações de
direções simbólicas** com consultas ordinárias em ε=2^-40. Estas últimas são
um teste de estabilização nessa escala para a família testada, não um limite
universal para infinitésimos. Nenhuma falha ocorreu. Também passaram as
suítes mantidas `tpp-convex-cycle-tests` e
`tpp-convex-cycle-certificate-tests`.
