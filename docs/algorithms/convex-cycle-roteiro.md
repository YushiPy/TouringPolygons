# Roteiro de estudo: o oráculo do ciclo convexo de ordem fixa

Mapa para entender e explicar o método (T26). Cada item diz o enunciado, a ideia da
prova, onde ela está escrita, onde está no código e onde é testada. As provas
completas estão em [`convex-cycle.md`](convex-cycle.md) e
[`convex-cycle-certificate.md`](convex-cycle-certificate.md); este arquivo não as
repete, só as ordena.

O problema: dadas regiões convexas fechadas `P_1, ..., P_k` numa ordem cíclica fixa,
achar `q_i ∈ P_i` que minimizem `L(q) = Σ_i |q_(i+1) − q_i|`, com `q_(k+1) = q_1`.
É um programa convexo (soma de normas sobre um produto de convexos), e o ótimo
existe porque as regiões são compactas.

## A estrutura em uma frase

**O oráculo propõe uma solução com construções geométricas e prova limites
`L ≤ OPT ≤ U` com um certificado que não depende de como a solução foi obtida.** As
âncoras na borda são o caminho exato, que garante terminação com gap zero.

Isso importa para o artigo. Nas rodadas congeladas da armijo, 97,4% das chamadas
fecham pela prova do candidato da construção e 2,6% pelo polimento. O caminho
exato, onde estão as âncoras, não rodou em nenhuma das cerca de 179 milhões de
chamadas que fecham as 558 instâncias; rodou uma única vez, numa execução de
300 s do caso 189
([números](../../benchmarks/results-saved/README.md#fekete-reference-armijo-2026-10-17)).

## Parte I — O certificado: por que os limites estão certos

Esta parte sustenta sozinha a correção de todo `L` e `U` devolvido.

1. **Condições de otimalidade (KKT), necessárias e suficientes.**
   - *Enunciado:* seja `d_i = q_(i+1) − q_i`, com `u_i = d_i/|d_i|` se
     `d_i ≠ 0` e qualquer `|u_i| ≤ 1` se `d_i = 0`. Então `q` é ótimo se, e
     só se, existem esses `u_i` com `(u_(i−1) − u_i)·(v − q_i) ≥ 0` para todo
     vértice `v` de `P_i`.
   - *Ideia:* somar as desigualdades de subgradiente das normas dá
     `L(z) ≥ L(q)` (suficiência). A condição de subgradiente com cone normal num
     problema convexo dá a necessidade.
   - *Escrito:* [certificado §1](convex-cycle-certificate.md#necessary-and-sufficient-support-conditions).
   - *Código:* [`tpp_convex_verify_cycle_certificate`](../../packages/convex-tpp/cpp/include/tpp/convex/cycle_certificate.h#L71)
     em [`cycle_certificate.cpp`](../../packages/convex-tpp/cpp/src/solvers/cycle_certificate.cpp).
     Os sinais são decididos sem raiz quadrada: `sign(a/√A − b/√B)` compara
     `a²B` com `b²A`.
   - *Teste:* `main-cycle_certificate_tests.cpp` (1.040 casos de dois raios).
2. **Blocos de contatos coincidentes (ligações nulas).**
   - *Enunciado:* num bloco de contatos iguais, a existência dos `u_i` com
     `|u_i| ≤ 1` se decide exatamente propagando conjuntos alcançáveis
     `R_j = D ∩ (R_(j−1) + N_j)`, onde `D` é o disco unitário e `N_j` o cone
     normal.
   - *Ideia:* três invariantes. Os pontos extremos ficam no círculo; cada passo
     só precisa de no máximo dois semiplanos novos; as direções continuam
     racionais.
   - *Escrito:* [certificado §2](convex-cycle-certificate.md#complete-zero-block-test-by-diskcone-reachability).
   - *Código:* [`zero_contact_certificate.h`](../../packages/convex-tpp/cpp/src/solvers/zero_contact_certificate.h)
     e [`cycle_zero_certificate.h`](../../packages/convex-tpp/cpp/src/solvers/cycle_zero_certificate.h).
3. **Limite inferior por dualidade fraca.**
   - *Enunciado:* para quaisquer `u_i` com `|u_i| ≤ 1`, vale
     `OPT ≥ D(u) = Σ_i min_(v ∈ P_i) (v − r)·(u_(i−1) − u_i)`.
   - *Ideia:* `|d_i| ≥ u_i·d_i` e reagrupamento cíclico.
   - *Escrito:* [certificado, "Bounds and status"](convex-cycle-certificate.md#bounds-and-status).
   - *Código:* `dual_lower` em [`cycle_certificate.cpp:113`](../../packages/convex-tpp/cpp/src/solvers/cycle_certificate.cpp#L113),
     racional, e a versão binary64 em [`binary_dual.cpp`](../../packages/convex-tpp/cpp/src/solvers/binary_dual.cpp)
     (`support_bounds`).
4. **Teorema dos limites em binary64.**
   - *Enunciado:* se o estágio binary64 devolve `L` e `U`, então
     `L ≤ OPT ≤ U`, qualquer que seja o candidato proposto.
   - *Ideia:*
     - a pertinência de cada contato é provada com sinais exatos;
     - `U` é a soma intervalar dos comprimentos;
     - `L` é `D(u)` com vetores provados dentro do disco e mínimos de suporte
       alargados por um limite a priori do erro de arredondamento.
   - *Escrito:* [convex-cycle.md, "Correctness of the bounds"](convex-cycle.md#correctness-of-the-bounds).
   - *Código:* [`tpp_convex_solve_cycle_float_certified`](../../packages/convex-tpp/cpp/src/solvers/cycle_float_oracle.cpp#L88).

**O que explicar:** com 1 a 4, um `L` ou `U` errado é impossível, mesmo que a
construção ou o polimento falhem. Uma falha só leva a chamada ao caminho exato
ou deixa o gap aberto.

## Parte II — As propostas: de onde vêm os candidatos

Esta parte não tem peso na correção; ela decide a eficiência.

5. **Construção por contatos ativos.** Começa nos centroides e faz atualizações
   por coordenada: cada contato minimiza as duas ligações vizinhas dentro da sua
   região. Daí sai um tipo combinatório (vértice, aresta ou passagem reta). Com
   ele, desdobram-se as reflexões, e a reta resultante dá os contatos em forma
   fechada.
   - São no máximo `k + 1` varreduras: o trabalho é finito por construção. Por
     isso a T19 (terminação do refinamento) não é necessária para a correção.
     O que se afirma é empírico: ela fecha 97,4% das chamadas.
   - *Escrito:* [convex-cycle.md, "Fast construction by active contacts"](convex-cycle.md#fast-construction-by-active-contacts).
   - *Código:* [`CycleRefinement`](../../packages/convex-tpp/cpp/src/solvers/cycle_refinement.h#L11),
     compartilhado entre racional e binary64.
6. **Polimento por pontos interiores.** Newton com barreira logarítmica no
   comprimento suavizado.
   - *Proposição:* num ponto estacionário com parâmetro `μ`, as direções
     suavizadas dão `D(u) ≥ comprimento − σμ(k+F)`.
   - Ela só justifica a escolha de `μ_end`. Os limites devolvidos passam de
     novo pelo item 4.
   - *Escrito:* [convex-cycle.md, "Why the polish produces good lower bounds"](convex-cycle.md#why-the-polish-produces-good-lower-bounds).
   - *Código:* [`cycle_float_oracle.cpp:159`](../../packages/convex-tpp/cpp/src/solvers/cycle_float_oracle.cpp#L159)
     e `chain_interior_point` em [`float_chain.h`](../../packages/convex-tpp/cpp/src/solvers/float_chain.h).
     Os parâmetros (T30) estão nos passos de [convex-cycle.md](convex-cycle.md#steps).

## Parte III — O caminho exato: terminação com gap zero

É o método apresentado em 16/10 (âncoras na borda).

7. **Redução à borda.**
   - *Enunciado:* se as regiões não têm ponto comum, algum ciclo ótimo positivo
     tem um contato na borda de alguma região.
   - *Ideia:* se todos os contatos fossem interiores, a estacionariedade (agrupando
     contatos coincidentes) forçaria todas as ligações não nulas a ter a mesma
     direção, e um ciclo positivo não fecha.
   - Com interseções, é preciso examinar a borda de **todas** as regiões. Uma
     região que contém outra pode não ter contato ótimo na borda. Se as regiões
     são disjuntas, basta a menor.
   - *Escrito:* [convex-cycle.md, "General boundary reduction"](convex-cycle.md#general-boundary-reduction-and-correctness).
   - *Código:* [`common_region`](../../packages/convex-tpp/cpp/src/solvers/cycle.cpp#L396)
     (ciclo zero) e [`search_intersecting_boundaries`](../../packages/convex-tpp/cpp/src/solvers/cycle.cpp#L405).
8. **A função de ancoragem `F`.**
   - *Enunciado:* `F(x)` é o TPP com `s = t = x` pelas outras regiões. Ela é
     convexa e 2-Lipschitz no plano, portanto convexa em cada segmento de aresta.
   - *Ideia:* minimizar parcialmente uma função conjuntamente convexa preserva
     convexidade.
   - **Não é convexa ao longo da borda inteira.** A parametrização é periódica, e
     há mínimos locais não globais; o próprio caso da reunião tem um. Por isso
     cada aresta é cortada nas interseções com as outras bordas, e cada pedaço é
     tratado separadamente.
   - **Diferente do TPP, o ótimo pode não ser único.** No TPP de ordem fixa por
     convexos o caminho mais curto é único (Dror et al.), mas um ciclo não tem
     ponto fixo e pode haver uma família contínua de ciclos ótimos, todos com as
     mesmas direções de ligação. Então `F` é constante num pedaço da borda; um
     mínimo plano contém um minimizador racional, e a terminação vale igual.
   - *Código:* os cortes em [`cycle.cpp:432`](../../packages/convex-tpp/cpp/src/solvers/cycle.cpp#L432).
   - *Teste e figura:* `rotated_squares()` e [`funcao-ancoragem`](../reports/funcao-ancoragem/README.md).
9. **Derivada e busca no interior da aresta.**
   - *Enunciado:* no interior de um pedaço, `f'(t) = (u_in − u_out)·e`, e nas
     pontas a mesma expressão é um subgradiente.
   - *Ideia:* as direções das ligações não nulas incidentes são únicas entre os
     ótimos ancorados, pela convexidade estrita da norma na direção.
   - Se nenhuma ponta resolve o problema restrito ao pedaço, os sinais nas pontas
     são estritos, e há uma raiz interior: a bissecção a acha.
   - *Código:* `derivative` em [`cycle.cpp:424`](../../packages/convex-tpp/cpp/src/solvers/cycle.cpp#L424)
     e o teste das pontas `restricted` em [`cycle.cpp:455`](../../packages/convex-tpp/cpp/src/solvers/cycle.cpp#L455).
10. **Terminação exata.**
    - *Enunciado:* há finitos tipos combinatórios de caminho ancorado. Em cada um,
      o comprimento é uma soma de duas distâncias a imagens fixas, ou
      `|(A − I)x + c|`. Logo o ponto estacionário é racional, de altura `Q`.
    - *Ideia:* a bissecção seguida do racional mais simples no intervalo (frações
      contínuas) o acha em `O(1 + log Q)` sondagens.
    - *Escrito:* [convex-cycle.md, "Exact termination"](convex-cycle.md#exact-termination).
    - *Código:* [`simplest_rational`](../../packages/convex-tpp/cpp/src/solvers/cycle.cpp#L36),
      usado em [`cycle.cpp:474`](../../packages/convex-tpp/cpp/src/solvers/cycle.cpp#L474).
    - **Hipótese:** o TPP de origem fixa precisa estar certo. Para regiões
      disjuntas, é Dror et al. Para regiões que se intersectam, usamos os mapas
      corrigidos de Tan e Jiang, implementados e testados, mas **não provados**
      ([intersecting-tpp-correction.md](intersecting-tpp-correction.md)).

## Parte IV — No B&B do TSPN

11. **Cada nó tem um limite inferior válido.** O nó guarda uma sequência cíclica
    parcial. O ciclo ótimo por ela é um limite inferior de todo completamento,
    porque tirar visitas não alonga o ciclo. O oráculo devolve `L`, e o nó é podado
    quando `L` atinge o incumbente (`CertifiedBound`).
    - *Escrito:* [tspn.md, "Cyclic relaxation and branching"](tspn.md#cyclic-relaxation-and-branching).
    - *Código:* a chamada em [`unordered.cpp:779`](../../packages/nonconvex-tpp/cpp/src/solvers/unordered.cpp#L779).
12. **"Resolvido" quer dizer gap fechado com limites provados**
    (`UB ≤ 1,001·LB`). Não é ótimo algébrico; esse exige o caminho exato com gap
    zero.

## O que dá para afirmar no artigo, e com que hipóteses

- **Teorema A (correção dos limites):** os itens 1 a 4. Não depende das
  âncoras, da construção, do polimento nem dos mapas com interseção. Está escrito
  e implementado.
- **Teorema B (terminação com gap zero):** os itens 7 a 10. Depende do contrato
  dos mapas de origem fixa, que com interseções não está provado. No enquadramento
  "propor e certificar", o Teorema A basta para a correção do TSPN. O Teorema B só
  é necessário para garantir gap zero.
- **Fatos empíricos, não teoremas:** a construção fecha 97,4% das chamadas, o
  polimento fecha o resto, e o caminho exato quase nunca roda.

## Para conseguir explicar sozinho

1. Por que o certificado do item 1 prova otimalidade sem confiar na construção.
2. Por que um `u` qualquer no disco dá um limite inferior (item 3), e por que
   arredondar exige o alargamento do item 4.
3. Por que algum contato ótimo está na borda (item 7) e por que, com interseções,
   uma região só não basta.
4. Por que `F` é convexa em cada aresta mas não na borda inteira (item 8, figura
   da T27).
5. Por que a bissecção termina com aritmética exata (item 10) e qual hipótese
   ainda está em aberto.

## Lacunas

- **Mapas de origem fixa com interseção:** a prova está em aberto. Ela afeta
  apenas o Teorema B.
- **Primal, dual e critérios de encerramento escritos para o artigo (T28):** o
  conteúdo está nos itens 1 a 4; falta a redação.
- **Pior caso do B&B (T04):** no máximo `Π m_i` folhas, mais o custo das
  chamadas.
