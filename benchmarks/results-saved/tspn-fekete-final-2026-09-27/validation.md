# TSPN: entrega e comparação final, 27/09/2026

O B&B mantido agora resolve TSPN por `tpp_nonconvex_tspn_solve` ou
`tpp-unordered --cycle`: ordem livre, ciclo fechado, nenhum ponto fixo,
polígonos simples possivelmente não convexos, com interseção ou contenção.
A fila, decomposição, escolha do próximo polígono, incumbente e contabilidade
dos limites são os mesmos do TPP de extremos fixos. Não foi criado outro B&B.
Buracos não são representados pela API.

## Algoritmo e garantia

A raiz fixa apenas a identidade da região 0. Cada região ausente é inserida
em todos os intervalos cíclicos pertinentes, incluindo o de fechamento;
a simetria de reversão elimina um dos dois triângulos equivalentes. Regiões
não convexas são refinadas pelas peças da decomposição existente. A relaxação
de cada nó usa o solver de ciclo convexo e seu certificado independente.
O novo limite dual de inserção é calculado racionalmente e arredondado para
baixo. A prova e o contrato estão em [tspn.md](../../../docs/algorithms/tspn.md).

O pior caso do B&B continua exponencial: até `(n-1)!/2` ordens não orientadas,
para n >= 3, multiplicadas pelas escolhas de peças. A especialização adicionada
nesta conclusão calcula ciclos de dois polígonos por projeções vértice/aresta,
em O(mn) operações aritméticas. A construção é compartilhada por racionais e
doubles, e continua submetida ao certificado. A complexidade geral do oráculo
está em [convex-cycle.md](../../../docs/algorithms/convex-cycle.md).

**Exatidão:** o oráculo racional é exato em suas entradas, com comprimento
representado como soma de raízes de racionais. A construção convexa não usa
epsilon de otimização. O B&B completo preserva a normalização, geometria
numérica, tolerância de visita e gap do B&B anterior. Portanto, `exact=true`
significa fechamento do gap solicitado; não é uma promessa de TSPN global
com erro racional zero. Os limites do solver externo também são numéricos.

## Protocolo reproduzível

```bash
python3 benchmarks/tpp.py tspn-benchmark \
  --output NEW_DIRECTORY --repetitions 3 --seconds 3
```

São 25 instâncias: 17 pequenas/sintéticas e as primeiras duas instâncias de
cada tamanho 5, 10, 15 e 20 do corpus alemão mantido. Este é um recorte, não
uma avaliação completa do corpus. Os quatro exemplos originalmente usados
para ciclos de ordem fixa são aqui resolvidos com ordem livre por ambos.

O comparador executa o B&B completo do fork Fekete, backend `socp`, revisão
`f4aa78c631545e4e894732a0fe8aef45f455c34c`, sem modificar o submódulo.
Medições nativas incluem preparação, heurística/raiz, busca e extração;
excluem início do processo e inicialização do ambiente Gurobi, aquecido numa
instância independente. Execução sequencial, uma thread por solver, ordem dos
backends alternada entre instâncias, três repetições. Tempos abaixo são medianas.

Gap comparado: `UB <= (1 + 1e-6) LB`; em nosso contrato isso corresponde a
gap absoluto zero e relativo `1e-6/(1+1e-6)`. Tolerância geométrica comum
`1e-8`; validação externa independente `1e-7`. No Gurobi foram usados
`FeasibilityTol=OptimalityTol=1e-9`, `BarConvTol=BarQCPConvTol=1e-10`;
`SPANNING_TOLERANCE=1e-9`. São ajustes explícitos de precisão do comparador,
não os defaults da publicação. Demais estratégias e parâmetros estão em
[config.json](config.json); o comando também oferece `--socp-defaults`.

## Corretude observada

- 75/75 rotas nossas e 75/75 externas passaram na validação independente.
  Ela usa predicados racionais de interseção/contenção e diagnósticos numéricos
  de distância/comprimento, sem comparar coordenadas de contato.
- Os intervalos reportados se sobrepõem nas 25 instâncias. Isso é uma
  verificação cruzada, não uma prova de exatidão global dos limites numéricos.
- Nosso gap fechou em 24/25 instâncias, nas três repetições. O intervalo
  reportado pelo externo fechou o mesmo critério em 18/25. Os sete restantes
  incluem um ciclo zero com resíduo de comprimento `3.39e-14`; não se deve
  interpretar todos como soluções ruins ou presumir certificado mais forte.
- A maior diferença de objetivo entre casos com nosso gap fechado foi
  `0.0001179601`, em `german_4_n10`, dentro do gap relativo solicitado.
  A diferença de `11.1936583` em `german_6_n15` pertence a uma execução nossa
  encerrada por tempo, não a uma solução declarada ótima.
- Maior afastamento de uma região: `3.31e-15` nas nossas rotas e `4.21e-10`
  nas externas. A cobertura estrita dos doubles foi confirmada em 57/75 e
  54/75 rotas, respectivamente; as demais passaram com a tolerância declarada.
  Comprimento recalculado e UB diferiram no máximo `5.69e-14`.

## Desempenho e limites encontrados

Fomos mais rápidos em **18/25** instâncias pelo tempo de retorno. No recorte
em que **ambos** os intervalos fecharam o gap, vencemos **13/17**, com média
geométrica da razão `tempo externo / tempo nosso` de **2.09**. No corpus
alemão vencemos apenas **3/8**; não há evidência de competitividade geral.

| Instância | Nosso tempo | Fekete/SOCP | Observação |
|---|---:|---:|---|
| seeded_boxes_8 | 2.739 ms | 19.227 ms | ambos fecharam gap |
| seeded_concave_12 | 7.747 ms | 35.396 ms | ambos fecharam gap |
| german_2_n5 | 24.015 ms | 0.212 ms | gap externo reportado 4.83e-6 |
| german_3_n10 | 19.026 s | 6.580 ms | gap externo reportado 7.94e-6 |
| german_5_n15 | 43.696 ms | 28.799 ms | ambos fecharam gap |
| german_6_n15 | 4.619 s | 31.351 ms | nosso gap ficou em 10.07% |
| german_7_n20 | 77.966 ms | 91.786 ms | gap externo reportado 1.71e-6 |

O limite de três segundos é cooperativo: uma chamada convexa já iniciada
pode ultrapassá-lo. `german_3_n10` fechou o gap depois de cerca de 19 segundos;
`german_6_n15` retornou `time_limit`. Tempos de retorno com gaps diferentes
não equivalem a tempos até uma mesma garantia de otimalidade.

O perfil de `german_3_n10` atribui 19.1279 dos 19.1291 segundos da primeira
repetição a apenas nove chamadas convexas: o gargalo não é uma árvore enorme.
Para ciclos, o campo de tempo geométrico inclui também a certificação e as
recuperações racionais; os subcampos zerados não medem sua ausência.

A especialização de dois polígonos reduziu `german_2_n5` de **215.438 ms**
na [campanha anterior](../tspn-fekete-2026-09-27/analysis.md) para **24.015 ms**,
aproximadamente **9 vezes**. Ela não resolve os casos lentos de três ou mais
regiões. Permanecem como trabalho de otimização a construção desses ciclos e
a interrupção interna do oráculo. A integração e a comparação estão entregues;
a meta de superar ou igualar Fekete em geral **não foi atingida**.

Entradas, dados brutos, configurações e hashes estão preservados nesta pasta.
A tabela completa está em [analysis.md](analysis.md); os testes finais e suas
limitações estão em [tests.md](tests.md).
