# Análise estratificada — German comparison

## Escopo e rótulos

A análise de runtime usa apenas as **550 instâncias concluídas por ambos**; as 8 sem resultado do Fekete foram excluídas, como na comparação anterior. As oito são OSM, todas com 60 polígonos. `speedup = runtime(Fekete) / runtime(nosso)`, então valores acima de 1 favorecem nosso solver.

Os 558 rótulos foram recuperados do ZIP simplificado do submódulo e associados ao `instances.bin` por uma impressão digital exata das coordenadas: sem depender da ordem dos polígonos, do vértice inicial ou da orientação. A verificação passou para 558/558 instâncias. `random` vem do metadado `source=random`; `tessellation` vem de `source=public_instance_set`; `OSM` vem da presença de `geo_information`.

A recuperação encontrou 320 OSM, 160 aleatórias e 78 tessellations, exatamente os totais descritos na Seção 4.2 do artigo (`docs/bibliography/TSPN-B&B-Michael/original.pdf`). O artigo diz que as tessellations são regiões de Voronoi derivadas de conjuntos CG:SHOP, TSPLIB e Salzburg; os casos OSM são pegadas de edifícios de 20 cidades. A tabela de runtime abaixo usa somente os pares comuns concluídos.

## Runtime por fonte

A média geométrica é a métrica principal para comparar razões multiplicativas. A média aritmética aparece como complemento e é mais afetada pelos grandes speedups. `Vitórias` conta instâncias em que nosso runtime foi menor.

| Fonte | Corpus | Casos pareados | Sem Fekete | Speedup geométrico | Mediana | Média aritmética | Vitórias do nosso |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OSM | 320 | 312 | 8 | 7.605× | 7.048× | 14.548× | 306/312 |
| random | 160 | 160 | 0 | 3.996× | 3.809× | 6.350× | 148/160 |
| tessellation | 78 | 78 | 0 | 1.114× | 0.994× | 2.096× | 38/78 |

A separação por fonte muda a leitura do efeito do tamanho. Em especial, a coorte tessellation/Voronoi passa a favorecer Fekete nas faixas maiores, enquanto OSM segue favorecendo nosso solver; a tendência agregada mistura esses perfis.

## Speedup geométrico por fonte e tamanho

`n` é o número de pares concluídos na célula. Células vazias não têm instâncias.

| Faixa de polígonos | OSM | Aleatórias | Voronoi/tessellation |
| --- | ---: | ---: | ---: |
| 4–10 | 16.198× (n=80) | 8.637× (n=40) | 6.797× (n=8) |
| 11–20 | 5.805× (n=80) | 4.441× (n=40) | 1.530× (n=16) |
| 21–30 | 5.805× (n=40) | 2.960× (n=20) | 1.172× (n=15) |
| 31–40 | 4.959× (n=40) | 2.247× (n=20) | 0.700× (n=14) |
| 41–50 | 5.741× (n=40) | 2.039× (n=20) | 0.732× (n=15) |
| 51–60 | 7.670× (n=32) | 3.256× (n=20) | 0.525× (n=10) |

Por fonte, OSM fica relativamente estável depois da primeira faixa; aleatórias caem até 41–50 polígonos e têm leve recuperação em 51–60; tessellation cai de 6.80× para abaixo de 1× nas faixas 31–40, 41–50 e 51–60. Portanto, a impressão de que o speedup cresce com o número de polígonos não é geral: ela depende fortemente da origem da instância. As oito falhas do Fekete em OSM com 60 polígonos não entram no speedup, então a última faixa OSM representa apenas os 32 casos concluídos.

## Relação entre sobreposição e speedup

A classificação usa a geometria simplificada efetivamente presente na campanha. `interior_disjoint` permite compartilhamento de arestas ou vértices, desde que não haja interseção de área positiva; `strictly_disjoint` também exclui esses contatos. As classes abaixo são mutuamente exclusivas.

| Relação geométrica | Composição | Pares | Speedup geométrico | Mediana | Vitórias do nosso |
| --- | --- | ---: | ---: | ---: | ---: |
| Estritamente disjuntas | OSM 156, random 21 | 177 | 10.488× | 9.560× | 177/177 |
| Sem sobreposição de área, com contato | OSM 147, tessellation 78 | 225 | 3.239× | 3.584× | 179/225 |
| Com sobreposição de área | OSM 9, random 139 | 148 | 3.433× | 3.334× | 136/148 |

Separando por fonte, o speedup geométrico é:

| Fonte | Sem sobreposição de área | Com sobreposição de área |
| --- | ---: | ---: |
| OSM | 7.701× (n=303) | 4.985× (n=9) |
| random | 12.793× (n=21) | 3.352× (n=139) |
| tessellation | 1.114× (n=78) | — |

No conjunto pareado, os casos sem sobreposição de área têm média geométrica maior que os casos com sobreposição (5.43× contra 3.43×). Mas `disjunto` não explica sozinho o resultado: os 78 casos Voronoi não têm sobreposição de área e mesmo assim ficam em 1.11×; todos têm contatos de fronteira entre células. Já as 177 instâncias estritamente disjuntas — 156 OSM e 21 aleatórias — ficam em 10.49×. As 21 aleatórias estritamente disjuntas são pequenas (18 com 5 polígonos, uma com 9 e duas com 10), então tamanho e fonte também confundem esse contraste.

## Menores speedups

Os 12 menores speedups das 550 instâncias pareadas são todos tessellation/Voronoi. O pior é o Caso 452 (`sbgdb-20200507-pntset-0000060`), com 60 polígonos: 246.63 s no nosso solver, 18.83 s no Fekete e speedup de 0.076× (Fekete cerca de 13.1× mais rápido).

## Método e limitações geométricas

O classificador testa pares de polígonos com Boost.Geometry. Vértices consecutivos separados por até `1e-12 × max(1, extensão da caixa delimitadora)` são removidos antes da validação; a interseção é considerada de área positiva somente se superar `max(1e-12, 1e-10 × menor área dos dois polígonos)`. Áreas menores que isso são registradas como contato de fronteira. Essa tolerância evita tratar ruído numérico como sobreposição, mas casos com áreas na vizinhança do limite dependem dela.

A classificação de disjunção é descritiva e correlacional. As classes diferem também em fonte e tamanho; ela não isola causalmente o efeito da sobreposição. Para Voronoi, compartilhar fronteiras é esperado, portanto disjunção estrita não é uma categoria aplicável a essas regiões fechadas.

O CSV por instância mantém os rótulos da fonte, nomes/IDs originais, status, runtimes, speedup e classificação de contatos. A proveniência do artigo está documentada junto à campanha.

SHA-256 do ZIP de instâncias simplificadas: `210841184500cb444f332537c4291aeff502dde4e3adc4a4ee307a80db21711f`.
Revisão do submódulo: `f4aa78c631545e4e894732a0fe8aef45f455c34c`.
