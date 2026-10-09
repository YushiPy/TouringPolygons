# Roteiro de 2 minutos (cinco slides, ~250 palavras, ~1 min 45 s a 140 palavras por minuto)

**Slide 1 (~15 s).** Qual é o menor caminho que visita todas estas regiões? Aqui
elas são as letras do título, e a rota colorida é a solução ótima. Sou o Gabriel,
orientado pelo Ernesto Birgin, com bolsa da FAPESP.

**Slide 2 (~20 s).** Formalmente: dados um ponto inicial, um final e vários
polígonos, queremos o menor caminho que toca todos. Por exemplo, um drone do IME e
51 regiões da USP: cerca de 10^104 rotas possíveis, inviável testar todas. Nosso
solver acha a ótima em 0,23 segundos.

**Slide 3 (~25 s).** Como? Com branch and bound. Começamos com uma rota rápida, o vizinho mais próximo,
só como referência. Depois montamos rotas região por região e abandonamos qualquer
rota parcial já pior que a melhor conhecida. Na instância de 60 regiões, a força
bruta precisaria de 10^117 cálculos; nós fazemos cerca de 2 mil, em 0,1 segundo.
Isso vale também para a ordem livre, novidade desde o resumo.

**Slide 4 (~30 s).** Comparamos com Fekete e colaboradores, um grupo alemão que
usa um solver genérico, o Gurobi, nas 558 instâncias do artigo deles. No pôster, a
média era 10 vezes. Depois da submissão melhorei o solver: agora a média é 123
vezes mais rápido, e somos mais rápidos nas 558 instâncias: em 553 medimos o
speedup, e nas outras 5 o Fekete nem terminou. Também provamos 558 ótimos, contra 553.

**Slide 5 (~15 s).** Venham ao pôster: lá explico por que o solver geométrico
vence, onde ganhamos menos e como a busca evita as 10^117 combinações. E o QR code
leva ao app, com centenas de instâncias e a simulação da busca.

## Perguntas prováveis
- *Por que os números mudaram desde o pôster?* Melhorei o solver depois de enviar o pôster. A campanha nova rodou na dantzig (i9-12900K): o Fekete sem limite de tempo (02/10) e o nosso solver em 08/10, já com o polimento em ponto flutuante; o pôster usava a campanha anterior, com teto de 6 h. Os resultados estão preservados em `benchmarks/results-saved`.
- *O pior caso não era 3×?* Era, com o solver de 06/10. O pior caso continua o 452 (Voronoi, 60 regiões), mas nosso tempo caiu de 7,4 s para 2,9 s com o polimento em ponto flutuante: 25,2 s do Fekete dão 8,7×.
- *O que significa "ótimo"?* Gap relativo fechado na tolerância: 0,1 % nos dois solvers nesta campanha. É certificado numérico, não prova racional.
- *Por que a média e não a mediana?* É a mesma métrica do pôster (média aritmética dos speedups: 10,4× lá, 122,6× agora), então a comparação é direta. A média é puxada por poucos casos extremos (o maior é 2.302×); por isso o gráfico mostra também a mediana (57,6×). A média geométrica é 65,4×.
- *E os 5 casos em aberto do Fekete?* Foram interrompidos manualmente depois de dias; nosso solver os resolveu (o maior leva 978 s). Ficam fora da comparação de tempo. Três dos 8 casos que estavam abertos fecharam depois: 131 (11,5 h contra 23 s nossos), 420 (6,8 h contra 45 s) e 558 (25,3 h contra 47 s).
- *Por que as letras valem como instância?* Tocar uma letra equivale a tocar seu contorno externo, então os buracos não alteram a rota ótima.
