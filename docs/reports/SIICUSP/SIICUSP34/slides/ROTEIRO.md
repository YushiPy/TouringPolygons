# Roteiro de 2 minutos (cinco slides, ~270 palavras)

**Slide 1 (~15 s).** Qual é o menor caminho que visita todas estas regiões? Aqui
as regiões são as letras do título, e o caminho colorido é a solução ótima, do
começo ao fim. Sou o Gabriel, orientado pelo Ernesto Birgin, com bolsa da FAPESP.

**Slide 2 (~25 s).** O problema: dados um ponto inicial s, um final t e polígonos
P₁ a Pₖ, queremos o caminho de s a t de menor comprimento que toca cada polígono.
Escolhemos a ordem e onde tocar. Por exemplo, um drone sai do IME e precisa passar
por 51 regiões da USP: são cerca de 10^104 rotas possíveis, testar todas é
inviável, e o nosso solver acha a ótima em 0,23 segundos.

**Slide 3 (~30 s).** Como? Numa busca em que descartamos ramos inteiros. Em um
exemplo real de 5 regiões: estimamos o caminho mais curto possível, acrescentamos
uma região por vez, e abandonamos o ramo que já passa da melhor rota conhecida.
Na instância de 60 regiões, a força bruta precisaria de 10^117 cálculos, uns 10^100
anos; nosso método faz cerca de 1,4 mil cálculos, em 0,1 segundo. O resumo cobria
ordem fixa; depois da submissão, estendemos para ordem livre.

**Slide 4 (~35 s).** Comparamos com Fekete e colaboradores, um grupo alemão que
publicou este ano uma solução com solver genérico, o Gurobi; a nossa é um solver
geométrico exato. Usamos as 558 instâncias do artigo deles. No pôster a média era 10,4× e éramos
mais rápidos em 492 de 550. Depois da submissão melhorei o solver: agora a mediana
é 40,8×, e somos mais rápidos nos 550 casos que ambos resolvem. Também provamos
558 ótimos contra 550 (está na nota de rodapé). O histograma mostra o speedup de cada instância em escala logarítmica: nenhuma fica à esquerda de 1×.
Mesma tolerância de 0,1 % nos dois, sem limite de tempo, uma thread.

**Slide 5 (~15 s).** Se ficou curioso, venha ver o pôster: lá explico por que um
solver geométrico exato supera um solver genérico, em quais tipos de instância
vencemos menos e como a busca evita testar 10^117 combinações. E, para ir além,
o QR code leva ao app, com centenas de instâncias e a simulação passo a passo.

## Perguntas prováveis
- *Por que os números mudaram desde o pôster?* Melhorei o solver depois de enviar o pôster. A campanha nova rodou na dantzig (i9-12900K), sem limite de tempo; o pôster usava a campanha anterior, com teto de 6 h. Os dois resultados estão preservados em `benchmarks/results-saved`.
- *O que significa "ótimo"?* Gap relativo fechado na tolerância: 0,1 % nos dois solvers nesta campanha. É certificado numérico, não prova racional.
- *Mediana ou média?* A mediana (40,8×) resiste a valores extremos; a média geométrica é 49,0× e a aritmética 106,7×. O pôster usava a média aritmética (10,4×; mediana 5,1× nos mesmos dados).
- *E os 8 casos em aberto do Fekete?* Foram interrompidos manualmente depois de dias; nosso solver os resolveu (o maior leva 722 s). Ficam fora da comparação de tempo.
- *Por que as letras valem como instância?* Tocar uma letra equivale a tocar seu contorno externo, então os buracos não alteram a rota ótima.
