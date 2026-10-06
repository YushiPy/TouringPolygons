# Roteiro de 2 minutos (cinco slides, ~270 palavras)

**Slide 1 (~15 s).** Qual é o menor caminho que visita todas estas regiões? Aqui
as regiões são as letras do título, e o caminho colorido é a solução ótima, do
começo ao fim. Sou o Gabriel, orientado pelo Ernesto Birgin, com bolsa da FAPESP.

**Slide 2 (~25 s).** Formalmente: dados um ponto inicial s, um final t e polígonos
P₁ a Pₖ, queremos o caminho de s a t de menor comprimento que toca cada
polígono. Não escolhemos só a ordem, mas também onde tocar. Por exemplo, um drone
sai do IME e fotografa 51 regiões da USP: a rota ótima tem 4,97 quilômetros.

**Slide 3 (~30 s).** Testar todas as rotas é impossível, então montamos a rota
numa árvore de busca. Em cada nó, um solver geométrico exato para o caso convexo
dá um limite inferior L. Ramificamos inserindo o próximo polígono ou refinando em
peças convexas. Se L já supera uma rota viável U, descartamos o ramo. Um caso com
60 regiões tem 10^117 combinações; resolvemos em 2,47 s. O resumo cobria ordem
fixa; depois da submissão, estendemos para ordem livre.

**Slide 4 (~35 s).** Comparamos com o algoritmo de Fekete e colaboradores, que usa
Gurobi, nas 558 instâncias do artigo deles. No pôster a média era 10,4× e éramos
mais rápidos em 492 de 550. Depois da submissão melhorei o solver: agora a mediana
é 40,8×, somos mais rápidos nos 550 casos que ambos resolvem, e provamos 558 ótimos
contra 550. Cada ponto é uma instância; abaixo da diagonal, somos mais rápidos.
Mesma tolerância de 0,1 % nos dois, sem limite de tempo, uma thread.

**Slide 5 (~15 s).** Limitações: o pior caso continua exponencial e as regiões são
alvos, não obstáculos. Próximo passo: roteirização de veículos. Escaneiem o QR code
para resolver desafios e investigar os 558 casos.

## Perguntas prováveis
- *Por que os números mudaram desde o pôster?* Melhorei o solver depois de enviar o pôster. A campanha nova rodou na dantzig (i9-12900K), sem limite de tempo; o pôster usava a campanha anterior, com teto de 6 h. Os dois resultados estão preservados em `benchmarks/results-saved`.
- *O que significa "ótimo"?* Gap relativo fechado na tolerância: 0,1 % nos dois solvers nesta campanha. É certificado numérico, não prova racional.
- *Mediana ou média?* A mediana (40,8×) resiste a valores extremos; a média geométrica é 49,0× e a aritmética 106,7×. O pôster usava a média aritmética (10,4×; mediana 5,1× nos mesmos dados).
- *E os 8 casos em aberto do Fekete?* Foram interrompidos manualmente depois de dias; nosso solver os resolveu (o maior leva 722 s). Ficam fora da comparação de tempo.
- *Por que as letras valem como instância?* Tocar uma letra equivale a tocar seu contorno externo, então os buracos não alteram a rota ótima.
