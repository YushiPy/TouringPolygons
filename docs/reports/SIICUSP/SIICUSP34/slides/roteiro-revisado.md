# Roteiro de 2 minutos (cinco slides, ~282 palavras, ~1min55 a 150 palavras por minuto)

**Slide 1 (~15 s).** Olá a todos, meu nome é Gabriel, fui orientado pelo Ernesto, e a pergunta que eu quero responder é: qual o menor caminho que visita um conjunto de regiões no plano? O próprio título dessa apresentação é uma instância do Problema de Visita de Polígonos, para o qual estou propondo um algoritmo exato.

**Slide 2 (~20 s).** Formalmente: dados um ponto inicial, um final e vários polígonos, a gente quer o menor caminho que toca todos. Por exemplo, um drone que parte do IME visita 51 regiões da USP, temos cerca de 10^104 rotas possíveis, claramente inviável testar todas. Bem, o nosso solver acha a rota ótima em 0,23 segundos.

**Slide 3 (~25 s).** Como? Com branch and bound. A gente monta as rotas região por região, até aí, nada melhor que força bruta, no entanto, se notarmos que uma ordenação, mesmo incompleta, está pior que a melhor rota encontrada até agora, a gente descarta ela. Numa instância de 60 regiões, a força bruta precisaria de 10^117 cálculos, isso levaria 10^100 anos pra terminar; nós fazemos cerca de 2 mil, em 0,1 segundo.

**Slide 4 (~30 s).** Afinal, essa abordagem é boa? Comparamos com Fekete e colaboradores, um grupo alemão que publicou ainda esse ano, usando um solver genérico, o Gurobi, nas 558 instâncias do próprio artigo deles. No pôster, a média ainda era de 10 vezes. Depois da submissão melhorei o solver: agora a média é 123 vezes mais rápido, e somos mais rápidos em todas as instâncias, no pior dos casos 9x e no melhor, 2300x.

**Slide 5 (~12 s).** Venham ao pôster, onde explico por que nosso solver vence, onde ganhamos menos e mais, e como a busca evita as 10^117 combinações. O QR code leva ao app, com centenas de instâncias e simulações passo a passo da busca.
