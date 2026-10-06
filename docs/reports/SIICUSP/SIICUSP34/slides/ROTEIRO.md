# Roteiro de 2 minutos (~270 palavras)

**Slide 1 (~20 s).** Imagine um drone que sai do IME e precisa fotografar 51
regiões da USP. Qual é o menor caminho? Ele não precisa passar por pontos
fixos: basta tocar cada região, e a ordem das visitas também é escolha nossa.
Esse é o Problema de Visita de Polígonos, e esta é a rota ótima que calculamos:
4,97 quilômetros.

**Slide 2 (~35 s).** Testar todas as rotas é impossível. Nosso método monta a
rota aos poucos, numa árvore de busca. Em cada nó, um solver geométrico exato
para o caso convexo calcula um limite inferior L. Ramificamos inserindo o
próximo polígono em cada posição ou refinando em peças convexas. Se L já é
pelo menos o de uma rota viável U, o ramo inteiro é descartado. Num caso com 60
regiões, são 10^117 combinações; resolvemos em 2,47 segundos.

**Slide 3 (~40 s).** Comparamos com o algoritmo de Fekete e colaboradores, que
usa o solver comercial Gurobi, nas 558 instâncias do artigo deles, com uma
thread. Cada ponto é uma instância; abaixo da diagonal, somos mais rápidos.
Em média, 10,4 vezes mais rápidos, e vencemos em 492 de 550 casos. Provamos
558 ótimos; eles, 550. A ressalva: nas instâncias Voronoi eles ganham em 40 de
78.

**Slide 4 (~25 s).** O ponto central é usar geometria exata dentro de uma busca
combinatória. O resumo cobria ordem fixa; depois da submissão estendemos para
ordem livre. Limitações: o pior caso continua exponencial e regiões são alvos,
não obstáculos. Próximo passo: roteirização de veículos. Escaneiem o QR code
para explorar. Agradeço à FAPESP pelo apoio.

## Perguntas prováveis
- *O que significa "ótimo"?* Gap relativo 1e-7 no nosso solver (1e-3 em Fekete); certificado numérico, não prova racional.
- *Por que perde em Voronoi?* Polígonos adjacentes: a relaxação convexa é fraca e a poda ajuda menos.
- *10,4× é média de quê?* Média aritmética das razões tempo Fekete / nosso tempo, nos 550 casos concluídos por ambos.
