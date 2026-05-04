
## Feitos até agora

- Implementamos A-star para aproximar a solução ótima.
- Comentar sobre **tentar** verificar se uma solução é ótima dada a solução.
- Comentar sobre bolsa BEPE.
- Comentar sobre a monitoria na greve.
- Comentar sobre a outra implementação do TPP.
	- Calcula todas as regiões do mapa de primeiro contato de uma vez.
	- Remove o fator logaritmico, mas não permite computação preguiçosa.
	- Interessante se quase todas as regiões forem necessárias.
	- É uma melhoria local, assim, podemos misturar com a abordagem anterior.
- Comentar sobre a versão interativa web do TPP.

- Tour convexo:
	- Fácil (custo O(min|P_i|)) determinar a aresta ótima, mas como determinar o ponto ótimo?


## Estratégia Futuras

- Quebrar o fecho convexo em multiplas partes.
	- Dificil de implementar, pois não sabemos escolher os pontos de quebra.
	- Não garante que seja mais rápido.

## Novas ideias

- Mais limitantes
- Tentar explorar a árvore em outras ordens, não precisamos começar no primeiro polígono
- Explorar a árvore em subconjuntos do polígono original, em vez de explorar a árvore do polígono inteiro.
	- Inicialmente, podemos fazer hardcoding para certas instâncias para verificar se isso é útil.
- O incumbente já é muito bom, então talvez seja melhor focar em melhorar os limitantes.
- Deixar pra lá o problema de um tour em polígonos convexos, pode não ter solução boa.
- Implementar a solução melhor do TPP não é prioridade.
- Instâncias maiores
