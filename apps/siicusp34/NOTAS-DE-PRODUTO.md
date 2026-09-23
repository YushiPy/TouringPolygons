# SIICUSP 34 — decisões e ideias para o site

Este documento resume as discussões de setembro de 2026 sobre a experiência pública ligada ao pôster. Ele registra decisões, evidências e ideias para as próximas iterações; não é uma especificação fechada.

## Objetivo

O site é a extensão interativa do pôster. Seu papel é fazer o visitante:

1. entender visualmente o Touring Polygons Problem;
2. perceber que escolher ordem e pontos de visita é difícil;
3. ver um caminho real sendo construído;
4. acompanhar como o algoritmo conclui a busca pelo menor caminho no modelo;
5. sair com uma evidência experimental clara e defensável.

O `TPP Research Workbench` é uma ferramenta interna. Ele pode ajudar pesquisadores e colaboradores, mas não deve aparecer na experiência pública. O site do QR code deve ter uma única entrada, sem rotas antigas ou ferramentas de laboratório expostas.

## O que precisa ser preservado

- Abrir diretamente com uma instância real, não com uma ilustração genérica.
- Oferecer imediatamente o playback do caminho visitando os polígonos.
- Mostrar segmentos e pontos de contato geometricamente plausíveis. Curvas decorativas e contatos arbitrários enfraquecem a credibilidade.
- Manter os exemplos navegáveis, a simulação da busca, a tabela de resultados, o método, as referências e o contato.
- Preservar a densidade científica sem transformar a página em um workbench.
- Fazer mudanças incrementais sobre a experiência existente. Antes de uma reformulação visual ampla, validar o que será perdido.

`apps/siicusp34` é a publicação estática autocontida do evento, sem backend ou WebAssembly. Sua implementação atual deve ser examinada diretamente; materiais antigos de outras aplicações não são fonte para esta página.

## Fluxo desejado para o visitante

O fluxo da página após a revisão de setembro de 2026 é:

1. explorar a rota da USP, uma demonstração independente do corpus;
2. tentar os três desafios;
3. ver o problema, o método e a simulação registrada do algoritmo;
4. conhecer os trabalhos anteriores;
5. comparar os resultados no corpus de 558 instâncias;
6. conversar com o autor e consultar as referências.

A simulação fica na etapa do algoritmo e usa um pseudocódigo didático com
palavras-chave coloridas e destaque da linha associada ao evento visível.
O registro não mostra cada instrução executada. A demonstração da USP usa
contornos OSM provisórios; IME e FEA precisam de revisão manual antes de
servirem como ilustração cartográfica confiável.

A explicação do algoritmo começa com uma definição curta do TPP de ordem livre,
segue a busca por ordem parcial, inserção em todas as posições, refinamento em
peças convexas e poda por limite inferior. Antes da simulação, explica por que
o subproblema convexo de ordem fixa é tratável e por que `L ≥ U` permite podar.
A página relata conclusão da busca, status e tolerâncias; não promete ao
visitante um certificado independente para download ou verificação.

Os três desafios têm funções diferentes:

- **Desafio 1:** regiões convexas; o visitante escolhe somente a ordem.
- **Desafio 2:** regiões não convexas decompostas; a ordem é fixa e o visitante escolhe uma peça por região.
- **Desafio 3:** o visitante escolhe peças e ordem.

Devemos manter os três. Se o primeiro for óbvio demais, o visitante pode concluir erroneamente que o problema inteiro é fácil. As instâncias devem ser pequenas o bastante para interação imediata, mas difíceis o bastante para escolhas locais intuitivas frequentemente falharem.

As instâncias sintéticas atuais são geradas por `scripts/build_challenge_data.mjs`, independentemente do corpus de 558 casos. Para avaliar a escolha local, parte-se de `S` e escolhe-se a próxima região não visitada mais próxima da anterior. Nos desafios com peças, escolhe-se a peça (e, no terceiro, também a região) mais próxima da escolha anterior. A distância entre os retângulos é a distância euclidiana entre conjuntos, sem usar `T` antes da última visita. O script enumera as 24 ordens, 27 escolhas de peças e 1.944 combinações, calcula cada caminho com o solver local e exige diferença relevante entre a regra local e o menor caminho, alternativas próximas e margens de escolha não microscópicas. A interface mostra a comparação da regra local depois da tentativa do visitante e oferece botões recolhidos como alternativa aos alvos finos do mapa no celular.

## Solução geométrica no navegador

Os três desafios usam uma implementação JavaScript compacta do TPP convexo com ordem fixa, mantida em `tpp-solver.js` e `tpp-vector2.js` ao lado da página. Essa cópia foi congelada a partir da implementação histórica do visualizador, para que o evento seja autocontido e possa ser publicado copiando apenas `apps/siicusp34`. Para instâncias tão pequenas, JavaScript é suficiente e evita:

- WebAssembly;
- workers e carregamento adicional;
- binários ou backend;
- dependências e pontos extras de falha.

O navegador pode enumerar as poucas ordens e escolhas dos desafios e chamar o solver de ordem fixa para cada alternativa. Isso mantém a comparação real, em vez de usar respostas visuais hardcoded.

## Simulação do branch-and-bound

A simulação do branch-and-bound é conceitualmente separada dos desafios. O solver JavaScript resolve a escolha do visitante; a simulação explica o algoritmo de ordem livre e regiões não convexas.

Decisão atual:

- não executar o solver completo via WASM durante a visita;
- reproduzir uma execução nativa real, registrada previamente;
- filtrar eventos pouco informativos, mas nunca inventar decisões, bounds ou podas;
- deixar explícito que se trata de uma reprodução fiel de uma execução.

Uma narrativa como “a decisão local obrigou o caminho a voltar para C” só deve aparecer quando puder ser derivada dos dados registrados. Não devemos explicar a árvore com histórias hardcoded que não correspondam ao algoritmo.

Pode existir futuramente uma segunda página, mais técnica, com mais instâncias, árvore completa e ferramentas para orientador ou colaboradores. Ela não deve competir com o fluxo principal do QR code.

## Comparação com Fekete et al.

A comparação é importante. Dizer apenas “resolvemos 558 instâncias” não explica por que essas instâncias são relevantes. O corpus ganha valor porque está ligado ao trabalho de Fekete, Kniep, Krupke e Perk e foi apresentado como desafiador.

Pontos que devem ser respeitados:

- O solver de Fekete et al. resolve TSPN com ordem livre. A descrição anterior de que ele resolveria apenas ordem fixa estava errada.
- A comparação deve colocar os dois solvers no mesmo problema adaptado, com extremos fixos e ordem livre.
- Cada instância deve usar uma única thread. Rodar várias instâncias simultaneamente não deixa de ser uma comparação single-core por instância.
- Multithreading interno não deve ser usado como argumento principal de desempenho, pois aumentar núcleos seria uma forma fácil e potencialmente enganosa de melhorar o número.
- Devem ser publicados hardware, limite de tempo, tolerância de gap, número de threads, versão/revisão e critérios de término e limites numéricos.
- “Exato” e “gap menor ou igual a 0,1%” não são a mesma coisa e devem aparecer separados.
- Resultados parciais da campanha de seis horas não devem ser apresentados como resultado final.

Resultados publicados na versão atual:

- nosso solver: 558/558 instâncias concluídas com `termination=optimal` e gap dentro das tolerâncias declaradas;
- 477/558 instâncias do nosso solver foram resolvidas em menos de 10 segundos;
- no conjunto comum concluído, o speedup mediano Fekete/nosso é 5,11× e nosso solver é mais rápido em 492/550 casos;
- solver de Fekete et al.: 550/558 concluídas; 8 instâncias não foram concluídas no limite de seis horas.

Os caminhos da nova rodada estão em `benchmarks/results-saved/german-comparison/ours.csv` e incluem a trajetória final, a ordem livre e o SHA-256 de cada instância. O app é regenerado por `apps/siicusp34/scripts/build_event_data.py`.

O relatório `touring-polygons-benchmark-report.pdf` mencionado na conversa não serve como comparação com os alemães: a análise interpretada anteriormente comparava variantes internas do nosso solver e, em parte, problemas diferentes. Não reutilizar a afirmação de que “liberar a ordem melhorou 488 de 498 casos” como evidência contra Fekete et al.

Quando a campanha terminar, a atualização pública deve ser pequena: números, rótulos, metodologia e, se necessário, um gráfico/tabela. Não redesenhar a página para atualizar o benchmark.

## Campanha externa de seis horas

O runner está em:

`python3 benchmarks/tpp.py run-fekete`

Comando a partir da raiz:

```bash
python3 benchmarks/tpp.py run-fekete --workers 8
```

Para uma máquina dedicada com 12 threads:

```bash
python3 benchmarks/tpp.py run-fekete --workers 12
```

Cada worker executa um processo independente com uma thread. O runner grava checkpoints atômicos, reaproveita certificados compatíveis do benchmark de 10 segundos e não repete instâncias concluídas. Ao interromper, perde-se apenas o trabalho das instâncias que estavam rodando naquele momento.

Oito workers foram escolhidos como padrão conservador para preservar responsividade e margem térmica. Doze são aceitáveis se a máquina estiver dedicada à campanha.

Também foi considerada uma rodada do nosso solver com parada em gap de 0,1%, para comparação simétrica de tempo. Ela pode fortalecer a análise, mas não é necessária para mostrar que nosso solver concluiu as 558 instâncias sob as tolerâncias declaradas. Não vale atrasar o site ou o pôster esperando essa rodada.

## Estado atual da versão estática

`apps/siicusp34/index.html` é uma exportação estática da experiência de `/evento`:

- CSS e JavaScript embutidos;
- 558 instâncias embutidas;
- 186 traces embutidos;
- nenhum `fetch`, endpoint de API, backend ou WASM;
- favicon local como o único arquivo adicional.

O HTML tem aproximadamente 16,8 MiB. Esse tamanho foi aceito temporariamente para preservar a versão anterior sem novas alterações. Os dados e o solver local são arquivos estáticos; não há dependência de `apps/visualizer-local` ou de outro app do repositório. Antes da publicação, podemos reduzir o carregamento mantendo o site estático, por exemplo:

- embutir apenas os casos de abertura e carregar o restante de JSONs estáticos sob demanda;
- separar traces raramente vistos;
- comprimir os dados no processo de deploy;
- manter o primeiro exemplo e seu playback disponíveis imediatamente.

Qualquer otimização deve preservar o funcionamento offline ou ter uma versão offline equivalente.

## Melhorias futuras possíveis

Prioridade alta:

- revisar todas as afirmações da comparação para garantir equivalência de problema e critério;
- escolher os três melhores exemplos para abertura, variedade geométrica e valor didático;
- revisar textos com o orientador;
- testar o fluxo completo em celular e na rede do evento;
- gerar o QR code somente depois de estabilizar a URL pública.

Prioridade média:

- relacionar o desafio final à execução registrada do branch-and-bound;
- oferecer explicações curtas para incumbente, limite inferior, ramificação e poda;
- mostrar claramente a diferença entre caminho viável, busca concluída com gap numérico fechado e solução apenas factível;
- reduzir o peso inicial sem remover conteúdo;
- considerar uma página técnica separada para colaboradores.

Evitar:

- hero genérico que esconda a demonstração real;
- caminhos curvos ou pontos de contato apenas decorativos;
- remover a tabela ou resultados em nome de minimalismo;
- expor o Research Workbench ao público;
- comparar problemas diferentes;
- publicar números provisórios como conclusão;
- usar WASM quando dados registrados ou JavaScript simples resolvem o mesmo objetivo;
- recomeçar o design sem primeiro preservar e validar os elementos já apreciados.

## Critério de sucesso

Ao escanear o QR code, o visitante deve entender em poucos segundos que está vendo uma instância real e poder tocar em “Veja o caminho”. Depois, deve conseguir escolher entre experimentar, acompanhar a busca ou examinar resultados. A página deve impressionar pela geometria e pela evidência, não por efeitos que contradigam o problema.
