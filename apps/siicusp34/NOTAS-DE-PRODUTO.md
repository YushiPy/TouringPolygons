# SIICUSP 34 — decisões e ideias para o site

Este documento resume as discussões de setembro de 2026 sobre a experiência pública ligada ao pôster. Ele registra decisões, evidências e ideias para as próximas iterações; não é uma especificação fechada.

## Objetivo

O site é a extensão interativa do pôster. Seu papel é fazer o visitante:

1. entender visualmente o Touring Polygons Problem;
2. perceber que escolher ordem e pontos de visita é difícil;
3. ver um caminho real sendo construído;
4. acompanhar como o algoritmo encontra e certifica uma solução;
5. sair com uma evidência experimental clara e defensável.

O `TPP Research Workbench` é uma ferramenta interna. Ele pode ajudar pesquisadores e colaboradores, mas não deve aparecer na experiência pública. O site do QR code deve ter uma única entrada, sem rotas antigas ou ferramentas de laboratório expostas.

## O que precisa ser preservado

- Abrir diretamente com uma instância real, não com uma ilustração genérica.
- Oferecer imediatamente o playback do caminho visitando os polígonos.
- Mostrar segmentos e pontos de contato geometricamente plausíveis. Curvas decorativas e contatos arbitrários enfraquecem a credibilidade.
- Manter os exemplos navegáveis, a simulação da busca, a tabela de resultados, o método, as referências e o contato.
- Preservar a densidade científica sem transformar a página em um workbench.
- Fazer mudanças incrementais sobre a experiência existente. Antes de uma reformulação visual ampla, validar o que será perdido.

A rota `/evento` do `benchmark-dashboard` é a referência visual e funcional atual. A pasta `apps/siicusp34` contém uma versão estática congelada dessa experiência, sem backend ou WebAssembly.

## Fluxo desejado para o visitante

O fluxo discutido foi:

1. ver exemplos e entender o problema;
2. tentar tomar as decisões manualmente;
3. perceber que o desafio final é difícil;
4. observar o branch-and-bound resolvendo esse mesmo desafio;
5. conhecer os resultados no corpus de benchmark;
6. aprofundar-se no método, histórico e referências se tiver interesse.

Os três desafios têm funções diferentes:

- **Desafio 1:** regiões convexas; o visitante escolhe somente a ordem.
- **Desafio 2:** regiões não convexas decompostas; a ordem é fixa e o visitante escolhe uma peça por região.
- **Desafio 3:** o visitante escolhe peças e ordem.

Devemos manter os três. Se o primeiro for óbvio demais, o visitante pode concluir erroneamente que o problema inteiro é fácil. As instâncias devem ser pequenas o bastante para interação imediata, mas difíceis o bastante para escolhas locais intuitivas frequentemente falharem.

## Solução geométrica no navegador

Os três desafios podem usar apenas uma implementação JavaScript compacta do TPP convexo com ordem fixa. A implementação existente em `apps/visualizer-local` é a referência. Para instâncias tão pequenas, JavaScript é suficiente e evita:

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
- Devem ser publicados hardware, limite de tempo, tolerância de gap, número de threads, versão/revisão e critérios de certificação.
- “Exato” e “gap menor ou igual a 0,1%” não são a mesma coisa e devem aparecer separados.
- Resultados parciais da campanha de seis horas não devem ser apresentados como resultado final.

Resultados conhecidos quando este documento foi criado:

- nosso solver: 558/558 instâncias certificadas exatamente;
- baseline externo de 10 segundos: 437/558 dentro do gap de 0,1%;
- campanha externa de seis horas: em andamento;
- última leitura intermediária: 526/558 certificadas, 32 pendentes — número provisório, não publicável como conclusão.

O relatório `touring-polygons-benchmark-report.pdf` mencionado na conversa não serve como comparação com os alemães: a análise interpretada anteriormente comparava variantes internas do nosso solver e, em parte, problemas diferentes. Não reutilizar a afirmação de que “liberar a ordem melhorou 488 de 498 casos” como evidência contra Fekete et al.

Quando a campanha terminar, a atualização pública deve ser pequena: números, rótulos, metodologia e, se necessário, um gráfico/tabela. Não redesenhar a página para atualizar o benchmark.

## Campanha externa de seis horas

O runner está em:

`benchmarks/scripts/run_fekete_6h.py`

Comando a partir da raiz:

```bash
python3 benchmarks/scripts/run_fekete_6h.py --workers 8
```

Para uma máquina dedicada com 12 threads:

```bash
python3 benchmarks/scripts/run_fekete_6h.py --workers 12
```

Cada worker executa um processo independente com uma thread. O runner grava checkpoints atômicos, reaproveita certificados compatíveis do benchmark de 10 segundos e não repete instâncias concluídas. Ao interromper, perde-se apenas o trabalho das instâncias que estavam rodando naquele momento.

Oito workers foram escolhidos como padrão conservador para preservar responsividade e margem térmica. Doze são aceitáveis se a máquina estiver dedicada à campanha.

Também foi considerada uma rodada do nosso solver com parada em gap de 0,1%, para comparação simétrica de tempo. Ela pode fortalecer a análise, mas não é necessária para provar que nosso solver certificou exatamente as 558 instâncias. Não vale atrasar o site ou o pôster esperando essa rodada.

## Estado atual da versão estática

`apps/siicusp34/index.html` é uma exportação estática da experiência de `/evento`:

- CSS e JavaScript embutidos;
- 558 instâncias embutidas;
- 186 traces embutidos;
- nenhum `fetch`, endpoint de API, backend ou WASM;
- favicon local como o único arquivo adicional.

O HTML tem aproximadamente 16,8 MiB. Esse tamanho foi aceito temporariamente para preservar a versão anterior sem novas alterações. Antes da publicação, podemos reduzir o carregamento mantendo o site estático, por exemplo:

- embutir apenas os casos de abertura e carregar o restante de JSONs estáticos sob demanda;
- separar traces raramente vistos;
- comprimir os dados no processo de deploy;
- manter o primeiro exemplo e seu playback disponíveis imediatamente.

Qualquer otimização deve preservar o funcionamento offline ou ter uma versão offline equivalente.

## Melhorias futuras possíveis

Prioridade alta:

- substituir os números preliminares pelos resultados finais da campanha de seis horas;
- revisar todas as afirmações da comparação para garantir equivalência de problema e critério;
- escolher os três melhores exemplos para abertura, variedade geométrica e valor didático;
- revisar textos com o orientador;
- testar o fluxo completo em celular e na rede do evento;
- gerar o QR code somente depois de estabilizar a URL pública.

Prioridade média:

- tornar os desafios ligeiramente menos óbvios;
- relacionar o desafio final à execução registrada do branch-and-bound;
- oferecer explicações curtas para incumbente, limite inferior, ramificação e poda;
- mostrar claramente a diferença entre caminho viável, ótimo certificado e solução dentro de tolerância;
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

