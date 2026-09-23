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
50 contornos desenhados manualmente no QGIS e exportados da suíte local
`benchmarks/suites/usp-butanta-50`. O autor confirmou que `ime-a`, `ime-b` e
`ime-c` representam os blocos A, B e C do IME; os três são dourados. Outros
rótulos provisórios e a licença de redistribuição dos contornos ainda exigem
revisão antes da publicação externa.

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

O card após o desafio final usa uma contagem **formal** do corpus: para `n` regiões, `n! × ∏ p_i`, onde `p_i` é o número de peças convexas da região `i` na decomposição guardada em `data/event-data.js`. Não é o número de ramos examinados nem de rotas geométricas distintas. As 558 linhas foram conferidas; o caso 1 não tem decomposição armazenada, mas seus 48 polígonos são convexos (`p_i = 1`). O caso 49 tem 60 regiões, produto de peças `≈ 1,8767 × 10^35`, espaço formal `≈ 1,5616 × 10^117` e terminou em 2,473421125 s com uma thread. O maior espaço formal do corpus é o caso 130, `≈ 4,7851 × 10^123`; sua execução terminou em 20.862,645174292 s. O card usa o caso 49 para comparar o espaço formal com uma execução rápida. A analogia agora concede um computador a **cada átomo da Terra**, cada um verificando `10^9` combinações por segundo. A massa da Terra (`5,9722 × 10^24 kg`, NASA) dividida pela massa de um próton (`1,67262192595 × 10^-27 kg`, NIST) dá um teto generoso de `3,571 × 10^51` átomos/computadores, pois um átomo tem pelo menos um próton. Mesmo sob esse teto, a enumeração exigiria mais de `1,38 × 10^49` anos (ano de 365,25 dias), mais de `2,77 × 10^39` vezes os cerca de 5 bilhões de anos até a fase de gigante vermelha do Sol, conforme as fontes citadas na página. A analogia não sugere que o solver tenha enumerado todas as combinações.

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

A página já usa a rodada salva da campanha. Atualizações dos resultados devem preservar juntos entrada, configuração, dados brutos, análise e proveniência; depois, regenerar `data/event-data.js` e revisar as afirmações públicas.

## Estado atual da publicação

A versão atual é formada por arquivos separados em `apps/siicusp34`:

- `index.html` (~51 kB), `styles.css` (~88 kB), `app.js` (~116 kB) e os SVGs;
- `data/usp-demo.js` (~63 kB): 50 regiões do campus, fora do corpus;
- `data/challenge-data.js` (~270 kB): três desafios sintéticos;
- `data/event-data.js` (~10,5 MB): 558 casos e comparação experimental;
- `data/trace-data.js` (~5,5 MB): 186 registros de execução.

Não há backend, WASM, `fetch` ou mapas carregados em tempo de visita. Os dados
estáticos grandes são scripts externos carregados pela página; o total bruto
inicial ainda é de aproximadamente 16,6 MB. A página funciona com uma pasta
estática, mas esse peso deve ser medido em um celular real na rede do evento.
O [README](README.md) contém os comandos de prévia local e na mesma rede Wi-Fi.

## Pendências e prioridades

Antes de publicar a URL do QR code:

1. Conferir os nomes ainda provisórios em `polygons.csv`, os 50 contornos e a
   permissão de publicação das geometrias desenhadas sobre o mapa-base OSM.
2. Testar a rota, os três desafios, a simulação, a tabela e o contato em um
   celular físico; verificar carregamento, zoom, leitura e funcionamento na
   rede do evento.
3. Revisar com o orientador a linguagem do algoritmo e a equivalência da
   comparação experimental, incluindo formulação, tolerâncias e status.
4. Fixar a URL pública antes de gerar o QR code do pôster e dos slides.

Depois do fluxo central estar estável, medir o custo dos dois arquivos grandes
e considerar carregamento sob demanda dos casos e traces, preservando uma
abertura imediata da rota da USP. Uma página técnica separada e a ligação entre
o desafio final e uma árvore registrada são opcionais.

O prazo do pôster é 1º de outubro. Isso torna revisão científica, URL e teste
em celular prioritários em relação a novas interações ou expansão de conteúdo.

## Critério de sucesso

Ao escanear o QR code, o visitante deve entender em poucos segundos que está vendo uma instância real e poder tocar em “Veja o caminho”. Depois, deve conseguir escolher entre experimentar, acompanhar a busca ou examinar resultados. A página deve impressionar pela geometria e pela evidência, não por efeitos que contradigam o problema.
