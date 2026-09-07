# Plano de trabalho para o 34º SIICUSP

## Objetivo de entrega

Preparar uma apresentação clara, rigorosa e visualmente marcante do trabalho **Algoritmos Exatos Para o Problema de Visita de Polígonos**, mantendo coerência com o resumo submetido e incorporando os resultados mais recentes como continuação do trabalho.

O pacote mínimo seguro de entrega contém:

- pôster A0 em PDF;
- apresentação curta em PDF, inicialmente planejada com capa e dois slides de conteúdo;
- instância demonstrativa estável, com solução e estatísticas reproduzíveis;
- roteiro de dois minutos e respostas para perguntas prováveis;
- QR code testado, somente se o destino estiver publicado e funcionar bem em celular.

## Regras confirmadas

### Edital geral do 34º SIICUSP

- O título deve ser idêntico no formulário de inscrição, no resumo e no material de apresentação.
- A avaliação considera domínio do problema e dos objetivos, virtudes e limitações da metodologia, compreensão dos resultados, relação entre conclusões e dados, relevância, clareza oral, qualidade das respostas, interesse despertado e coerência com o resumo.
- A seleção para a Etapa Internacional considera conteúdo e comunicação.
- O resumo deve apresentar declaração de conflito de interesses.

### Orientação da primeira etapa do IME

- Pôster em formato A0, entregue como PDF.
- Nome do arquivo: nome completo do autor.
- Slides entregues como PDF.
- Pitch oral de dois minutos.
- Pôsteres expostos de 7 a 14 de outubro de 2026.
- Apresentações e visita aos pôsteres em 13 e 14 de outubro de 2026.
- Prazo público do pôster: 28 de setembro de 2026.
- Prazo público dos slides: 6 de outubro de 2026.

### Pontos ainda não confirmados

- A documentação pública consultada não fixa o número de slides. Usaremos três, capa e dois slides, conforme a informação recebida por Gabriel, até confirmação da CPqI.
- O PDF obrigatório não preserva vídeo nem animações. A entrega principal será completamente funcional como PDF estático. Qualquer mídia será tratada como demonstração adicional, nunca como parte necessária da fala.
- A meta de trabalho considera uma primeira entrega em quatro dias, mesmo que o prazo público do IME seja posterior.

## Estratégia narrativa

### Pergunta central

> Como encontrar o menor caminho entre dois pontos que visita várias regiões poligonais, escolhendo onde e, na variante mais geral, em que ordem visitá-las?

### História em quatro movimentos

1. Uma situação concreta apresenta o problema: um drone deve sobrevoar institutos da USP.
2. A geometria explica a diferença para visitar pontos: o caminho escolhe onde tocar cada região.
3. A metodologia mostra como o solver convexo exato alimenta uma busca Branch and Bound para regiões não convexas e ordem livre.
4. Os resultados mostram redução do espaço de busca, certificados de otimalidade, validação independente e limitações atuais.

### Relação com o resumo submetido

O material não deve contradizer o resumo. A extensão para ordem livre será apresentada como avanço recente produzido a partir da estrutura descrita no resumo. Formulação recomendada:

> O resumo descreve o solver convexo e o Branch and Bound para ordem fixa. Desde a submissão, estendemos a mesma arquitetura para também decidir a ordem de visita.

## Cobertura dos critérios de avaliação

| Critério | Evidência no material | Preparação oral |
| --- | --- | --- |
| Problema e objetivos | Definição visual e instância da USP | Explicar entrada, saída e diferença para visitar pontos |
| Virtudes e limitações | Certificado de limites, pior caso exponencial, tolerâncias numéricas | Distinguir solução ótima certificada de busca interrompida |
| Resultados | Números reproduzíveis e uma instância completa | Explicar exatamente o que cada número mede |
| Conclusões apoiadas pelos dados | Conclusão curta junto aos resultados | Evitar alegações universais de superioridade |
| Relevância | Relação com roteirização e generalização de visita a pontos | Explicar por que regiões tornam o problema mais expressivo |
| Clareza e interesse | Grande figura central, pouco texto, pitch ensaiado | Abrir com a pergunta do drone e terminar com contribuição concreta |
| Coerência com o resumo | Separação explícita entre trabalho submetido e avanço recente | Saber justificar a atualização sem reescrever a história do projeto |

## Conteúdo proposto

### Pôster A0

Composição recomendada, orientação vertical:

1. Cabeçalho com título oficial, autor, orientador, IME-USP, FAPESP e logotipos exigidos.
2. Grande figura da instância da USP com a rota calculada.
3. Definição compacta do problema e comparação entre pontos e regiões.
4. Quadro com os dois eixos de dificuldade: ordem fixa ou livre, regiões convexas ou não convexas.
5. Pipeline do algoritmo: solver convexo, relaxação, ramificação e poda.
6. Resultados com no máximo três números principais, todos reproduzíveis.
7. Limitações e próximos passos em poucas linhas.
8. QR code com chamada clara e URL curta.

### Apresentação de três slides

#### Slide 1, capa e pergunta

- título oficial;
- pergunta do drone;
- mapa da USP sem excesso de explicações;
- nome, orientador, IME-USP e FAPESP.

Meta de fala: 20 segundos.

#### Slide 2, dificuldade e método

- comparação visual entre pontos e regiões;
- matriz ordem fixa ou livre versus convexa ou não convexa;
- uma árvore pequena mostrando inserção, refinamento e poda;
- frase principal: o solver convexo fornece limites para controlar a busca combinatória.

Meta de fala: 55 segundos.

#### Slide 3, resultado e contribuição

- rota final na instância principal;
- dois ou três resultados quantitativos;
- explicação visual de `LB <= OPT <= UB` e fechamento do gap;
- conclusão: a arquitetura resolve conjuntamente escolhas geométricas e combinatórias e certifica otimalidade quando os limites coincidem.

Meta de fala: 45 segundos.

## Plano de execução em quatro ciclos

Cada ciclo deve terminar com um artefato utilizável. Ideias ambiciosas entram somente depois que o marco seguro do ciclo estiver concluído.

### Sequenciamento no tempo

O primeiro dia deve produzir uma versão completa de contingência. Os dias restantes melhoram essa versão, sem substituir um arquivo estável por trabalho incompleto.

| Momento | Entrega mínima ao final |
| --- | --- |
| Primeiras 2 horas | Regras, narrativa e afirmações quantitativas congeladas |
| Restante do primeiro dia | Pôster e três slides completos, mesmo com uma instância provisória |
| Segundo dia | Instância da USP e figuras definitivas inseridas |
| Terceiro dia | Revisão do orientador incorporada, dashboard e QR code decididos |
| Quarto dia | Revisão final, ensaio, exportação e cópias de segurança |

Os quatro resets disponíveis podem corresponder aos quatro ciclos abaixo. Nenhuma entrega obrigatória deve depender de ainda haver um reset: ao final de cada ciclo, salvar uma versão candidata completa e registrar o que falta. Usar o maior esforço de raciocínio nos ciclos 1 e 2, que concentram decisões científicas e validação. Usar os ciclos 3 e 4 para produção, revisão e acabamento.

### Ciclo 1, conteúdo e evidência

Objetivo: congelar o que será afirmado.

- [ ] Confirmar com a CPqI ou com a mensagem original se há limite formal de três slides.
- [ ] Confirmar o prazo interno de quatro dias e distingui-lo do prazo público de 28 de setembro.
- [ ] Registrar a versão exata do código usada nos resultados.
- [ ] Selecionar no máximo três resultados principais.
- [ ] Reexecutar ou validar os comandos que produzem esses resultados.
- [ ] Preparar uma tabela privada com instância, máquina, limite, tolerância e significado de cada métrica.
- [ ] Escrever a resposta curta para objetivo, método, virtudes, limitações, resultados e relevância.
- [ ] Definir a frase que conecta o resumo conservador ao avanço de ordem livre.

Marco seguro: uma página de conteúdo aprovada, mesmo sem design final.

### Ciclo 2, instância principal e figuras

Objetivo: obter a imagem central do material.

- [ ] Escolher de 8 a 12 institutos ou edifícios reconhecíveis da USP.
- [ ] Definir ponto inicial e ponto final coerentes com a narrativa do drone.
- [ ] Obter polígonos em projeção métrica adequada.
- [ ] Guardar a fonte e a licença dos dados geográficos.
- [ ] Criar a instância no formato do solver.
- [ ] Resolver com limites generosos e guardar ordem, caminho, limites e estatísticas.
- [ ] Validar geometricamente o caminho.
- [ ] Exportar uma figura vetorial e uma versão raster de alta resolução.
- [ ] Criar uma segunda figura simples mostrando decomposição convexa e poda.

Plano de contingência: se a instância real da USP atrasar ou produzir uma figura ruim, usar uma instância esquemática inspirada no campus, identificada explicitamente como ilustração.

Marco seguro: uma figura principal pronta para impressão, com legenda e resultado reproduzível.

### Ciclo 3, pôster e slides

Objetivo: produzir arquivos completos antes de polir recursos extras.

- [ ] Criar a primeira versão do pôster A0.
- [ ] Verificar legibilidade a 100%, a 25% e numa impressão A4 de teste.
- [ ] Criar os três slides em formato 16:9.
- [ ] Exportar os slides em PDF e conferir que nenhum significado depende de animação.
- [ ] Manter o título oficial exatamente igual ao resumo.
- [ ] Conferir nomes, afiliações, bolsa FAPESP e logotipos.
- [ ] Solicitar revisão do orientador com perguntas específicas sobre correção científica e prioridades.

Marco seguro: pôster e slides completos, corretos e entregáveis.

### Ciclo 4, experiência e ensaio

Objetivo: aumentar o impacto sem colocar a entrega em risco.

- [ ] Criar uma página pública simplificada para visitantes.
- [ ] Carregar automaticamente uma instância interessante.
- [ ] Exibir claramente rota, ordem, limite inferior, limite superior, gap e estado da certificação.
- [ ] Testar a página em celular, Wi-Fi e dados móveis.
- [ ] Gerar QR code com correção de erro e URL curta impressa abaixo.
- [ ] Ensaiar o pitch até ficar entre 1 minuto e 45 segundos e 1 minuto e 55 segundos.
- [ ] Gravar dois ensaios e revisar ritmo, clareza e postura.
- [ ] Simular perguntas técnicas com o orientador ou com um colega.
- [ ] Fazer uma revisão final separada de conteúdo, visual e conformidade.

Plano de contingência: remover o QR code se a página não estiver estável. O pôster precisa funcionar sozinho.

Marco seguro: apresentação ensaiada, PDFs finais e cópias de segurança.

## Priorização

### P0, obrigatório

- título oficial idêntico;
- afirmações verificadas;
- pôster A0 em PDF;
- slides em PDF;
- pitch de dois minutos;
- domínio das limitações e resultados;
- revisão do orientador.

### P1, alto impacto

- instância da USP;
- visualização da árvore e das podas;
- QR code para página móvel simplificada;
- material suplementar para perguntas.

### P2, somente se P0 e P1 estiverem estáveis

- vídeo;
- animação ao vivo;
- execução do solver durante o pitch;
- comparação ampla com solver externo;
- redesign completo do dashboard técnico.

## Gates de segurança

- Não usar no pôster nenhum número que não possa ser reproduzido e explicado.
- Não chamar uma solução interrompida de ótima.
- Não depender de internet, vídeo, animação ou execução ao vivo durante o pitch.
- Não fazer afirmações universais a partir de uma única suíte.
- Não deixar mudanças no solver depois do congelamento contaminarem silenciosamente as figuras ou os benchmarks.
- Não sacrificar legibilidade para incluir detalhes técnicos.
- Preservar uma versão final candidata antes de cada rodada de mudanças ambiciosas.

## Perguntas que o material deve permitir responder

- Qual é exatamente o problema resolvido?
- Por que visitar regiões é diferente de visitar pontos?
- Por que o caso convexo é importante?
- Como o limite inferior é obtido?
- Por que a poda não remove a solução ótima?
- O que torna o caso não convexo difícil?
- Como a ordem livre entra na árvore de busca?
- O que significa “exato” com tolerâncias numéricas?
- Quais instâncias ainda não são resolvidas dentro do limite?
- O que foi contribuição própria e o que veio da literatura?
- Como os resultados foram validados?
- Como o trabalho recente se relaciona ao resumo submetido?

## Checklist final

- [ ] `GabrielFreireUshijima.pdf`, pôster A0, abre corretamente.
- [ ] `GabrielFreireUshijima.pdf`, slides, abre corretamente no ambiente de apresentação.
- [ ] O número de páginas dos slides está confirmado.
- [ ] Fontes estão incorporadas nos PDFs.
- [ ] Figuras continuam nítidas com zoom.
- [ ] QR code foi testado em pelo menos dois celulares.
- [ ] Existe uma versão sem QR code caso a página saia do ar.
- [ ] Existe cópia local, na nuvem e em pendrive.
- [ ] O pitch cabe em dois minutos sem acelerar no final.
- [ ] As respostas sobre limitações, validação e contribuição estão ensaiadas.

## Fontes normativas consultadas

- `edital.pdf`, Edital do 34º SIICUSP, incluindo os Anexos I e II.
- `criterios.pdf`, formulário de apoio à avaliação.
- página do SIICUSP no IME-USP e chamada da primeira etapa do IME para 2026.
