# Plano de trabalho: TPP com ordem livre

Atualizado em 5 de setembro de 2026. Este arquivo é um backlog para continuar o projeto com tarefas delimitadas e modelos adequados ao custo de cada etapa. A criação desta lista não inicia as tarefas.

## Estado de partida

Já existem B&B não convexo com ordem livre, oráculo convexo com verificação primal/dual e método auxiliar numérico, testes por enumeração, validação independente com Gurobi e integração com `apps/benchmark-dashboard`. A interface permite escolher ordem fixa ou livre, executar campanhas e inspecionar os resultados registrados.

Na comparação registrada de 60 instâncias, com uma thread e dois segundos por caso: nosso solver declarou 41 ótimos, contra 40 do externo; os tempos totais foram 47,80 s e 46,52 s. As tolerâncias diferem e o teste de extremos do adaptador externo falhou em 27 casos. Portanto, ainda não está demonstrada superioridade global.

`exact` significa fechamento do gap nas tolerâncias numéricas configuradas. Não significa certificação em aritmética racional ou intervalar. As APIs antigas de ordem fixa ainda têm problemas conhecidos em algumas sequências com interseções.

Leia primeiro [o contexto do projeto](../context.md) e [a documentação do algoritmo](algorithms/unordered-tpp.md). Os resultados acima são uma referência histórica, não uma previsão de novas execuções.

## Como usar os modelos

As recomendações abaixo são escolhas de engenharia para este projeto, não resultados de um benchmark entre modelos. High aumenta o esforço dentro de um modelo; não garante equivalência com outro modelo.

- **Terra Medium:** dashboard, scripts, execução de experimentos e organização dos resultados.
- **Sol Medium:** instrumentação e investigação com escopo conhecido.
- **Sol High:** implementação e depuração do núcleo do solver.
- **Astra Medium:** revisão matemática, novos bounds e dificuldades numéricas resistentes.
- **Sol Light/Low:** documentação e mudanças pequenas já especificadas.
- **Luna Light/Low:** extração e formatação determinística de resultados. Usar High apenas se o resultado em esforço baixo for insuficiente.

Para economizar, execute uma tarefa por vez, reutilize os artefatos existentes e aumente o modelo ou o esforço quando surgir uma dificuldade concreta. Não peça uma investigação indefinida de performance. Cada tarefa deve terminar com arquivos alterados, evidências, limitações e próximo passo registrados neste documento.

## Ordem recomendada

| Prioridade | Tarefa | Modelo inicial | Dependência |
| --- | --- | --- | --- |
| P0 | 1. Consolidar a referência reproduzível | Terra Medium | Nenhuma |
| P0 | 2. Medir os gargalos reais | Sol Medium | 1 |
| P0 | 3. Tornar a comparação mais justa | Sol High | 1 |
| P0 | 4. Revisar as garantias do B&B | Astra Medium | 1 |
| P1 | 5. Reduzir chamadas ao método auxiliar | Sol High | 2 e 4 |
| P1 | 6. Melhorar a busca e o incumbente | Sol High | 2 e 4 |
| P1 | 7. Fortalecer limites inferiores | Astra Medium | 2 e 4, se os bounds forem um gargalo |
| P1 | 8. Ampliar e repetir os benchmarks | Terra Medium | 3 e cada melhoria aceita |
| P1 | 9. Mostrar os gargalos no dashboard | Terra Medium | 2 |
| P1 | 10. Garantir procedência e atualização dos relatórios | Terra High | 1 |
| P1 | 11. Preparar material para o SIICUSP | Sol Medium | 3 e 4; atualizar após 8 |
| P2 | 12. Corrigir as APIs antigas de ordem fixa | Sol High | 4 |

P0 organiza a evidência e verifica correção antes de otimizar. P1 melhora o resultado e sua apresentação. P2 é importante, mas pode ficar fora da entrega de ordem livre se o prazo estiver apertado. As tarefas 5, 6 e 7 são alternativas guiadas pelo perfil, não uma obrigação de implementar todas.

## Tarefas detalhadas

### 1. Consolidar a referência reproduzível

- [x] Concluída

**Modelo:** Terra Medium. **Entrega:** manifesto da configuração e comandos reproduzíveis.

**Instruções:**

1. Leia a documentação e identifique os comandos realmente usados para compilar, validar e comparar. Confirme os caminhos dos artefatos locais, sem repetir experimentos já documentados apenas para redescobrir seu resultado.
2. Registre revisão Git, existência de alterações locais, compilador, flags, dependências, máquina, threads, tolerâncias e limites. Uma revisão Git sozinha não identifica um executável produzido com alterações não commitadas; registre também um identificador dos fontes ou do binário.
3. Preserve os relatórios históricos. Novas execuções devem usar diretórios próprios.
4. Documente qual ambiente Python executar em cada comando e quais verificações exigem Gurobi/licença.

**Aceite:** outra pessoa consegue identificar a configuração de cada resultado e executar os comandos sem adivinhar ambientes ou sobrescrever a referência.

### 2. Medir os gargalos reais

- [x] Concluída

**Modelo:** Sol Medium; subir para Sol High se a instrumentação exigir mudanças substanciais. **Entrega:** métricas no JSON e relatório de perfil por instância.

**Instruções:**

1. Instrumente `packages/nonconvex-tpp/cpp/src/solvers/unordered.cpp` e `packages/convex-tpp/cpp/src/solvers/certified.cpp`.
2. Separe tempo do solver geométrico, verificação do certificado, método auxiliar, decomposição, verificação de visitas, heurística inicial e manutenção da busca. Identifique quais tempos são inclusivos; não some categorias que se sobrepõem.
3. Registre chamadas, motivos de fallback e uso de precisão ampliada. Os contadores existentes não revelam a distribuição do tempo.
4. Propague as métricas pela CLI e pelos scripts. Compare uma execução instrumentada com a referência para detectar overhead relevante.
5. Execute a suíte de desenvolvimento e selecione casos representativos: rápidos, lentos resolvidos, timeouts e uso intenso de fallback.

**Aceite:** o relatório responde onde o tempo é gasto e propõe uma melhoria sustentada pelas medições. Instrumentação não altera a regra de poda nem os critérios numéricos.

### 3. Tornar a comparação com o TSPN mais justa

- [x] Concluída

**Modelo:** Sol High. **Entrega:** protocolo comparativo e nova rodada controlada.

**Instruções:**

1. Inspecione `tspn-comparison/benchmarks/run_comparison.py` e o solver externo para confirmar semântica de caminho, extremos fixos, tolerâncias geométricas e critério de otimalidade.
2. Investigue as 27 falhas de extremos com exemplos reproduzíveis. Determine se vêm da trajetória, da exportação ou do teste. Não remova as flags nem afrouxe a validação para aumentar a contagem de sucessos.
3. Exporte os caminhos externos e aplique o mesmo validador independente aos dois: extremos, visita a todos os polígonos e comprimento recalculado.
4. Iguale as tolerâncias quando as APIs permitirem. Quando não permitirem, registre as diferenças e separe os resultados correspondentes.
5. Use instâncias com hashes iguais, mesmo orçamento, mesma máquina e uma thread. Preserve separadamente o tempo de resolução e o custo de inicialização, quando medidos.

**Aceite:** o relatório distingue otimalidade declarada, viabilidade verificada e resultados não comparáveis. A nova configuração não é misturada silenciosamente com a rodada histórica.

### 4. Revisar as garantias do B&B

- [ ] Concluída

**Modelo:** Astra Medium. **Entrega:** revisão matemática curta e regressões para problemas encontrados.

**Instruções:**

1. Revise completude das inserções e do refinamento por peças, visitas incidentais e contabilização da fronteira durante interrupções.
2. Confira a derivação e a implementação do limite dual, margens numéricas e correção aplicada à remoção de vértices quase coincidentes.
3. Examine sobreposição, segmentos de comprimento zero, extremos internos, `start == target` e escalas de coordenadas muito diferentes.
4. Reutilize os testes por enumeração e a validação independente. A enumeração que reutiliza o mesmo oráculo não é uma verificação independente desse oráculo.
5. Para cada falha, produza um caso mínimo, uma correção e um teste que falhe antes dela. Documente as hipóteses que os testes não provam.

**Aceite:** não há falha conhecida de correção sem tratamento explícito; resultados inconclusivos preservam bounds e não declaram otimalidade. A documentação distingue argumento matemático de garantia numérica.

### 5. Reduzir chamadas ao método auxiliar

- [x] Concluída

**Modelo:** Sol High; revisão Astra Medium se mudar o certificado. **Entrega:** uma otimização isolada, com comparação antes/depois.

**Instruções:**

1. Use o perfil da tarefa 2 para selecionar o motivo mais frequente ou caro de fallback.
2. Separe falhas do caminho geométrico de dificuldades em construir direções duais, sobretudo em contatos coincidentes.
3. Avalie uma intervenção por vez: corrigir o caminho, melhorar a recuperação do certificado ou reaproveitar informação numérica compatível do nó pai.
4. Preserve a verificação primal/dual e o fallback quando necessários. Reduzir precisão ou relaxar tolerâncias não conta como ganho equivalente de algoritmo.
5. Meça tempo total, frequência e custo dos fallbacks, validade e gaps nas mesmas instâncias.

**Aceite:** melhoria mensurável sem perda de validade ou regressão numérica identificada. Se a hipótese falhar, registre o resultado negativo e não a incorpore por expectativa.

### 5A. Consolidar e modularizar a implementação

- [x] Concluída

**Entrega:** cleanup isolado antes dos próximos experimentos algorítmicos.

O oráculo certificado foi separado em fluxo, geometria/certificado e refinamento;
a geometria pura da busca sem ordem foi isolada; os runners Python passaram a usar
um protocolo compartilhado. Builds regeneráveis sem referência foram removidos,
mantendo resultados e baselines documentados. O aceite exige equivalência de
objetivos, bounds e estrutura da busca, além de ausência de regressão mensurável.

### 6. Melhorar a busca e o incumbente

- [ ] Concluída

**Modelo:** Sol High. **Entrega:** experimento controlado sobre uma política de busca.

**Instruções:**

1. Use nós, tamanho da fila, evolução do incumbente e dos bounds para distinguir dificuldade em achar uma boa solução de dificuldade em provar otimalidade.
2. Compare a política atual com variantes delimitadas: intervalo de descida, escolha da região não visitada ou melhoria da solução inicial. Não mude todas simultaneamente.
3. Para heurísticas, valide o caminho contra os polígonos originais antes de aceitar seu comprimento como limite superior.
4. Preserve todas as posições de inserção e peças necessárias à completude. Reordenar a exploração não autoriza descartar filhos sem limite válido.
5. Registre configurações e sementes. Separe instâncias usadas para escolher parâmetros das usadas para avaliar a melhoria.

**Aceite:** tabela de ablação com tempo, nós, memória/fila, ótimos e gaps. A política escolhida tem justificativa empírica e passa nos testes de interrupção.

### 7. Fortalecer limites inferiores

- [ ] Concluída

**Modelo:** Astra Medium para derivação; Sol High para implementar a proposta revisada. **Entrega:** bound demonstrado e avaliação de custo-benefício.

**Instruções:**

1. Comece apenas se o perfil indicar que bounds fracos contribuem para os casos difíceis.
2. Proponha um limite barato que considere regiões ainda não inseridas e os extremos fixos. Considere explicitamente sobreposições e visitas incidentais.
3. Derive a validade antes de usá-lo em podas. Não some dois limites válidos sem provar que a soma permanece válida; custos podem ser contados duas vezes.
4. Teste o candidato contra ótimos independentes de instâncias pequenas e verifique comportamento sob limites de execução.
5. Meça custo por nó e redução da árvore; um bound mais forte pode aumentar o tempo total.

**Aceite:** prova sob hipóteses explícitas, regressões relevantes e ganho medido. Se não houver ganho, mantenha a proposta como experimento documentado.

### 8. Ampliar e repetir os benchmarks

- [ ] Concluída

**Modelo:** Terra Medium; Luna Light/Low pode formatar tabelas a partir de agregados já verificados. **Entrega:** resultados e gráficos reproduzíveis.

**Instruções:**

1. Inclua famílias distintas: regiões disjuntas, sobrepostas, não convexas e geometrias reais disponíveis. Separe quantidade de polígonos de complexidade da decomposição.
2. Faça primeiro uma rodada curta para estimar o custo. Defina um orçamento total antes de ampliar instâncias, limites e repetições.
3. Alterne a ordem de execução dos solvers e repita as medições para estimar variação. Não execute concorrentes simultaneamente disputando a máquina.
4. Reporte tempo total, ótimos, viabilidade, gaps, timeouts e dispersão. Mostre o speedup nos casos resolvidos por ambos como estatística condicionada, sem estendê-lo à suíte inteira.
5. Use um conjunto separado para avaliar parâmetros ajustados na suíte de desenvolvimento. Gere gráficos científicos com ferramentas de plotting e preserve os dados de origem.

**Aceite:** conclusões são reproduzíveis e delimitam onde cada solver ganha. Nenhum caso difícil desaparece dos agregados sem explicação.

### 9. Mostrar os gargalos no dashboard

- [ ] Concluída

**Modelo:** Terra Medium; Sol Light/Low para ajustes pequenos de apresentação. **Entrega:** visualização das métricas da tarefa 2.

**Instruções:**

1. Amplie `apps/benchmark-dashboard/static/free-order-report.js` para mostrar decomposição do tempo, fallbacks e estatísticas da busca, quando presentes.
2. Permita ordenar ou filtrar casos por tempo, gap e uso do método auxiliar. Exponha configuração e procedência da execução.
3. Preserve compatibilidade com relatórios antigos: ausência de métrica deve aparecer como indisponível, nunca como zero.
4. Mantenha ordem fixa e livre identificadas. Atualize o CSV para incluir as novas métricas.
5. Valide no navegador carregamento, filtros, caminhos e alternância de ordem; use uma campanha pequena para testar a execução real.

**Aceite:** o usuário consegue localizar um caso lento e entender seu perfil sem abrir JSON. Os fluxos existentes continuam funcionando.

### 10. Garantir procedência e atualização dos relatórios

- [ ] Concluída

**Modelo:** Terra High. **Entrega:** cache e seleção de relatórios compatíveis com a execução atual.

**Instruções:**

1. Revise `benchmarks/scripts/free_order_campaign.py` e `apps/benchmark-dashboard/dashboard/dashboard_free_order.py`.
2. Verifique se a chave de reutilização inclui identificação dos solvers/fontes, instâncias, parâmetros e tolerâncias. Acrescente o que faltar para impedir reutilização após alterações relevantes.
3. Verifique como o dashboard escolhe o relatório mais recente. Se a campanha tiver sido editada, sinalize o resultado antigo ou deixe de apresentá-lo como resultado da geometria atual.
4. Preserve execuções históricas e permita identificar qual configuração gerou cada uma.
5. Teste mudança de geometria, mudança de solver/configuração, execução interrompida e tentativa de reutilização.

**Aceite:** um resultado antigo não se apresenta silenciosamente como medição do código ou da campanha atual; execuções compatíveis ainda podem ser reutilizadas.

### 11. Preparar material para o SIICUSP

- [ ] Concluída

**Modelo:** Sol Medium; Astra Medium apenas para revisar afirmações matemáticas controversas. **Entrega:** roteiro, figuras e texto de resultados.

**Instruções:**

1. Confira formato, duração e regras da apresentação nos materiais disponíveis. Se faltarem, escreva um roteiro modular e registre a informação pendente, sem inventar limites.
2. Organize a narrativa: problema, solvers de ordem fixa, busca da ordem, certificado, experimentos e limitações.
3. Produza um exemplo pequeno em que a ordem livre melhore o caminho e uma figura que mostre inserção e refinamento de um polígono não convexo.
4. Use apenas medições rastreáveis. Diferencie ganho por chamada convexa de ganho no algoritmo completo.
5. Prepare respostas para: por que é exato nas tolerâncias? Como trata interseções? O que ocorre no timeout? Quando perde para o externo? Qual é a contribuição em relação ao TSPN?

**Aceite:** cada afirmação de correção ou desempenho aponta para argumento, teste ou resultado. O roteiro cabe no tempo quando ele for conhecido. A apresentação não depende de executar uma instância difícil ao vivo.

### 12. Corrigir as APIs antigas de ordem fixa

- [ ] Concluída

**Modelo:** Sol High; Astra Medium se a correção afetar o argumento geométrico. **Entrega:** correção isolada e análise de impacto.

**Instruções:**

1. Localize os casos com interseções em que a API antiga retorna caminho inviável ou valor incorreto. Minimize-os e registre regressões.
2. Determine se o defeito está na geometria, recuperação do caminho ou pressupostos da API.
3. Corrija a causa quando possível. Avalie separadamente o custo de redirecionar chamadas ao oráculo certificado; não trate essa troca como gratuita.
4. Preserve interfaces ou documente explicitamente mudanças de contrato. Execute testes de ordem fixa e livre relevantes.

**Aceite:** casos conhecidos corrigidos, validade verificada e impacto no desempenho documentado. Não é necessário bloquear a apresentação de ordem livre por uma reformulação ampla dessas APIs.

## Instrução reutilizável para iniciar uma tarefa

```text
Execute somente a tarefa [N] de docs/plano-de-trabalho-tpp.md.
Leia o estado documentado e os arquivos relevantes antes de alterar código.
Preserve alterações locais e resultados históricos.
Não enfraqueça tolerâncias nem critérios de validação para obter melhores números.
Use saídas compactas de ferramentas e rode verificações proporcionais à mudança.
Conclua pelos critérios de aceite. Se uma hipótese falhar, registre a evidência.
Ao terminar, atualize a tarefa com arquivos, comandos, resultados e pendências.
Não inicie automaticamente a próxima tarefa.
```

## Registro de conclusão

Preencha após cada tarefa. Marque a caixa somente quando o aceite estiver atendido.

| Tarefa | Data | Modelo | Arquivos/artefatos | Validação e resultado | Pendências |
| --- | --- | --- | --- | --- | --- |
| 1 | 2026-09-05 | Terra Medium | `docs/unordered-reference-2026-09-05.md`; manifesto dos quatro artefatos históricos, executável e fontes | Conferidos revisão, diff local, hashes SHA-256 dos artefatos e binário, ambientes Python, dependências e comandos sem reexecutar a rodada histórica. A referência é identificável e novas rodadas usam diretórios próprios. | 8 depende da tarefa 3; 9 depende da tarefa 2. A equivalência byte a byte entre recompilação nova e binário histórico não foi presumida. |
| 2 | 2026-09-05 | GPT-5 | `packages/nonconvex-tpp/cpp/{include/tpp/nonconvex/unordered.h,src/solvers/unordered.cpp,src/main-unordered.cpp,src/main-unordered_tests.cpp}`; `packages/convex-tpp/cpp/{include/tpp/convex/certified.h,src/solvers/certified.cpp}`; `benchmarks/scripts/{unordered_benchmark.py,summarize_unordered.py,unordered_validation.py}`; `docs/unordered-profile-2026-09-05.md`; resultados em `benchmarks/results/unordered/task-2-profile-20260905/` | Build isolado em `.build/unordered-instrumented`; 86 casos exaustivos e 344 interrupções passaram. Cinco casos representativos foram medidos contra o binário histórico. O oráculo consumiu 92,0% a 98,7% nos quatro casos não triviais, e o fallback, 88,7% a 94,3%; o motivo dominante foi caminho geométrico inválido. Overhead observado até 1,3%, sem mudança de objetivos, bounds ou chamadas nos casos concluídos. | A medição é uma execução por binário e não estima dispersão. Próximo passo sugerido: tarefa 5, focada na recuperação do caminho que dispara fallback; investigar precisão ampliada separadamente nos casos 3, 12, 55 e 59. |
| 3 | 2026-09-05 | GPT-5 | `tspn-comparison/benchmarks/run_comparison.py`; `benchmarks/scripts/{unordered_validation.py,unordered_benchmark.py,summarize_unordered.py,free_order_campaign.py}`; `docs/unordered-comparison-protocol-2026-09-05.md`; resultados em `benchmarks/results/unordered/task-3-comparison-20260905/` | Os 27 índices históricos reproduziram 27/27 falhas de extremos; 26/27 também falharam visita a `1e-7`, e encaixar extremos recuperou apenas 1/27. Rodada controlada de seis hashes, uma thread e 2 s/caso: próprio 5 ótimos declarados e 6 caminhos válidos; externo estrito 0 ótimos declarados, 0 caminhos brutos válidos e 2 válidos após encaixe. Tempos de resolução e processo foram preservados separadamente. | O externo não expõe o mesmo critério misto de gap e sua trajetória SOCP mantém resíduos mesmo com tolerância de cobertura `1e-8`; não houve caso simultaneamente ótimo e validado para estimar speedup justo. Próximo passo: parametrização numérica ou reparação certificada no externo, sem contar simples encaixe como sucesso. |
| 5 | 2026-09-05 | GPT-5 | `packages/convex-tpp/cpp/{include/tpp/convex/certified.h,src/solvers/certified.cpp}`; `packages/nonconvex-tpp/cpp/{include/tpp/nonconvex/unordered.h,src/solvers/unordered.cpp,src/main-unordered.cpp,src/main-unordered_tests.cpp}`; `benchmarks/scripts/summarize_unordered.py`; `docs/{algorithms/unordered-tpp.md,unordered-performance-2026-09-05.md}`; resultados em `benchmarks/results/unordered/task-5-{warm-start,contact-repair}-20260905/` | Reparação interior pequena, seguida pelas mesmas verificações primal/dual, reduziu o caso 37 de 1,757 s para 0,623 s e tornou o caso 9 ótimo em 1,613 s, antes timeout. Casos 12 e 55 melhoraram 5,9% em tempo e 7,3% em chamadas; 30/30 caminhos válidos. A suíte recompilada passou 86 casos exaustivos e 344 verificações interrompidas. O warm start testado separadamente regrediu e foi removido. | Medição restrita a cinco casos de desenvolvimento, três repetições e 2 s por caso. Próximo passo sugerido: tarefa 6 em conjunto separado; não iniciada. |
| 5A | 2026-09-05 | GPT-5 | `packages/convex-tpp/cpp/{CMakeLists.txt,src/solvers/certified{,_geometry,_refinement}.cpp,src/solvers/certified_internal.h}`; `packages/nonconvex-tpp/cpp/{CMakeLists.txt,src/solvers/unordered{,_geometry}.cpp,src/solvers/unordered_geometry.h}`; `benchmarks/scripts/{unordered_runner.py,unordered_benchmark.py,free_order_campaign.py}`; `docs/{algorithms/unordered-tpp.md,unordered-cleanup-2026-09-05.md}`; resultados em `benchmarks/results/unordered/task-cleanup-20260905/` | Três repetições alternadas em cinco casos mantiveram objetivos, bounds e estrutura da busca nos casos concluídos; variação de -1,3% a +1,6% nos casos não triviais e -0,8% de throughput no único timeout. 30/30 caminhos válidos; 86 casos exaustivos e 344 verificações interrompidas passaram. Sete builds regeneráveis foram removidos, reduzindo `.build` para 22 MB, sem apagar resultados ou baselines documentados. | O executável histórico `a378f45b...034a` já não estava em `.build/unordered` no início do cleanup; seu hash e resultados permanecem documentados. Próximo passo: tarefa 6, não iniciada. |

Referência para a escolha de modelos: [orientação oficial do Codex](https://learn.chatgpt.com/pt-BR/docs/models). Atribuições específicas ao TPP são recomendações para este projeto e devem ser ajustadas conforme os resultados observados.
