# Prompt para revisão independente da refatoração

Você é uma pessoa engenheira sênior revisando uma refatoração grande de um
monorepo de pesquisa em geometria computacional. Faça uma revisão crítica da
branch `codex/repository-refactor` do repositório TouringPolygons. Não assuma
que uma remoção é correta apenas porque reduziu o número de arquivos.

## Contexto e objetivos

O repositório mantém:

- primitivas geométricas;
- TPP convexo de ordem fixa, inclusive uma implementação legada ainda usada;
- TPP não convexo de ordem fixa;
- TPP não convexo de ordem livre;
- benchmarks, documentação de pesquisa e um dashboard local.

A refatoração removeu experimentos e saídas geradas antigas, aposentou o
`apps/visualizer-server/`, removeu o `apps/visualizer-local/` e tornou o app
estático do SIICUSP autocontido. O SIICUSP agora possui uma cópia local e
congelada do solver JavaScript convexo de ordem fixa em
`apps/siicusp34/tpp-solver.js`, com `tpp-vector2.js` como dependência. O
`apps/siicusp34/app.js` deve executar essa cópia para calcular as escolhas dos
desafios; os dados embutidos continuam sendo referência da publicação.

O dashboard deve continuar usando o solver C++/WASM quando apropriado. Não
introduza um solver JavaScript geral no dashboard só para compartilhar código
com o evento.

## Checklist de revisão

1. Leia `AGENTS.md`, `README.md`, `DEVELOPMENT.md`,
   `docs/architecture.md`, `docs/reports/repository-refactor-2026-09.md` e
   `docs/third-party.md`.
2. Inspecione `git diff --stat`, `git diff --summary`, todos os arquivos
   removidos e todos os arquivos novos. Procure referências quebradas a apps,
   scripts, imports, rotas, comandos e links.
3. Verifique se o SIICUSP abre como uma pasta estática isolada: os únicos
   recursos de solver devem ser os arquivos ao lado dele, sem import de
   `visualizer-local`, `visualizer-server`, backend ou WASM.
4. Verifique que o SIICUSP realmente usa `tpp-solver.js` ao comparar desafios,
   e não apenas contém o arquivo sem chamá-lo. Confira os casos vazio, múltiplas
   ordens, peças convexas e todas as combinações publicadas.
5. Verifique o dashboard offline e o mapa de último passo: disjunção estrita,
   polígonos tocando, sobreposição, polígonos não convexos, degenerados,
   instância vazia, falha de WASM e fallback visual. Confirme que nenhum solver
   JS geral foi reintroduzido.
6. Analise se algo removido de `experiments/`, `output/`, `benchmarks/scripts/`
   ou `apps/visualizer-server/` ainda é chamado por CI, CMake, scripts,
   documentação ou workflows. Recomendação de exclusão deve citar evidência.
7. Confira se `docs/algorithms/`, `docs/reports/` e as campanhas preservadas em
   `benchmarks/results-saved/` ficaram navegáveis e se os READMEs não prometem
   apps que não existem. Resultados de benchmark devem manter dados e análise
   juntos; argumentos duráveis de correção devem estar em `docs/algorithms/`.
8. Revise a política de `tspn-comparison/` e `paula-tspn/`: eles devem ser
   checkouts locais ignorados, com origem, SHA e patches de compatibilidade
   registrados, nunca cópias silenciosas commitadas.
9. Revise privacidade: gravações e transcrições não devem entrar no Git; um
   `resumo-publico.md` só é aceitável após anonimização e remoção de detalhes
   administrativos desnecessários.
10. Execute, quando as dependências estiverem disponíveis:

    ```bash
    node --test apps/siicusp34/test-solver.mjs
    RUN_BROWSER=1 npm --prefix apps/benchmark-dashboard run test:all
    git diff --check
    ```

## Formato da resposta

Não faça alterações nem commit antes de apresentar os achados. Classifique cada
problema como P0 (quebra ou perda de dados), P1 (regressão funcional), P2
(manutenção/documentação) ou P3 (melhoria opcional). Para cada item, informe:

- arquivo e linha ou símbolo relevante;
- evidência observada;
- impacto;
- correção recomendada;
- teste que comprovaria a correção.

Ao final, forneça: (a) veredito sobre se a branch está pronta para merge, (b)
comandos que passaram ou falharam e por quê, (c) lista de riscos que dependem
de decisão do mantenedor e (d) uma ordem curta de trabalho para a próxima
iteração. Separe fatos verificados de hipóteses e não trate arquivos gerados
locais ignorados como parte do commit sem confirmar sua origem.
