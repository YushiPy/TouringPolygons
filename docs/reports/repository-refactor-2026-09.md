# Refatoração e poda do repositório — setembro de 2026

Este relatório registra a poda feita na branch `codex/repository-refactor` e o
estado deixado para uma revisão posterior. A branch ainda não foi commitada nem
mesclada.

## Objetivo

Reduzir fragmentos de tentativas anteriores sem perder os solvers que ainda têm
uso, separar código de produção de artefatos de pesquisa e deixar o caminho do
evento SIICUSP independente dos visualizadores antigos.

## O que foi feito

- Mantivemos explícitos os quatro eixos algorítmicos relevantes: primitivas
  geométricas, TPP convexo de ordem fixa (incluindo o legado), TPP não convexo
  de ordem fixa e TPP não convexo de ordem livre.
- Removemos do checkout mantido `experiments/`, `output/` e scripts de benchmark
  que não tinham chamada atual. Os resultados e campanhas gerados continuam
  sendo tratados como artefatos locais/ignorados.
- Aposentamos o código rastreado de `apps/visualizer-server/`. A implementação
  WASM e os testes que ainda tinham valor foram movidos para
  `apps/benchmark-dashboard/wasm/`.
- Removemos `apps/visualizer-local/`. Antes da remoção, verificamos que ele era
  a origem histórica do solver JavaScript usado como referência no antigo
  material do evento; ele não era importado diretamente pelo SIICUSP.
- Tornamos `apps/siicusp34/` autocontido:
  - `tpp-solver.js` contém a cópia congelada da primitiva TPP convexa de ordem
    fixa necessária pelos três desafios;
  - `tpp-vector2.js` é sua dependência local mínima;
  - `app.js` importa esse solver e calcula a rota escolhida no navegador, em vez
    de apenas procurar uma rota pré-calculada nos dados;
  - os dados pré-calculados continuam sendo usados como referência/certificação
    visual do desafio e para a experiência pública registrada;
  - `test-solver.mjs` reexecuta todas as 24 ordens, as 27 escolhas do desafio de
    peças e as 1.944 combinações do desafio combinado.
- Migramos o editor offline para `apps/benchmark-dashboard/offline-editor/` e
  mantivemos o mapa de último passo com o solver WASM para instâncias convexas
  disjuntas de ordem fixa. O dashboard não recebeu um solver JavaScript novo.
- Mantivemos o código legado que ainda tem uso. O antigo tutorial HTML do
  visualizador foi removido; uma limpeza posterior transferiu contratos e
  contraexemplos duráveis para `docs/algorithms/` e passou a manter análises de
  benchmark junto das campanhas preservadas em `benchmarks/results-saved/`.
- Criamos `AGENTS.md`, atualizamos a arquitetura e os READMEs, e documentamos a
  política para repositórios de terceiros, reuniões e artefatos gerados.

## Por que o SIICUSP precisava da cópia local

A inspeção do HTML mostrou que a página carregava apenas os três arquivos de
dados e `app.js`; não havia import de `visualizer-local`. O app apresentava
soluções embutidas e a frase “compare com o solver”, mas não executava um solver
no navegador. Isso era frágil: apagar o visualizador antigo deixaria a
experiência sem a implementação JavaScript que os desafios pressupunham.

A solução adotada é deliberadamente limitada ao evento. Ela não é uma nova
implementação de produção, não executa branch-and-bound de ordem livre e não
substitui os solvers C++. É uma cópia congelada, testada contra os dados
pedagógicos embutidos, para que a pasta `apps/siicusp34/` possa ser publicada
ou arquivada sem carregar outro app do monorepo.

## Privacidade e reuniões

Resumos de reuniões podem ser versionados somente depois de revisão manual e
anonimização, usando `docs/meetings/AAAA-MM-DD/resumo-publico.md`. Gravações,
transcrições e notas brutas permanecem fora do Git. O fato de um resumo não
conter áudio não elimina possíveis dados pessoais, então decisões técnicas e
resultados reproduzíveis devem ser separados de nomes, contatos e contexto
administrativo.

## Validação realizada

- O teste local do solver do evento passou: 4 testes, cobrindo 24 + 27 + 1.944
  combinações, além dos casos vazio e separado.
- A suíte completa do dashboard passou após a migração do SIICUSP, incluindo o
  smoke test browser e a regressão nativa:

  ```bash
  RUN_BROWSER=1 npm --prefix apps/benchmark-dashboard run test:all
  ```

- A suíte inclui lint/format Python, sintaxe e lint JavaScript, testes do
  editor, smoke test de browser, regressão nativa convexa e `git diff --check`.

## O que ainda falta

1. Revisar o diff completo e fazer um commit coeso da branch. Em especial,
   conferir se cada remoção em `experiments/`, `output/` e nas antigas ferramentas de benchmark
   e `apps/visualizer-server/` está coberta por histórico ou documentação.
2. Revisar os artefatos ignorados que ainda podem existir localmente sob
   `apps/visualizer-server/` — banco local, ambientes virtuais, caches e
   imagens. O código rastreado foi removido, mas esta poda não deve apagar
   dados locais sem uma decisão explícita sobre recuperabilidade.
3. A interação de linha de vértices do antigo visualizador não foi migrada.
   Ela foi reconhecida como útil, mas depende de uma decisão de produto sobre
   permitir novamente a inserção de vértices; o editor offline atual não deve
   ganhar essa função por acidente.
4. A política de terceiros foi resolvida posteriormente: `third_party/paula-tspn/` permanece
   local por não haver autorização de redistribuição, enquanto o solver alemão
   modificado passou a ser um fork MIT fixado como submódulo.
5. Fazer uma revisão de conteúdo do evento SIICUSP, inclusive o tamanho dos
   dados estáticos e a diferença entre resultados registrados e resultados
   calculados ao vivo. O evento está funcional, mas ainda é uma publicação
   congelada, não o front-end principal do dashboard.
6. Repetir `scripts/sanity_check.sh --no-install` em uma máquina com as
   dependências nativas disponíveis. A falha anterior por ausência de
   `libomp`, Eigen e Boost é um problema de ambiente, não uma evidência de
   falha desta poda.
