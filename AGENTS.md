# Instruções para agentes e colaboradores

Este arquivo define as regras de trabalho para alterações neste repositório.
Ele se aplica à árvore inteira, salvo instruções mais específicas em um
subdiretório.

## Objetivo do projeto

O repositório implementa o Touring Polygons Problem (TPP) e suas primitivas.
As linhas mantidas são:

- TPP convexo com ordem fixa, incluindo a implementação legada ainda usada;
- TPP não convexo com ordem fixa;
- TPP não convexo com ordem livre;
- primitivas geométricas e ferramentas de validação, benchmark e visualização
  que sustentam esses solvers.

No código C++, `tpp_geometry` é a base geométrica, `tpp_convex` fornece o
solver convexo, `optimal_convex_partition` fornece a decomposição e
`tpp_nonconvex` integra essas peças para os casos não convexos. O C++ é a
referência para o núcleo algorítmico. O código Python de `packages/` é legado
ou experimental cercado: preserve-o quando ainda houver uso explícito, mas não
crie uma segunda implementação sem justificar a compatibilidade.

## Limites de manutenção

- `apps/benchmark-dashboard` é a aplicação principal mantida e também hospeda
  o solver WebAssembly opcional.
- `apps/siicusp34` é uma aplicação congelada de apresentação e é autocontida;
  sua cópia JavaScript do solver convexo de ordem fixa não deve ser confundida
  com uma segunda implementação de produção.
- `apps/visualizer-server` foi aposentada; não reintroduza dependências ou
  referências a ela.
- `experiments/` e `output/` não fazem parte do checkout mantido. Resultados
  gerados devem ficar nos diretórios ignorados documentados em
  `benchmarks/` ou em artefatos locais fora do Git.
- `.build*`, ambientes virtuais, `node_modules/`, WebAssembly gerado,
  campanhas, resultados de benchmark e arquivos locais de execução não devem
  ser adicionados ao commit.

## Repositórios de terceiros

`tspn-comparison/` e `paula-tspn/` são checkouts locais usados para comparação.
Eles não devem ser commitados como cópias, subtrees ou snapshots automáticos.
Preserve-os ignorados, registre revisões e mudanças de compatibilidade em
`benchmarks/patches/` e documente a origem e o procedimento de reprodução em
`docs/third-party.md` ou no relatório do benchmark correspondente.

## Documentação e pesquisa

- `docs/algorithms/` descreve contratos e algoritmos que ainda fazem parte do
  sistema.
- `docs/research/` preserva evidência histórica, protocolos e resultados
  experimentais. Não apague um relatório apenas por ser antigo: primeiro
  verifique se ele é duplicado, se suas conclusões foram incorporadas em uma
  fonte canônica e se os artefatos ainda são necessários para reprodução.
- Relatórios devem declarar data, escopo, formulação, tolerâncias, status de
  exatidão e limitações. “Ótimo” não deve ser usado para um resultado apenas
  factível ou interrompido por orçamento.
- Evite novos Markdown soltos na raiz. Prefira o índice apropriado em `docs/`,
  atualize README desatualizado quando a estrutura mudar e remova prompts ou
  planos somente quando a informação útil tiver sido preservada.

## Registros de reuniões e privacidade

Gravações, transcrições, notas brutas, `resumo.md` e informações administrativas
das reuniões são materiais privados e permanecem ignorados. Nunca adicione ao
Git nomes completos, contatos, passaporte, valores financeiros, processos
institucionais, informação médica ou outros detalhes pessoais.

Um resumo pode ser versionado somente depois de revisão manual e anonimização,
com o nome `docs/meetings/AAAA-MM-DD/resumo-publico.md`. Ele deve conter apenas
decisões técnicas, tarefas de pesquisa, resultados reproduzíveis e contexto
necessário ao projeto. A ausência de gravação no repositório não elimina o
risco de privacidade do texto derivado.

## Desenvolvimento e validação

Antes de alterar o núcleo, leia `README.md`, `DEVELOPMENT.md`, a arquitetura e
a documentação do algoritmo envolvido. Preserve mudanças locais não
relacionadas e confirme o estado do Git antes de operações destrutivas.

Validações mínimas, conforme o escopo:

```bash
./scripts/sanity_check.sh --no-install
cd apps/benchmark-dashboard && RUN_BROWSER=0 npm run test:all
cd apps/benchmark-dashboard && node wasm/test-intersections.mjs
```

Para mudanças C++, use o preset/documentação correspondente e execute os testes
nativos afetados. Para mudanças no dashboard, rode também Ruff, ESLint e os
testes JavaScript quando aplicável. Não trate binários gerados locais como
prova de que uma árvore nova é reproduzível.

## Fluxo Git

Refatorações amplas devem ocorrer em uma branch `codex/<descricao-curta>` e só
devem ser mescladas depois de uma revisão do diff, dos arquivos removidos e dos
testes. Não use `git reset --hard`, `git checkout --` ou remoções amplas sem
autorização explícita. Em caso de dúvida sobre apagar documentação ou dados,
prefira mover, marcar como histórico ou pedir confirmação.
