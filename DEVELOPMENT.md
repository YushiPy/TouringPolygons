# Guia de desenvolvimento

Este é um monorepo de pesquisa. Código mantido, artefatos gerados, campanhas e
dependências externas têm fronteiras diferentes.

## Preparação

```bash
git submodule update --init --recursive
./scripts/install_dependencies.sh
```

O núcleo C++ usa CMake, C++23, Eigen, Boost e CGAL. Gurobi é opcional e serve
como baseline. O dashboard mantém seus ambientes Python e Node próprios.

## Entradas mantidas

- `CMakePresets.json`: presets dos solvers C++;
- `apps/benchmark-dashboard`: aplicação principal e WASM opcional;
- `benchmarks/tpp.py`: única CLI pública de geração e benchmarks;
- `scripts/sanity_check.sh`: validação ampla de um checkout;
- `scripts/verify_unordered.sh`: verificação focada no solver de ordem livre.

Os módulos em `benchmarks/_internal/` implementam a CLI e podem ser importados
por testes e apps, mas não são interfaces de usuário estáveis. Um fluxo novo
deve entrar como subcomando de `benchmarks/tpp.py`, reutilizando esses módulos,
em vez de criar mais um script executável.

## Fronteiras dos diretórios

- `packages/`: código mantido, testes locais dos pacotes e fixtures pequenas;
- `apps/`: aplicações; ferramentas exclusivas de uma aplicação ficam dentro
  dela;
- `benchmarks/suites/`: corpora canônicos rastreados;
- `benchmarks/campaigns/` e `benchmarks/results/`: trabalho local ignorado;
- `benchmarks/results-saved/`: campanhas publicadas com evidência completa;
- `third_party/`: código externo redistribuível, fixado por submódulo;
- `docs/algorithms/`: contratos e argumentos de correção duráveis;
- `docs/reports/`: publicações e relatórios, não dados soltos de benchmark.

Builds, ambientes virtuais, caches, módulos Node, WASM gerado e resultados de
campanhas permanecem fora do Git. Para tornar um caso uma regressão durável,
extraia apenas a entrada mínima para `benchmarks/suites/` e documente a origem.

## Dependências C++

```text
tpp_geometry -> tpp_convex -> optimal_convex_partition -> tpp_nonconvex
```

Comportamento compartilhado deve subir para o pacote responsável. Não copie
geometria ou solver convexo para pacotes consumidores.

## Terceiros e privacidade

O fork alemão está em `third_party/tspn-socg` e a coleção da Paula permanece
local e ignorada em `third_party/paula-tspn`. Regras de licença, reprodução e privacidade estão em
[`docs/third-party.md`](docs/third-party.md).

Gravações, transcrições, notas brutas e dados pessoais não pertencem ao Git.
As regras de publicação de reuniões estão em `docs/meetings/README.md`.

## Validação

Antes de integrar mudanças amplas:

```bash
./scripts/sanity_check.sh --no-install
cd apps/benchmark-dashboard && RUN_BROWSER=0 npm run test:all
cd apps/benchmark-dashboard && node wasm/test-intersections.mjs
```

Durante refactors das ferramentas de benchmark:

```bash
python3 -m compileall benchmarks/_internal benchmarks/tpp.py \
  apps/siicusp34/scripts
python3 benchmarks/tpp.py generate-suites
```

Consulte [`benchmarks/README.md`](benchmarks/README.md) para protocolos,
limites e interpretação dos resultados.
