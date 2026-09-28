# Verificações da entrega TSPN

Testes executados sobre o código desta campanha:

| Verificação | Resultado |
|---|---|
| `tpp-tspn-tests` | 19 casos por enumeração exaustiva, 76 buscas interrompidas, decomposição não convexa e 240 referências arbitrárias para limites duais: passaram |
| `tpp-unordered-tests` | 86 casos por enumeração exaustiva e 344 buscas interrompidas, mais regressões do oráculo: passaram |
| `main-cycle_tests` | 67 comparações disjuntas racional/double e 96 casos semeados com interseções, além das regressões determinísticas: passaram |
| `main-cycle_certificate_tests` | 1040 pares de raios com links nulos, além das regressões determinísticas: passaram |
| `python3 -m unittest discover -s apps/benchmark-dashboard/tests -p test_benchmark_tools.py` | 10 testes: passaram |
| `node wasm/test-intersections.mjs` | 7 verificações: passaram |
| `./scripts/sanity_check.sh --no-install --threads 2` | completou as suítes convexas e o smoke benchmark não convexo de 30 instâncias; saída em `sanity-check.txt` |
| `git diff --check` | passou |
| `UV_OFFLINE=1 RUN_BROWSER=0 npm run test:all` | bloqueado antes dos testes: `ruff==0.16.5` ausente do cache local |

O modo offline evita repetir downloads de dependências indisponíveis nesta
sessão. A suíte completa do dashboard não foi validada; seu log de falha foi
preservado. O teste novo do validador independente foi executado diretamente.

Builds C++ usaram os diretórios externos `/private/tmp/tpp-tspn-build` e
`/private/tmp/tpp-cycle-build`; o benchmark construiu sua própria cópia em
`/private/tmp/tpp-tspn-comparison-build`. Os arquivos `*-tests.txt`,
`unordered-regression.txt`, `wasm-intersections.txt` e `dashboard-tests.txt`
guardam as saídas. Os hashes em `config.json` foram conferidos contra os
arquivos finais e coincidem. O submódulo externo permaneceu limpo.
