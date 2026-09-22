# Instruções para agentes

Estas regras se aplicam a todo o repositório. Antes de alterar o núcleo, leia
`README.md`, `DEVELOPMENT.md`, `docs/architecture.md` e a documentação do
algoritmo envolvido.

## Regras não negociáveis

- O C++ é a referência algorítmica. Preserve o fluxo
  `tpp_geometry -> tpp_convex -> optimal_convex_partition -> tpp_nonconvex` e
  não crie implementações paralelas sem uma exigência explícita de
  compatibilidade.
- `apps/benchmark-dashboard` é a aplicação mantida; `apps/siicusp34` é uma
  publicação congelada. Não reintroduza `apps/visualizer-server` no código
  ativo.
- `benchmarks/tpp.py` é a única CLI pública de benchmark. Reutilize módulos em
  `benchmarks/_internal/` em vez de criar scripts soltos.
- Campanhas e resultados gerados são locais. Só preserve uma campanha em
  `benchmarks/results-saved/` com entradas, dados brutos, análise, configuração
  e proveniência juntas.
- Não adicione builds, ambientes, caches, `node_modules`, WASM gerado,
  campanhas locais ou resultados ad hoc.

## Terceiros e privacidade

- O fork alemão é o submódulo `third_party/tspn-socg`. Não adicione ao
  submódulo ambientes, builds, licenças comerciais ou resultados.
- `third_party/paula-tspn/` é material local sem autorização explícita de redistribuição;
  mantenha-o ignorado.
- Nunca versione gravações, transcrições, notas brutas ou dados pessoais.
  Resumos de reunião só podem ser publicados conforme
  `docs/meetings/README.md`.

## Trabalho seguro

- Preserve mudanças locais não relacionadas e examine o estado do Git antes
  de operações destrutivas.
- Refatorações amplas usam branch `codex/<descrição>` e exigem revisão do diff,
  dos arquivos removidos e dos testes.
- Análises de benchmark pertencem à campanha; conclusões duráveis sobre
  contratos e correção pertencem a `docs/algorithms/`.
- Relatórios devem declarar formulação, tolerâncias, status de exatidão e
  limitações. Não chame de ótimo um resultado apenas factível ou limitado por
  orçamento.

## Validação mínima

```bash
./scripts/sanity_check.sh --no-install
cd apps/benchmark-dashboard && RUN_BROWSER=0 npm run test:all
cd apps/benchmark-dashboard && node wasm/test-intersections.mjs
```

Para uma mudança menor, execute ao menos os testes diretamente afetados e
registre claramente qualquer verificação que não pôde ser executada.
