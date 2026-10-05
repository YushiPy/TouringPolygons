# Implementação interna de benchmarks

Este diretório implementa [`../tpp.py`](../tpp.py). Seus módulos são usados
pela CLI, pelos testes e pelo dashboard, mas não constituem comandos públicos
estáveis.

Ao adicionar um fluxo de benchmark:

1. reutilize `benchmark_cases.py`, `unordered_runner.py` e
   `unordered_validation.py` para leitura, execução e validação;
2. mantenha `main(argv)` separado da lógica importável;
3. exponha o fluxo como subcomando de `benchmarks/tpp.py`;
4. obtenha caminhos de `workspace.py` (`campaign_path`, `run_path`) e compile
   binários com `native_build.ensure_tool`; envolva a execução em
   `workspace.recorded_run` para gravar `run.json`;
5. preserve uma campanha somente em `benchmarks/results-saved/`, junto com
   seus dados, análise e proveniência.

As famílias internas são:

- campanhas: `create_synthetic_campaign.py`, `run_generated.py` e
  `free_order_campaign.py`;
- suites: `generate_algorithm_suites.py`, `build_algorithm_suites.py` e os
  geradores canônicos de ordem livre;
- execução e validação: `benchmark_cases.py`, `unordered_*` e `bench.py`;
- comparação externa: `tspn_run_comparison.py`, `tspn_oracle_backends.py` e
  `run_fekete.py`;

Na comparação com Fekete, `--time-limit -1` significa execução sem limite de
tempo. O watchdog do processo pai é desativado nesse modo e o valor `-1` é
mantido nos manifestos e resultados; a conversão para o limite inteiro nativo
é feita somente dentro do worker, para compatibilidade com bindings antigos.
- infraestrutura: `workspace.py` (caminhos, proveniência, migração),
  `native_build.py` (alvos nativos e dependências), `jobs.py` e `remote.py`;
- conversão: `convert_instances.py`, `convert_tspn_native_instances.py` e
  `normalize_polygon_orientation.py`.

Ferramentas exclusivas do SIICUSP ficam em `apps/siicusp34/scripts/`. A análise
da campanha alemã preservada fica em
`benchmarks/results-saved/fekete-comparison/`.
