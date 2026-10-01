# Benchmarks

`benchmarks/tpp.py` é a única interface pública para geração, execução,
conversão e comparação. Os módulos de `benchmarks/_internal/` são detalhes de
implementação importáveis, não uma coleção de comandos independentes.

## Organização

- `tpp.py`: CLI estável;
- `_internal/`: implementação da CLI;
- `suites/`: corpora canônicos rastreados e suites derivadas ignoradas;
- `campaigns/`: entradas e execuções locais reproduzíveis;
- `results/`: saídas locais e descartáveis;
- `results-saved/`: campanhas deliberadamente preservadas com dados e análise.

Use `python3 benchmarks/tpp.py --help` e acrescente `--help` após um subcomando
para consultar todos os parâmetros.

## Campanha sintética

```bash
python3 benchmarks/tpp.py create smoke \
  --vertices 8 --polygons 20 --instances 100 --shape star
python3 benchmarks/tpp.py run smoke \
  --threads 8 --max-calls 1000000 --max-seconds 30 --timeout 3600
python3 benchmarks/tpp.py status smoke
```

A campanha fica em `benchmarks/campaigns/smoke/` com manifesto, entradas,
preview e resultados. Execuções concluídas são reutilizadas ao retomar.

Para gerar uma matriz baseada em OpenStreetMap:

```bash
python3 benchmarks/tpp.py generate-matrix sao-paulo \
  packages/instance-generation/regions/sao-paulo.osm.pbf \
  --instances 100 --sample-size 40 --seed 42
```

## Suites algorítmicas

As suites `algorithm-dev-v1.bin` e `canonical-v1.bin` são derivadas do corpus
rastreado `suites/nonconvex/test_cases.bin`:

```bash
python3 benchmarks/tpp.py generate-suites
python3 benchmarks/tpp.py benchmark \
  --suite benchmarks/suites/algorithm-dev-v1.bin
```

Para reconstruir o corpus a partir do arquivo alemão fixado no submódulo:

```bash
python3 benchmarks/tpp.py convert-german
```

## Ordem livre

Uma campanha completa usa:

```bash
python3 benchmarks/tpp.py free-order NOME_DA_CAMPANHA --help
```

Para comparar os dois solvers com oito threads dentro de cada instância, sem
executar instâncias diferentes em paralelo, a campanha aceita:

```bash
python3 benchmarks/tpp.py free-order german-instances \
  --solver unordered --solver tspn --workers 1 --threads-per-instance 8 \
  --max-seconds 21600 --max-calls 10000000 \
  --absolute-gap 0 --relative-gap 0.001 --eps 0.001 \
  --sampled-perimeter-initial --convex-initial-refinement --bidirectional-initial
```

`--workers` controla instâncias simultâneas; `--threads-per-instance` controla
threads dentro de cada solver. Campanhas com a mesma configuração retomam os
casos já registrados e guardam `report.json`, os dados externos brutos e
`comparison.md` em `benchmarks/campaigns/<nome>/results/free-order/`.

Para medir o efeito das threads, compare runs de uma thread e multithread dos
dois solvers. A análise pareia instâncias pelo SHA-256, grava `comparison.csv`,
`summary.md` e `manifest.json`, e resume speedups apenas quando ambos os runs
fecharam a tolerância. Os argumentos aceitam CSV, `report.json` ou diretório de
campanha; o diretório da campanha multithread pode ser usado enquanto o CSV do
Fekete ainda está sendo preenchido.

```bash
python3 benchmarks/tpp.py compare-threads \
  --ours-single benchmarks/results/free-order-gap-6h/runs.csv \
  --ours-single-variant fekete_gap \
  --ours-multi benchmarks/campaigns/german-instances/results/free-order/ID_DA_RUN \
  --fekete-single benchmarks/results-saved/german-comparison/fekete.csv \
  --fekete-multi benchmarks/campaigns/german-instances/results/free-order/ID_DA_RUN \
  --output benchmarks/results/thread-scaling/german
```

O speedup é `tempo(1 thread) / tempo(multithread)`. Diferenças detectáveis de
limite, gap e workers aparecem no relatório: uma comparação histórica com
configurações diferentes é indicativa, não uma medição causal do ganho de
threads.

Para executar diretamente uma suite binária:

```bash
python3 benchmarks/tpp.py free-order-run \
  --suite benchmarks/suites/algorithm-dev-v1.bin \
  --solver .build/unordered/tpp \
  --seconds 3 --max-calls 10000000 --workers 4 \
  --output benchmarks/results/free-order-dev.jsonl
```

Para medir o custo de provar otimalidade quando o incumbente inicial já é bom,
`free-order-run` aceita `--initial-paths` com um CSV separado por `;` contendo
`case_index`, `sha256` e `path` (JSON de pontos). O hash de cada instância é
conferido antes de executar. Exemplo com os caminhos preservados da campanha alemã:

```bash
python3 benchmarks/tpp.py free-order-run \
  --suite benchmarks/results-saved/german-comparison/instances.bin \
  --initial-paths benchmarks/results-saved/german-comparison/ours.csv \
  --solver .build/unordered/tpp --seconds 21600 --max-calls 100000000 --workers 1 \
  --output benchmarks/results/german-good-initial.jsonl
```

Repita sem `--initial-paths`, com os mesmos limites e máquina, gravando em outro
JSONL. O arquivo `ours.csv` é lido apenas nas colunas de identificação e caminho.

Os caminhos entram apenas como soluções factíveis iniciais. `exact` só é verdadeiro
quando o solver fecha o gap com os limites inferiores. Compare `profile.search_seconds` e
`calls` com uma rodada padrão sob a mesma configuração. O tempo de leitura do CSV,
validação do caminho, preparação da instância e processo não desaparece numa
aplicação real; por isso, a diferença entre rodadas é um limite otimista para o
ganho obtido ao melhorar a heurística. Os tempos dependem da máquina e da carga.

Para uma comparação pareada de binários próprios:

```bash
python3 benchmarks/tpp.py free-order-ablation \
  --suite benchmarks/suites/algorithm-dev-v1.bin \
  --solver baseline=.build/unordered-baseline/tpp \
  --solver candidate=.build/unordered-candidate/tpp \
  --seconds 3 --repeats 3 \
  --output benchmarks/results/free-order-comparison.jsonl
```

Use `--resume` with the same command to reuse exact solver/case results and rerun
only pairs that did not finish with an optimality proof. The campaign metadata
must match the existing output; each resume keeps the latest row per pair and
prints a summary over all preserved results.

Para comparar a tolerância atual do solver com a tolerância equivalente ao
critério de gap de Fekete (`UB <= 1.001 * LB`):

```bash
python3 benchmarks/tpp.py compare-gaps \
  --time-limit 1 --workers 8 --case 0
```

A campanha fica em `benchmarks/results/free-order-gap-comparison/`. Ela guarda
checkpoints por instância e configuração em cada conclusão. Repetir o comando
retoma o trabalho; aumentar `--time-limit` reexecuta somente as instâncias que
ainda não fecharam o gap no limite anterior. Cada execução do solver usa uma
thread; `--workers` controla a concorrência entre instâncias. `runs.csv` contém
todas as tentativas, `comparison.csv` resume o resultado mais recente por caso,
e `summary.md` compara as duas configurações. Use `--output` para separar outras
campanhas ou suítes.

Os comandos `generate-free-order-canon`, `summarize-free-order` e
`free-order-metamorphic` cobrem, respectivamente, a campanha canônica, a
comparação pareada de resultados e os testes metamórficos.

## Solver alemão

O fork fixado em `third_party/tspn-socg` é uma dependência de comparação, não
um pacote do projeto. Prepare o ambiente dele conforme
[`docs/third-party.md`](../docs/third-party.md) e execute:

```bash
python3 benchmarks/tpp.py compare-external --help
python3 benchmarks/tpp.py compare-oracles --help
```

A campanha longa e retomável usa:

```bash
python3 benchmarks/tpp.py run-fekete --workers 8
```

Ela grava checkpoints em `benchmarks/results/fekete-free-order-6h/`. No macOS,
uma execução não supervisionada pode ser iniciada com:

```bash
caffeinate -i python3 benchmarks/tpp.py run-fekete --workers 8
```

## Preservação de resultados

Resultados novos permanecem ignorados até virarem evidência deliberadamente
publicada. Uma pasta em `results-saved/` deve conter:

- entradas exatas ou uma receita determinística para obtê-las;
- hashes das entradas, executáveis e revisões relevantes;
- configuração, formulação, tolerâncias e orçamento;
- resultados brutos, inclusive falhas e limites;
- análise reproduzível e limitações conhecidas.

A comparação alemã atual está em
`results-saved/german-comparison/`; sua análise vive na mesma pasta. Não copie
uma análise de campanha para `docs/research/`.

## Interpretação

No benchmark não convexo, observe principalmente:

- chamadas ao solver convexo e nós explorados;
- diferença entre incumbente inicial e resultado final;
- motivo de término e gap final;
- tempo medido por caso e tempo de parede da campanha;
- tolerância e validação independente da trajetória.

“Ótimo” significa que o solver fechou o gap dentro da tolerância declarada.
Um caminho factível interrompido por tempo ou chamadas não deve ser rotulado
como ótimo. Tempos obtidos com números diferentes de workers ou sob contenção
não são comparações diretas de desempenho.

## Campanhas diagnósticas de TSPN

O subcomando `tspn-benchmark` adapta a instrumentação do TPP de ordem livre ao
**ciclo fechado sem extremos fixos**, comparando o B&B mantido com o SOCP de
Fekete. O benchmark de caminho com extremos fixos continua em `free-order-run`.
Os perfis abaixo usam o ZIP SOCG simplificado, classificação pelos metadados e
número efetivo de polígonos após simplificação. O ZIP contém 558 entradas;
360 é o número de instâncias alteradas pelo pré-processamento no artigo.

Execute na raiz do checkout desejado, sem outros benchmarks concorrentes:

```bash
python3 benchmarks/tpp.py tspn-benchmark --profile quick \
  --output benchmarks/results/tspn-diagnostic-quick --resume

caffeinate -i python3 benchmarks/tpp.py tspn-benchmark --profile overnight \
  --output benchmarks/results/tspn-diagnostic-overnight --resume
```

`caffeinate` é opcional e específico do macOS. A primeira execução compila em
`.build/tspn-comparison`, fora do submódulo, e usa as dependências e licença
Gurobi já instaladas. Em worktrees com submódulo vazio, localiza o checkout
primário automaticamente. Não instala dependências nem copia licenças.

| Perfil | Casos por estrato | Repetições | Limite nativo por execução | Teto de processo | Máximo de processos |
|---|---:|---:|---:|---:|---:|
| `quick` | 1 (até 12 casos) | 1 | 3 s | 10 s | 4 min |
| `overnight` | 8 (até 96 casos) | 2 | 60 s | 75 s | 8 h |

São 12 estratos: OSM/random/tessellation × 5–10/11–20/21–40/41–60 polígonos.
A seleção é uniforme sem reposição, com seed 20260930, anterior às medições.
O perfil rápido é um subconjunto do noturno. Os estratos são intercalados e
uma rodada cobre todos os casos antes de iniciar a próxima repetição. Assim,
uma interrupção não concentra a amostra em uma única classe. Os tetos da tabela
somam os limites firmes dos processos; **build, validação e relatórios ficam
fora desse teto**. Instâncias fáceis podem encerrar a campanha muito antes.
Não é uma campanha para provar otimalidade de todos os casos grandes.

Ambos os perfis selecionam `cache + features + root + interval`, uma thread por
solver e execução sequencial. `--portfolio` é opcional, acrescenta `memo` ao
preset e registra dois workers nativos contra um externo; mantenha esse ensaio
em outra pasta. `--seconds`, `--external-timeout`, `--per-stratum`,
`--repetitions`, `--seed` e `--instances-zip` permitem substituir o preset.
`--dry-run` mostra a seleção e o orçamento sem compilar ou executar solvers.

Cada execução concluída é persistida em `raw.jsonl`. Repetir **o mesmo comando**
com `--resume` pula registros existentes, inclusive timeouts já observados;
valida configuração, entradas e hashes dos executáveis, e não recompila.
Mudanças nos executáveis exigem outra pasta de resultados. Ctrl-C preserva os
registros completos e gera relatórios parciais; uma gravação final truncada é
recuperada na retomada. Não execute duas campanhas simultâneas na mesma pasta.

Os artefatos para análise são:

- `instances.json` e `config.json`: entradas, seed, população por estrato,
  formulação, tolerâncias, configuração, hashes de fontes e executáveis;
- `raw.jsonl` e `runs.csv`: trajetórias e métricas brutas; o CSV expõe todos os
  campos escalares e métricas derivadas, sem traços por nó de tamanho ilimitado;
- `summary.json/csv`, `strata.json/csv`, `analysis.md`: comparações por instância,
  classe/tamanho e diagnóstico de gargalos;
- `progress.json`: quantidade de registros concluídos e status.

As métricas incluem fases do solver, chamadas/nós, atualizações do incumbente,
qualidade inicial, gap final, decomposição, filas, podas, cache e contatos
reutilizados. A nova telemetria mede chamada máxima, histogramas de quantidade
**e tempo** e tempo das chamadas que usaram recuperação racional. Este último
inclui a chamada inteira, não apenas a recuperação. Os subtempos herdados do
oráculo de caminho não separam corretamente as fases internas do ciclo; não
os use para atribuir custo geométrico ou de certificação no TSPN.

Speedups só usam casos em que todas as repetições dos dois solvers validam a
trajetória, fecham o gap solicitado e têm intervalos reportados compatíveis.
Timeouts de processo são censurados, não recebem um tempo fictício de solução;
limites nativos mantêm métricas e gap, quando o processo retorna. A razão
incumbente inicial/final mede melhoria, não garante aproximação enquanto o gap
estiver aberto. Os limites do baseline SOCP são numéricos. Mantemos gap relativo
comparável de 1e-6, factibilidade de 1e-8 e validação independente de 1e-7;
o benchmark não adiciona tolerância ao oráculo convexo certificado.

Para reconstruir relatórios sem executar solvers:

```bash
python3 benchmarks/tpp.py tspn-benchmark --report-only \
  --output benchmarks/results/tspn-diagnostic-overnight
```

Envie a pasta da campanha para análise. Não é necessário enviar builds ou
arquivos de licença. Os resultados continuam locais e ignorados pelo Git.
