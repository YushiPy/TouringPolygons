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
- `results-saved/`: resumos compactos e fixtures pequenos indispensáveis;
  dados brutos e campanhas completas ficam locais e ignorados.

Use `python3 benchmarks/tpp.py --help` e acrescente `--help` após um subcomando
para consultar todos os parâmetros.

## Ambiente próprio

Prepare uma vez, a partir da raiz do repositório:

```bash
python3 benchmarks/tpp.py setup
```

O comando usa `uv` para criar `benchmarks/.venv` com Python 3.12 e instalar as
versões fixadas em `benchmarks/uv.lock`: Shapely para validação independente,
Matplotlib e PyOsmium para geração/visualização, CMake para builds e Ruff para
verificação das ferramentas. Repetir o comando sincroniza o mesmo ambiente;
ele não compila nem executa solvers. `uv` é um pré-requisito preparado pelo
instalador geral `./scripts/install_dependencies.sh`.

Não é necessário ativar a venv: `python3 benchmarks/tpp.py COMANDO` seleciona
esse Python e coloca suas ferramentas no `PATH`. A CLI pede novo setup se o
lockfile ou manifesto mudar; não instala dependências durante uma campanha.
Ajuda e setup funcionam antes de preparar o ambiente. Use `setup --offline`
quando os pacotes já estiverem em cache, ou `setup --python /caminho/python`
para escolher outro intérprete compatível. Compilador, Eigen, Boost, OpenMP,
GMP e CGAL continuam sendo dependências nativas, preparadas separadamente.

O ambiente em `third_party/tspn-socg/.venv` fica restrito aos workers e ao
binding nativo de Fekete; não é necessário para executar os benchmarks
próprios com um binário já compilado. Os comandos externos aceitam
`--external-python` quando esse ambiente está em outro local.

## Instâncias locais de Paula

O importador considera exclusivamente os 235 arquivos de `npol-le-100`:
45 de `gtsplib` e 190 de `momlib`. Conserva coordenadas, regiões e ordem do arquivo,
e acrescenta `start == target` no centro da bbox global. É TPP com depot fixo;
os ótimos de ciclo livre publicados no paper não são ótimos de referência para
essa formulação. Pontos e segmentos usam um e dois vértices, respectivamente.
Polígonos não convexos são tratados pela decomposição existente, com peças
alternativas para a mesma região.

```bash
python3 benchmarks/tpp.py convert-paula paula-center
python3 benchmarks/tpp.py free-order-run \
  --suite benchmarks/campaigns/paula-center/inputs/paula-center.bin \
  --solver .build/unordered/tpp --seconds 30 --max-calls 1000000 \
  --output benchmarks/campaigns/paula-center/run.jsonl
python3 benchmarks/tpp.py verify-socp \
  --solver .build/unordered/tpp \
  --manifest benchmarks/campaigns/paula-center/paula-manifest.json \
  --output benchmarks/campaigns/paula-center/verification \
  --reference-python third_party/tspn-socg/.venv/bin/python
```

O manifesto associa cada caso ao arquivo original e seu SHA-256, à bbox e aos
extremos adicionados. Os dados de terceiros e toda a campanha derivada devem
permanecer locais e ignorados, pois não há autorização de redistribuição.

A referência SOCP requer um Python com Gurobi licenciado e Shapely >= 2.1;
`--reference-python` seleciona esse ambiente, sem mudar as dependências do solver.
Executa 32 casos sintéticos e, com o manifesto, 12 subconjuntos determinísticos
de até três regiões. Seleciona regiões curtas e casos de menor custo de enumeração,
conservando o depot do caso original. **Esses subconjuntos não são as instâncias
completas.** `--subset-size 0` pede instâncias completas, sujeito ao limite de
20.000 folhas; a enumeração cresce com ordens e peças, não apenas com regiões.
Veja o [contrato e as tolerâncias](../docs/algorithms/unordered-tpp.md#referência-independente-socp).
`verification.json` registra resultados, status numérico da referência e as
validações independentes dos caminhos. Uma execução curta de `free-order-run`
pode terminar por tempo/chamadas; consulte `exact`, limites e `termination`.

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

Para reconstruir o corpus a partir do arquivo de instâncias fixado no submódulo:

```bash
python3 benchmarks/tpp.py convert-fekete
```

## Ordem livre

Uma campanha completa usa:

```bash
python3 benchmarks/tpp.py free-order NOME_DA_CAMPANHA --help
```

No laboratório, `scripts/run_comparison.sh` chama esta CLI para executar os
558 casos de Fekete et al. Selecione os solvers com `--solver tpp-ours`,
`--solver tpp-fekete` ou `--solver both` (padrão). A pasta da campanha depende
de `--threads-per-instance`, não de `--workers` nem do solver selecionado:
uma thread reutiliza `fekete-free-order-comparison-v1`, e outras contagens usam
pastas próprias, como `fekete-free-order-comparison-8threads`. `--campaign`
permite escolher outro nome. O script cria a pasta e o manifesto na primeira
execução; repetir a mesma configuração retoma os casos concluídos. No macOS,
se o CMake não reconhecer o AppleClang como compatível com C++23, o setup tenta
o LLVM do Homebrew. No Linux, tenta GCC
16/15/14 e Clang 20/19/18 instalados se o compilador padrão não passar a
verificação; variáveis `CC` e `CXX` definidas pelo usuário são respeitadas.

Para executar somente nosso solver com oito threads por instância, use:

```bash
scripts/run_comparison.sh --solver tpp-ours --threads-per-instance 8
```

Isso cria a campanha local
`benchmarks/campaigns/fekete-free-order-comparison-8threads/`. Esse modo não
prepara nem verifica Fekete ou a licença Gurobi; reutiliza os pacotes C++ já
baixados em `third_party/tspn-socg/.conan/release` para compilar nosso solver.
`--solver tpp-fekete` prepara e valida apenas Fekete e exige licença Gurobi;
`--solver both` prepara e executa os dois. O comando
`scripts/run_comparison.sh --setup-only --solver tpp-ours --threads-per-instance 8`
verifica apenas o setup necessário para nosso solver, sem começar os casos.
Falhas transitórias de download no setup do Fekete são repetidas até três vezes.
`Ctrl+C` grava trajetórias incumbentes e limites
parciais disponíveis, marcando os casos ativos como `interrupted` para serem
reexecutados ao retomar. Durante o encerramento cooperativo, o runner informa
que aguarda a chamada geométrica ativa; um segundo `Ctrl+C` força o processo
nativo a parar e preserva os checkpoints concluídos.

Para comparar os dois solvers com oito threads dentro de cada instância, sem
executar instâncias diferentes em paralelo, a campanha aceita:

```bash
python3 benchmarks/tpp.py free-order fekete-instances \
  --solver tpp-ours --solver tpp-fekete --workers 1 --threads-per-instance 8 \
  --max-seconds 21600 --max-calls 10000000 \
  --absolute-gap 0 --relative-gap 0.000999000999000999 --eps 0.001 \
  --sampled-perimeter-initial --convex-initial-refinement --bidirectional-initial
```

Os casos pendentes entram em uma fila compartilhada: primeiro os casos do tpp-ours
em ordem de índice, depois os do tpp-fekete em ordem de índice. `--workers`
controla quantos casos dessa fila podem rodar ao mesmo tempo; quando um worker
termina um caso do tpp-ours, ele já pega o próximo do tpp-fekete, mesmo que
outro worker ainda esteja no tpp-ours. `--threads-per-instance` controla as
threads internas de cada caso. O pico é de até
`--workers × --threads-per-instance` threads de solver. Campanhas compatíveis
retomam os casos já registrados e guardam `report.json`, o CSV externo bruto e
`comparison.md` em `benchmarks/campaigns/<nome>/results/free-order/`.

Se uma campanha falhar por problema no ambiente Python, a retomada reexecuta
as linhas com erro e preserva as execuções concluídas, incluindo resultados
parciais por limite de tempo. Para conservar os binários e seus hashes, use
diretamente `benchmarks/tpp.py free-order` com `--no-build` e os mesmos
parâmetros da campanha, em vez de repetir o setup dos solvers. Use
`python3 benchmarks/tpp.py`; a CLI seleciona a entrada da venv própria sem
resolver seu link para um executável versionado do Homebrew. Trajetórias
armazenadas sem validação são revalidadas na retomada;
isso não altera os caminhos nem os tempos registrados. Os ratios no resumo
exigem que ambos os solvers fechem o gap e que ambos os caminhos originais
passem na tolerância de validação independente registrada.

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
  --ours-multi benchmarks/campaigns/fekete-instances/results/free-order/ID_DA_RUN \
  --fekete-single benchmarks/results-saved/fekete-comparison/fekete.csv \
  --fekete-multi benchmarks/campaigns/fekete-instances/results/free-order/ID_DA_RUN \
  --output benchmarks/results/thread-scaling/fekete
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
  --suite benchmarks/results-saved/fekete-comparison/instances.bin \
  --initial-paths benchmarks/results-saved/fekete-comparison/ours.csv \
  --solver .build/unordered/tpp --seconds 21600 --max-calls 100000000 --workers 1 \
  --output benchmarks/results/fekete-good-initial.jsonl
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

O solver de ordem livre aceita `--sequence-storage native|packed|deltas`.
Para isolar o cache de despacho exato do oráculo, compare duas variantes do
mesmo binário e adicione
`--solver-argument baseline=--no-oracle-dispatch-cache` somente à referência.
O candidato usa o cache por padrão. O perfil informa o tempo de despacho e
os acertos nas consultas de pares. Sem corte de tempo, caminhos, limites e
contagens de busca devem coincidir; ambos mantêm as mesmas tolerâncias.
O cache de vértices normalizados da proposta intervalar pode ser isolado com
`--solver-argument baseline=--no-oracle-interval-geometry-cache`;
`convex_proposal_preparation_seconds` registra essa preparação. Combine os
dois argumentos na referência para medir o efeito total dos caches, ou use
uma variante intermediária para separar suas contribuições.
`--no-prepared-visits` isola a preparação de geometria e o reaproveitamento
de contatos para o último caminho. Compare `visit_query_evaluations`,
`visit_query_cache_hits` e `search_visit_check_seconds` junto com a igualdade
de caminhos/limites/contagens. `--relocate-initial` e
`--interpolated-zero-dual` são opções experimentais independentes; a primeira
pode mudar a árvore e a segunda pode fortalecer limites. Nesses ensaios,
tempo sob um teto de chamadas não é tempo de conclusão: compare gap e status,
além do custo total e da heurística inicial.
Para isolar memória, compare esses modos com o mesmo binário usando
`--solver-argument LABEL=--sequence-storage --solver-argument LABEL=MODO`.
A CLI de ablação conserva o pico de RSS e os bytes de sequência retornados
pelo solver e mostra RSS máximo/mediano no resumo. Prefira `--workers 1`
para esse diagnóstico; resultados limitados por orçamento exigem comparar
também chamadas e limites, pois podem explorar quantidades diferentes de nós.

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

## solver de Fekete et al.

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

Resultados e dados brutos novos permanecem locais e ignorados. Em
`results-saved/`, preserve somente um resumo curto com formulação, população e
seleção da amostra, orçamento/tolerâncias, métricas agregadas, status de gap e
exatidão, limitações e hashes/revisões mínimos para identificar a medição.
Não salve raws por repetição, cópias de polígonos, logs, builds ou patches de
experimentos. Entradas só ficam no Git quando forem fixtures pequenas exigidas
por testes ou ferramentas.

`results-saved/fekete-comparison/` mantém o corpus completo porque os CSVs e o
arquivo de instâncias alimentam o material SIICUSP. O fixture pequeno
`convex-cycle-gurobi-reference-2026-09-25/instances.json` é carregado pelos
testes e benchmarks de ciclo. Todos os demais resultados ficam em resumos; a
comparação, tolerâncias e limitações de cada um estão no índice
`results-saved/README.md`.

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

Para o **benchmark completo das 558 instâncias originais** na formulação TSPN
(tour fechado, ordem livre, sem ponto fixo) contra o SOCP de Fekete, use o
runner de campanha:

```bash
scripts/run_tspn_comparison.sh --seconds 60 --external-timeout 75 --repetitions 1
```

Ele equivale a `python3 benchmarks/tpp.py tspn-benchmark --all
--instances-zip third_party/tspn-socg/instances/instances_socg_simplified.zip
--output benchmarks/campaigns/tspn-fekete-comparison-v1 --seconds 60
--external-timeout 75 --repetitions 1` com os `--cycle-optimization` padrão,
e retoma registros concluídos ao repetir o comando. O `--all` desativa a
amostragem estratificada e os filtros de faixa de tamanho, selecionando as
558 entradas do ZIP sem substituição de reposição.

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
inclui a chamada inteira, não apenas a recuperação. Novas execuções também
expõem `cycle_construction_seconds`, `cycle_certification_seconds` e
`cycle_rational_recovery_seconds`, exclusivos. A certificação inclui os testes
feitos durante a recuperação racional; a recuperação exclui esse custo.
Registros antigos sem esses campos não permitem reconstruir essa divisão.
Chamadas interrompidas cooperativamente entram nos contadores; processos
encerrados à força continuam censurados.

O adaptador Fekete registra `termination_reason` como `gap_criterion`,
`frontier_exhausted` ou `time_limit`, observando o fluxo do solver fixado.
`frontier_has_next` registra o estado final separadamente. Uma parada com gap
aberto e fronteira esgotada não é reclassificada como timeout. Registros antigos
sem motivo explícito permanecem `unknown`.

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


### Captura e reprodução de chamadas caras

Use uma lista focal de entradas locais, mantendo as opções da campanha:

```bash
python3 benchmarks/tpp.py tspn-benchmark \
  --inputs benchmarks/results/focal-inputs.json \
  --output benchmarks/results/focal-capture \
  --seconds 10 --external-timeout 15 --repetitions 1 \
  --cycle-optimization cache --cycle-optimization features \
  --cycle-optimization root --cycle-optimization interval --capture-oracles

python3 benchmarks/tpp.py cycle-replay \
  --capture benchmarks/results/focal-capture/oracle-captures/000-0.jsonl \
  --min-seconds 0.01 --seconds 10 --repetitions 2 \
  --cache --features --interval --output benchmarks/results/focal-replay
```

A captura grava e descarrega cada `begin` antes de entrar no oráculo e um
`end` com seus tempos e limites quando ele retorna. `--call-id ID` também
seleciona chamadas sem `end`, úteis depois de um timeout firme. Captura e
replay permanecem locais e ignorados. O I/O da captura não serve para medir
speedup; use execuções sem captura para comparar o B&B completo. O replay
reutiliza o adaptador C++ do B&B, incluindo propostas herdadas, corte e eventual
refinamento racional. Sua verificação independente fica fora do tempo medido.
Um orçamento de tempo esgotado preserva limites certificados e é reportado
como incompleto, nunca como ótimo.
