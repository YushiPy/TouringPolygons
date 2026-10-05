# Benchmarks

`benchmarks/tpp.py` é a única interface pública para geração, execução,
conversão e comparação. Os módulos de `benchmarks/_internal/` são detalhes de
implementação importáveis, não uma coleção de comandos independentes.

## Montar e executar um benchmark

`scripts/benchmark.sh` abre uma interface de terminal (stdlib, sem dependências)
que monta o comando por você: navegue com as setas, edite um campo com Enter
(seletor para opções finitas, texto para números e caminhos) e, ao terminar,
imprima, copie para a área de transferência ou execute o comando. Os últimos
valores e presets nomeados ficam em `benchmarks/workspace/tui-state.json`.

Nos campos numéricos só é possível digitar números e `+ - * / ** ( )`; o valor
pode ser uma expressão (`10 ** -7`, `1 / 100`, `(10 - 5) * 2 / 3`), com
pré-visualização do resultado, e um valor fora do intervalo do campo (por exemplo
negativo, exceto `-1` onde significa "sem limite") não é aceito. A expressão é
interpretada, nunca executada, com limite de tamanho para expoentes. `Delete` ou
`Backspace` sobre um campo apaga o valor e abre a edição; `Shift+Enter` faz o
mesmo nos terminais que o distinguem de Enter (kitty, WezTerm, Ghostty, iTerm2 com
CSI u, com `TPP_TUI_KITTY_KEYS=1`), e `Alt+Enter` funciona na maioria dos demais.
Se alguma tecla não responder, `scripts/benchmark.sh --debug-keys` mostra o que o
terminal envia.

O comando produzido é `python3 benchmarks/tpp.py bench --problem P [opções]`.
Há três problemas, executados pelos módulos já existentes:

| `--problem` | Executa | Instâncias |
|---|---|---|
| `fixed-order` | `run` (B&B de ordem fixa) | campanha `--campaign` |
| `free-order` | `free-order` (nosso solver e/ou Fekete) | campanha `--campaign` |
| `tspn` | `tspn-compare` (ciclo fechado, 558 casos de Fekete) | ZIP fixo da Fekete |

As opções comuns têm um único nome (`--time-limit`, `--oracle-calls`,
`--threads`, `--workers`, `--repetitions`, `--resume/--no-resume`, `--dry-run`,
etc.). Cada uma só vale para os problemas em que existe, e o padrão pode variar
por problema (por exemplo, 60 s no TSPN, o protocolo da campanha). `bench --help`
lista tudo; ele é gerado do mesmo esquema (`_internal/run_spec.py`) que alimenta
a interface, então a ajuda não diverge do comportamento. `--no-resume` reinicia:
reexecuta (`--force`) em ordem fixa e livre e move a campanha TSPN para o lado.
Os subcomandos antigos continuam disponíveis.

## Acompanhar uma execução longa

Enquanto uma instância roda, o solver de ordem livre e o TSPN (`tpp-ours`)
imprimem a cada `--progress-interval` segundos (padrão 60; 0 desliga; campo
"Progress report" no `scripts/benchmark.sh`) uma linha como

```text
[free case 1/2 tpp-ours] 00:00:06  LB 107.06  UB 136.957  gap 21.8%  calls 318,871  nodes 45,932  open 269,361 (growing)  time limit 43% used
```

Nela estão o tempo decorrido, o limite inferior (LB) e o superior (UB), o gap
relativo `(UB - LB) / UB`, as chamadas ao oráculo e os nós explorados, a fila
(`open`, com a tendência: crescendo, estável ou diminuindo) e a fração usada dos
limites de tempo e de chamadas. Com `--portfolio` as duas buscas são combinadas
(melhor LB e melhor UB). Quando o gap vem diminuindo, aparece também uma
estimativa de quando o gap-alvo seria atingido. Ela é **otimista** (supõe que o
gap continue caindo no mesmo ritmo, o que o branch and bound muitas vezes não faz).
Trate-a como um palpite; LB, UB e a tendência da fila são os sinais confiáveis.

Os mesmos dados ficam em `live.json` (no diretório da execução: a campanha
TSPN, ou `results/EXECUÇÃO/`), reescrito de forma atômica a cada ~5 s. Ele não
depende do terminal que iniciou a execução: de qualquer outro terminal,

```bash
python3 benchmarks/tpp.py live            # tudo o que está rodando no workspace
python3 benchmarks/tpp.py live NOME       # só uma campanha
python3 benchmarks/tpp.py live --once     # imprime o estado atual e sai
```

o `live` imprime o estado de cada instância e **continua rodando**, mostrando
cada novo report assim que ele chega (Ctrl+C para sair), e avisa quando uma
execução termina. Snapshots deixados por processos que já não existem nesta máquina
(execução morta ou terminal fechado) são apagados, com uma linha avisando
quantos; `--include-gone` os mantém e mostra. Snapshots de outra máquina
(workspace compartilhado) nunca são apagados. Se o terminal for fechado, o shell costuma matar a
execução; para execuções longas, inicie-a em `tmux`/`screen` ou com `nohup`.

### Quando um solver é interrompido, morto ou fica sem memória

Cada report também é **anexado** a `results/EXECUÇÃO/progress.jsonl` (tempo, LB, UB,
gap, chamadas), que sobrevive a um terminal perdido, ao pai morto e ao fim da
execução: dá o histórico de convergência de cada caso. Além disso, o resultado parcial
vai para o relatório mesmo que o solver não termine:

| O que aconteceu | `status` da linha | O que fica registrado |
|---|---|---|
| `tpp.py stop` (SIGINT no tpp-ours, SIGTERM no Fekete) ou Ctrl+C | `interrupted` | incumbente (caminho), LB e UB |
| processo morto por sinal (SIGKILL, OOM killer) | `killed` | LB, UB, tempo e chamadas do último report; no Fekete também o último incumbente que ele havia gravado |
| acima do limite de memória (`--max-memory-gb`) | `memory_limit` | incumbente, LB e UB (o solver é parado, não morto) |

Esses casos contam como erro de solver (`completed_with_errors`) e a retomada os
refaz do zero, pois nenhum solver continua de um estado salvo; o parcial serve
para saber onde parou e estimar o tempo que faltava.

Para parar só algumas instâncias, sem encerrar a execução:

```bash
python3 benchmarks/tpp.py stop                # lista as instâncias em execução e seus pids
python3 benchmarks/tpp.py stop --case 130     # o número mostrado por `live` (free case 130/558)
python3 benchmarks/tpp.py stop --all
```

`--max-memory-gb N` (campo "Memory limit (GB)" no `scripts/benchmark.sh`) confere a
memória residente de cada solver a cada 5 s e o interrompe com SIGINT/SIGTERM
antes que o sistema o mate. O limite vale por instância, não pelo total. Para a
execução sobreviver ao terminal fechado, inicie-a com `tpp.py jobs start`, `tmux`
ou `nohup`.

O `live` avisa quando uma instância deixa de reportar (uma chamada longa ao
oráculo ou um solver travado). O relatório só observa a busca: o resultado, as chamadas e
os nós são idênticos com ele ligado ou desligado (há teste C++ para isso). O
estado vem do laço do branch and bound, então uma única chamada muito longa
atrasa a próxima linha. Execuções iniciadas com um binário anterior a este
recurso não reportam nada. A ordem fixa (`run`) ainda não tem relatório periódico.

## Organização

- `tpp.py`: CLI estável; `python3 benchmarks/tpp.py --help` lista os comandos
  agrupados por tarefa;
- `_internal/`: implementação da CLI;
- `suites/`: corpora canônicos rastreados e suites derivadas ignoradas;
- `results-saved/`: resumos compactos e fixtures pequenos indispensáveis;
- `workspace/`: **todo** dado gerado localmente (ignorado pelo Git).

### Workspace

Tudo o que a CLI, o dashboard ou uma máquina remota produzem fica em um único
diretório, `benchmarks/workspace/` (ou `$TPP_WORKSPACE`, por exemplo um disco
de scratch com mais cota):

```text
workspace/
├── campaigns/<nome>/        conjuntos de instâncias com campaign.json e suas
│                            execuções em results/<execução>/ (ver abaixo)
├── campaigns/<nome>@<host>/ campanhas trazidas de outra máquina
├── runs/<nome>/             saídas de comandos sem campanha (suites, ablações,
│                            comparações)
├── experiments/<nome>/      investigações manuais: notas, logs, dados ad hoc
├── jobs/<id>/               execuções destacadas (job.json, output.log)
├── regions/                 extratos OpenStreetMap (.osm.pbf) e caches
└── history.jsonl            uma linha por comando executado
```

Dentro de uma campanha, **toda execução tem a própria pasta**
`results/<AAAAMMDD-HHMMSS-id>/`, qualquer que seja o problema: a ordem livre
guarda um único `report.json`; a ordem fixa guarda `run-index.csv` e os
`.csv`/`.md`/`.log`/`.done` de cada entrada; o TSPN guarda `config.json`,
`raw.jsonl`, os relatórios e os `shard-N/`. O tipo é dado pelo que a pasta
contém, não pelo caminho. Repetir um comando com as mesmas configurações
continua a execução mais recente (ordem fixa: se o que ela terminou usou as
mesmas configurações e o mesmo binário); configurações diferentes ou
`--no-resume` abrem uma pasta nova e nunca sobrescrevem a anterior.
`results/comparisons/` guarda comparações derivadas, não execuções. Campanhas
criadas antes (`results/free-order/<execução>/` e arquivos soltos em `results/`)
continuam sendo lidas; `python3 benchmarks/tpp.py workspace migrate-results
[--dry-run]` as move para o formato atual.

`campaign.json` registra `origin` (`cli`, `dashboard` ou `remote:<host>`).
Cada execução grava `run.json` com comando, máquina (host, CPU, memória,
carga), revisão Git e hashes de binários e entradas; retomadas acrescentam
tentativas. `python3 benchmarks/tpp.py ls` lista tudo com tipo, origem e
status. Checkouts antigos são migrados por
`python3 benchmarks/tpp.py workspace migrate` (`--dry-run` mostra o plano),
que deixa `benchmarks/campaigns` e `benchmarks/results` como links de
compatibilidade.

### Binários nativos

`python3 benchmarks/tpp.py build [FERRAMENTA...]` compila em `.build/tools/bin/`
(ou `.build/tools-gurobi/bin/`). Cada `src/main-*.cpp` é um alvo próprio
(`tpp-unordered`, `tpp-bnb-workload-benchmark`, `tpp-convex-…`; veja
`build --list`), então comandos diferentes nunca reconfiguram nem sobrescrevem o
binário uns dos outros, e builds concorrentes esperam um lock. `doctor` verifica
compilador, Eigen/Boost, Gurobi e ferramentas já compiladas.

## Outras máquinas (rede IME)

Localmente bastam `ssh` e `rsync`. A máquina remota precisa de Python 3.12+ com
`uv` (instalável em `~/.local/bin` sem root), CMake (vem da venv) e um
compilador C++23 (por exemplo `g++-14`). Eigen e Boost são usados só como
headers: sem pacotes do sistema, `--fetch-deps` baixa versões fixadas (com
SHA-256 conferido) para `.cache/deps` do checkout remoto. CGAL e GMP são
opcionais. Nada usa sudo.

```bash
python3 benchmarks/tpp.py remote push USUARIO@MAQUINA --campaign NOME
python3 benchmarks/tpp.py remote setup USUARIO@MAQUINA --fetch-deps
python3 benchmarks/tpp.py remote run USUARIO@MAQUINA -- \
  free-order NOME --threads-per-instance 8 --max-seconds 3600
python3 benchmarks/tpp.py remote jobs USUARIO@MAQUINA            # lista
python3 benchmarks/tpp.py remote jobs USUARIO@MAQUINA log ID -f  # acompanha
python3 benchmarks/tpp.py remote pull USUARIO@MAQUINA NOME
```

`push` envia os arquivos rastreados do working tree (inclusive alterações não
commitadas) e um carimbo `.tpp-source.json` com revisão, estado sujo e hash do
diff, que substitui o Git nos `run.json` remotos. `--with-external` inclui o
submódulo de Fekete. `--dir` e `--workspace` escolhem os diretórios remotos e
ficam lembrados em `workspace/remotes.json`. `run` inicia um job destacado:
ele sobrevive ao fim da sessão SSH; `jobs … stop ID` envia Ctrl+C, que os
runners tratam gravando checkpoint, e `--force` encerra. `pull` traz campanhas
e runs como `NOME@MAQUINA`, reescrevendo caminhos absolutos remotos, e o
dashboard as mostra junto das locais.

O mesmo mecanismo de jobs funciona localmente, para campanhas longas:
`python3 benchmarks/tpp.py jobs start -- run NOME --threads 8`.

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
  --suite benchmarks/workspace/campaigns/paula-center/inputs/paula-center.bin \
  --solver .build/tools/bin/tpp-unordered --seconds 30 --max-calls 1000000 \
  --output benchmarks/workspace/campaigns/paula-center/run.jsonl
python3 benchmarks/tpp.py verify-socp \
  --solver .build/tools/bin/tpp-unordered \
  --manifest benchmarks/workspace/campaigns/paula-center/paula-manifest.json \
  --output benchmarks/workspace/campaigns/paula-center/verification \
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

A campanha fica em `benchmarks/workspace/campaigns/smoke/` com manifesto, entradas,
preview e resultados. Execuções concluídas são reutilizadas ao retomar.

Para gerar uma matriz baseada em OpenStreetMap:

```bash
python3 benchmarks/tpp.py generate-matrix sao-paulo \
  benchmarks/workspace/regions/sao-paulo.osm.pbf \
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
`benchmarks/workspace/campaigns/fekete-free-order-comparison-8threads/`. Esse modo não
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
retomam os casos já registrados e guardam um único arquivo por execução,
`report.json`, em `benchmarks/workspace/campaigns/<nome>/results/<execução>/`.
Ele reúne a configuração, as linhas dos dois solvers e, nas do Fekete, a
telemetria completa do runner externo (`external`: iterações, ramificações,
chamadas e tempos SOCP...). A geometria não é copiada: as linhas guardam o hash
da instância e o `.bin` da campanha é a única cópia. O resumo comparativo é
calculado sob demanda por `python3 benchmarks/tpp.py report <campanha|execução>`.
Execuções feitas antes dessa mudança (com `external/*.csv`, `geometry/` e
`comparison.md`) continuam legíveis, e retomar uma delas importa o CSV antigo
apenas para as linhas que o relatório não tem.

O `status` do `report.json` diz como a campanha terminou, separado de como cada
solver foi: `completed` (todos os casos rodaram sem erro), `completed_with_errors`
(todos foram tentados, mas um solver falhou em alguns; `solver_errors` lista, por
solver, os casos e o primeiro erro, e a CLI sai com código 2), `interrupted`
(parada pelo usuário, retomável, código 130) e `failed` (a própria campanha
quebrou: exceção ou casos que nunca rodaram, código 1). Retomar reexecuta só os
casos com erro.

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
campanha; o diretório da campanha multithread pode ser usado enquanto a execução
ainda está em andamento (em execuções antigas, usa o CSV parcial do Fekete).

```bash
python3 benchmarks/tpp.py compare-threads \
  --ours-single benchmarks/workspace/runs/free-order-gap-6h/runs.csv \
  --ours-single-variant fekete_gap \
  --ours-multi benchmarks/workspace/campaigns/fekete-tpp_free_order/results/ID_DA_RUN \
  --fekete-single benchmarks/results-saved/fekete-comparison/fekete.csv \
  --fekete-multi benchmarks/workspace/campaigns/fekete-tpp_free_order/results/ID_DA_RUN \
  --output benchmarks/workspace/runs/thread-scaling/fekete
```

O speedup é `tempo(1 thread) / tempo(multithread)`. Diferenças detectáveis de
limite, gap e workers aparecem no relatório: uma comparação histórica com
configurações diferentes é indicativa, não uma medição causal do ganho de
threads.

Para executar diretamente uma suite binária:

```bash
python3 benchmarks/tpp.py free-order-run \
  --suite benchmarks/suites/algorithm-dev-v1.bin \
  --solver .build/tools/bin/tpp-unordered \
  --seconds 3 --max-calls 10000000 --workers 4 \
  --output benchmarks/workspace/runs/free-order-dev.jsonl
```

Para medir o custo de provar otimalidade quando o incumbente inicial já é bom,
`free-order-run` aceita `--initial-paths` com um CSV separado por `;` contendo
`case_index`, `sha256` e `path` (JSON de pontos). O hash de cada instância é
conferido antes de executar. Exemplo com os caminhos preservados da campanha alemã:

```bash
python3 benchmarks/tpp.py free-order-run \
  --suite benchmarks/results-saved/fekete-comparison/instances.bin \
  --initial-paths benchmarks/results-saved/fekete-comparison/ours.csv \
  --solver .build/tools/bin/tpp-unordered --seconds 21600 --max-calls 100000000 --workers 1 \
  --output benchmarks/workspace/runs/fekete-good-initial.jsonl
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
  --output benchmarks/workspace/runs/free-order-comparison.jsonl
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

A campanha fica em `benchmarks/workspace/runs/free-order-gap-comparison/`. Ela guarda
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

Ela grava checkpoints em `benchmarks/workspace/runs/fekete-free-order-6h/`. No macOS,
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
  --output benchmarks/workspace/runs/tspn-diagnostic-quick --resume

caffeinate -i python3 benchmarks/tpp.py tspn-benchmark --profile overnight \
  --output benchmarks/workspace/runs/tspn-diagnostic-overnight --resume
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

O script é um atalho para `python3 benchmarks/tpp.py tspn-compare`, que faz toda
a preparação: confere o SHA-256 do ZIP e a revisão fixada do submódulo, valida as
dependências Conan, escolhe o padrão C++ e compila uma vez. Depois executa
`tspn-benchmark --all --instances-zip …/instances_socg_simplified.zip --output
benchmarks/workspace/campaigns/NOME/results/<execução>` com os
`--cycle-optimization` padrão, e retoma registros concluídos ao repetir o
comando (continua a execução mais recente da campanha; `--no-resume` abre uma
nova pasta e mantém a anterior). Com `--workers N`, divide os casos em N shards (cada solve continua com
uma thread), executa-os em paralelo e funde `raw.jsonl`, configuração e
relatórios. `--setup-only` só verifica o ambiente e `--dry-run` mostra o plano. O `--all` desativa a
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
Mudanças nos executáveis exigem uma nova execução (`--no-resume`). Ctrl-C preserva os
registros completos e gera relatórios parciais; uma gravação final truncada é
recuperada na retomada. Não execute duas campanhas simultâneas com o mesmo nome.

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
  --output benchmarks/workspace/runs/tspn-diagnostic-overnight
```

Envie a pasta da campanha para análise. Não é necessário enviar builds ou
arquivos de licença. Os resultados continuam locais e ignorados pelo Git.


### Captura e reprodução de chamadas caras

Para comparar alterações internas com trabalho fixo, use
`tspn-benchmark --solver ours --max-calls N` e um orçamento de tempo suficiente
para atingir esse limite. O padrão continua em 100 milhões de chamadas.
Um limite diferente do padrão é exclusivo de `--solver ours`, porque Fekete
não oferece o mesmo orçamento. Compare instâncias, opções, chamadas, nós,
limites e trajetórias entre os binários congelados. Um speedup para a mesma
busca parcial não equivale a um speedup até fechar o gap.

As opções experimentais `--cycle-optimization proposal-bound` e
`--cycle-optimization primal-starts` controlam, respectivamente, uma fase de
proposta certificada antes da recuperação completa e partidas adicionais da
heurística de incumbente. São independentes e desativadas por padrão: medir a
combinação também é necessário, pois um incumbente inicial melhor não garante
uma busca mais rápida. O oráculo mantém a certificação exata; a aceitação de um
intervalo parcial pelo B&B usa somente o contrato numérico já declarado.

Use uma lista focal de entradas locais, mantendo as opções da campanha:

```bash
python3 benchmarks/tpp.py tspn-benchmark \
  --inputs benchmarks/workspace/runs/focal-inputs.json \
  --output benchmarks/workspace/runs/focal-capture \
  --seconds 10 --external-timeout 15 --repetitions 1 \
  --cycle-optimization cache --cycle-optimization features \
  --cycle-optimization root --cycle-optimization interval --capture-oracles

python3 benchmarks/tpp.py cycle-replay \
  --capture benchmarks/workspace/runs/focal-capture/oracle-captures/000-0.jsonl \
  --min-seconds 0.01 --seconds 10 --repetitions 2 \
  --cache --features --interval --output benchmarks/workspace/runs/focal-replay
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
