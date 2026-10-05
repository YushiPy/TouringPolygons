# Scripts do repositório

Esta pasta contém utilitários usados pelo projeto Touring Polygons:

- `review_meeting_transcript.py`: aplica correções contextuais recorrentes à transcrição bruta deste projeto.
- `install_dependencies.sh`: prepara as dependências do repositório.
- `sanity_check.sh`: verifica ferramentas, dependências, geração de instâncias, compilação e testes básicos.
- `benchmark.sh`: interface de terminal que monta (e imprime, copia ou executa) um comando `tpp.py bench`.
- `run_comparison.sh`: compila os solvers e executa ou retoma a comparação alemã de ordem livre.
- `run_tspn_comparison.sh`: atalho para `tpp.py tspn-compare`, que prepara, compila e executa ou retoma a campanha TSPN completa (558 casos de Fekete).
- `verify_unordered.sh`: executa a verificação focada do solver de ordem livre.

## Comparar os solvers de ordem livre

Em um clone completo do repositório, execute:

```bash
scripts/run_comparison.sh
```

Por padrão, `scripts/run_comparison.sh` prepara e executa ambos os solvers.
Use `--solver tpp-ours`, `--solver tpp-fekete` ou `--solver both` para escolher.
A pasta da campanha é determinada por `--threads-per-instance`, independente
do solver selecionado e de `--workers`. Uma thread usa
`fekete-free-order-comparison-v1`; outras contagens usam uma pasta própria,
como `fekete-free-order-comparison-8threads`. `--campaign` pode definir outro
nome. O runner cria a pasta e seu manifesto na primeira execução. Repetir a
configuração reutiliza os resultados; mudar apenas `--workers` também retoma a
mesma campanha.

O modo `--solver tpp-ours` não prepara nem verifica Fekete ou a licença Gurobi.
Ele usa os pacotes C++ já baixados em
`third_party/tspn-socg/.conan/release` para compilar nosso solver. Assim, se
esses pacotes já existem na máquina, é possível rodar nosso solver sem esperar
pela instalação ou licença Gurobi. `--solver tpp-fekete` prepara e valida
Fekete e exige uma licença Gurobi; `--solver both` prepara e executa os dois.
Para preparar apenas nosso solver sem iniciar casos, use
`scripts/run_comparison.sh --setup-only --solver tpp-ours`.

O setup de Fekete inicializa o submódulo na revisão fixada pelo repositório,
prepara o ambiente Python 3.12+ e compila o binding C++. Dois patches locais
versionados corrigem o header de `fmt` e compilam as variantes racional e
double do oráculo TPP embutido. Enquanto aplicados, aparecem como alterações
locais no submódulo; o commit fixado permanece o mesmo. Fingerprints locais
fazem o setup pular a resolução Conan e a compilação quando fontes e
configuração não mudaram. O setup de nosso solver verifica C++23; isso permite
usar GCC 13 distribuído com Ubuntu 24.04. Os runners sincronizam o ambiente
próprio com `python3 benchmarks/tpp.py setup` e usam o CMake fixado em
`benchmarks/.venv`. O Python do Fekete é usado apenas pelos workers externos.
Outros targets mantêm C++26. No
macOS e Linux, se o compilador padrão não passar, o script tenta toolchains
compatíveis instalados. `CC` e `CXX` definidos pelo usuário são respeitados.

`Ctrl+C` pede encerramento cooperativo ao nosso solver e salva a trajetória
incumbente e os limites dos casos ativos. Ele informa que está aguardando o
solver terminar a chamada geométrica em andamento; um segundo `Ctrl+C` força o
encerramento do processo nativo e preserva os checkpoints já gravados. Para o solver Fekete, o runner salva
periodicamente sua melhor trajetória e os limites conhecidos; interrupções
ficam marcadas como `interrupted` e são tentadas novamente ao retomar.

Por exemplo, `--solver both --workers 2 --threads-per-instance 12` permite até
dois casos simultâneos, cada um com 12 threads internas. Quando um deles termina
um caso pendente do tpp-ours, passa ao próximo caso do tpp-fekete sem esperar o
outro.
O pico é de até `workers × threads-per-instance` threads de solver. Alterar
apenas `--workers` retoma os resultados concluídos da mesma campanha.

Resultados e checkpoints ficam localmente em
`benchmarks/workspace/campaigns/fekete-free-order-comparison-v1/`. Rodar o comando de
novo reutiliza builds compatíveis sem alterações e retoma a campanha compatível. Para
começar um relatório novo sem apagar o anterior, use `--force`. Opções como `--workers 2`,
`--threads-per-instance 2`, `--build-jobs 12` e `--campaign outro-nome` podem
ajustar a execução. O número de tarefas paralelas deve considerar a memória
disponível; o padrão conservador é um worker.

## Benchmark TSPN completo (558 casos de Fekete)

No laboratório, `scripts/run_tspn_comparison.sh` chama `tpp.py tspn-compare` para executar
todas as 558 instâncias do arquivo
`third_party/tspn-socg/instances/instances_socg_simplified.zip` na formulação
original (tour fechado, ordem livre, sem ponto fixo), comparando nosso solver
com o B&B SOCP de Fekete:

```bash
scripts/run_tspn_comparison.sh --seconds 60 --external-timeout 75 --repetitions 1
```

Os artefatos ficam em
`benchmarks/workspace/campaigns/tspn-fekete-comparison-v1/`: `raw.jsonl` com a
telemetria completa (fases do solver, chamadas, nós, limites, heurísticas,
oráculo de ciclo, etc.), `summary.csv` e `strata.csv` resumidos,
`analysis.md` e `progress.json`. Os padrões são `--cycle-optimization
cache,features,root,interval`; repita o mesmo comando para retomar registros
concluídos. Opções `--portfolio`, `--search-strategy` e `--capture-oracles`
direcionam para o mesmo CSV bruto. Em máquinas remotas, exporte `GUROBI_HOME`
quando o Gurobi não estiver no local padrão e rode o setup do Fekete
(`scripts/run_comparison.sh --setup-only --solver tpp-fekete`) antes da
primeira execução `--solver both`/`--solver tpp-fekete`. O `--force` move a
campanha anterior para uma pasta de backup em vez de apagá-la.

## Revisar uma transcrição

```bash
python3 scripts/review_meeting_transcript.py \
	"docs/meetings/AAAA-MM-DD/transcrição-bruta.txt"
```

O resultado padrão é criado na mesma pasta com o nome `transcrição-revisada.txt`.

Opções:

- `transcript`: caminho da transcrição bruta, obrigatório.
- `--output`: caminho alternativo para a versão revisada.
- `--end-time`: ignora segmentos iniciados depois do instante informado em segundos.

Esse script aplica substituições recorrentes definidas em `REPLACEMENTS`, além de algumas normalizações por expressão regular. A revisão automática não garante uma transcrição literal perfeita; trechos incertos devem ser conferidos usando os timestamps e a gravação original.
