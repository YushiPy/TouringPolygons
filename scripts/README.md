# Scripts do repositório

Esta pasta contém utilitários usados pelo projeto Touring Polygons:

- `review_meeting_transcript.py`: aplica correções contextuais recorrentes à transcrição bruta deste projeto.
- `install_dependencies.sh`: prepara as dependências do repositório.
- `sanity_check.sh`: verifica ferramentas, dependências, geração de instâncias, compilação e testes básicos.
- `run_comparison.sh`: compila os solvers e executa ou retoma a comparação alemã de ordem livre.
- `verify_unordered.sh`: executa a verificação focada do solver de ordem livre.

## Comparar os solvers de ordem livre

Em um clone completo do repositório, execute:

```bash
scripts/run_comparison.sh
```

Para preparar as dependências, verificar o runtime/licença do Gurobi e compilar
ambos os solvers sem iniciar os casos, execute
`scripts/run_comparison.sh --setup-only`. Depois inicie ou retome a campanha
com `scripts/run_comparison.sh`. O setup repete até três vezes falhas
transitórias de download de dependências (HTTP 502/503/504 ou timeout).

O script inicializa o submódulo Fekete na revisão fixada pelo repositório,
prepara o ambiente Python 3.12+, compila o binding C++ do Fekete e compila o
nosso solver. Dois patches locais versionados corrigem o header de `fmt` e
compilam as variantes racional e double do oráculo TPP embutido. Enquanto
aplicados, eles aparecem como alterações locais no submódulo; o commit fixado
permanece o mesmo. Fingerprints locais fazem o setup pular a resolução Conan e
a compilação quando as fontes e a configuração não mudaram; um build
interrompido não é marcado como pronto. O setup testa a configuração e a
compilação de C++26 antes de construir o solver. No macOS, se o CMake do
ambiente não reconhecer o AppleClang instalado para `cxx_std_26`, ele tenta o
LLVM do Homebrew. No Linux, se o compilador padrão não passar a verificação,
ele tenta GCC 16, 15 e 14 e Clang 20, 19 e 18 que estejam instalados. `CC` e
`CXX` definidos pelo usuário são respeitados. Depois executa os 558
casos do corpus alemão com endpoints fixos e ordem de visita livre, primeiro
no nosso solver e depois no de Fekete. O
limite padrão por caso é ilimitado; a configuração padrão usa um caso e uma
thread por vez. A máquina precisa ter compilador compatível com C++26, OpenMP,
Eigen3, headers Boost e uma licença acadêmica válida do Gurobi.

`Ctrl+C` pede encerramento cooperativo ao nosso solver e salva a trajetória
incumbente e os limites dos casos ativos. Ele informa que está aguardando o
solver terminar a chamada geométrica em andamento; um segundo `Ctrl+C` força o
encerramento do processo nativo e preserva os checkpoints já gravados. Para o solver Fekete, o runner salva
periodicamente sua melhor trajetória e os limites conhecidos; interrupções
ficam marcadas como `interrupted` e são tentadas novamente ao retomar.

Resultados e checkpoints ficam localmente em
`benchmarks/campaigns/german-free-order-comparison-v1/`. Rodar o comando de
novo reutiliza builds compatíveis sem alterações e retoma a campanha compatível. Para
começar um relatório novo sem apagar o anterior, use `--force`. Opções como `--workers 2`,
`--threads-per-instance 2`, `--build-jobs 12` e `--campaign outro-nome` podem
ajustar a execução. O número de tarefas paralelas deve considerar a memória
disponível; o padrão conservador é um worker.

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
