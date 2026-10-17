# Scripts do repositório

Utilitários de linha de comando. A lógica de benchmark vive em `benchmarks/_internal/`
e é exposta por `python3 benchmarks/tpp.py` (a única CLI pública de benchmark); os
atalhos `.sh` abaixo só preparam o Python do `tpp.py` e o chamam.

| Script | O que faz |
|---|---|
| `install_dependencies.sh` | prepara as dependências do repositório, **sem sudo**: baixa o que falta e pode ser obtido sem administrador (uv, Python 3.12, headers, o solver original de Fekete et al. e seus pacotes); só lista o que exige administrador. Usa o Homebrew no macOS quando existe. `--solvers-only` pula Node, dashboard e Playwright |
| `sanity_check.sh` | verifica dependências, compila os solvers e roda os testes C++ e Python em cerca de um minuto (com o build pronto). `--full` acrescenta as suítes convexas de estresse, cerca de 25 minutos a mais. Não roda benchmarks |
| `verify_unordered.sh` | compila e executa a verificação focada do solver de ordem livre |
| `benchmark.sh` | atalho para `tpp.py tui`: monta (e imprime, copia ou executa) um comando de benchmark |
| `run_comparison.sh` | atalho para `tpp.py free-compare`: prepara e executa/retoma a comparação de ordem livre com os 558 casos de Fekete |
| `run_tspn_comparison.sh` | atalho para `tpp.py tspn-compare`: prepara e executa/retoma a campanha TSPN completa |
| `run_paula_campaign.sh` | atalho para `tpp.py paula-campaign`: prepara tudo sem sudo (uv, headers, GMP, solver e seus testes) e executa/retoma a campanha longa de ILS + B&B nas instâncias da Paula (`--detach`, `--report`) |
| `run_fekete_paper.sh [SEGUNDOS]` | configuração **A** do código original de Fekete et al. nas 558 instâncias do artigo (raiz CHR, uma execução por vez com uma thread por CPU lógica dos núcleos de performance, 3 repetições; `--threads N` escolhe outro número); em segundo plano e retomável |
| `run_fekete_single_thread.sh [SEGUNDOS]` | configuração **B** (uma thread, raízes CHR e LEFP, uma execução por núcleo de performance, 3 repetições); espera a A terminar se ela estiver rodando |
| `run_ours_single_thread.sh [SEGUNDOS]` | configuração **C**: o nosso solver no protocolo da B, com as opções da comparação de 16/10; o diretório leva o commit do nosso código |
| `run_fekete_paper_lefp.sh [SEGUNDOS]` | configuração **D**: como a A, com a raiz LEFP |
| `run_ours_threads.sh [SEGUNDOS]` | configuração **E**: o nosso solver no protocolo da A, com as mesmas threads |
| `machine_specs.sh` | imprime, em Markdown e sem sudo, o hardware e o software desta máquina para `benchmarks/machines.md` |
| `review_meeting_transcript.py` | revisão contextual de transcrições de reunião (abaixo) |

Os atalhos aceitam as mesmas opções dos subcomandos (`scripts/run_comparison.sh --help`).
`TPP_PYTHON` escolhe o Python (3.12 ou mais novo). Os comandos, as opções, os
resultados e o setup do Fekete estão documentados em
[`../benchmarks/README.md`](../benchmarks/README.md).

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

O script aplica substituições recorrentes definidas em `REPLACEMENTS`, além de algumas
normalizações por expressão regular. A revisão automática não garante uma transcrição
literal perfeita; trechos incertos devem ser conferidos usando os timestamps e a gravação
original. Ele fica aqui porque `docs/meetings/` é privado e ignorado pelo Git.
