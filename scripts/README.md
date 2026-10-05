# Scripts do repositório

Utilitários de linha de comando. A lógica de benchmark vive em `benchmarks/_internal/`
e é exposta por `python3 benchmarks/tpp.py` (a única CLI pública de benchmark); os
atalhos `.sh` abaixo só preparam o Python do `tpp.py` e o chamam.

| Script | O que faz |
|---|---|
| `install_dependencies.sh` | prepara as dependências do repositório (sistema, Python, Node) |
| `sanity_check.sh` | verifica ferramentas, dependências, geração de instâncias, compilação e testes básicos |
| `verify_unordered.sh` | compila e executa a verificação focada do solver de ordem livre |
| `benchmark.sh` | atalho para `tpp.py tui`: monta (e imprime, copia ou executa) um comando de benchmark |
| `run_comparison.sh` | atalho para `tpp.py free-compare`: prepara e executa/retoma a comparação de ordem livre com os 558 casos de Fekete |
| `run_tspn_comparison.sh` | atalho para `tpp.py tspn-compare`: prepara e executa/retoma a campanha TSPN completa |
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
