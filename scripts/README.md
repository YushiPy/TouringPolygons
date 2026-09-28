# Scripts do repositório

Esta pasta contém utilitários usados pelo projeto Touring Polygons:

- `review_meeting_transcript.py`: aplica correções contextuais recorrentes à transcrição bruta deste projeto.
- `install_dependencies.sh`: prepara as dependências do repositório.
- `sanity_check.sh`: verifica ferramentas, dependências, geração de instâncias, compilação e testes básicos.
- `verify_unordered.sh`: executa a verificação focada do solver de ordem livre.

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
