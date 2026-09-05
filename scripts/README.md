# Scripts

Esta pasta contém utilitários do repositório. Os scripts relacionados às gravações de reuniões são:

- `transcribe_meeting.py`: extrai a faixa de áudio de um vídeo e produz a transcrição bruta em TXT e JSON.
- `review_meeting_transcript.py`: aplica correções contextuais recorrentes à transcrição bruta e gera a transcrição revisada.
- `sanity_check.sh`: verifica ferramentas, dependências, geração de instâncias, compilação e testes básicos do projeto. Ele não faz parte do fluxo de transcrição.

## Dependências

O fluxo de transcrição requer Python 3.12 ou mais recente, `ffmpeg` e `faster-whisper`.

No macOS, `ffmpeg` pode ser instalado com Homebrew:

```bash
brew install ffmpeg
```

Considerando o ambiente virtual `.venv` na raiz do repositório, instale a dependência Python com:

```bash
uv pip install --python .venv/bin/python faster-whisper
```

O modelo Whisper escolhido é baixado automaticamente na primeira execução e mantido no cache local.

## Fluxo completo

Execute os comandos a partir da raiz do repositório.

### 1. Converter o vídeo e transcrever

```bash
.venv/bin/python scripts/transcribe_meeting.py \
	"docs/meetings/AAAA-MM-DD/gravação-vídeo.mov" \
	--model medium \
	--language pt \
	--prompt "Reunião sobre TPP, TSPN, Branch and Bound, CGAL, Gurobi, Dror, Tan e Jiang."
```

Por padrão, os resultados são gravados na mesma pasta do vídeo:

```text
gravação-áudio.m4a
transcrição-bruta.txt
transcrição-bruta.json
```

O áudio é extraído sem recodificação. Isso preserva o fluxo AAC existente no vídeo e evita uma nova perda de qualidade.

O argumento `--prompt` é opcional, mas ajuda o modelo a reconhecer nomes próprios, siglas e termos técnicos. Ele deve conter apenas contexto provável, sem tentar antecipar frases inteiras da reunião.

### 2. Revisar a transcrição

```bash
.venv/bin/python scripts/review_meeting_transcript.py \
	"docs/meetings/AAAA-MM-DD/transcrição-bruta.txt"
```

O resultado padrão é criado na mesma pasta com o nome `transcrição-revisada.txt`.

Esse script aplica substituições recorrentes definidas em `REPLACEMENTS`, além de algumas normalizações por expressão regular. A lista atual é voltada ao vocabulário deste projeto e pode ser ampliada quando novos erros sistemáticos forem encontrados. A revisão automática não garante uma transcrição literal perfeita; trechos incertos devem ser conferidos usando os timestamps.

## Opções de `transcribe_meeting.py`

- `video`: caminho do vídeo de entrada, argumento obrigatório.
- `--model`: modelo do Whisper. O padrão é `small`; `medium` tende a reconhecer melhor fala técnica, mas exige mais memória e processamento.
- `--language`: código do idioma falado. O padrão é `pt`.
- `--output-dir`: pasta de saída. Quando omitida, usa a pasta do vídeo.
- `--prompt`: contexto inicial com vocabulário esperado.
- `--skip-audio`: reutiliza `gravação-áudio.m4a` da pasta de saída em vez de extraí-lo novamente.

Exemplo reutilizando um áudio já extraído:

```bash
.venv/bin/python scripts/transcribe_meeting.py \
	"docs/meetings/AAAA-MM-DD/gravação-vídeo.mov" \
	--skip-audio \
	--model medium
```

## Opções de `review_meeting_transcript.py`

- `transcript`: caminho da transcrição bruta, argumento obrigatório.
- `--output`: caminho alternativo para a versão revisada. Quando omitido, gera `transcrição-revisada.txt` ao lado do arquivo de entrada.
- `--end-time`: ignora segmentos iniciados depois do instante informado em segundos. É útil quando a gravação continua após o encerramento da reunião.

Exemplo com destino personalizado:

```bash
.venv/bin/python scripts/review_meeting_transcript.py \
	"/caminho/transcrição-bruta.txt" \
	--output "/caminho/transcrição-revisada.txt"
```

Exemplo limitando a revisão à primeira hora e quatro minutos:

```bash
.venv/bin/python scripts/review_meeting_transcript.py \
	"/caminho/transcrição-bruta.txt" \
	--end-time 3852
```

## Sobrescrita de arquivos

Os scripts substituem arquivos de saída com os mesmos nomes. Preserve alterações manuais importantes em outro arquivo antes de executar novamente o fluxo.
