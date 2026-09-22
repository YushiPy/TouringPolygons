# Reuniões de orientação

Esta pasta reúne registros locais das reuniões de orientação relacionadas ao
projeto Touring Polygons. Cada reunião deve ficar em uma subpasta com a data no
formato `AAAA-MM-DD`.

Por padrão, seu conteúdo é privado e ignorado pelo Git. Gravações,
transcrições, anotações, `resumo.md` e informações administrativas não devem
ser publicados neste repositório. Somente um `resumo-publico.md` revisado e
anonimizado pode ser versionado.

## Estrutura

```text
meetings/
├── README.md
├── ACOMPANHAMENTO.md
└── AAAA-MM-DD/
    ├── gravação-vídeo.mov
    ├── gravação-áudio.m4a
    ├── transcrição-bruta.txt
    ├── transcrição-bruta.json
    ├── transcrição-revisada.txt
    ├── resumo.md              # privado, ignorado
    ├── resumo-publico.md      # opcional, revisado para o Git
    └── informacoes-extras.md
```

Nem toda reunião precisa conter todos esses arquivos. Reuniões sem gravação, por exemplo, podem ter apenas um resumo e informações complementares.

## Significado dos arquivos

- `ACOMPANHAMENTO.md`: painel cumulativo com contexto, estado atual, tarefas abertas, atividades concluídas, decisões e próxima reunião.
- `gravação-vídeo.mov`: gravação original da reunião, incluindo imagem e áudio.
- `gravação-áudio.m4a`: faixa de áudio extraída do vídeo. Quando o vídeo contém AAC, a extração é feita sem recodificação, evitando perda adicional de qualidade.
- `transcrição-bruta.txt`: saída direta do Whisper, dividida em segmentos com timestamps. Pode conter erros de reconhecimento, especialmente em nomes próprios e termos técnicos.
- `transcrição-bruta.json`: representação estruturada da transcrição. Contém o caminho do áudio, o modelo utilizado, o idioma detectado, a duração e os segmentos com tempos inicial e final.
- `transcrição-revisada.txt`: versão com correções contextuais de nomes e vocabulário técnico. Os timestamps são preservados para permitir conferência no áudio.
- `resumo.md`: síntese privada da reunião, normalmente com pontos discutidos,
  decisões, ações, agenda e contexto administrativo.
- `resumo-publico.md`: versão opcional e revisada, limitada a decisões
  técnicas, tarefas de pesquisa e resultados que possam ser publicados.
- `informacoes-extras.md`: observações ou contexto complementar que não pertencem diretamente ao resumo.

## Convenções

- A data pertence ao nome da subpasta e não deve ser repetida nos nomes dos arquivos.
- Os nomes descrevem a função do arquivo, não o assunto específico da reunião.
- O vídeo é a fonte original. Áudio, transcrições e resumo são artefatos
  derivados e podem ser regenerados.
- Remova nomes completos, contatos, valores, documentos, viagens, saúde,
  processos institucionais e qualquer outro detalhe pessoal antes de criar um
  `resumo-publico.md`.
- A transcrição revisada não substitui a conferência do áudio em passagens ambíguas.
- Após cada reunião, o `ACOMPANHAMENTO.md` deve ser atualizado com as novas tarefas, mudanças de estado e decisões duradouras.

Os comandos para gerar o áudio e as transcrições estão documentados em [`../../scripts/README.md`](../../scripts/README.md).
