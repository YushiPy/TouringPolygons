# Pôster A0 — 34º SIICUSP

Fonte editável: `poster.tex`. O PDF tem uma página A0 em retrato (84,1 × 118,9 cm).
`poster.pdf` é a saída local; `Gabriel Freire Ushijima.pdf` é a cópia de entrega
com o nome solicitado pela organização. Os PDFs e intermediários não são
versionados.

## Compilar

Requer XeLaTeX, Arial, Latin Modern Math e os pacotes usados no preâmbulo.
A fonte matemática é indicada pelo arquivo `latinmodern-math.otf`, encontrado
pelo TeX Live, para evitar depender do cadastro de fontes do sistema.

A partir desta pasta:

```sh
mkdir -p /tmp/siicusp34-poster-build
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/siicusp34-poster-build poster.tex
cp /tmp/siicusp34-poster-build/poster.pdf poster.pdf
cp poster.pdf 'Gabriel Freire Ushijima.pdf'
```

A captura da figura tem procedimento e proveniência em
[`figures/README.md`](figures/README.md). Compilar o pôster não recaptura nem
altera o app congelado.

## Decisões editoriais

- Título idêntico ao resumo. A primeira seção apresenta a pergunta do drone e
  distingue o caso não convexo de ordem fixa, submetido no resumo, da extensão
  posterior de ordem livre.
- A segunda seção define TSP, TPP e ordem livre antes de apresentar a base
  geométrica e a oportunidade identificada no trabalho de Fekete et al.
- As implementações de Dror, Tan–Jiang e Greene aparecem explicitamente como
  trabalho do grupo. O texto atribui meses à implementação dos dois métodos,
  sem atribuir mais de um ano apenas a Dror.
- A rota da USP é a figura central. As anotações aproveitam sua margem vazia;
  a geometria e sua aparência vêm do JavaScript original do app. A demonstração
  está separada do corpus de 558 casos.
- O método começa pela escala da enumeração e ilustra ordem parcial, relaxação,
  limites, condição de poda e as duas ramificações em uma só árvore. O exemplo
  IF → Central → IAG não é usado como prova de corte geométrico.
- A legenda da instância da USP deixa explícito que as regiões são alvos, não
  obstáculos, e que regras de voo não são modeladas.
- Os resultados vêm depois do método. Razões, denominadores, protocolo,
  tolerâncias e limitações acompanham os números. “Ótimo” significa fechamento
  numérico dos limites globais e visita verificada.
- Uma faixa de conclusão resume o alcance dos resultados: limites fechados e
  visitas validadas nos 558 casos desta campanha, sem generalização para outros
  corpora.
- As questões abertas mantêm `s = t` ao propor extremos regionais. Não se
  afirma que extremos independentes resolvam todo o TSPN. Chazelle–Dobkin é
  apresentado como possibilidade de partição com pontos de Steiner, sem
  afirmações sobre inexistência de implementações.
- O convite é construir uma rota e compará-la com a solução certificada; não
  sugere superar o ótimo. E-mails aparecem junto aos nomes no cabeçalho; a logo
  da FAPESP e o número do processo ficam no alto à direita, logo abaixo da
  marca do SIICUSP. Referências permanecem no rodapé.

A composição usa texto principal de 30 pt, títulos de seção de 42 pt,
referências de 22 pt e uma figura a aproximadamente 302 dpi. As decisões de
hierarquia, respiro e densidade foram orientadas pelo
[guia de pôster do IME](https://www.ime.usp.br/~kon/guia-poster-ime.html),
respeitando a ordem narrativa escolhida pelo autor.

## Fontes e conferências de conteúdo

- [Resumo submetido](../resumo/resumo-pt.tex), [plano de trabalho](../PLANO-DE-TRABALHO.md),
  edital e critérios locais.
- [Contrato da ordem livre](../../../../algorithms/unordered-tpp.md).
- [Campanha preservada](../../../../../benchmarks/results-saved/german-comparison/README.md)
  e seu `report.md`; números de tempo reconferidos por `case_index` e SHA-256
  diretamente em `ours.csv` e `fekete.csv`.
- [Dados da demonstração](../../../../../apps/siicusp34/data/USP-DEMO.md) e app:
  51 regiões, 4969,523045289922 m; 186 registros de busca; três desafios.
- No mapa, as 51! ordens possíveis e o produto das alternativas da decomposição
  convexa ótima de cada região dão aproximadamente 3,87 × 10^103 combinações
  formais: 1,55 × 10^66 ordens e 2,49 × 10^37 escolhas de peças. O registro da
  demonstração informa uma resolução em 8,771744792 s, em uma execução local e
  uma thread. O solver não enumerou esse espaço.
- Caso 49 no pôster corresponde a `case_index = 48`, com 60 regiões e tempo
  preservado de 2,473421125 s. A contagem formal foi reconferida chamando a
  biblioteca C++ existente pela interface `apps/siicusp34/scripts/partition_usp.cpp`:
  `60! × produto(p_i) ≈ 1,56166322 × 10^117`. Essa conta não representa nós
  efetivamente enumerados e não é uma nova execução de benchmark.
- A métrica de precisão é o máximo, sobre as regiões, da distância euclidiana
  mínima entre cada região e a polilinha inteira. Os denominadores 558 e 551
  contam trajetórias disponíveis, não apenas casos concluídos em comum.
- [Chazelle–Dobkin, 1985](https://www.cs.princeton.edu/~chazelle/pubs/OptimalConvexDecomp.pdf)
  confirma o problema de partição convexa mínima com pontos de Steiner.

## Verificação desta revisão (25/09/2026)

- Compilação XeLaTeX/latexmk concluída; ausência de erros, caixas excedentes e
  caracteres ausentes no log.
- Cabeçalho com e-mails logo abaixo dos nomes e FAPESP/processo abaixo da marca
  SIICUSP, no alto à direita.
- Inspeção visual da página completa e de recortes ampliados do cabeçalho,
  quadro combinatório no mapa, método, resultados, referências e QR; medidas
  dos blocos verificadas contra sobreposição e limites da página.
- A caixa da instância USP usa os dados registrados em `usp-demo.js`; os
  números de ordens e de escolhas de peças são contagens formais do espaço,
  não nós efetivamente visitados pela busca.
- Uma página A0; todas as fontes incorporadas. Figura da rota: 6048 × 3456,
  302 dpi; logos SIICUSP: 119 dpi e FAPESP: 712 dpi no tamanho atual.
- Captura real do app sem erros de JavaScript; 51 polígonos; rota laranja
  original com `stroke-width: 3px` e sem filtro. Fontes e imagem com hashes.
- `node --check capture_usp_route.mjs` aprovado.
- QR decodificado por Core Image/macOS a partir de uma renderização do PDF:
  `https://yushipy.github.io/TouringPolygons/apps/siicusp34/`.
- Destino do QR e `data/trace-data.js` responderam HTTP 200. Isso é verificação
  digital; não foi realizado teste com câmera de celular nem impressão física.
- Não foram executadas as suítes gerais de C++/dashboard: nenhuma implementação
  algorítmica, dado de campanha ou aplicação foi alterada. As verificações
  diretamente afetadas foram compilação, captura, conteúdo e apresentação.
