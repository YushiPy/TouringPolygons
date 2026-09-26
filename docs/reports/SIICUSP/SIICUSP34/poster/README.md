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
- O método começa pela escala da enumeração e explica a integração de ordem e
  peças, a relaxação, os limites e a poda. O exemplo IF → Central → IAG não é
  usado como prova de corte geométrico.
- Os resultados vêm depois do método. Razões, denominadores, protocolo,
  tolerâncias e limitações acompanham os números. “Ótimo” significa fechamento
  numérico dos limites globais e visita verificada.
- As questões abertas mantêm `s = t` ao propor extremos regionais. Não se
  afirma que extremos independentes resolvam todo o TSPN. Chazelle–Dobkin é
  apresentado como possibilidade de partição com pontos de Steiner, sem
  afirmações sobre inexistência de implementações.
- O convite é construir uma rota e compará-la com a solução certificada; não
  sugere superar o ótimo. Referências, contatos, FAPESP e processo foram mantidos.

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
- Inspeção visual da página completa e de recortes ampliados de método,
  resultados, referências e QR; medidas dos blocos verificadas contra
  sobreposição e limites da página.
- Uma página A0; todas as fontes incorporadas. Figura da rota: 6048 × 3456,
  302 dpi; logotipo SIICUSP: 167 dpi; FAPESP: 630 dpi.
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
