# Pôster A0 — 34º SIICUSP

Fonte editável: `poster.tex`. O PDF tem uma página A0 em retrato (84,1 × 118,9 cm).
`poster.pdf` é a saída local. Os PDFs e intermediários não são versionados.

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
```

A captura da figura tem procedimento e proveniência em
[`figures/README.md`](figures/README.md). Compilar o pôster não recaptura nem
altera o app congelado.

## Decisões editoriais

- O título usa a grafia “Algoritmo Exatos Para o Problema de Visita de
  Polígonos”. A primeira seção apresenta a pergunta do drone, resume o caso
  submetido e distingue o TPP da extensão de ordem livre.
- A primeira seção reúne a introdução e os cards de TSP, TPP e TPP de ordem
  livre. A segunda seção fica dedicada aos trabalhos anteriores.
- As implementações de Dror, Tan–Jiang e Greene aparecem explicitamente como
  trabalho do grupo. O texto atribui meses à implementação dos dois métodos,
  sem atribuir mais de um ano apenas a Dror.
- A rota quadrada da USP preserva as 51 regiões e o caminho completo. Um texto
  abaixo da imagem, limitado à largura dela, descreve a instância, a distância,
  as contagens formais, o número de chamadas ao oráculo e o tempo registrado. A
  geometria e sua aparência vêm do JavaScript original do app. A demonstração está separada
  do corpus de 558 casos.
- O método começa pela escala da enumeração e ilustra ordem parcial, relaxação,
  limites, condição de poda e as duas ramificações em uma só árvore. O exemplo
  IF → Central → IAG não é usado como prova de corte geométrico.
- A legenda da instância da USP deixa explícito que as regiões são alvos, não
  obstáculos, e que regras de voo não são modeladas.
- Os resultados vêm depois do método. Quatro métricas ficam em uma faixa
  compacta, com a explicação do conjunto de instâncias ao lado. A conclusão
  compara o solver às próprias instâncias de Fekete et al.
- As questões abertas mantêm `s = t` ao propor extremos regionais. Não se
  afirma que extremos independentes resolvam todo o TSPN. Chazelle–Dobkin é
  apresentado como possibilidade de partição com pontos de Steiner, sem
  afirmações sobre inexistência de implementações.
- O convite aponta para os três desafios, as instâncias e a simulação do
  algoritmo no webapp. E-mails aparecem junto aos nomes no cabeçalho. A marca
  vertical do IME ocupa a coluna esquerda; SIICUSP e FAPESP ficam na direita,
  com o número do processo alinhado à direita sob a FAPESP. Referências ficam
  no rodapé.

A composição usa corpo de texto de 24 pt, títulos de seção de 42 pt,
referências de 22 pt e uma figura quadrada de 29,5 cm, a aproximadamente 314
dpi. As decisões de
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

## Verificação anterior (25/09/2026)

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
- Uma página A0; todas as fontes incorporadas. A figura quadrada mede
  3456 × 3456 pixels e era usada a 30 cm (cerca de 293 dpi). No cabeçalho, as
  marcas SIICUSP, IME-USP e FAPESP são usadas a aproximadamente 223, 708 e
  828 dpi, respectivamente.
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

## Revisão visual (26/09/2026)

- O TSP está definido como uma ordenação de pontos que minimiza o comprimento
  do caminho; a definição não fixa um ponto inicial nem acrescenta retorno a
  ele. A hifenização automática em português permite que palavras quebrem entre
  linhas sem hífens inseridos manualmente.
- O corpo dos parágrafos usa uma única fonte de 24 pt em todas as seções. Títulos,
  rótulos, métricas, nomes e referências mantêm estilos próprios de hierarquia.
- A seção 1 reúne o contexto e os cards de TSP e TPP, com setas para o card de
  ordem livre. A seção 2 apresenta os resultados de Dror et al. (2003), Tan e
  Jiang (2017) e Fekete et al. (2026), e descreve o uso do solver especializado.
- A seção 3 apresenta o mapa quadrado e, logo abaixo, um texto limitado à sua
  largura; os antigos cards de regiões, distância e espaço formal foram
  removidos. O recorte de 3456 × 3456 pixels é usado a 29,5 cm (cerca de
  314 dpi) e conserva a rota completa e as 51 regiões. O texto registra 6.003
  combinações, correspondentes às chamadas ao oráculo. A legenda permanece
  sobre a borda inferior do mapa, sem linha separadora, com os textos
  “Caminho encontrado”, “s partida · t chegada” e “Cor: primeiras → últimas
  visitas”.
- A compilação XeLaTeX produziu uma página A0. A página completa e as seções 1
  a 5 foram conferidas visualmente; não há caixas excedentes nem caracteres
  ausentes. As quebras naturais de linha foram verificadas no PDF.
- O subtítulo é “Caminhos mínimos para visitar vizinhanças difíceis”; os nomes
  têm o mesmo tamanho e aparecem mais próximos. Os títulos de seção usam
  números sem zero à esquerda; a última seção diz “Tente você mesmo!”.
- Na síntese do resultado polinomial, o texto explicita “No entanto, no TPP
  (ordem fixa)” e define $k$ e $n$ com a redação solicitada pelo autor.
- A nova captura por navegador foi refeita após remover o separador da legenda.
  A cena mantém a grade contínua por trás da legenda e não redesenha os
  polígonos nem a rota.
- Na instância USP, o solver executou 6.003 chamadas ao oráculo usando os
  mesmos hashes registrados para a entrada e o executável. A reexecução
  terminou como `optimal` em 9,168 s. O pôster preserva os 8,77 s da execução
  gravada no `usp-demo.js`.
- A seção 5 usa quatro cards contíguos, separados por barras verticais, com
  barras superiores coloridas e a explicação do corpus ao lado. O pôster foi
  recompilado e conferido em uma renderização A0 de página única.
- O cabeçalho usa a marca vertical simplificada oficial do IME à esquerda e as
  marcas SIICUSP e FAPESP empilhadas à direita. O processo FAPESP fica alinhado
  à direita sob a marca.
