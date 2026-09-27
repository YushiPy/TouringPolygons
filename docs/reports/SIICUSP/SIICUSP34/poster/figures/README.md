# Figuras do pôster

## Rota da USP

`usp-route-print.png` é uma **captura da renderização JavaScript real** de
`apps/siicusp34`, produzida por `../capture_usp_route.mjs` com Playwright/Chromium.
A captura tem **6048 × 3456 pixels** e serve como fonte. `usp-route-square.png`
é a captura quadrada da mesma cena, com **3456 × 3456 pixels**, usada no pôster
a 29,5 cm (cerca de **314 dpi**). Não há um segundo renderizador de polígonos.

O script abre a publicação estática local em 1440 × 1100 CSS pixels, com fator
de pixels 6, espera `app.js` construir o SVG e captura somente `#route-map`.
Mantém a rota completa, as 51 regiões, o gradiente original, os blocos do IME
em dourado, o fundo e a grade. Não altera geometria, projeção, cores ou
espessuras. O único controle ocultado é o zoom flutuante. Não adiciona contorno
à rota laranja. A imagem quadrada recorta a cena em x=216..792 CSS px e
y=0..576 CSS px: conserva a rota e todas as regiões, retirando as margens
laterais vazias. O texto vetorial abaixo do mapa apresenta a instância, a
distância, as contagens formais, as chamadas ao oráculo e o tempo, limitado à
largura da imagem; não há cards laterais sobrepostos ao mapa.

`usp-route-capture.json` registra versão do Chromium, dimensões das duas
imagens, propriedades da cena, limites da solução e hashes SHA-256 das fontes
e das imagens. A origem
dos contornos e a formulação estão em
[`USP-DEMO.md`](../../../../../../apps/siicusp34/data/USP-DEMO.md).

Para refazer a captura, a partir da raiz do repositório:

```sh
python3 apps/siicusp34/scripts/serve.py --port 8773
# Em outro terminal, com o Playwright do dashboard já instalado:
node docs/reports/SIICUSP/SIICUSP34/poster/capture_usp_route.mjs
```

A geometria, a projeção e a rota vêm do visualizador existente; esta revisão
altera somente a apresentação da legenda. O antigo `render_usp_route.py`, que
redesenhava a cena e acrescentava um contorno à rota, foi substituído pelo
capturador.

## Marca FAPESP

`fapesp-logo.svg` é a
[variante preta da marca FAPESP publicada no Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Logo_fapesp_em_preto.svg),
atribuída à FAPESP. `fapesp-logo.png` é a conversão para inclusão no PDF.
O autor confirmou autorização para utilizar a marca neste pôster.

## Marca IME-USP

`ime-usp-vertical-logo.png` é a marca vertical simplificada obtida pela página
oficial de [Identidade Institucional do IME-USP](https://www.ime.usp.br/identidade-visual/).
Ela integra a efígie de Arquimedes e a sigla IME.

Nenhuma das propostas genéricas `image.png`, `image2.png` ou `image3.png` foi usada.
