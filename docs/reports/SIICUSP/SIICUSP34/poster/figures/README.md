# Figuras do pôster

## Rota da USP

`usp-route-print.png` é uma **captura da renderização JavaScript real** de
`apps/siicusp34`, produzida por `../capture_usp_route.mjs` com Playwright/Chromium.
A captura tem **6048 × 3456 pixels**, cerca de **302 dpi** na largura de 50,8 cm
usada no A0. Não há um segundo renderizador de polígonos.

O script abre a publicação estática local em 1440 × 1100 CSS pixels, com fator
de pixels 6, espera `app.js` construir o SVG e captura somente `#route-map`.
Mantém a rota completa, as 51 regiões, o gradiente original, os blocos do IME
em dourado, o fundo e a grade. Não altera geometria, projeção, cores ou
espessuras. O único controle ocultado é o zoom flutuante. Não adiciona contorno
à rota laranja. As anotações à esquerda da figura são texto vetorial do LaTeX,
na margem vazia da cena.

`usp-route-capture.json` registra versão do Chromium, dimensões, propriedades
da cena, limites da solução e hashes SHA-256 das fontes e da imagem. A origem
dos contornos e a formulação estão em
[`USP-DEMO.md`](../../../../../../apps/siicusp34/data/USP-DEMO.md).

Para refazer a captura, a partir da raiz do repositório:

```sh
python3 apps/siicusp34/scripts/serve.py --port 8773
# Em outro terminal, com o Playwright do dashboard já instalado:
node docs/reports/SIICUSP/SIICUSP34/poster/capture_usp_route.mjs
```

O app congelado não é modificado. O antigo `render_usp_route.py`, que redesenhava
a cena e acrescentava um contorno à rota, foi substituído pelo capturador.

## Marca FAPESP

`fapesp-logo.svg` é a
[variante preta da marca FAPESP publicada no Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Logo_fapesp_em_preto.svg),
atribuída à FAPESP. `fapesp-logo.png` é a conversão para inclusão no PDF.
O autor confirmou autorização para utilizar a marca neste pôster.

Nenhuma das propostas genéricas `image.png`, `image2.png` ou `image3.png` foi usada.
