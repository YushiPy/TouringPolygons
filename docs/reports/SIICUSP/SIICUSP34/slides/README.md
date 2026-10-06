# Slides — 34º SIICUSP

Quatro slides 16:9 para o pitch de 2 minutos, seguindo o
[guia de slides do IME](https://www.ime.usp.br/~kon/guia-slides-ime.html):
pouco texto, uma figura grande por slide, fundo claro, tamanhos equivalentes a
≥ 24 pt em slide de 33,87 cm e financiamento FAPESP visível.

| # | Tempo | Conteúdo |
| --- | --- | --- |
| 1 | ~20 s | Pergunta do drone + rota ótima da USP (51 regiões, 4,97 km) |
| 2 | ~35 s | Árvore de busca: limite inferior pelo solver convexo, ramificar, podar |
| 3 | ~40 s | Gráfico único: tempo nosso × Fekete et al. nos 550 casos comuns |
| 4 | ~25 s | Matriz ordem fixa/livre × convexo/não convexo, limitações, próximo passo, QR |

O roteiro cronometrável está em [`ROTEIRO.md`](ROTEIRO.md).

## Compilar

```sh
python3 make_scatter_data.py   # só se a campanha de Fekete mudar
mkdir -p /tmp/siicusp34-slides
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/siicusp34-slides slides.tex
cp /tmp/siicusp34-slides/slides.pdf slides.pdf
```

Requer XeLaTeX, Arial e os pacotes `pgfplots`, `qrcode` e `adjustbox`.
O mapa vem de `../poster/figures/usp-route-square.png`; os números vêm do pôster
e de `benchmarks/results-saved/fekete-comparison`. `scatter-data.tex` é gerado
por `make_scatter_data.py`.

## Título como instância (slide 1, protótipo)

O título do slide 1 é uma instância de TPP: cada componente conexa de glifo
(Arial Bold) é uma região e a rota laranja é o caminho mínimo de ordem livre que
as toca, com `s = t` à esquerda. Buracos das letras são ligados ao exterior por
uma fenda fina para manter polígonos simples. As letras são coloridas pela ordem de visita (azul → verde) e a rota pelo
progresso (amarelo → vermelho). São 52 regiões e 1840 vértices (curvas com
tolerância de 0,003 em); o solver (`.build/unordered/tpp`) certificou o ótimo
(gap numérico ≤ 1e-7) em cerca de 90 s. É uma demonstração, não um resultado de desempenho.

```sh
# precisa de fontTools e shapely em qualquer venv
python3 make_title_instance.py --solver ../../../../../.build/unordered/tpp
# regenera title-instance.json e title-art.tex; title-art-body.tex é a
# versão sem as duas primeiras linhas (\def) de title-art.tex
```
