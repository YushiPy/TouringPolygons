# Slides — 34º SIICUSP

Quatro slides 16:9 para o pitch de 2 minutos, seguindo o
[guia de slides do IME](https://www.ime.usp.br/~kon/guia-slides-ime.html):
pouco texto, uma figura grande por slide, fundo claro, tamanhos equivalentes a
≥ 24 pt em slide de 33,87 cm e financiamento FAPESP visível.

| # | Conteúdo |
| --- | --- |
| 1 | Capa: título como instância de TPP (letras = regiões, rota ótima), pergunta, autor, orientador, FAPESP e processo |
| 2 | O problema: dados (s, t, P₁…Pₖ) e objetivo, com a instância da USP e a rota ótima |
| 3 | Método: busca real (caso 17 do corpus, 5 regiões) com mapas de cada nó, limite inferior L, ramificar e podar |
| 4 | Resultados atualizados (dantzig; nossos tempos de 2026-10-08, Fekete de 2026-10-02): histograma log do speedup (Fekete/nosso) nos 553 casos comuns, por tipo de instância, com os números do pôster ao lado |
| 5 | Convite ao pôster (três perguntas que ele responde) e QR code do app (com a marca do app) |

São cinco slides (o plano de trabalho previa quatro); a capa, o slide 2 e o slide 5 são curtos.
O roteiro cronometrável está em [`ROTEIRO.md`](ROTEIRO.md).

## Compilar

```sh
python3 make_hist_data.py   # só se a campanha mudar (float-recovery-dantzig-2026-10-08)
mkdir -p /tmp/siicusp34-slides
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/siicusp34-slides slides.tex
cp /tmp/siicusp34-slides/slides.pdf slides.pdf
```

Requer XeLaTeX, Arial e os pacotes `pgfplots`, `qrcode` e `adjustbox`.
O mapa vem de `../poster/figures/usp-route-square.png`; os números vêm do pôster
e de `benchmarks/results-saved/float-recovery-dantzig-2026-10-08/per-case.csv`. `hist-data.tex` é gerado
por `make_hist_data.py`.

## Título como instância (slide 1, protótipo)

O título do slide 1 é uma instância de TPP: cada componente conexa de glifo
(Arial Bold) é uma região e a rota laranja é o caminho mínimo de ordem livre que
as toca, com `s = t` à esquerda. O solver recebe só o contorno externo de cada
componente: tocar o glifo equivale a tocar seu contorno externo, então os buracos
(a, e, o, g, P…) não alteram a rota; o slide os desenha normalmente. As letras são coloridas pela ordem de visita (teal claro → azul-petróleo) e a rota pelo
progresso (laranja claro → escuro). São 52 regiões e 1380 vértices (curvas com
tolerância de 0,003 em); o solver (`.build/unordered/tpp`) certificou o ótimo
(gap numérico ≤ 1e-7) em cerca de 1 min. É uma demonstração, não um resultado de desempenho.

```sh
# precisa de fontTools e shapely em qualquer venv
python3 make_title_instance.py --solver ../../../../../.build/unordered/tpp
# regenera title-instance.json e title-art.tex; title-art-body.tex é a
# versão sem as duas primeiras linhas (\def) de title-art.tex
```

`make_usp_art.py` redesenha a instância da USP (dados de `apps/siicusp34/data/usp-demo.js`, somente leitura) com a paleta do deck; `palette.py` guarda as rampas de cor.

`make_trace_art.py` desenha a árvore do slide 3 a partir de um rastro real: caso 17 do corpus (conferido por SHA-256) e `apps/siicusp34/data/trace-data.js`, ambos somente leitura.
No slide 3 os comprimentos são relativos à rota ótima (=1). A rota inicial é a do vizinho mais próximo registrada no rastro real (1,26); o ramo descartado mostra o comprimento exato da melhor rota para aquela ordem (1,42).
