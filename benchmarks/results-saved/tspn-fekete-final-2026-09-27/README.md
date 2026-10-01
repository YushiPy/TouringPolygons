# TSPN — recaptura do baseline Fekete (2026-09-27)

Recaptura de 25 instâncias, três repetições, com B&B nativo e B&B SOCP de
Fekete. O nativo retornou tours válidos nos 25 casos; um caso de 15 regiões
permaneceu com gap aberto. Fekete também retornou tours válidos, mas não fechou
o gap em diversos casos. Comparações temporais só são válidas em pares com gap
fechado por ambos; bounds do SOCP são numéricos, não certificados exatos.

A campanha posterior de 33 instâncias após correções de contatos ativos está em
[`tspn-active-contacts-2026-09-28`](../tspn-active-contacts-2026-09-28/README.md).
Este resumo substitui os raws e relatórios duplicados.
