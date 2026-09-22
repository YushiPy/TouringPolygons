# Contraexemplos das podas geométricas de ordem livre

Estes casos se aplicam ao solver de **ordem livre**. Os protótipos de poda não
foram mantidos no solver de produção.

## Cruzamento em uma relaxação parcial

Arquivo reproduzível:
`packages/nonconvex-tpp/cpp/tests/unordered-partial-crossing-counterexample.txt`.

O nó que não pode ser podado é:

```text
node = 11
parent = 5
sequência parcial = [3, 2, 5, 0]
lower bound = 31.897787422310348
incumbente naquele instante = 33.90214812381971
caminho =
  (0, 0)
  (2.877892972001312, 5.240053825127741)       P3
  (0.4034433097024684, 4.90775689710519)       P2
  (0.25329225192562765, -3.7204849013977306)   P5
  (10.76293634693957, -4.203823805715689)      P0
  (10, 0)
```

Os segmentos 0 e 2 cruzam-se propriamente em
`(0.3284442414374704, 0.5980297114693436)`. Mesmo assim, esse nó parcial
precisa permanecer na árvore. Os polígonos ainda ausentes `P4` e `P1` podem ser
inseridos entre os elementos existentes, substituindo os segmentos que se cruzam.

A extensão ótima tem ordem:

```text
[4, 3, 2, 1, 5, 0]
```

e comprimento `32.69461492061567`. A poda agressiva elimina o nó 11 e certifica
incorretamente `32.816118329243004`.

Isto não é uma falha da versão realmente segura: se um caminho já visita todos
os polígonos, o 2-opt de um cruzamento próprio preserva as visitas. O erro é aplicar
essa conclusão à relaxação de uma sequência **parcial**.

## Saída e reentrada no mesmo polígono

Arquivo reproduzível:
`packages/nonconvex-tpp/cpp/tests/unordered-reentry-counterexample.txt`.

É o caso 28 de `benchmarks/suites/canonical-v1.bin`, SHA-256
`c867e685ea1dd7877b7332135aa3348b5e6e93ec8fe7c40fa642f10cb11dd1f3`.

O primeiro nó podado incorretamente — e o que pertence à ordem ótima — é:

```text
node = 105
parent = 104
sequência parcial = [2, 3, 12, 5, 16, 6, 17]
lower bound = 183.41732412531567
incumbente naquele instante = 192.19805172803103
caminho =
  (0, 0)
  (7.64, 18.19)       P2
  (36.14, 19)         P3
  (50.93, 14)         P12
  (65.98, 19.08)      P5 e primeira visita incidental a P6
  (87.03, 13.21)      P16
  (73.56, 32.29)      segunda visita a P6
  (66.18, 57.54)      P17
  (95.07, 71.68)
```

O caminho toca `P6` primeiro no ponto compartilhado com `P5`, sai para visitar
`P16` e volta a `P6`. As posições acumuladas das duas visitas no caminho são
aproximadamente `79.7374` e `124.9462`.

A sequência do nó 105 é uma subsequência da ordem ótima completa:

```text
[2, 0, 9, 10, 1, 3, 11, 12, 4, 5, 13, 14, 15, 16, 7, 8, 6, 17, 18, 19]
```

O ótimo é `190.0004629646404`. Ao eliminar reentradas, o solver remove esse ramo
e certifica incorretamente `191.3412595263247`.

Esse exemplo também mostra por que a regra não pode considerar toda visita
incidental como parte da ordem: `P6` é tocado na fronteira compartilhada com `P5`
antes de sua visita que representa a posição escolhida na sequência.
