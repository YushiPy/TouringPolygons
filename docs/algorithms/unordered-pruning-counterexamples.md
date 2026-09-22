# Contraexemplos de podas no TPP de ordem livre

Estes casos protegem o solver de ordem livre contra duas podas geométricas
incorretas. Os protótipos não fazem parte do solver de produção; os fixtures são
mantidos para impedir sua reintrodução sem uma nova prova.

## Cruzamento em uma relaxação parcial

Fixture:
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
`(0.3284442414374704, 0.5980297114693436)`. Mesmo assim, esse nó parcial deve
permanecer na árvore. Os polígonos ainda ausentes `P4` e `P1` podem ser
inseridos entre elementos existentes, substituindo os segmentos que cruzam.

A extensão ótima tem ordem `[4, 3, 2, 1, 5, 0]` e comprimento
`32.69461492061567`. A poda por cruzamento certificaria incorretamente
`32.816118329243004`.

Se um caminho completo já visita todos os polígonos, o 2-opt de um cruzamento
próprio preserva as visitas. O erro é estender essa conclusão à relaxação de
uma sequência parcial.

## Saída e reentrada no mesmo polígono

Fixture: `packages/nonconvex-tpp/cpp/tests/unordered-reentry-counterexample.txt`.

O primeiro nó podado incorretamente que pertence à ordem ótima é:

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

O caminho toca `P6` na fronteira compartilhada com `P5`, sai para visitar
`P16` e retorna a `P6`. A sequência do nó é subsequência da ordem ótima:

```text
[2, 0, 9, 10, 1, 3, 11, 12, 4, 5, 13, 14, 15, 16, 7, 8, 6, 17, 18, 19]
```

O ótimo é `190.0004629646404`. Eliminar reentradas remove esse ramo e
certifica incorretamente `191.3412595263247`. Uma visita incidental não pode
ser tratada automaticamente como a posição escolhida do polígono na ordem.
