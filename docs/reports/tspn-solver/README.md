# O solver do TSPN

Relatório técnico (em português) que explica, de forma didática, o solver de
TSPN: o ciclo convexo de ordem fixa (certificado dual do ciclo, redução à
fronteira, construção por contatos ativos e estágio `binary64`), o
branch-and-bound (âncora de ponto, raiz, ramificação cíclica, limites de
inserção provados em `binary64`, política de busca e memória da fronteira) e a
busca local iterada (subproblema de um contato, vizinhanças, aceitação e
janelas exatas). Pressupõe o relatório `docs/reports/oraculo-convexo` e o
algoritmo de ordem livre em `docs/algorithms/unordered-tpp.md`.

Compilar a partir de qualquer diretório:

```bash
bash docs/reports/tspn-solver/build.sh
```

O script grava `tspn-solver.pdf` nesta pasta (ignorado pelo Git) e os
arquivos auxiliares em `.build/tspn-solver/`.

- `main.tex`: texto, pseudocódigo e bibliografia.
- `figures.tex`: figuras TikZ/pgfplots. Os desenhos geométricos usam
  coordenadas exatas dos exemplos do texto, com a mesma escala nos dois eixos.
- `dados.tex`: a série da figura do piso da fronteira (100i1000-410, dois
  orçamentos de memória, 10/10/2026), com o protocolo no comentário.

Os exemplos numéricos foram conferidos com o solver: as três caixas
(`4+4√2`, ciclo), a inserção da quarta caixa (limite `12,3682`, ótimo
`2√41`, ciclo) e a armadilha da descida por coordenadas (`2√10`, caminho de
ordem livre). Os números de medição vêm de `benchmarks/results-saved/README.md`;
as instâncias da Paula são locais e não estão no repositório.
