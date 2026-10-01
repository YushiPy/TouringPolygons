# Fixture Gurobi para ciclos convexos

`instances.json` contém quatro casos pequenos de polígonos convexos disjuntos,
na ordem cíclica fixa. O fixture é consumido pelos testes C++ e pelos comandos
`cycle-benchmark` e `tspn-benchmark`; por isso é o único dado de entrada
preservado fora de `fekete-comparison`.

No comparativo independente, os ótimos racionais passaram o certificado exato.
A maior diferença reportada entre os objetivos C++ e Gurobi foi cerca de
`1,02e-6`; intervalos certificados sobrepuseram os resultados Gurobi. Bounds
Gurobi são numéricos. A implementação double padrão permite recuperação
racional; a variante double pura pode atingir `FloatingPointLimit` ou falhar.

SHA-256 do fixture: `fbee6f57c3c4cc4133f32b51b57555792cc3e565bd78b1aaf2e415b77c41ffbf`.
O C++ certifica os ótimos; o fixture não preserva licenças, executáveis, logs
ou saída bruta do solver externo.
