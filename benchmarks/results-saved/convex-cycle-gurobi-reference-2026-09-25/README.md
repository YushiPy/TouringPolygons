# Fixture Gurobi para ciclos convexos

`instances.json` contém quatro casos pequenos de polígonos convexos disjuntos,
na ordem cíclica fixa. O fixture é consumido pelos testes C++ e pelos comandos
`cycle-benchmark` e `tspn-benchmark`. `reference.json` preserva somente os
campos necessários à comparação dos testes: nome, status numérico, objetivo,
bound e contatos factíveis, mais as tolerâncias absoluta e relativa. Os
polígonos não são duplicados nesse resumo.

No comparativo independente, os ótimos racionais passaram o certificado exato.
A maior diferença reportada entre os objetivos C++ e Gurobi foi cerca de
`1,02e-6`; intervalos certificados sobrepuseram os resultados Gurobi. Bounds
Gurobi são numéricos. A implementação double padrão permite recuperação
racional; a variante double pura pode atingir `FloatingPointLimit` ou falhar.

O estado `2` dos registros significa `OPTIMAL` segundo o status numérico de
Gurobi 13.0.3. Isso não é um certificado exato: os objetivos e bounds externos
são numéricos, e `feasible_contacts` são apenas candidatos que os testes
certificam independentemente em aritmética racional. A origem dos valores
compactados é o commit `48b2729f5fc8c2d2ca0d78a66dc85af04c285a91`, anterior à
compactação dos resultados. O C++ certifica os ótimos; o fixture não preserva
licenças, executáveis, logs ou saída bruta do solver externo.
