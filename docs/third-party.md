# Repositórios e dados de terceiros

## Solver alemão

O solver de Fekete, Kniep, Krupke e Perk é distribuído sob a licença MIT. A
integração modificada é mantida no fork
[`YushiPy/TSPN-SoCG-2026`](https://github.com/YushiPy/TSPN-SoCG-2026), branch
`codex/touring-polygons-oracle`, e fixada neste repositório pelo submódulo
`tspn-comparison/solver-oracle`. A primeira revisão integrada é
`94db8fa3279f428484e598a3c40c04c470d03600`.

O fork parte do upstream `tubs-alg/TSPN-SoCG-2026` e acrescenta o backend do
oráculo convexo deste projeto, propagação do tempo restante, instrumentação e a
interface Python correspondente. O backend SOCP original continua sendo o
padrão. A licença e o copyright do upstream permanecem no fork.

Clone este projeto com o submódulo:

```bash
git clone --recurse-submodules https://github.com/YushiPy/TouringPolygons.git
```

Em um checkout existente:

```bash
git submodule update --init --recursive
```

Ambientes virtuais, builds, Gurobi, arquivos de licença, caches e resultados não
pertencem ao submódulo nem ao repositório principal.

## Instâncias da Paula

`paula-tspn/` contém material obtido da página pessoal da autora com permissão
para uso e modificação local, mas sem autorização explícita de redistribuição.
Por isso, a pasta inteira permanece ignorada e não deve ser publicada, copiada
para um fork ou adicionada ao histórico Git.

Quando uma instância for necessária para um resultado público, verifique
separadamente a licença e a proveniência do conjunto de dados original. A
permissão sobre o repositório da autora não implica permissão para redistribuir
datasets de terceiros contidos nele.
