# Repositórios e dados de terceiros

## solver de Fekete et al.

O solver de Fekete, Kniep, Krupke e Perk é distribuído sob a licença MIT. A
integração modificada é mantida no fork
[`YushiPy/TSPN-SoCG-2026`](https://github.com/YushiPy/TSPN-SoCG-2026), branch
`codex/touring-polygons-oracle`, e fixada neste repositório pelo submódulo
`third_party/tspn-socg`. A primeira revisão integrada é
`94db8fa3279f428484e598a3c40c04c470d03600`; a fixada hoje é
`e91e936502ad32a5d5b3cf3f5a002cda739948e2`, que acrescenta duas correções de
compatibilidade (o header de `fmt` nos bindings Python e a variante binary64 do
mapa direcional no oráculo embutido) sem mudar o solver.

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

As medições de referência do lado do Fekete (`tpp.py fekete-paper` e
`fekete-single-thread`) não usam o fork. Elas rodam o upstream sem modificações,
`tubs-alg/TSPN-SoCG-2026` no commit `5cd84507c2409caf47c524f60bdd3b96453f0e15`
(o `main` deles e a base do fork). O código é clonado e compilado em
`.build/fekete-upstream/` por `tpp.py fekete-setup`, que faz parte de
`scripts/install_dependencies.sh`. Veja
[`benchmarks/README.md`](../benchmarks/README.md#código-original-como-no-artigo-configurações-a-e-b).

O submódulo continua sendo um projeto externo, não um pacote mantido deste
monorepo. Adaptadores, campanhas e resultados de comparação ficam em
`benchmarks/`; mudanças necessárias ao solver externo são feitas no fork e
fixadas aqui por revisão.

Ambientes virtuais, builds, Gurobi, arquivos de licença, caches e resultados não
pertencem ao submódulo nem ao repositório principal.

## Instâncias da Paula

`third_party/paula-tspn/` contém material obtido da página pessoal da autora com permissão
para uso e modificação local, mas sem autorização explícita de redistribuição.
Por isso, a pasta inteira permanece ignorada e não deve ser publicada, copiada
para um fork ou adicionada ao histórico Git.

Quando uma instância for necessária para um resultado público, verifique
separadamente a licença e a proveniência do conjunto de dados original. A
permissão sobre o repositório da autora não implica permissão para redistribuir
datasets de terceiros contidos nele.
