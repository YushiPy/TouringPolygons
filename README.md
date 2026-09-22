# Touring Polygons Problem

Implementações e experimentos para o **Touring Polygons Problem (TPP)**: dados
um ponto inicial, um ponto final e regiões poligonais, encontrar o menor caminho
euclidiano que visita todas as regiões. O projeto cobre tanto ordem fixa quanto
ordem livre e preserva a implementação convexa legada ainda utilizada.

O núcleo mantido é C++ e segue esta cadeia de dependências:

```text
tpp_geometry -> tpp_convex -> optimal_convex_partition -> tpp_nonconvex
```

- `tpp_geometry`: primitivas geométricas compartilhadas;
- `tpp_convex`: solver exato para polígonos convexos em ordem fixa;
- `optimal_convex_partition`: decomposição convexa com CGAL;
- `tpp_nonconvex`: ordem fixa e Branch and Bound de ordem livre.

O dashboard mantido fica em `apps/benchmark-dashboard`. A aplicação
`apps/siicusp34` é uma publicação estática e autocontida.

## Estrutura

```text
apps/                       aplicações mantidas e publicação do SIICUSP
benchmarks/
├── tpp.py                  única CLI pública de benchmarks
├── _internal/              implementação importável da CLI
├── suites/                 conjuntos de entrada canônicos
└── results-saved/          campanhas deliberadamente preservadas
docs/
├── algorithms/             contratos, algoritmos e auditorias atuais
└── reports/                relatórios do projeto
packages/                   bibliotecas e solvers mantidos
third_party/tspn-socg/      fork alemão fixado como submódulo
```

Veja [`docs/architecture.md`](docs/architecture.md) para responsabilidades e
ciclo de vida dos componentes e [`docs/third-party.md`](docs/third-party.md)
para proveniência, licença e dados externos.

## Instalação e verificação

Clone incluindo o solver externo:

```bash
git clone --recurse-submodules https://github.com/YushiPy/TouringPolygons.git
cd TouringPolygons
./scripts/install_dependencies.sh
./scripts/sanity_check.sh
```

O instalador é idempotente. Ele instala as dependências de sistema conhecidas,
sincroniza os ambientes Python e Node do dashboard e instala o Chromium usado
pelos testes de navegador.

Para verificar um checkout já preparado sem tentar instalar pacotes:

```bash
./scripts/sanity_check.sh --no-install
```

## Benchmarks

`benchmarks/tpp.py` é a interface pública. Exemplo de campanha sintética:

```bash
python3 benchmarks/tpp.py create smoke \
  --vertices 8 --polygons 20 --instances 100 --shape star
python3 benchmarks/tpp.py run smoke \
  --threads 8 --max-calls 1000000 --max-seconds 30
python3 benchmarks/tpp.py status smoke
```

Use `python3 benchmarks/tpp.py --help` para geração, conjuntos canônicos,
ordem livre, conversões e comparações externas. Resultados novos são locais por
padrão; uma campanha só deve ser preservada quando entradas, resultados brutos,
análise, configuração e proveniência estiverem juntas em
`benchmarks/results-saved/<campanha>/`.

## Documentação

- [`DEVELOPMENT.md`](DEVELOPMENT.md): setup, comandos de desenvolvimento,
  limites dos diretórios e validação;
- [`docs/architecture.md`](docs/architecture.md): arquitetura mantida e decisões
  de ciclo de vida;
- [`docs/algorithms/`](docs/algorithms): especificações e argumentos de
  correção;
- [`docs/third-party.md`](docs/third-party.md): repositórios e dados externos;
- [`benchmarks/README.md`](benchmarks/README.md): protocolos de benchmark.

## Referências

- Dror, M., Efrat, A., Lubiw, A., Mitchell, J. S. B. (2003). *Touring a
  sequence of polygons*. STOC 2003. https://doi.org/10.1145/780542.780612
- Tan, X., Jiang, B. (2017). *Efficient algorithms for touring a sequence of
  convex polygons and related problems*. TAMC 2017.
  https://doi.org/10.1007/978-3-319-55911-7_44
- Arkin, E. M., Fekete, S. P., Mitchell, J. S. B. (2005). *The Traveling
  Salesman Problem with Neighborhoods: A Survey*.
