# Touring Polygons Problem

Algoritmos exatos e experimentos para dois problemas de rota mínima que visita
regiões no plano:

- **Touring Polygons Problem (TPP):** dados um ponto inicial, um ponto final e
  regiões poligonais, encontrar o menor caminho euclidiano que toca todas, em
  ordem fixa ou em ordem livre;
- **TSPN** (*Traveling Salesman Problem with Neighborhoods*): o menor ciclo
  fechado que toca todas as regiões, em ordem livre e sem ponto fixo.

As regiões podem ser não convexas, encostar umas nas outras ou se sobrepor; o
TSPN também aceita pontos e segmentos. A busca é um *branch and bound* cujos
limites inferiores e superiores são provados. "Resolvido" quer dizer gap
numérico fechado (por exemplo, `UB ≤ 1,001·LB`). Otimalidade algébrica, com gap
zero, usa o caminho em aritmética racional exata.

## Núcleo

O núcleo mantido é C++ e segue esta cadeia de dependências:

```text
tpp_geometry -> tpp_convex -> optimal_convex_partition -> tpp_nonconvex
```

- `tpp_geometry`: primitivas geométricas compartilhadas;
- `tpp_convex`: caminhos e ciclos de ordem fixa por polígonos convexos, a partir
  dos mapas de última etapa de Dror et al., estendidos a polígonos que se
  intersectam. O oráculo constrói a solução em `double`, prova
  `L ≤ OPT ≤ U` com aritmética intervalar, recorre a um polimento por pontos
  interiores quando a prova não fecha e mantém o caminho racional exato com
  certificado KKT como reserva;
- `optimal_convex_partition`: decomposição convexa ótima de polígonos simples
  (implementação própria, sem CGAL, do `optimal_convex_partition_2` do CGAL);
- `tpp_nonconvex`: ordem fixa com decomposição convexa, e o *branch and bound*
  de ordem livre para o TPP e para o TSPN.

O dashboard mantido fica em `apps/benchmark-dashboard`. A aplicação
`apps/siicusp34` é a página estática e autocontida do 34º SIICUSP.

## Resultados principais

Cada item resume uma seção de
[`benchmarks/results-saved/README.md`](benchmarks/results-saved/README.md), que
registra formulação, tolerâncias e limitações.

- **TSPN, 558 instâncias de Fekete et al.** Lado a lado, 300 s e uma thread:
  557/558 resolvidas, contra 536/558 do solver deles pelo critério de parada
  dele. Nas 536 resolvidas pelos dois, o nosso é mais rápido em 535, com média
  geométrica de 41× ([`tspn-fekete-558-2026-10-16`](benchmarks/results-saved/README.md#tspn-fekete-558-2026-10-16)).
- **TPP de ordem livre com extremos fixos, mesmas instâncias.** 558/558
  resolvidas, contra 553/558. Nos 553 casos comuns, a média geométrica é de 65×
  (de 8,7× a 2.302×). Os tempos deles vêm de outra campanha na mesma máquina, não
  lado a lado ([`float-recovery-dantzig-2026-10-08`](benchmarks/results-saved/README.md#float-recovery-dantzig-2026-10-08)).
- **Coleção de TSPN da Paula (235 instâncias).** 229 certificadas com gap
  relativo de 10⁻⁶: nelas, o melhor tour da Paula é ótimo dentro dessa
  tolerância. Os dados são locais, sem autorização de redistribuição
  ([`tspn-paula-long-armijo-2026-10-14`](benchmarks/results-saved/README.md#tspn-paula-long-armijo-2026-10-14)).

## Estrutura

```text
apps/                       dashboard mantido e página do SIICUSP
benchmarks/
├── tpp.py                  única CLI pública de benchmarks
├── _internal/              implementação importável da CLI
├── suites/                 conjuntos de entrada canônicos
├── results-saved/          resumos dos resultados e fixtures pequenos
└── workspace/              dados locais gerados (ignorado; ver benchmarks/README.md)
docs/
├── algorithms/             contratos, algoritmos e argumentos de correção
└── reports/                relatórios do projeto
packages/                   bibliotecas e solvers mantidos
scripts/                    instalação e verificação
third_party/tspn-socg/      fork do solver de Fekete et al., fixado como submódulo
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

O instalador é idempotente e nunca usa `sudo`. O que falta e pode ser obtido sem
administrador é baixado para a pasta pessoal ou para o repositório: uv, Python 3.12,
o ambiente Python dos benchmarks, headers C++ e o solver original de Fekete et al.
com seus pacotes. O que exige administrador (compilador, `git`, `make`) é só
listado, e o script para. No macOS, ele usa o Homebrew quando existe; sem ele, o
nosso solver fica de fora, porque precisa da `libomp`. Com Node.js presente,
também prepara o dashboard e o Chromium dos testes de navegador.

O `sanity_check` compila os solvers e roda os testes em cerca de um minuto com
o build já feito. Para verificar um checkout já preparado sem tentar instalar
pacotes, ou para incluir as suítes de estresse geradas (cerca de 25 minutos a
mais):

```bash
./scripts/sanity_check.sh --no-install
./scripts/sanity_check.sh --full
```

## Benchmarks

`benchmarks/tpp.py` é a interface pública. Para preparar apenas seu ambiente
Python, execute `python3 benchmarks/tpp.py setup`. A CLI usa
`benchmarks/.venv` automaticamente, com dependências fixadas em `uv.lock`.
Exemplo de campanha sintética:

```bash
python3 benchmarks/tpp.py create smoke \
  --vertices 8 --polygons 20 --instances 100 --shape star
python3 benchmarks/tpp.py run smoke \
  --threads 8 --max-calls 1000000 --max-seconds 30
python3 benchmarks/tpp.py status smoke
```

`bench --problem` monta uma execução de TPP de ordem fixa, TPP de ordem livre ou
TSPN a partir das mesmas opções. `free-compare` prepara e roda a comparação de
ordem livre com o solver de Fekete et al., e `paula-campaign` a campanha longa
de TSPN na coleção da Paula. Use `python3 benchmarks/tpp.py --help` para os
demais comandos. Resultados e dados brutos novos são locais por padrão;
`benchmarks/results-saved/` mantém resumos concisos e fixtures mínimos, e
`fekete-comparison` é a exceção com corpus completo, porque seus arquivos são
usados pelo material do SIICUSP.

## Documentação

- [`DEVELOPMENT.md`](DEVELOPMENT.md): setup, comandos de desenvolvimento,
  limites dos diretórios e validação;
- [`docs/architecture.md`](docs/architecture.md): arquitetura mantida e decisões
  de ciclo de vida;
- [`docs/algorithms/`](docs/algorithms): especificações e argumentos de
  correção, entre eles o [oráculo convexo certificado](docs/algorithms/certified-convex-oracle.md),
  os [ciclos convexos](docs/algorithms/convex-cycle.md) e o
  [TSPN](docs/algorithms/tspn.md);
- [`docs/reports/oraculo-convexo/`](docs/reports/oraculo-convexo): relatório
  técnico do oráculo convexo (construção, prova intervalar, polimento e
  correção dos limites);
- [`docs/third-party.md`](docs/third-party.md): repositórios e dados externos;
- [`benchmarks/README.md`](benchmarks/README.md): protocolos de benchmark.

## Referências

- Dror, M., Efrat, A., Lubiw, A., Mitchell, J. S. B. (2003). *Touring a
  sequence of polygons*. STOC 2003, pp. 473–482.
  https://doi.org/10.1145/780542.780612
- Tan, X., Jiang, B. (2017). *Efficient algorithms for touring a sequence of
  convex polygons and related problems*. TAMC 2017, LNCS, pp. 614–627.
  https://doi.org/10.1007/978-3-319-55911-7_44
- Fekete, S. P., Kniep, R., Krupke, D., Perk, M. (2026). *A Branch-and-Bound
  Algorithm for the Traveling Salesman Problem with Difficult Neighborhoods*.
  SoCG 2026, LIPIcs 367, pp. 46:1–46:20.
  https://doi.org/10.4230/LIPIcs.SoCG.2026.46 (código:
  https://github.com/tubs-alg/TSPN-SoCG-2026)
- Arkin, E. M., Hassin, R. (1994). *Approximation algorithms for the geometric
  covering salesman problem*. Discrete Applied Mathematics 55(3), pp. 197–218.
  https://doi.org/10.1016/0166-218X(94)90008-6
- Greene, D. H. (1983). *The decomposition of polygons into convex parts*. Em
  F. P. Preparata (ed.), *Computational Geometry*, Advances in Computing
  Research 1, pp. 235–259. JAI Press.
