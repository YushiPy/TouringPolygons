# USP Butantã — 50 edifícios

Uma instância do problema Touring Polygons com os 50 contornos não vazios do
GeoPackage `qgis/predios.gpkg`, camada `predios`, no estado exportado em
23 de setembro de 2026.

## Arquivos

- `usp-butanta-50.bin`: uma instância binária, com 50 polígonos e o caminho de
  referência encontrado pelo solver C++.
- `polygons.csv`: associação entre a ordem dos polígonos no binário, o `fid` e
  o `id` original do QGIS, e um nome legível. Alguns rótulos estão marcados
  como provisórios para confirmação.
- `qgis/`: projetos QGIS e GeoPackage de origem, agrupados para manter o
  caminho relativo entre eles. Os projetos são `predios-usp.qgz` e
  `predios-usp-simples.qgz`.

Editar o GeoPackage não altera o arquivo `.bin`. Se os contornos mudarem, a
instância binária e o mapeamento devem ser regenerados/revisados.

## Geometria e formulação

Os polígonos foram desenhados manualmente no QGIS pelo usuário. OpenStreetMap
foi usado como mapa de fundo; as geometrias não são uma exportação oficial de
edifícios do OSM. O GeoPackage registra WGS 84 (EPSG:4326). Para o binário, as
coordenadas foram projetadas para um plano tangente local em metros, com origem
em 23.557° S, 46.732° W.

Início e destino coincidem, 3 m para fora da entrada do IME — Bloco B. O
caminho de referência tem 31 pontos. A execução de origem reportou status
`optimal` e certificado de exatidão.

## Proveniência

A camada atual foi copiada com backup SQLite consistente do GeoPackage do
QGIS. Os nomes dos edifícios foram normalizados a partir dos nomes do usuário
quando possível e conferidos em mapas e páginas institucionais. `polygons.csv`
indica a referência usada para cada nome e marca os nomes provisórios.
A camada original não tinha metadados de licença para os polígonos; o projeto
não declara uma licença nova para essas geometrias. O crédito do mapa-base do
OpenStreetMap é de seus contribuidores, conforme a licença ODbL.
