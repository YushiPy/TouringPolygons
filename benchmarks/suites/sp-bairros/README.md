# São Paulo — 32 subprefeituras

Uma instância do Touring Polygons formada pelas 32 subprefeituras do município
de São Paulo. O conjunto escolhido é uma divisão oficial e reconhecível da
capital, e fica dentro do limite de 50–60 polígonos do solver. São subprefeituras,
não os 96 distritos nem os bairros individuais.

## Arquivos

- `sp-bairros.bin`: instância binária com os 32 polígonos e um caminho de
  referência de 12 pontos.
- `polygons.csv`: associação entre o índice do polígono no binário, o código e
  a sigla oficiais, o nome, o `fid` da fonte e as contagens de vértices.
- `geosampa/subprefeituras.gpkg`: cópia da camada oficial usada como fonte,
  preservada em GeoPackage.

Os índices de `polygons.csv` começam em zero e seguem o código oficial de
subprefeitura, de `01` a `32`. Se a fonte for atualizada, revise o CSV e gere
novamente o binário.

## Geometria e formulação

A camada `subprefeitura` foi obtida em 23 de setembro de 2026 pelo WFS do
GeoSampa (`geoportal:subprefeitura`). Ela contém 32 polígonos sem geometria
nula, no sistema SIRGAS 2000 / UTM zona 23S (EPSG:31983). Os anéis originais
somam 84.016 vértices.

Para manter a instância leve o bastante para o solver, cada contorno foi
generalizado com Douglas–Peucker, tolerância de 20 m. O binário tem 4.455
vértices no total; Parelheiros, o maior polígono, tem 673. O GeoPackage mantém
a fonte integral. A generalização é adequada à visualização didática, mas não
representa uma nova delimitação administrativa oficial; pequenas diferenças
podem ocorrer nas fronteiras compartilhadas.

As coordenadas do binário estão em metros no mesmo UTM, transladadas pela
origem do retângulo envolvente da cidade (E 337002.651043 m; N 7379935.423214
m). Essa translação não altera as distâncias euclidianas. Início e destino
coincidem no centro geométrico do polígono Sé (código `09`), em
(-4389.022361 m; 14881.113746 m) no sistema local do binário.

A ordem de visita é livre. Uma região é considerada visitada quando o caminho
a intercepta; portanto atravessar uma subprefeitura conta como visitá-la. O
caminho de referência retorna ao ponto de partida.

## Resultado do solver

O solver C++ reportou `termination=optimal` e `exact=true` para esta geometria
generalizada, com limite inferior e superior de 113819.936637155 m, 223 chamadas
do oráculo e 3.589 s (limites de execução: 1.000.000 chamadas e 300 s). “Exato”
aqui significa que o certificado numérico atingiu a tolerância configurada; não
é uma prova em aritmética racional. Consulte
[`docs/algorithms/unordered-tpp.md`](../../../docs/algorithms/unordered-tpp.md)
para a formulação, tolerâncias e limitações.

## Proveniência e crédito

Fonte: Prefeitura do Município de São Paulo, SMUL/Geoinfo, camada
“Subprefeituras” do [catálogo de metadados GeoSampa](https://metadados.geosampa.prefeitura.sp.gov.br/geonetwork/srv/api/records/289e684e-b2be-453d-8d2d-68ec629cd3dc),
serviço [WFS GeoSampa](https://wfs.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wfs).
O registro informa revisão até 13 de abril de 2026, escala nominal 1:5.000 e
EPSG:31983. A camada foi publicada sob licença CC BY-SA indicada pelo
[GeoSampa](https://novogeosampa.prefeitura.sp.gov.br/); esta licença se aplica
aos dados geoespaciais municipais e à geometria generalizada deste binário.
Crédito: Prefeitura do Município de São Paulo — SMUL/Geoinfo, GeoSampa,
“Subprefeituras”.
