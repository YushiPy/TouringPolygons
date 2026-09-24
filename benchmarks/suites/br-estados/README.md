# Brasil — 27 unidades federativas

Instância didática do Touring Polygons com as 26 unidades estaduais e o
Distrito Federal. Cada UF é um alvo; o caminho parte de Brasília, atravessa
cada alvo e retorna ao ponto inicial. Como atravessar conta como visitar, não é
necessário pousar em cada região.

## Arquivos

- `br-estados.bin`: uma instância fechada com 27 polígonos e a rota de
  referência calculada pelo solver C++.
- `polygons.csv`: liga cada polígono do binário ao código, à sigla e ao nome
  oficial da UF; registra também o número da feição no shapefile, os contornos
  da fonte e as contagens de vértices.
- `ibge/BR_UF_2025.zip`: arquivo original do IBGE, preservado para reproduzir a
  conversão.

Os índices dos polígonos começam em zero e seguem o geocódigo numérico da UF.
`source_feature_index` também é zero-based e mantém a associação com a ordem das
feições no shapefile original.

## Geometria e formulação

A fonte é a malha de unidades federativas do IBGE, edição 2025, arquivo
[`BR_UF_2025.zip`](https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2025/Brasil/BR_UF_2025.zip).
O ZIP contém 27 feições no sistema geográfico SIRGAS 2000 (EPSG:4674). O
shapefile representa componentes insulares e outros anéis como partes
separadas. Para manter uma região por UF, a conversão escolhe o maior anel
exterior de cada feição; os demais componentes desconectados e os buracos ficam
fora desta versão. O Distrito Federal é incluído, portanto são 27 unidades
federativas, não 27 estados.

As coordenadas foram projetadas para SIRGAS 2000 / Brazil Polyconic
([EPSG:5880](https://epsg.org/crs_5880/SIRGAS-2000-Brazil-Polyconic.html)), em
metros, e transladadas pela origem do retângulo envolvente.
Cada contorno principal foi generalizado por Douglas–Peucker com tolerância de
5.000 m. O binário resultante tem 2.678 vértices, entre 7 e 202 por UF. A
tolerância foi escolhida para a escala nacional do desenho e para manter os
contornos simples. A generalização e a seleção de anéis são aproximações
didáticas, não uma nova delimitação administrativa oficial.

O depósito é Brasília, nas coordenadas geográficas (-47,8828°, -15,7939°),
dentro do polígono do Distrito Federal. Início e destino coincidem. O solver
minimiza distância euclidiana no plano Brazil Polyconic; o valor não deve ser
interpretado como distância geodésica sobre a superfície da Terra. Uma UF é
visitada quando o caminho toca ou atravessa seu contorno principal.

## Resultado do solver

O solver C++ reportou `termination=optimal` e `exact=true`: limite inferior de
10.039.051,367910659 m e superior de 10.039.051,367910660 m, 111 chamadas do
oráculo e aproximadamente 0,08 s. O caminho de referência tem 11 pontos e
retorna a Brasília.
“Exato” significa que o intervalo numérico atingiu as tolerâncias configuradas
(gap absoluto de `1e-7` m e relativo de `1e-9`); não é uma prova em aritmética
racional. A decomposição convexa ótima foi calculada pelo código C++ de Greene:
881 peças ao todo. Todas as 27 UFs têm mais de uma peça, então a camada
“Decomposição” fica habilitada no app. Consulte
[`docs/algorithms/unordered-tpp.md`](../../../docs/algorithms/unordered-tpp.md)
para a formulação e suas limitações.

## Reprodução

Com o solver em `.build/unordered/tpp` e um compilador C++20, execute na raiz do
repositório:

```bash
python3 apps/siicusp34/scripts/build_br_estados_demo.py \
  --solver .build/unordered/tpp
```

O script lê o ZIP, reconstrói `polygons.csv` e `br-estados.bin`, calcula a rota
e a decomposição convexa e atualiza `apps/siicusp34/data/br-estados-demo.js`.
Não requer dependências GIS no navegador ou no Python.

## Proveniência

Fonte: Instituto Brasileiro de Geografia e Estatística, Malhas Territoriais,
Malha Municipal Digital 2025, feições de Unidades da Federação. Consulte a
[página oficial de Malhas Territoriais](https://www.ibge.gov.br/geociencias/organizacao-do-territorio/malhas-territoriais/15774-malhas.html)
e a nota metodológica vigente antes de reutilizar os dados. O IBGE informa que
os downloads são públicos; este pacote preserva o arquivo original e a
atribuição à fonte. A instância simplificada serve à visualização e à pesquisa
do TPP, não substitui a malha oficial para uso legal, administrativo ou de
planejamento.

SHA-256 do arquivo fonte: `cdbbf05f79153802cbfa74d0c29814cd76a9c0b925aea910c9f04dffc28e6167`.
