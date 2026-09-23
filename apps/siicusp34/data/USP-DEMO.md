# Demonstração da rota da USP

A rota é uma demonstração independente dos 558 casos do corpus. O navegador
reproduz dados e uma solução pré-calculada, sem backend, mapas externos ou
execução do solver no visitante.

## Alvos e proveniência

Os **50 polígonos** vêm da instância local
`benchmarks/suites/usp-butanta-50/usp-butanta-50.bin`, exportada da camada
`predios` do GeoPackage `qgis/predios.gpkg` em 23/09/2026. Os contornos foram
desenhados manualmente no QGIS pelo autor do projeto, sobre um mapa-base OpenStreetMap;
não são uma exportação oficial de edifícios da USP nem um extrato direto das
geometrias do OSM. O mapa-base não é incluído na página. A camada original não
registra licença de redistribuição para os contornos; seu uso nesta publicação
foi autorizado pelo autor, mas não declaramos uma licença aberta nova.

O arquivo `polygons.csv` da suíte liga cada região ao `fid`, ao identificador
do QGIS, ao nome e à fonte usada para conferir o nome. Vários rótulos são
provisórios. Em particular, os IDs `ime-a` e `ime-c` **não confirmam** a
existência de blocos A e C no IME: a página os apresenta como edifícios
próximos ao IME ainda sem identificação. O `ime-b` é o alvo destacado junto à
partida. Há alvos compostos e a Praça dos Bancos, portanto “50 regiões” é
mais preciso que “50 prédios”. Nenhum contorno deve ser interpretado como
limite institucional ou representação completa de uma unidade.

O binário guarda vértices e uma rota de referência. O exportador confere a
contagem e a ordem das 50 linhas do CSV, o SHA-256 das fontes e a consistência
da rota de referência com a execução do solver. O JSON de navegação embutido em
`usp-demo.js` contém os vértices já projetados, a rota, a ordem, os primeiros
contatos e a proveniência; é suficiente para publicar a página separadamente
da suíte e do repositório de benchmarks.

## Formulação e resultado

As coordenadas da suíte estão em metros no plano tangente WGS84 linearizado em
23,557° S e 46,732° W. O início e o destino são o mesmo ponto, próximo à
entrada do IME e fora do contorno `ime-b`. Cada região conta como visitada ao
ser tocada ou atravessada. O modelo minimiza comprimento euclidiano plano em
ordem livre; regiões são **alvos, não obstáculos**. Não modela vias, altura,
pátios, autorização ou condições reais de voo.

O exportador chama o solver C++ de ordem livre com até 5.000.000 chamadas e
60 segundos. Na geração desta página, retornou `termination=optimal`,
`exact=true`, `LB≈UB≈4919,804690657526 m`, 4.352 chamadas e uma rota de 31
pontos. O fechamento do gap usa tolerância absoluta de `1e-7 m` e relativa de
`1e-9` do comprimento; é um resultado **numérico**, não uma prova em
aritmética exata. O tempo mostrado é de uma execução local em uma thread, não
uma comparação de desempenho com os 558 casos.

Além de conferir os extremos e o comprimento, o exportador calcula o primeiro
contato de cada polígono com a rota e exige que todos os 50 sejam visitados na
ordem reportada pelo solver. O binário e a rota de referência não são alterados.
Como o percurso é fechado, apresentamos sua orientação inversa, de mesmo
comprimento, para visitar o IME primeiro na animação. Os contatos e a ordem
exibidos são recalculados para essa orientação.

Para regenerar os dados estáticos, a partir da raiz do repositório e com o
solver compilado:

```bash
python3 apps/siicusp34/scripts/build_usp_demo.py --solver .build/unordered/tpp
```

Os arquivos `usp-preview.svg` e `usp-preview-mobile.svg` mostram a mesma rota
enquanto o JavaScript carrega. Os números de visita ficam ocultos por padrão
porque 50 rótulos cobririam os polígonos em um celular. A lista de alvos e a
ordem estão disponíveis em um painel recolhido.

## Revisão antes da publicação

1. Conferir os 50 rótulos no local ou em fonte institucional, sobretudo
   `ime-a`, `ime-c` e os marcados como provisórios em `polygons.csv`.
2. Conferir os contornos, a posição de partida e a licença/publicabilidade
   das geometrias desenhadas a partir do mapa-base.
3. Caso qualquer vértice ou extremo mude, regenerar a suíte e esta exportação;
   um novo desenho define outro problema matemático.
4. Inspecionar o SVG em celular: o recorte amplo favorece contexto do campus,
   mas reduz a legibilidade de polígonos pequenos sem zoom.
