# Demonstração da rota da USP

Esta é uma instância didática independente. Ela não pertence aos 558 casos do
corpus, não participa da comparação com Fekete et al. e não tem trace da busca
registrado. A página reproduz um resultado calculado previamente pelo solver C++.

## Alvos e extremos

Os oito alvos são contornos de edifícios na Cidade Universitária, apresentados
separadamente: IME, Bloco B (`way/154079142`); Poli Biênio (`way/147219057`);
Produção (`way/147219058`); Elétrica (`way/158960184`); Civil e Ambiental
(`way/158960183`); Mecânica, Mecatrônica e Naval (anel exterior de
`relation/2062864`, `way/158966883`); Minas e Petróleo (anel exterior de
`relation/2062859`, `way/158960190`); Metalurgia e Materiais
(`way/158960185`). Os nomes institucionais foram conferidos com a
[lista de prédios da POLI](https://www.poli.usp.br/a-poli-2/como-chegar/).

O ponto inicial é o centroide do contorno do Terminal de Ônibus da Cidade
Universitária (`way/158966878`). O ponto final é o centroide do Monumento a Ramos
de Azevedo (`way/588882619`). Assim, `s` e `t` representam dois marcos nomeados,
não posições operacionais de decolagem e pouso.

## Fonte, licença e transformação

[`usp-footprints.json`](usp-footprints.json) guarda os vértices em longitude e
latitude WGS84, identificadores, versões e datas dos objetos da consulta de
22/09/2026 ao OpenStreetMap. Dados © contribuidores do OpenStreetMap,
[ODbL 1.0](https://www.openstreetmap.org/copyright). O recorte transformado
em [`usp-demo.js`](usp-demo.js) deriva desses dados e conserva a mesma atribuição
e licença de dados. O desenho não usa tiles, imagens de satélite ou serviços de
mapa em tempo de execução.

O exportador usa um plano tangente WGS84 linearizado em 23,557° S,
46,732° W. As coordenadas do problema e o comprimento são em **metros planos**;
não são distâncias geodésicas. Os vértices não foram simplificados. O desenho
SVG é apenas uma projeção visual dessas mesmas coordenadas.

O TPP desta demonstração minimiza comprimento euclidiano de `s` a `t`, com ordem
livre e escolha dos pontos de contato. Tocar ou atravessar um contorno conta
como visita. Edifícios são **alvos, não obstáculos**. O modelo não representa
ruas, paredes, altura, zonas de voo, autorização ou segurança de drones. Para
Mecânica e Minas, a API recebe somente o anel exterior dos multipolígonos OSM:
os pátios internos ficam preenchidos no modelo. Os contatos da rota exportada
foram conferidos fora desses pátios, mas essa idealização continua uma limitação
da formulação mostrada.

## Resultado e validação

O exportador [`build_usp_demo.py`](../scripts/build_usp_demo.py) alimenta o
solver C++ de ordem livre com limite de 5.000.000 chamadas e 60 segundos. Na
execução preservada, o solver retornou `termination=optimal`, `exact=true`,
`LB=1298.5673652278992 m` e `UB=1298.5673652278995 m`. A tolerância de
fechamento é `1e-7 + 1e-9 * abs(UB)` metro. **“Certificado” é numérico sob essa
tolerância**, não uma prova em aritmética racional ou intervalar. O exportador
recalcula o comprimento e confere extremos, primeira visita e interseção da
polilinha com todos os oito polígonos a `1e-7` metro. O tempo mostrado na página
é de uma execução local, sem pretensão de benchmark.

Para regenerar, a partir da raiz do repositório, com o solver já compilado:

```bash
python3 apps/siicusp34/scripts/build_usp_demo.py --solver .build/unordered/tpp
```

O JSON exportado registra SHA-256 da fonte, do binário e da entrada enviada ao
solver, além de formulação, projeção, limites e tolerâncias. O exportador também
produz `usp-preview.svg` e `usp-preview-mobile.svg`, usados enquanto os scripts
estáticos carregam.
