# Demonstração da rota da USP

Esta instância didática é independente dos 558 casos do corpus. Não participa das
estatísticas ou da comparação com Fekete et al. A página reproduz um resultado
calculado previamente pelo solver C++ e continua estática, sem serviço de mapas.

## Modelo e escolha dos alvos

O cenário narrativo é um drone que sai de um ponto junto à entrada principal do
IME (Bloco B), visita contornos de edifícios da USP e retorna ao mesmo ponto:
`s = t`. O IME **é um dos 24 alvos**. O ponto de partida e retorno foi derivado
do nó OSM [`node/6204649436`](https://www.openstreetmap.org/node/6204649436),
marcado `entrance=main` no contorno do
[`way/154079142`](https://www.openstreetmap.org/way/154079142), e deslocado
3 m para fora na direção do centroide à entrada. O exportador confere que o
ponto está fora do IME; esse deslocamento é uma escolha geométrica ilustrativa,
não uma indicação de acesso ou operação de drone.

Os sete prédios da POLI continuam separados. A versão ampla também inclui as
unidades solicitadas (FEA, ECA, IAG, IFUSP e Geociências) e outros edifícios
com contornos identificáveis no recorte. Ela foi feita para seleção posterior;
24 números podem ficar densos em um celular, por isso a camada de rótulos pode
ser ligada separadamente.

| Alvo | Contorno | Vínculo institucional no recorte |
| --- | --- | --- |
| IME · Bloco B | [`way/154079142`](https://www.openstreetmap.org/way/154079142) | área OSM `way/154079145` |
| Poli · Biênio | [`way/147219057`](https://www.openstreetmap.org/way/147219057) | nome/contorno OSM |
| Poli · Produção | [`way/147219058`](https://www.openstreetmap.org/way/147219058) | nome/contorno OSM |
| Poli · Elétrica | [`way/158960184`](https://www.openstreetmap.org/way/158960184) | nome/contorno OSM |
| Poli · Civil | [`way/158960183`](https://www.openstreetmap.org/way/158960183) | nome/contorno OSM |
| Poli · Mecânica | [`way/158966883`](https://www.openstreetmap.org/way/158966883) | nome/contorno OSM |
| Poli · Minas e Petróleo | [`way/158960190`](https://www.openstreetmap.org/way/158960190) | nome/contorno OSM |
| Poli · Metalurgia | [`way/158960185`](https://www.openstreetmap.org/way/158960185) | nome/contorno OSM |
| FAU · Vilanova Artigas | [`way/158966879`](https://www.openstreetmap.org/way/158966879) | área OSM `way/153922310` |
| IFUSP · Alessandro Volta | [`way/322203748`](https://www.openstreetmap.org/way/322203748) | área OSM `way/158789266` |
| FEA · prédio 2 | [`way/153921989`](https://www.openstreetmap.org/way/153921989) | área OSM `way/158966874` |
| ECA · Cinema, Rádio e TV | [`way/153921471`](https://www.openstreetmap.org/way/153921471) | área OSM `way/153922309` |
| IAG · Administração | [`way/401961914`](https://www.openstreetmap.org/way/401961914) | área OSM `way/152732604` |
| Geociências · edifício | [`way/1447059783`](https://www.openstreetmap.org/way/1447059783) | área OSM `way/154246451` |
| Biociências · André Dreyfus | [`way/153921991`](https://www.openstreetmap.org/way/153921991) | área OSM `way/158966876` |
| ICB · prédio I | [`way/153489174`](https://www.openstreetmap.org/way/153489174) | área OSM `way/34381459` |
| Química · bloco 1 | [`way/152420864`](https://www.openstreetmap.org/way/152420864) | área OSM `way/34317133` |
| Oceanográfico · edifício | [`way/393899762`](https://www.openstreetmap.org/way/393899762) | área OSM `way/403735379` |
| Psicologia · bloco D | [`way/153959724`](https://www.openstreetmap.org/way/153959724) | área OSM `way/34382548` |
| Relações Internacionais | [`way/291407687`](https://www.openstreetmap.org/way/291407687) | nome/contorno OSM |
| FFLCH · História e Geografia | [`way/44972294`](https://www.openstreetmap.org/way/44972294) | área OSM `way/158966875` |
| Farmácia · Administração | [`way/152420887`](https://www.openstreetmap.org/way/152420887) | área OSM `way/52050177` |
| Energia e Ambiente · Alta Tensão | [`way/34317212`](https://www.openstreetmap.org/way/34317212) | área OSM `way/34381700` |
| Educação · bloco A | [`way/401293199`](https://www.openstreetmap.org/way/401293199) | área OSM `way/34317182` |

Os prédios da POLI também foram conferidos com a
[lista de prédios da escola](https://www.poli.usp.br/a-poli-2/como-chegar/).
O nome do prédio no OSM e/ou sua posição dentro de uma área mapeada para a
unidade fundamentam o rótulo. Nas entradas com nome genérico ou sem nome
próprio, o vínculo é **inferência espacial** e deve ser conferido antes de usar
o nome como afirmação institucional. A seleção foi cruzada com o
[mapa da Cidade Universitária divulgado pelo IFUSP](https://portal.if.usp.br/fge/sites/portal.if.usp.br.fge/files/Mapa%20-%20Cidade%20Universitaria.pdf)
e o [mapa do campus da Prefeitura da USP](https://puspc.usp.br/mobilidade/mapas/).

## Fonte, licença e transformação

[`usp-footprints.json`](usp-footprints.json) guarda os vértices em longitude e
latitude WGS84, IDs, versões e datas dos objetos OSM consultados em
22/09/2026. Dados © contribuidores do OpenStreetMap,
[ODbL 1.0](https://www.openstreetmap.org/copyright). O recorte transformado
em [`usp-demo.js`](usp-demo.js) deriva desses dados e mantém a atribuição e
a licença de dados. A página não consulta tiles, imagens de satélite nem APIs
em tempo de execução.

O exportador usa um plano tangente WGS84 linearizado em 23,557° S,
46,732° W. As coordenadas e o comprimento são **metros planos**, não distâncias
geodésicas. Os vértices não foram simplificados. Os SVGs são uma apresentação
das mesmas coordenadas. O modelo minimiza o comprimento euclidiano da rota
fechada, com ordem livre e pontos de contato escolhidos pelo solver.
Tocar ou atravessar um contorno conta como visita.

Edifícios são **alvos, não obstáculos**. O modelo não representa ruas, paredes,
altura, zonas de voo, autorização ou segurança de drones. Para os contornos da
POLI Mecânica e Minas, só o anel exterior dos multipolígonos OSM é usado: os
pátios internos ficam preenchidos no modelo. Os contatos da rota exportada
foram conferidos fora desses pátios, mas a idealização permanece.

## Resultado e validação

O exportador [`build_usp_demo.py`](../scripts/build_usp_demo.py) chama o solver
C++ de ordem livre com teto de 5.000.000 chamadas e 60 s. Nesta geração, o
solver retornou `termination=optimal`, `exact=true`,
`LB=6522.836611259418 m` e `UB=6522.836611259418 m`.
A tolerância de fechamento é `1e-7 + 1e-9 * abs(UB)` metro.
**“Certificado” é numérico sob essa tolerância**, não uma prova em aritmética
racional ou intervalar. O exportador recalcula o comprimento e verifica `s=t`,
a posição do ponto externo, a primeira visita e a interseção da polilinha com
todos os 24 polígonos a `1e-7` m. O tempo apresentado é de uma execução
local e não deve ser interpretado como benchmark.

Para regenerar, a partir da raiz do repositório, com o solver compilado:

```bash
python3 apps/siicusp34/scripts/build_usp_demo.py --solver .build/unordered/tpp
```

O JSON exportado registra SHA-256 da fonte, do binário e da entrada do solver,
além de formulação, projeção, limites e tolerâncias. O exportador também cria
`usp-preview.svg` e `usp-preview-mobile.svg`, visíveis enquanto os scripts
estáticos carregam.
