# SIICUSP34 — Touring Polygons Problem

Esta é a página interativa do 34º SIICUSP. O visitante explora uma rota por mais de 50
regiões da USP, tenta três desafios, acompanha uma execução comentada do
algoritmo e consulta os resultados de 558 instâncias da pesquisa. A rota da USP
é uma demonstração independente do corpus.

A publicação é **estática e autocontida**: `index.html`, CSS, JavaScript, SVGs
e dados ficam nesta pasta. O navegador não precisa do solver C++, de uma API,
de WebAssembly nem de serviços de mapas. O pequeno servidor abaixo serve apenas
para testar os arquivos localmente.

## Versão em inglês (Etapa Internacional)

`en/` é a versão em inglês para o QR code do pôster da Etapa Internacional
(25–26/11/2026), aberta em `http://127.0.0.1:8765/en/`. Ela reutiliza o CSS,
o solver dos desafios e os dados das demonstrações desta pasta (`../`), começa
pelo destaque Brasil e não tem a simulação da busca, cujos registros vêm do
solver antigo de extremos fixos. O Brasil da versão em inglês é um TSPN, sem
ponto fixo (`en/data/br-estados-tspn.js`, gerado por
`scripts/build_br_estados_tspn.py` a partir dos mesmos 27 polígonos de
`data/br-estados-demo.js`): 9.962 km, contra 10.039 km do tour que sai de
Brasília na versão em português. `en/en.css` guarda só os acréscimos.

Os resultados seguem o resumo em inglês e a medição de referência na armijo
([`fekete-reference-armijo-2026-10-30`](../../benchmarks/results-saved/fekete-reference-armijo-2026-10-30/README.md)):
ciclos de ordem livre (TSPN), o código original de Fekete et al. com uma
thread (configuração B, raiz CHR) contra o nosso com uma thread (F).
`en/data/event-data.js` é gerado da raiz do repositório por

```bash
python3 apps/siicusp34/scripts/build_event_data_en.py
```

que lê o `per-case.csv` da campanha, os tours da campanha local
`benchmarks/workspace/experiments/fekete-reference@armijo`, o ZIP de instâncias
do submódulo `third_party/tspn-socg`. As distâncias do painel de precisão vêm
dos registros, sem o arredondamento do CSV; a ordem de visita é a do primeiro
contato ao longo do tour; as decomposições vêm da partição C++ compilada na
geração, sem os vértices repetidos a menos de 10⁻⁹ que 100 regiões têm (com
eles, a partição devolve a região inteira como uma peça não convexa). Ambos usam U−L ≤ 10⁻³U + 10⁻³; nossos limites são provados. Em F, 557
instâncias fecham em 300 s; a restante continua aberta em 900 s. Para ela,
a página mostra o tour factível e o orçamento, sem tempo de fechamento.

Os seis cartões do método são tours fechados nos desenhos dos desafios, e não
caminhos de s a t como na versão em português. Eles saem de
`scripts/build_tspn_cards.py` (matplotlib e shapely), que resolve cada tour com
`--cycle`, confere os exemplos e grava `en/figures/*.svg` e, com `--pdf-dir`,
os PDFs do pôster.

## Abrir no computador

Na raiz do repositório, execute:

```bash
python3 apps/siicusp34/scripts/serve.py
```

Abra o endereço impresso no terminal, normalmente
`http://127.0.0.1:8765/`. Use Python 3.9 ou mais recente; não é necessário
instalar pacotes. Encerre com `Ctrl+C`.

## Abrir no celular

Com o computador e o celular na **mesma rede Wi-Fi**, execute na raiz:

```bash
python3 apps/siicusp34/scripts/serve.py --phone
```

O comando imprime um endereço `Celular (en0): http://IP-LOCAL:8765/` (a
interface pode ter outro nome). Digite esse endereço no navegador do celular.
Essa opção aceita conexões da rede local; se o sistema solicitar, permita que
o Python receba conexões. Algumas redes de convidados isolam os dispositivos e
impedem esse teste. Para usar outra porta, acrescente `--port 8766` e abra o
novo endereço impresso.

## Dados e manutenção

- `data/sp-bairros-demo.js` e `data/br-estados-demo.js` contêm as instâncias
  estáticas usadas nos destaques São Paulo e Brasil, com rota e decomposição
  convexa pré-calculadas. As fontes e os mapeamentos ficam em
  `benchmarks/suites/sp-bairros` e `benchmarks/suites/br-estados`. Para
  regenerar São Paulo, execute da raiz do repositório:

  ```bash
  python3 apps/siicusp34/scripts/build_sp_bairros_demo.py --solver .build/unordered/tpp
  ```

  Para regenerar a instância brasileira, com o solver C++ compilado e um
  compilador C++20 disponível, execute da raiz do repositório:

  ```bash
  python3 apps/siicusp34/scripts/build_br_estados_demo.py --solver .build/unordered/tpp
  ```

  O script também reconstrói `br-estados.bin` e `polygons.csv` a partir do ZIP
  oficial mantido em `benchmarks/suites/br-estados/ibge`.

- `data/usp-demo.js` e os SVGs de prévia contêm a rota estática da USP. A
  [proveniência e as limitações](data/USP-DEMO.md) estão documentadas à parte.
- `data/event-data.js` contém os 558 casos e resultados; `data/trace-data.js`
  contém os registros usados na simulação; `data/challenge-data.js` contém os
  três desafios. Nenhum desses arquivos é refeito ao iniciar o servidor.
- O exportador de rastros fica nesta aplicação e lê `data/event-data.js`.
  Para gerar um arquivo novo para revisão, sem substituir a publicação:

  ```bash
  benchmarks/.venv/bin/python apps/siicusp34/scripts/export_trace_data.py \
    --case 1 --case 3 --case 9 --output /tmp/siicusp34-traces-reviewed.js
  ```

  O solver deve estar compilado em `.build/tools/bin/tpp-unordered`. O comando
  é executado da raiz do repositório e recusa sobrescrever o destino.
- Para regenerar a demonstração da USP, é preciso ter a suíte local
  `benchmarks/suites/usp-butanta-50`, o solver C++ compilado e um compilador
  C++20. O script calcula a decomposição convexa ótima na build, usando o
  pacote C++ do repositório e sem adicionar dependências ao navegador. A partir
  da raiz:

  ```bash
  python3 apps/siicusp34/scripts/build_usp_demo.py --solver .build/unordered/tpp
  ```

- Os três desafios vêm de
  `benchmarks/campaigns/SIICUSP34 - Instances/manual-cases.json` e são
  regenerados com `node apps/siicusp34/scripts/build_challenge_data.mjs`.
- Para verificar a lógica local dos desafios: `node --test apps/siicusp34/test-solver.mjs`.

As [notas de produto](NOTAS-DE-PRODUTO.md) registram decisões, evidências e
pendências de apresentação. Para publicar, basta servir o conteúdo desta pasta
em hospedagem de arquivos estáticos; o comando `--phone` é somente para prévia
na rede local.
