# SIICUSP34 — Touring Polygons Problem

Esta é a página interativa do 34º SIICUSP. O visitante explora uma rota por mais de 50
regiões da USP, tenta três desafios, acompanha uma execução comentada do
algoritmo e consulta os resultados de 558 instâncias da pesquisa. A rota da USP
é uma demonstração independente do corpus.

A publicação é **estática e autocontida**: `index.html`, CSS, JavaScript, SVGs
e dados ficam nesta pasta. O navegador não precisa do solver C++, de uma API,
de WebAssembly nem de serviços de mapas. O pequeno servidor abaixo serve apenas
para testar os arquivos localmente.

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
