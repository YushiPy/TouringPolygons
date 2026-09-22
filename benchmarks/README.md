# Benchmarks

`benchmarks/tpp.py` é a única interface pública para geração, execução,
conversão e comparação. Os módulos de `benchmarks/_internal/` são detalhes de
implementação importáveis, não uma coleção de comandos independentes.

## Organização

- `tpp.py`: CLI estável;
- `_internal/`: implementação da CLI;
- `suites/`: corpora canônicos rastreados e suites derivadas ignoradas;
- `campaigns/`: entradas e execuções locais reproduzíveis;
- `results/`: saídas locais e descartáveis;
- `results-saved/`: campanhas deliberadamente preservadas com dados e análise.

Use `python3 benchmarks/tpp.py --help` e acrescente `--help` após um subcomando
para consultar todos os parâmetros.

## Campanha sintética

```bash
python3 benchmarks/tpp.py create smoke \
  --vertices 8 --polygons 20 --instances 100 --shape star
python3 benchmarks/tpp.py run smoke \
  --threads 8 --max-calls 1000000 --max-seconds 30 --timeout 3600
python3 benchmarks/tpp.py status smoke
```

A campanha fica em `benchmarks/campaigns/smoke/` com manifesto, entradas,
preview e resultados. Execuções concluídas são reutilizadas ao retomar.

Para gerar uma matriz baseada em OpenStreetMap:

```bash
python3 benchmarks/tpp.py generate-matrix sao-paulo \
  packages/instance-generation/regions/sao-paulo.osm.pbf \
  --instances 100 --sample-size 40 --seed 42
```

## Suites algorítmicas

As suites `algorithm-dev-v1.bin` e `canonical-v1.bin` são derivadas do corpus
rastreado `suites/nonconvex/test_cases.bin`:

```bash
python3 benchmarks/tpp.py generate-suites
python3 benchmarks/tpp.py benchmark \
  --suite benchmarks/suites/algorithm-dev-v1.bin
```

Para reconstruir o corpus a partir do arquivo alemão fixado no submódulo:

```bash
python3 benchmarks/tpp.py convert-german
```

## Ordem livre

Uma campanha completa usa:

```bash
python3 benchmarks/tpp.py free-order NOME_DA_CAMPANHA --help
```

Para executar diretamente uma suite binária:

```bash
python3 benchmarks/tpp.py free-order-run \
  --suite benchmarks/suites/algorithm-dev-v1.bin \
  --solver .build/unordered/tpp \
  --seconds 3 --max-calls 10000000 --workers 4 \
  --output benchmarks/results/free-order-dev.jsonl
```

Para uma comparação pareada de binários próprios:

```bash
python3 benchmarks/tpp.py free-order-ablation \
  --suite benchmarks/suites/algorithm-dev-v1.bin \
  --solver baseline=.build/unordered-baseline/tpp \
  --solver candidate=.build/unordered-candidate/tpp \
  --seconds 3 --repeats 3 \
  --output benchmarks/results/free-order-comparison.jsonl
```

Os comandos `generate-free-order-canon`, `summarize-free-order` e
`free-order-metamorphic` cobrem, respectivamente, a campanha canônica, a
comparação pareada de resultados e os testes metamórficos.

## Solver alemão

O fork fixado em `third_party/tspn-socg` é uma dependência de comparação, não
um pacote do projeto. Prepare o ambiente dele conforme
[`docs/third-party.md`](../docs/third-party.md) e execute:

```bash
python3 benchmarks/tpp.py compare-external --help
python3 benchmarks/tpp.py compare-oracles --help
```

A campanha longa e retomável usa:

```bash
python3 benchmarks/tpp.py run-fekete --workers 8
```

Ela grava checkpoints em `benchmarks/results/fekete-free-order-6h/`. No macOS,
uma execução não supervisionada pode ser iniciada com:

```bash
caffeinate -i python3 benchmarks/tpp.py run-fekete --workers 8
```

## Preservação de resultados

Resultados novos permanecem ignorados até virarem evidência deliberadamente
publicada. Uma pasta em `results-saved/` deve conter:

- entradas exatas ou uma receita determinística para obtê-las;
- hashes das entradas, executáveis e revisões relevantes;
- configuração, formulação, tolerâncias e orçamento;
- resultados brutos, inclusive falhas e limites;
- análise reproduzível e limitações conhecidas.

A comparação alemã atual está em
`results-saved/german-comparison/`; sua análise vive na mesma pasta. Não copie
uma análise de campanha para `docs/research/`.

## Interpretação

No benchmark não convexo, observe principalmente:

- chamadas ao solver convexo e nós explorados;
- diferença entre incumbente inicial e resultado final;
- motivo de término e gap final;
- tempo medido por caso e tempo de parede da campanha;
- tolerância e validação independente da trajetória.

“Ótimo” significa que o solver fechou o gap dentro da tolerância declarada.
Um caminho factível interrompido por tempo ou chamadas não deve ser rotulado
como ótimo. Tempos obtidos com números diferentes de workers ou sob contenção
não são comparações diretas de desempenho.
