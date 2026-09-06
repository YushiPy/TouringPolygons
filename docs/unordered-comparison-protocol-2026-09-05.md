# Protocolo comparativo com o TSPN externo

Investigação e rodada controlada realizadas em 5 de setembro de 2026. Nenhum
artefato histórico foi alterado.

## Semântica confirmada no solver externo 0.2.1

No modo `path`, os dois primeiros sites da `Instance` são origem e alvo. A raiz
mantém o índice 0 no início e o índice 1 no fim, e o SOCP modela sites pontuais por
igualdades nas coordenadas. A ordem dos polígonos restantes é livre. O critério de
parada do B&B é `UB <= (1 + eps) * LB`.

O solver expõe `FEASIBILITY_TOLERANCE`, usado na cobertura e na ramificação, mas
o adaptador não tem uma API única que reproduza o critério misto próprio
`UB - LB <= 1e-7 + 1e-9 * abs(UB)`. Na rodada controlada, o externo usa
`eps = 1e-9`, que é conservador em relação à parcela relativa própria, e ambos
usam tolerância de viabilidade `1e-8` onde a API permite. A validação independente
usa `1e-7` para ambos.

## Validador e classificação

`benchmarks/scripts/unordered_validation.py` recebe início, alvo, polígonos e uma
polilinha. Ele verifica, com Shapely, distância aos dois extremos, distância máxima
a qualquer polígono e comprimento recalculado. O adaptador externo exporta a
trajetória bruta orientada de início para alvo. Também exporta uma candidata
diagnóstica com apenas os extremos encaixados exatamente, mas mantém separados:

- `is_optimal`, declaração do solver nas tolerâncias configuradas;
- `raw_valid`, viabilidade independente da trajetória realmente retornada;
- `snapped_valid`, viabilidade depois do encaixe diagnóstico dos extremos;
- `recomputed_length` e `snapped_recomputed_length`;
- `solve_seconds`, que inclui a inicialização preguiçosa do ambiente Gurobi;
- `process_seconds`, custo completo do processo trabalhador.

Somente resultados declarados ótimos e aprovados na mesma validação podem entrar
em uma comparação de ótimos. O encaixe não transforma retroativamente a saída
bruta em resultado verificado.

## As 27 falhas históricas de extremos

Os mesmos 27 índices sinalizados na referência foram reexecutados com a configuração
histórica do algoritmo externo (`eps=1e-6`, `FEASIBILITY_TOLERANCE=0.001`) e com
exportação de caminho. Todos os 27 voltaram a falhar no teste histórico de extremos
a `1e-5`, logo a falha não veio da exportação antiga nem da seleção de casos.

Na validação comum a `1e-7`, nenhuma das 27 trajetórias brutas foi viável. A distância
mediana ao início foi $4,23\times10^{-6}$, e ao alvo,
$3,26\times10^{-5}$. Em 26 casos também houve distância excessiva a pelo menos um
polígono, com mediana $2,21\times10^{-5}$ e máximo $7,87\times10^{-4}$. Encaixar
os extremos deixou somente 1/27 caminho viável. O caso 3 é reproduzível com desvios
de aproximadamente $4,1\times10^{-6}$ no início, $3,3\times10^{-5}$ no alvo e
$2,1\times10^{-5}$ em uma região.

A causa está na trajetória produzida pelo SOCP: o modelo contém igualdades para os
pontos fixos, mas o ambiente Gurobi criado pelo solver não configura tolerâncias
numéricas próprias, e as coordenadas extraídas carregam resíduos. A tolerância de
cobertura do B&B externo não corrige o vetor exportado. Como 26/27 também falham em
regiões, alterar apenas a exportação ou encaixar extremos seria insuficiente.

## Rodada controlada

Foram escolhidos os hashes dos casos 2, 3, 12, 37, 55 e 59, cobrindo casos rápidos,
lentos, timeout e precisão ampliada. Os solvers rodaram sequencialmente na mesma
máquina, uma thread e 2 s por caso. O solver próprio preservou seus critérios
`1e-7 + 1e-9 * UB` e viabilidade `1e-8`; o externo usou `eps=1e-9` e viabilidade
`1e-8`. Os hashes foram conferidos pelo resumidor.

| Resultado | Próprio | Externo bruto | Externo com encaixe diagnóstico |
| --- | ---: | ---: | ---: |
| Casos | 6 | 6 | 6 |
| Otimalidade declarada | 5 | 0 | não aplicável |
| Viabilidade independente | 6 | 0 | 2 |
| Tempo de resolução total | 5,411 s | 6,824 s | não aplicável |

O tempo total dos processos externos foi 9,581 s, 2,757 s além de
`solve_seconds`; a sobrecarga mediana por processo foi 0,465 s. A inicialização
Gurobi ocorre preguiçosamente dentro de `solve_seconds`, portanto esse campo ainda
não é solver puro sem inicialização. `process_seconds` adiciona importação e criação
do trabalhador. Essa limitação permanece explícita.

Não há speedup comparável nessa amostra, pois nenhum caso foi simultaneamente
declarado ótimo e validado nos dois solvers. Os números não sustentam uma conclusão
de desempenho entre algoritmos, mas cumprem a separação entre declaração de
otimalidade, viabilidade e resultado não comparável.

## Comandos e artefatos

```bash
tspn-comparison/solver/.venv/bin/python tspn-comparison/benchmarks/run_comparison.py \
	--suite benchmarks/suites/algorithm-dev-v1.bin --mode path --threads 1 \
	--time-limit 2 --eps 0.000000001 --feasibility-tolerance 0.00000001 \
	--validation-tolerance 0.0000001 --case-index CASE --output OUTPUT

python3 benchmarks/scripts/summarize_unordered.py \
	benchmarks/results/unordered/task-3-comparison-20260905/ours.jsonl \
	benchmarks/results/unordered/task-3-comparison-20260905/external/20260905-192353/algorithm-dev-v1-tspn-path.csv \
	--output benchmarks/results/unordered/task-3-comparison-20260905/summary
```

Artefatos novos:

- `benchmarks/results/unordered/task-3-comparison-20260905/ours.jsonl`
- `benchmarks/results/unordered/task-3-comparison-20260905/external/20260905-192353/`
- `benchmarks/results/unordered/task-3-comparison-20260905/historical-failures-reproduced/20260905-192508/`
- `benchmarks/results/unordered/task-3-comparison-20260905/summary/`

Próximo passo limitado a esta comparação: expor e testar parâmetros numéricos do
SOCP externo, ou projetar uma reparação certificada que reotimize a trajetória com
extremos fixos e volte a validar todas as regiões. Apenas encaixar coordenadas não é
suficiente e não deve ser contado como sucesso.
