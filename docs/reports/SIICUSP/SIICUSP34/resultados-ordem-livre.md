# TPP não convexo com ordem livre: resultados para o 34º SIICUSP

Data: 6 de setembro de 2026. Revisão experimental: `86a47b57a9af1c6ed8a51a89edd5e4b51886ce5b`.

**Resultado utilizável:** na suíte de desenvolvimento de 60 instâncias, com 4 a 60 polígonos, todos os caminhos passaram na validação geométrica independente. Em uma thread e com orçamento cooperativo de 2 segundos por instância, 43 execuções fecharam o gap numérico e 17 terminaram por tempo. O total de resolução foi 44,27 segundos. Esses números descrevem esta suíte e esta máquina; não demonstram superioridade universal.

Todos os dados deste relatório foram produzidos nesta execução, salvo referências explicitamente identificadas como históricas. Não houve alteração do solver, commit, publicação, edição do resumo submetido, pôster ou slides.

## 1. Escopo do problema

Dadas regiões poligonais simples no plano e extremos fixos $s$ e $t$, procuramos um caminho de menor comprimento euclidiano que visite todas as regiões, escolhendo os pontos de contato e a ordem. Tocar a fronteira ou atravessar o interior conta como visita. São admitidas regiões não convexas, sobreposições, extremos internos e $s=t$.

As regiões são alvos, não obstáculos. A API não representa polígonos com buracos e não resolve restrições de velocidade, capacidade, múltiplos veículos ou desvio de obstáculos. O pior caso da busca permanece exponencial. Os comprimentos e tolerâncias estão nas unidades das coordenadas; a suíte não deve ser descrita automaticamente em metros.

## 2. Implementação auditada

O núcleo está em `packages/nonconvex-tpp/cpp/src/solvers/unordered.cpp`. Cada nó contém uma sequência parcial de regiões, representadas por fechos convexos ou peças de uma decomposição. O solver convexo certificado fornece limites para a sequência. O caminho candidato é testado contra todos os polígonos originais, permitindo visitas incidentais.

Um polígono ainda ausente é inserido em cada uma das $m+1$ posições possíveis. Um fecho que não garante visita ao polígono original é substituído, em filhos separados, por cada peça convexa. As peças cobrem o polígono, e as inserções cobrem as posições relativas possíveis. Sob decomposição correta e limites válidos, isso preserva as extensões do nó. O argumento combinatório não elimina as limitações da aritmética utilizada.

A busca usa fila por limite inferior, descidas periódicas (`dive_interval=128`) e heurística inicial. Os filhos ainda não avaliados herdam o limite do pai; a fronteira, a descida ativa e os nós já podados entram no limite final. A revisão não identificou perda de ramos nesses mecanismos.

O oráculo chama a implementação geométrica memoizada, recupera ou repara os contatos e verifica limites primal e dual. Se o gap não fecha, usa pontos interiores em `long double`, com precisão ampliada do Boost quando necessário. O comprimento geométrico retornado não é tomado automaticamente como limite inferior.

A decomposição atualmente passa por `packages/nonconvex-tpp/cpp/src/common.cpp` para a biblioteca própria `optimal_convex_partition`. Este build não vinculou CGAL nem Gurobi ao solver de produção. O texto geral do README que descreve decomposição via CGAL não identifica corretamente esse caminho atual; não foi usado para atribuir a implementação medida. As ideias de inserção e refinamento são atribuídas, na documentação do algoritmo, a Fekete, Kniep, Krupke e Perk. A integração e implementação próprias devem ser distinguidas da origem dessas ideias.

## 3. Significado dos estados

| Expressão | Significado adotado neste relatório |
| --- | --- |
| Caminho viável verificado | Extremos e visita a todas as regiões aprovados pelo Shapely a `1e-7`; comprimento recalculado compatível com UB |
| Ótimo declarado | O solver informa `exact=true` / `optimal` |
| Ótimo numericamente certificado | O intervalo retornado fecha segundo `UB-LB <= 1e-7 + 1e-9*abs(UB)` e a auditoria confirma consistência dos campos e viabilidade |
| Encerrado por limite | `time_limit` ou `call_limit`; a saída preserva um incumbente e limites, sem presumir otimalidade |
| Prova exata formal | Não produzida: não há certificação racional ou intervalar |

O verificador geométrico não certifica o LB. Na suíte de 60 casos, os 43 ótimos numericamente certificados são certificados pelo método primal/dual implementado, com checagens independentes de viabilidade, não 43 provas de otimalidade por outro solver.

## 4. Protocolo e procedência

Diretório da execução, doravante `RUN`:

[`benchmarks/results/unordered/siicusp34-20260906-200122/`](../../../../benchmarks/results/unordered/siicusp34-20260906-200122/)

| Item | Configuração |
| --- | --- |
| Máquina | MacBook Pro `Mac16,8`, Apple M4 Pro, 12 núcleos (8 de desempenho e 4 de eficiência), 24 GB |
| Sistema | macOS 26.6.2, build 25G83, arm64 |
| Compilação | Apple Clang 17.0.0, CMake 4.4.3, Release, C++26, `-O3`, Gurobi desativado; 8 jobs apenas no build |
| Dependências | Eigen 5.0.1, Boost 1.92.0, libomp 22.1.8; CGAL 6.2 instalado, mas não vinculado neste target |
| Benchmark/validação geométrica | Python 3.14.6 do dashboard, Shapely 2.1.2, GEOS 3.13.1 |
| SOCP e externo | Python 3.12.13, gurobipy 12.0.3, Shapely 2.1.2; licença funcionou nesta execução |
| Instâncias | Cópia preservada de `algorithm-dev-v1.bin`, 60 casos, índices começando em zero |
| Orçamento próprio | 2 s/caso, no máximo 10.000.000 chamadas ao oráculo; uma thread |
| Tolerâncias próprias | Gap absoluto `1e-7`, relativo `1e-9`, visita `1e-8` |
| Validação independente | Geometria a `1e-7`; diferença comprimento/UB até `1e-7 + 1e-9*abs(UB)` |
| Sementes | `342026` nos testes C++ e SOCP; benchmark lê entradas fixas, sem nova geração aleatória |
| Ambiente | `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1`; `PYTHONHASHSEED=0` |

O Git foi inspecionado antes de editar: nenhum diff rastreado; somente `PLANO-DE-TRABALHO.md` do SIICUSP não rastreado. Esse arquivo foi preservado. O diff inicial, hashes dos fontes, compilação, ambiente e cópias do resumo e plano estão em `RUN/provenance/`. O binário medido foi construído em `.build/siicusp34-20260906-200122/` e também foi copiado para a pasta de procedência.

| Identificador | SHA-256 |
| --- | --- |
| Suíte usada | `e7c4da86d21db93b91765f2b051d4d8357394d4c379219896f35b7dde2d74916` |
| Manifesto inicial de hashes dos fontes | `e652ec646ff9b99ac43fd1fe22bf78f4402a2fdefc3d50ea1e71ed4613139ad4` |
| Executável próprio | `bbef5c6fabcf6899785c9f18321da492a5e1274a837b40a86f2852a8e71e7fff` |
| JSONL da rodada adotada | `0976e9dae8b18f5228e08307200d10114d26f1cda879d01acf043725a11bd884` |

`RUN/SHA256SUMS` identifica os demais artefatos. O CSV por caso inclui o hash da entrada, LB, UB, gap, estado, validade, ordem exportada, chamadas e tempos por fase. `command.json` em cada etapa registra argumentos completos, diretório, ambiente, início, fim, tempo medido pelo wrapper e código de saída. Os fontes e binários não foram modificados entre as medições.

### Incidente na primeira rodada

A primeira campanha, `canonical/ours.jsonl`, foi atravessada pela interrupção da sessão. Há uma discrepância entre relógios: o caso 0 registrou 315,59 s no solver, e os timestamps de calendário do wrapper também divergem do tempo de `perf_counter`. Isso é compatível com suspensão ou pausa do ambiente, mas a causa exata não foi determinada.

Essa campanha foi preservada integralmente: 60 caminhos válidos, 42 declarações de ótimo e 18 encerramentos por tempo; soma registrada de 359,79 s. **Ela foi excluída das conclusões de desempenho.** A nova evidência justificou uma única repetição completa, `canonical-confirmation/ours.jsonl`. Nela, calendário e wrapper concordam em aproximadamente 44,65 s de processo. Essa é a rodada adotada. Nas medições seletivas e externas, `caffeinate -i` preveniu suspensão por inatividade enquanto o comando rodava.

## 5. Resultados da rodada adotada

| Medida | Resultado |
| --- | ---: |
| Casos e hashes distintos | 60 |
| Caminhos geometricamente válidos | 60/60 |
| Ótimos declarados, com fechamento numérico conferido | 43/60 (71,7%) |
| Encerramentos por tempo | 17/60 |
| Erros / `numerical_limit` / `call_limit` | 0 / 0 / 0 |
| Tempo total de resolução | 44,2695 s |
| Tempo do processo completo de benchmark | 44,6494 s |
| Maior tempo de resolução de um caso | 2,3114 s |
| Gap relativo médio, incluindo todos os 60 casos | 2,6797% |
| Maior gap relativo | 35,6406% (caso 39) |
| Maior distância caminho–polígono | `7,3734e-9` |
| Maior diferença entre comprimento recalculado e UB | `2,3283e-10` |
| Chamadas ao oráculo / fallbacks | 432.421 / 83.706 |

O gap relativo é `(UB-LB)/abs(UB)`; todos os UBs desta suíte são positivos. A média é por instância, não ponderada pelo comprimento. O limite temporal é cooperativo, por isso 2 s não é um teto rígido. Inicialização do processo e validação Python não estão incluídas em `seconds` do solver.

Alguns intervalos da rodada, para tornar visível a diferença entre solução e encerramento da prova:

| Caso | Polígonos | LB | UB | Gap relativo | Estado |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 4 | 55,469568852671 | 55,469568852823 | menor que `1e-8%` | ótimo nas tolerâncias |
| 9 | 40 | 1069,255251747683 | 1069,255251760257 | menor que `1e-8%` | ótimo nas tolerâncias |
| 33 | 50 | 115,763005520212 | 174,450360184541 | 33,6413% | tempo |
| 39 | 59 | 119,249440018965 | 185,286812535488 | 35,6406% | tempo |
| 55 | 40 | 189,394116657192 | 191,189985792474 | 0,9393% | tempo |

A tabela integral, com precisão exportada, está em [`RUN/summary/canonical/cases.csv`](../../../../benchmarks/results/unordered/siicusp34-20260906-200122/summary/canonical/cases.csv); os agregados estão em [`summary.json`](../../../../benchmarks/results/unordered/siicusp34-20260906-200122/summary/canonical/summary.json). Arredondamentos da tabela acima não devem ser usados para reavaliar o critério de gap.

### Comparação externa suplementar

Executamos também os mesmos 60 hashes no externo 0.2.1, com extremos fixos, uma thread, 2 s/caso, `eps=1e-6` e tolerância geométrica nativa `0.001`. O mesmo validador independente foi aplicado às trajetórias brutas a `1e-7`.

| Medida | Próprio | Externo |
| --- | ---: | ---: |
| Ótimos declarados | 43 | 40 |
| Caminhos brutos aprovados geometricamente | 60 | 5 |
| Declarados ótimos e com caminho bruto aprovado | 43 | 5 |
| Soma dos tempos de resolução | 44,27 s | 46,24 s |

O externo teve 20 encerramentos por limite, sem erros de processo. Encaixar seus extremos produziu 12 candidatos geometricamente válidos, mas isso não certifica as saídas originais nem seus gaps. O teste legado de extremos, mais permissivo que a validação comum, sinalizou 26 falhas. Os CSVs preservam ambos os critérios.

As tolerâncias de gap e de viabilidade são diferentes. O tempo de resolução externo inclui inicialização preguiçosa do Gurobi; a soma dos tempos de seus processos foi 71,93 s. Só cinco casos foram simultaneamente declarados ótimos e validados em ambos. Portanto, **não adotar uma frase de speedup ou superioridade para o pôster**. Razões condicionadas de tempo produzidas pelo resumidor legado são dados exploratórios, não a conclusão deste relatório. Os 60 índices, hashes, modos e resultados da validação externa foram novamente conferidos em `summary/artifact-audit/`.

O checkout local externo não possui `.git` próprio; consultar Git dentro dele retorna a revisão do repositório pai, não uma revisão upstream. Sua procedência é registrada pelos hashes dos módulos efetivamente importados e dos fontes locais em `provenance/external-modules.json` e `external-source-sha256.json`. Não inferimos equivalência entre fontes locais e extensão nativa instalada apenas pela versão 0.2.1.

## 6. Validações e resultado de cada execução

| Execução nova | Resultado | Artefatos em `RUN/` |
| --- | --- | --- |
| Configuração e build isolados | Passaram; targets `tpp` e `tpp-unordered-tests` | `01-configure/`, `02-build/`, `provenance/` |
| Suíte C++ | 86 casos por enumeração e 344 buscas com caps de chamadas 0, 1, 3 e 10 passaram | `03-tests/` |
| Primeira campanha própria | 60/60 válidos, 42 ótimos declarados; tempo contaminado pelo incidente, não adotado | `04-canonical/`, `canonical/`, `summary/interrupted-campaign/` |
| SOCP independente disponível | 24/24 passaram; 8 ramificações explícitas por decomposição | `05-independent-socp/` |
| Campanha própria adotada | 60/60 válidos; 43 ótimos nas tolerâncias, 17 limites de tempo | `06-canonical-confirmation/`, `canonical-confirmation/` |
| Perfil, repetição 1 | 5/5 válidos; 4 ótimos, 1 limite; soma 3,9396 s | `08-profile-1/`, `profile-1/` |
| Perfil, repetição 2 | 5/5 válidos; 4 ótimos, 1 limite; soma 3,9318 s | `08-profile-2/`, `profile-2/` |
| Perfil, repetição 3 | 5/5 válidos; 4 ótimos, 1 limite; soma 3,9010 s | `08-profile-3/`, `profile-3/` |
| Externo, 60 casos | 40 ótimos declarados, 20 limites; somente 5 caminhos brutos válidos | `09-external/`, `external/` |
| SOCP com captura dos resultados completos e checagens adicionais | 24 comparações repetidas para registrar entradas, caminhos e bounds; 168 execuções sob caps passaram | `10-socp-interruptions/`, `socp-interruptions/` |
| Auditoria dos artefatos | 60 registros próprios e 60 externos completos; 168 intervalos compatíveis com objetivo SOCP; 7 entradas corrompidas rejeitadas pelo resumidor | `14-artifact-audit/`, `summary/artifact-audit/` |
| Exemplo da CLI | `optimal`, 6 chamadas; LB `21.643550153774818`, UB `21.64355015382282` | `15-example/` |
| Sumários | Executados com sucesso; a versão final rejeita ausência/duplicação de casos, hash errado, geometria/comprimento inválidos e estados incoerentes | `07-*`, `11-*`, `12-*`, `13-*`, `16-*`, `summary/` |

As 86 comparações C++ usam a mesma decomposição e o mesmo oráculo que o B&B. Elas verificam integração e enumeração de objetivos, não independência dos componentes. Os testes não usam `assert` desativável por `NDEBUG`, mas exceções explícitas.

O SOCP enumera ordens e uniões de dois retângulos que formam um L, sem reutilizar nosso oráculo ou nossa decomposição. Usa uma thread, `BarQCPConvTol=1e-8`, `FeasibilityTol=1e-9`; o código permite nova tentativa com `BarQCPConvTol=1e-6` e `NumericFocus=3`. A tolerância de comparação de objetivos é `1e-5*(1+objetivo)`, maior que o gap nominal próprio. A maior diferença observada foi `6,0621e-8`.

As 168 execuções adicionais são 24 casos com sete configurações: caps de chamadas 0, 1, 3 e 10 (30 s), e caps de tempo 0, `1e-6` e `0.001` s (um milhão de chamadas). Foram verificados extremos, visitas, comprimento, finitude e ordenação dos bounds, cap de chamadas, coerência entre gap e estado e ordem como permutação. Houve 39 saídas `call_limit`, 50 `time_limit` e 79 `optimal`; um orçamento pequeno não obriga interrupção se o gap já fechou.

**Ressalva descoberta na captura independente:** em seis casos, o mínimo de `ObjBound` devolvido pela enumeração SOCP foi `-Infinity` (10, 11, 15, 18, 19 e 23). Isso foi preservado nos dados brutos. A checagem adicional decisiva usa o objetivo otimizado independente, exigindo `LB <= objetivo + 1e-6*(1+objetivo)` e `UB >= objetivo - 1e-5*(1+objetivo)`, e passou em 168/168. Não alegamos 24 certificados duais finitos independentes. Os JSONs brutos desse baseline contêm a extensão `-Infinity` aceita pelo leitor Python; o sumário final representa essa indisponibilidade explicitamente.

O verificador da ordem confere que ela é uma permutação dos polígonos. Não houve reconstrução independente da sequência exata das primeiras visitas, especialmente nos empates de contato.

## 7. Profiling e decisão sobre otimização

Usamos a instrumentação já existente, sem adicionar custo ao binário. Os casos 2, 9, 12, 55 e 59 cobrem um caso fácil, o caso concluído mais lento da rodada adotada, uso frequente de fallback, timeout e precisão ampliada. Foram três repetições seletivas, sequenciais, do mesmo binário; não há comparação antes/depois porque nenhuma otimização foi implementada.

| Caso | Estado nas três repetições | Tempo mediano [mín.; máx.] (s) | Chamadas | Oráculo (% do total) | Fallback (% do total) | Precisão ampliada (% do total) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | ótimo | 0,000076917 [0,000076; 0,000077333] | 3 | 9,5% | 0% | 0% |
| 9 | ótimo | 1,5047 [1,5034; 1,5083] | 32.549 | 83,6% | 75,4% | 0% |
| 12 | ótimo | 0,3523 [0,3420; 0,3532] | 1.878 | 96,6% | 88,5% | 18,1% |
| 55 | tempo | 2,0310 [2,0002; 2,0442] | 7.878–7.912 | 98,7% | 94,2% | 3,8% |
| 59 | ótimo | 0,04424 [0,04366; 0,04495] | 52 | 97,9% | 97,4% | 95,5% |

Os percentuais são medianas das frações de cada repetição. Oráculo inclui fallback; fallback inclui precisão ampliada. Não somar essas colunas. Pré-processamento, heurística inicial, busca e finalização são fases superiores disjuntas; `visit_check_seconds` atravessa essas fases e também não deve ser somado novamente.

Nos quatro casos concluídos, LB, UB, ordem e contagem de chamadas foram idênticos entre as três repetições. No caso 55, UB e ordem permaneceram iguais; o LB variou de `189.41080780786626` a `189.41724384481492`, acompanhando a quantidade de trabalho antes do timeout. Os 15 caminhos foram validados.

Na campanha completa adotada, o oráculo consumiu 94,1% do tempo e o fallback, 90,2%. O caso 59 mostra que uma etapa rara de precisão ampliada pode dominar um caso pequeno. O custo de fila ou decomposição não é o alvo principal indicado por esses dados.

**Decisão:** manter o núcleo. A reparação pequena de contatos já está implementada; seu histórico está em `docs/research/free-order-development-history-2026-09.md`. O gargalo restante exige investigar recuperação geométrica, condicionamento ou trabalho do método de pontos interiores. Não foi identificada uma intervenção adicional suficientemente pequena e justificada para aceitar antes da entrega. Reduzir precisão ou afrouxar tolerâncias para ganhar tempo alteraria a evidência científica. A contribuição desta rodada é a validação e o diagnóstico, não uma alegação de aceleração nova.

## 8. Comandos reproduzíveis

Execute na raiz do repositório. O bloco cria nomes novos; não reutilize um diretório existente. No Linux, omita `caffeinate -i` e use o ambiente Python com Shapely instalado. Os caminhos Python abaixo identificam os ambientes efetivamente usados no macOS.

```bash
RUN="benchmarks/results/unordered/siicusp34-$(date +%Y%m%d-%H%M%S)"
BUILD=".build/$(basename "$RUN")"
mkdir -p benchmarks/results/unordered
mkdir "$RUN"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0

cmake -S packages/nonconvex-tpp/cpp -B "$BUILD" \
	-DTARGET=main-unordered -DTPP_ENABLE_GUROBI=OFF
cmake --build "$BUILD" --target tpp tpp-unordered-tests -j 8
"$BUILD/tpp-unordered-tests"
"$BUILD/tpp" < packages/nonconvex-tpp/cpp/tests/unordered-example.txt

caffeinate -i apps/benchmark-dashboard/.venv/bin/python \
	benchmarks/scripts/unordered_benchmark.py \
	--suite benchmarks/suites/algorithm-dev-v1.bin --solver "$BUILD/tpp" \
	--seconds 2 --max-calls 10000000 --output "$RUN/canonical/ours.jsonl"

apps/benchmark-dashboard/.venv/bin/python \
	benchmarks/scripts/summarize_unordered_siicusp.py "$RUN/canonical/ours.jsonl" \
	--suite benchmarks/suites/algorithm-dev-v1.bin --output "$RUN/summary/canonical"

tspn-comparison/solver/.venv/bin/python \
	packages/nonconvex-tpp/cpp/tests/validate_unordered_gurobi.py \
	--solver "$BUILD/tpp" --cases 24

for REP in 1 2 3; do
	caffeinate -i apps/benchmark-dashboard/.venv/bin/python \
		benchmarks/scripts/unordered_benchmark.py \
		--suite benchmarks/suites/algorithm-dev-v1.bin --solver "$BUILD/tpp" \
		--seconds 2 --max-calls 10000000 \
		--case 2 --case 9 --case 12 --case 55 --case 59 \
		--output "$RUN/profile-$REP/ours.jsonl"
done

apps/benchmark-dashboard/.venv/bin/python \
	benchmarks/scripts/summarize_unordered_siicusp.py "$RUN"/profile-*/ours.jsonl \
	--suite benchmarks/suites/algorithm-dev-v1.bin \
	--case 2 --case 9 --case 12 --case 55 --case 59 --output "$RUN/summary/profile"

caffeinate -i tspn-comparison/solver/.venv/bin/python \
	tspn-comparison/benchmarks/run_comparison.py \
	--suite benchmarks/suites/algorithm-dev-v1.bin --mode path --threads 1 \
	--time-limit 2 --eps 0.000001 --feasibility-tolerance 0.001 \
	--validation-tolerance 0.0000001 --output "$RUN/external"

apps/benchmark-dashboard/.venv/bin/python benchmarks/scripts/summarize_unordered.py \
	"$RUN/canonical/ours.jsonl" "$RUN"/external/*/algorithm-dev-v1-tspn-path.csv \
	--output "$RUN/summary/comparison"
```

A suíte derivada é ignorada pelo Git. Para reproduzir exatamente esta seleção, use a cópia em `RUN` original e confira seu hash. Em clone novo, a geração por `python3 benchmarks/tpp.py generate-suites` está documentada, mas o resultado só deve ser tratado como a mesma suíte após conferir o hash. Não regravamos nem regeneramos a suíte histórica nesta execução.

Os comandos realmente executados, incluindo redirecionamentos e caminhos absolutos, estão em `RUN/commands.md` e nos `command.json`. O wrapper `record.py` arquivado registra stdout/stderr e se recusa a reutilizar o diretório de uma etapa. O driver adicional `audit_socp_interruptions.py` e o auditor `validate_artifacts.py` estão preservados em `RUN`; para repetir suas execuções, copie os drivers e o manifesto para outro diretório de campanha e atualize o caminho do build. Eles criam novas subpastas e não sobrescrevem as anteriores.

## 9. Limitações e relação com o resumo submetido

O resumo submetido apresenta o solver convexo e o B&B não convexo com ordem fixa; anuncia ordem livre como extensão em andamento. Formulação coerente para a apresentação:

> A partir da estrutura apresentada no resumo, estendemos a busca para também escolher a ordem de visita. Nesta etapa, verificamos caminhos e limites numéricos e medimos onde o solver gasta tempo.

Os cerca de 2 milhões de chamadas e 25 segundos mencionados no resumo pertencem a uma instância de ordem fixa. Não são resultados da ordem livre e não foram reproduzidos aqui. A alegação de validação da decomposição em mais de 16 mil polígonos também não foi reexecutada nesta rodada.

Limitações que precisam acompanhar as conclusões:

- Uma única suíte de desenvolvimento, já utilizada para melhorias anteriores, não mede generalização para conjuntos novos. Há apenas uma rodada completa com tempos adotados; três repetições seletivas não fornecem intervalos de confiança para toda a suíte.
- A máquina não foi isolada de toda atividade do sistema, frequência ou estado térmico. Os tempos são observações, não constantes reproduzíveis byte a byte.
- A margem de segurança dual usa ponto flutuante; em `certified.cpp` ela inclui `1e-12*scale*(n+1)`. Isso não substitui aritmética intervalar com arredondamento dirigido.
- Na conversão dos contatos refinados para `double`, a API convexa isolada não repete uma verificação completa de pertinência. O B&B verifica visitas antes de aceitar um incumbente. Escalas extremas e quase degenerescências não foram cobertas sistematicamente aqui.
- `numerical_limit` preserva resultados quando sobra gap em uma saída utilizável. Caminhos inválidos, falha de visita à peça atribuída ou falha interna do ponto interior podem lançar exceção e abortar sem devolver incumbente. Nenhuma ocorreu nas campanhas novas.
- As APIs antigas de ordem fixa têm problemas conhecidos com algumas interseções. A proteção da API certificada não foi propagada para todas elas.
- Os testes C++ existentes não cobrem explicitamente `dive_interval=0`; a auditoria adicional usa os defaults da CLI. Não é uma prova de todas as opções da API.
- Gurobi foi usado apenas na validação e no externo. A execução de produção foi compilada com a dependência comercial desativada.
- Os resultados brutos são locais e ignorados pelo Git. Devem ser arquivados junto ao relatório antes de limpar o workspace; não estão publicados.

## 10. Frases sustentáveis para o pôster

Selecionar no máximo três para manter a leitura simples. As cinco opções abaixo são sustentadas por esta execução:

1. **“Em 60 instâncias de teste, todos os caminhos visitaram as regiões exigidas e respeitaram os pontos de partida e chegada, segundo uma verificação geométrica independente.”** Na legenda: tolerância `1e-7`, suíte de desenvolvimento.
2. **“Com uma thread e orçamento de 2 segundos por instância, 43 dos 60 casos fecharam o gap de otimalidade nas tolerâncias numéricas; 17 terminaram por tempo.”** Na legenda: M4 Pro, tolerância `1e-7 + 1e-9*UB`, limite cooperativo.
3. **“Nos casos interrompidos, o programa devolve um caminho viável e limites que mostram quanto ainda pode faltar para provar a melhor solução.”** Referir-se às execuções testadas, não prometer recuperação após qualquer falha interna.
4. **“Em 24 pequenas instâncias, os comprimentos concordaram com uma formulação independente de otimização; a maior diferença observada foi cerca de `6,1e-8`.”** Na legenda: regiões em L, SOCP/Gurobi, comparação numérica.
5. **“Nesta suíte, cerca de 90% do tempo foi gasto no método numérico auxiliar que sustenta os limites da busca.”** É diagnóstico experimental do gargalo atual, não resultado de complexidade.

## 11. Afirmações que não devem ser feitas

- “Todos os 60 problemas foram resolvidos otimamente” ou “um timeout é ótimo”.
- “43 provas matemáticas exatas independentes”, “certificado racional” ou “todas as falhas numéricas preservam os limites”.
- “O solver é universalmente mais rápido”, inclusive usando as razões de tempo condicionadas do resumo externo.
- “O externo está errado em qualquer configuração”. O que falhou foi a viabilidade de muitas trajetórias exportadas nesta configuração e tolerância.
- “Resolve 60 polígonos em 25 s com ordem livre”, transplantando o exemplo de ordem fixa.
- “A enumeração C++ valida independentemente o oráculo”, “24 limites duais SOCP finitos” ou “a ordem de primeiras visitas foi toda reconstruída por outro código”.
- “Dispensa solvers comerciais em toda a avaliação”, “trata obstáculos” ou “permite regiões com buracos”.
- “A execução de hoje acelerou o algoritmo”. Nenhuma otimização do solver foi realizada.

## 12. Perguntas dos avaliadores e respostas curtas

| Pergunta | Resposta |
| --- | --- |
| Qual é a diferença para visitar pontos? | O caminho escolhe onde tocar cada região; esses contatos são variáveis contínuas. |
| Como decide a ordem? | Insere uma região em todas as posições possíveis da sequência parcial, controlando a busca com limites. |
| Por que o LB não excede o ótimo? | Omitir regiões e usar fechos convexos relaxa restrições; o limite dual do subproblema relaxado fornece um limite inferior, sujeito à avaliação numérica implementada. |
| Por que pode podar? | Se o LB do ramo já não melhora o melhor caminho além da tolerância, explorar esse ramo não é necessário para fechar o gap configurado. |
| Como lida com não convexidade? | Refina o fecho para peças convexas que cobrem o polígono original. |
| Uma região pode ser visitada por acaso? | Sim; todo candidato é conferido contra todas as regiões, mesmo as ausentes da sequência parcial. |
| O que significa exato aqui? | A busca fecha o intervalo entre limites na tolerância numérica declarada; não é prova em aritmética racional. |
| O que acontece em dois segundos? | O limite é cooperativo. O solver termina quando consegue checá-lo e mantém o melhor caminho e os limites nas interrupções testadas. |
| Como foi validado? | Enumeração para a integração, Shapely para caminhos e SOCP independente em 24 instâncias pequenas. Cada verificação tem alcance diferente. |
| Qual o pior resultado medido? | O maior gap relativo foi 35,64% após o orçamento, no caso 39; não chamamos essa saída de ótima. |
| Por que não otimizar a fila? | O profiling atribuiu aproximadamente 90% do tempo ao método auxiliar; a fila não é o gargalo dominante medido. |
| A contribuição é toda nova? | A implementação integra geometria, certificação numérica e decisões de ordem e peças; os algoritmos e estratégias vindos da literatura são atribuídos às fontes. |
| Como isso se relaciona ao resumo? | A extensão recente decide também a ordem, usando a estrutura cujo caso de ordem fixa foi apresentado na submissão. |

## 13. Decisões humanas restantes e arquivos

Não há bloqueio técnico para utilizar os resultados qualificados acima. Restam decisões de comunicação e escopo:

1. Escolher, com o orientador, até três frases e a redação da novidade/contribuição, mantendo explícita a natureza numérica de “exato”.
2. Decidir se a comparação externa merece espaço; a recomendação deste relatório é deixá-la como material suplementar, dada a falta de equivalência numérica e de viabilidade comum.
3. Confirmar a procedência dos exemplos de ordem fixa e da validação de decomposição antes de reutilizar esses números históricos.
4. Definir a instância ilustrativa da USP e suas fontes/licenças, além de confirmar número permitido de slides e prazo interno. Essas decisões pertencem à preparação posterior do material, não foram executadas aqui.
5. Arquivar o diretório de resultados antes de qualquer limpeza local e decidir posteriormente se será disponibilizado como suplemento. Nada foi publicado.

Arquivos criados ou modificados para esta entrega:

- Este relatório.
- `benchmarks/scripts/summarize_unordered_siicusp.py`: auditoria e sumários próprios por caso e repetição, com saídas em diretório novo.
- `docs/algorithms/unordered-tpp.md`: precisão sobre enumeração compartilhada, exceções e limite temporal; link para esta reprodução, preservando os números históricos.
- Dados e drivers locais em `RUN`, incluindo entradas, binários, logs, caminhos, ordens, bounds, hashes e sumários. O plano preexistente e o resumo submetido foram preservados.
