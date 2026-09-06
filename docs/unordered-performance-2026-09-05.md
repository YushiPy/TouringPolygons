# Redução de fallbacks do TPP sem ordem

Data: 2026-09-05. Suíte: `benchmarks/suites/algorithm-dev-v1.bin`.

## Intervenção mantida

O perfil da tarefa 2 mostrou que o fallback certificado consumia 88,7% a 94,3% do
tempo nos casos não triviais e que quase todas as chamadas eram causadas por falha
na recuperação do caminho geométrico. A mudança isolada tenta reparar esse caminho
somente quando ele contém exatamente um ponto por região. Cada ponto é combinado
com o centroide do polígono correspondente, primeiro por `1e-12`, depois `1e-10` e
`1e-8`. Cada candidato passa novamente pela recuperação ordenada de contatos, pelo
cálculo do comprimento primal e pelo mesmo limite dual. O fallback original é usado
se a viabilidade ou o gap não fechar. Tolerâncias, margens numéricas, bounds e poda
não foram relaxados.

A referência instrumentada foi preservada em
`.build/unordered-profiled-baseline/tpp`, SHA-256
`48cdd429763e2423d94f4ee84eaad1823711c83f37cf92d808de13fe69e896c9`.
O binário com reparação foi `.build/unordered-instrumented/tpp`, SHA-256
`8025b959ae429bc245aaf2d17909e035d9a156c9d73d101bc3685f92d8b40df7`.

Foram feitas três repetições alternadas por variante, com limite de 2 s e
10.000.000 chamadas. A tabela usa medianas. Nos casos ótimos, a variação é do tempo;
nos timeouts, a comparação principal é trabalho concluído dentro do mesmo limite.

| Caso | Referência | Reparação | Fallbacks ref. | Fallbacks reparação | Resultado |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 0,083 ms | 0,086 ms | 0 | 0 | Mesmo ótimo; diferença de 0,002 ms é ruído dominante |
| 9 | timeout, 19.075 chamadas | ótimo em 1,613 s, 32.549 chamadas | 8.314 | 4.624 | Passou a provar ótimo, `1069,2552517602574` |
| 12 | 0,405 s | 0,381 s | 1.644 | 1.393 | 5,9% mais rápido, mesmo ótimo e bounds |
| 37 | 1,757 s | 0,623 s | 9.617 | 2.390 | 64,5% mais rápido, mesmo ótimo e bounds |
| 55 | timeout, 6.822 chamadas | timeout, 7.321 chamadas | 5.972 | 5.813 | 7,3% mais chamadas; mesmo upper bound e lower bound mediano maior |

No caso 37, 7.227 chamadas foram certificadas após reparação e a frequência de
fallback caiu 75,1%. No caso 9, a frequência caiu de 43,6% para 14,2%, permitindo
encerrar a prova dentro do limite. Todos os 30 caminhos do benchmark foram validados
independentemente com `unordered_validation.py` no ambiente `.venv`; não houve falha.

Os tempos de `profile` são inclusivos conforme `profile.timing_semantics`: o oráculo
contém geometria, certificado e fallback; o fallback contém `long double` e precisão
ampliada. Essas categorias não devem ser somadas como se fossem disjuntas.

## Experimento rejeitado

Também foi testado inicializar o fallback `long double` com uma combinação de 99,999%
do caminho geométrico e 0,001% do centroide, sujeita à barreira estritamente viável.
Em três repetições alternadas, o caso 37 ficou 7,5% mais lento, o caso 12 ficou 2,4%
mais lento e os casos 9 e 55 processaram 6,1% e 2,6% menos chamadas em 2 s. Objetivos,
bounds e validade não mudaram nos casos concluídos. A mudança foi removida; os dados
foram preservados em `benchmarks/results/unordered/task-5-warm-start-20260905/`.

## Reprodução e validação

```bash
cmake --build .build/unordered-instrumented --target tpp tpp-unordered-tests -j 8
.build/unordered-instrumented/tpp-unordered-tests

python3 benchmarks/scripts/unordered_benchmark.py \
  --solver .build/unordered-instrumented/tpp --seconds 2 --max-calls 10000000 \
  --case 2 --case 9 --case 12 --case 37 --case 55 \
  --output benchmarks/results/unordered/task-5-contact-repair-20260905/repair-1.jsonl
```

O mesmo comando foi repetido três vezes para cada binário, alternando a ordem. A
suíte C++ recompilada passou 86 casos por enumeração exaustiva e 344 verificações de
busca interrompida. A validação Python cobriu as seis saídas JSONL, 30 linhas. A
CLI expõe `repaired_geometric_path_calls`; o benchmark preserva o campo e o resumo
comparativo o inclui por caso e no total.

Limitação: são cinco casos da suíte de desenvolvimento e três repetições curtas, não
uma campanha ampla nem uma estimativa de desempenho fora dessas famílias. Próximo
passo sugerido é a tarefa 6, avaliar política de busca em conjunto separado; ela não
foi iniciada aqui.
