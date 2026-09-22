# Comparacao no corpus de Fekete et al.

Pasta canonica para a comparacao single-core dos dois solvers nas 558
instancias. Os indices nos CSVs comecam em zero; portanto, `case_index = 0`
e o Caso 001 exibido ao usuario.

## Arquivos

- `instances.bin`: corpus original, SHA-256
  `aa442e0546567461621b7fcdb9596ba7b3cc4094929d23fb9bb38d1093c88737`.
- `ours.csv`: nosso solver exato, 558/558 otimos certificados, com `order` e `path` da rodada final. Separador `;`.
- `fekete.csv`: solver de Fekete et al., 550/558 instancias concluidas com
  gap relativo de no maximo 0,1%; as outras 8 atingiram o limite de 6 horas.

Cada processo usou uma thread. A campanha externa executou ate 10 processos
independentes em paralelo; isso reduz o tempo de parede da campanha, mas nao da
mais de um nucleo a uma instancia. O nosso solver resolveu todas as instancias;
477 delas em menos de 10 segundos.

A análise reproduzível está em `analyze.py`. Para regenerar `analysis/`:

```bash
python3 benchmarks/results-saved/german-comparison/analyze.py \
  --instances benchmarks/results-saved/german-comparison/instances.bin \
  --ours benchmarks/results-saved/german-comparison/ours.csv \
  --fekete benchmarks/results-saved/german-comparison/fekete.csv \
  --output-dir benchmarks/results-saved/german-comparison/analysis
```

O solver externo usado na campanha é o submódulo
`third_party/tspn-socg`, fixado na revisão documentada em
`docs/third-party.md`.

## Caso 001 corrigido

A geometria desse caso havia sido alterada acidentalmente numa copia de
trabalho. As duas linhas foram refeitas com o corpus original:

- nosso solver: otimo exato `19172.656985081398`, em `40.42125025` s;
- Fekete et al.: intervalo `[19153.785922982144, 19172.656993168614]`,
  gap `0.0984269952%`, em `23.386760834` s.

Os caminhos da rodada final estao preservados em `ours.csv` e podem ser
auditados geometricamente contra o corpus original.

## Proxima rodada

Ao melhorar nosso solver, substitua `ours.csv` nesta pasta. Mantenha
`case_index` e `sha256` para que a correspondencia com `fekete.csv` possa ser
validada sem depender da ordem fisica das linhas.
