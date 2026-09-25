# Oráculo convexo híbrido certificado

Implementação principal:
`packages/convex-tpp/cpp/src/solvers/hybrid.cpp`.
API pública: `tpp/convex/hybrid.h`.

## Contrato

O oráculo híbrido constrói primeiro uma solução em `double`, mas o modo seguro
somente aceita essa candidata depois de reproduzir exatamente sua proveniência
combinatória em aritmética racional binária. O resultado contém exatamente um
contato ordenado por polígono, sem incluir os extremos `start` e `target`, e
preserva contatos duplicados.

A validação segura:

1. reconstrói exatamente vértices, interseções com arestas e regiões da última
   etapa do mapa;
2. materializa um contato pertencente a cada polígono na ordem exigida;
3. verifica as condições locais de otimalidade convexa, inclusive cones
   factíveis de arestas e vértices e sequências de contatos coincidentes;
4. encerra o objetivo radical por limites díadicos dirigidos;
5. usa o solver racional correspondente quando qualquer etapa não pode ser
   certificada.

Polígonos dois a dois disjuntos usam como fallback a recorrência estabelecida
em aritmética racional. Casos com interseção usam mapas direcionais racionais.
Uma falha de certificação significa apenas que a candidata rápida não foi
provada; ela nunca autoriza usar seu comprimento como limite inferior.
Quando o chamador fornece um corte finito, uma candidata materializada mas
não certificada pode dispensar o fallback se seu **dual factível em aritmética
racional** já alcançar o corte. Um dual em `double` serve apenas como filtro
para decidir se vale calcular o dual racional; ele nunca causa poda sozinho.
Nesse retorno antecipado, os contatos fornecem um caminho factível para a
sequência, o limite superior é o comprimento arredondado para cima desse
caminho e o limite inferior é o dual racional, sem afirmar que a sequência
foi resolvida até fechar o gap.

As funções `tpp_convex_solve_hybrid_safe` e
`tpp_convex_solve_length_hybrid_safe` expõem esse contrato. As variantes
`hybrid_unchecked` omitem reconstrução, certificação e fallback exato: existem
somente para diagnóstico de desempenho e não fornecem um limite seguro para
branch-and-bound.

A sobrecarga de `tpp_convex_solve_hybrid` que recebe
`DynamicConvexTppWorkspace` reutiliza polígonos já convertidos e normalizados
em aritmética racional entre chamadas. O cache confere todas as coordenadas
binárias antes de reutilizar uma entrada, limita a retenção a 8192 vértices e
não altera os predicados, os limites ou a escolha do fallback.

## Casos de fronteira

O localizador de contato trata tangência, passagem por vértice, fechamento
circular, retas paralelas e sobreposição colinear com uma aresta de suporte.
Contatos coincidentes usam primeiro o testemunho de direções alinhadas; um
programa dinâmico exato limitado propaga direções admissíveis por cones normais
de arestas e vértices. Se não encontrar testemunho, o modo seguro faz fallback.

As suítes focadas devem continuar cobrindo cardinalidade dos contatos, contatos
duplicados, orientação invertida, polígonos repetidos, extremos estacionários,
tangência, sobreposição colinear, polígonos finos ou quase colineares e escalas
de coordenadas muito pequenas e muito grandes.

## Validação

```bash
cmake --preset convex-release -DTARGET=main-directional_tests
cmake --build --preset convex-release -j 4
.build/convex-release/packages/convex-tpp/cpp/tpp-convex \
  --random-boxes 1000 --random-convex 200

cmake --preset convex-release -DTARGET=main-intersection_tests
cmake --build --preset convex-release -j 4
.build/convex-release/packages/convex-tpp/cpp/tpp-convex

cmake --preset nonconvex-release -DTARGET=main-unordered_tests
cmake --build --preset nonconvex-release -j 4
.build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp
```

Resultados temporais e comparações entre versões não pertencem a este contrato.
Quando uma campanha precisar ser preservada, seus dados, configuração, análise e
procedimento de reprodução devem ficar juntos em `benchmarks/results-saved/`.
