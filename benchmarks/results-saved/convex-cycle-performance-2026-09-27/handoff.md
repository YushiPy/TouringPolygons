> Registro histórico da interrupção anterior. Os itens pendentes foram tratados na retomada; veja o [resultado final](../convex-cycle-final-2026-09-27/validation.md).

# Estado ao encerrar, 2026-09-27

Trabalho interrompido a pedido do usuário, sem commit, na branch `codex/convex-cycle-disjoint`.
O worktree temporário desaparecido foi reconstruído a partir do histórico de ferramentas. Os patches anteriores do solver, verificador e os dados brutos Gurobi foram recuperados. O antigo manifesto gerado de hashes não foi recuperado; a campanha nova contém proveniência própria.

## Implementado

- APIs gerais `tpp_convex_solve_cycle` e `tpp_convex_solve_cycle_double`, mantendo as APIs disjuntas anteriores.
- Aceleração compartilhada por racionais/doubles: atualização geométrica de contatos, movimento conjunto de blocos coincidentes e fechamento direto por reflexões. Todo candidato passa pelo verificador independente. A busca anterior permanece como fallback disjunto.
- Interseções: região comum (ótimo zero), contenção, toque e sobreposição; fallback usa os mapas direcionais existentes, agora com entrada/saída racional exata. Fronteiras são subdivididas nas interseções de arestas.
- Certificados de links zero com testemunhas duais de norma <= 1 em Q(sqrt(D)), sinais exatos sem epsilon. O verificador também admite regiões pontuais/segmentos para certificar subproblemas restritos.
- Validação convexa linear por giros locais e número de voltas das direções das arestas; verificação só de raios incidentes ao testar suporte em vértices.
- CLI `python3 benchmarks/tpp.py cycle-benchmark`, campanha com entradas, 15 repetições por caso, dados brutos, parâmetros e hashes. Validação/certificação incluídas nos tempos C++; Gurobi inclui construção do modelo e usa ambiente aquecido compartilhado.

## Correção e complexidade

A proposta por reflexões apenas acelera a construção: `Optimal` exige factibilidade e condições globais de suporte exatas. Somar essas desigualdades fornece um limite inferior igual ao comprimento do candidato. Não há tolerância de aceitação nem discretização. Comprimentos irracionais são representados por somas de raízes de comprimentos quadrados racionais.

Sem blocos coincidentes, cada varredura da aceleração custa O(N) operações aritméticas; são permitidas até k+1 varreduras antes do fallback. Fechamento por reflexões é linear. A validação de separação do fallback ainda custa O(N²). Mantém-se a complexidade documentada da redução disjunta. Para interseções, recorte de blocos/região comum pode custar O(N²), o arranjo pode ter O(N²) eventos e a busca de testemunhas duais pode crescer exponencialmente no tamanho de um bloco. Não há aqui uma nova garantia polinomial completa para todo o caso intersectante; custos de inteiros/racionais grandes não são constantes.

## Limitações e trabalho restante

**A implementação geral de interseções ainda não é completa.** A construção de testemunhas duais é suficiente, mas não comprovadamente completa. Intervalos de âncora com extremos não diferenciáveis ainda podem retornar `OracleFailure` com um candidato factível. Falta resolver esses casos e provar completude/terminação geral. Nunca são declarados ótimos por orçamento ou tolerância.

Double usa a mesma proposta geométrica, mas pode precisar reconstruir racionalmente uma configuração de contatos ou recorrer ao solver racional. As recuperações são contadas (`rational_feature_recoveries`, `rational_cycle_recoveries`, `rational_anchor_recoveries`) e podem ser desativadas. Não se afirma a mesma robustez para double puro. `FloatingPointLimit` não significa ótimo exato.

O benchmark cobre uma suíte sintética pequena, não dominância universal sobre Gurobi. Os limites numéricos de Gurobi não são certificados exatos; foram comparados com intervalos independentes após correção racional de factibilidade, sem exigir coordenadas iguais.

## Validação

- Testes de ciclos passaram: 66 comparações disjuntas racional/double; contatos e objetivos racionais; 96 casos aleatórios de interseção e fixtures de contato/contensão/sobreposição com rotações cíclicas.
- Testes independentes do certificado passaram.
- Logs de testes direcionais, sanity e dashboard acompanham este relatório. Sanity foi interrompido ao pedido de parada, sem afirmar sucesso da suíte completa.
- WASM: não executável porque `static/wasm/tpp_convex_wasm.js` não está gerado neste worktree.

Veja `analysis.md` para as medianas e `summary.json` para gaps/status.
