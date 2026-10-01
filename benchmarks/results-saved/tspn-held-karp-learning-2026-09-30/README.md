# TSPN — one-tree e branching aprendido (2026-09-30)

Triagem de tour fechado, ordem livre, sem ponto fixo; 2 s nativos e teto de
processo de 8 s, gap `1e-6`, factibilidade `1e-8`, validação `1e-7`. Fekete foi
reutilizado apenas com hashes/configuração compatíveis. Cada variante retornou
6/6 tours válidos nos dois conjuntos; fechou 6/6 gaps em small6 e 3/6 em
large6. Não houve timeout de processo.

O one-tree melhorou o bound inicial em 4/6 casos grandes, com cerca de 335 ms
de trabalho e sem fechar mais casos. Em OSM39, branching aprendido aumentou
chamadas de 580 para 1.297 e a mediana de `0,751 s` para `1,235 s`. Em amostra
maior, o gap de tessellation60 também piorou. São bounds válidos e heurísticas
de ordenação; os resultados não indicam benefício suficiente. As duas opções
permanecem opt-in.
