# TSPN — memo e cutoff certificado (2026-09-30)

Tour fechado, ordem livre, sem ponto fixo; gap alvo `1e-6`, factibilidade
`1e-8`, validação independente `1e-7`. Um tour válido com gap aberto é limite
de orçamento, não ótimo; bounds Fekete reutilizados são numéricos.

**Memo de uma busca.** Em 6.124 consultas não houve chaves repetidas nem hits.
`bound-first` registrou podas certificadas por cutoff, mas não mostrou ganho de
tempo estável; em casos grandes alguns gaps abertos pioraram.

**Portfólio com compartilhamento.** No screen large6, memo obteve 1.341 hits;
dois dos três casos fechados melhoraram e um ficou praticamente igual. No
follow-up OSM39 com o mesmo binário, 10 s e três repetições, todas as seis
execuções foram válidas e fecharam o gap. Mediana caiu de `0,8243 s` para
`0,7023 s` (14,8%); foram 368, 378 e 376 hits por execução. Evidência restrita
a esse caso; memo segue opt-in e não teve benefício demonstrado sem sharing.

No conjunto auditado houve 90 trajetórias nativas válidas, 66 gaps fechados e
zero timeouts de processo; três casos grandes ficaram sem objetivo fechado.
