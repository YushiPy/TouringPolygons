# Ciclo convexo — resultado final (2026-09-27)

**Formulação.** Ciclo euclidiano fechado, ordem fixa, 20 instâncias sintéticas
convexas com separação, interseções, contenção e casos degenerados. Quinze
repetições por instância, uma execução de aquecimento por backend e Gurobi com
uma thread. Tempos C++ incluem validação e certificado; Gurobi é numérico.

**Resultado.** Todas as 20 saídas racionais foram certificadas ótimas; as 20
saídas double foram factíveis e seus intervalos independentes compatíveis com
os racionais/Gurobi. Nas medianas por instância, racional foi 1,29–21,33× e
double 1,11–22,38× mais rápido que a chamada Gurobi completa.

**Exatidão e limites.** O certificado racional usa predicados exatos e retorna
o objetivo como soma de raízes de racionais. Double permite recuperação racional
local; não é uma execução exclusivamente binary64. Os limites Gurobi são
numéricos. Esta suíte finita não prova dominância universal nem cobertura de
todas as degenerescências.

Os testes de ciclo e certificado passaram, assim como WASM e `sanity_check` na
revisão descrita pela campanha. A suíte de navegador não foi validada por falha
de DNS ao instalar `contourpy`. As medições preliminares `complete`, `optimized`
e `performance` foram consolidadas aqui; dados por repetição foram removidos.
