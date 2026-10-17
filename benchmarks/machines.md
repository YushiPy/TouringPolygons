# Máquinas dos experimentos

Hardware e software das máquinas em que rodam os benchmarks, para citar em relatórios
e no artigo e para interpretar tempos de máquinas diferentes. Os registros de cada
execução (`run.json`, `plan.json`) guardam o host, o modelo da CPU e a carga; aqui fica
o resto. Para atualizar uma seção, rode na máquina e cole a saída no lugar da seção dela:

```bash
./scripts/machine_specs.sh
```

O script não usa sudo, por isso não lê a velocidade da memória no Linux, e não imprime
números de série, UUIDs nem o conteúdo de licenças.

## Resumo

| Máquina | CPU | Núcleos / CPUs lógicas | Memória | Sistema | Uso |
|---|---|---|---|---|---|
| Mac do Gabriel | Apple M4 Pro | 8 P + 4 E / 12 | 24 GiB LPDDR5 | macOS 26.6 | desenvolvimento, campanhas curtas |
| dantzig | Intel Core i9-12900K | 8 P (com SMT) + 8 E / 24 | 125,5 GiB | Ubuntu 24.04 | campanhas longas; compartilhada |
| armijo | AMD Ryzen 7 9700X | 8 (com SMT) / 16 | 123,5 GiB | Ubuntu 24.04 | campanhas longas |
| Fekete et al. (artigo) | AMD Ryzen 9 7900 (o artigo diz "Ryzen 7900") | 12 (com SMT) / 24 | 96 GiB DDR5-5200 | Ubuntu 24.04 | referência: os tempos do artigo deles |

## Notas para os benchmarks

- **Núcleos de performance.** No Linux, as configurações de uma thread (B e C das
  [medições de referência](README.md#medições-de-referência-o-fekete-original-e-o-nosso-configurações-a-a-e),
  `free-order`, `free-compare`) fixam cada execução num núcleo físico, presa aos dois
  hyperthreads dele.
  - Na dantzig, o i9-12900K é híbrido: só os 8 núcleos P são usados.
  - Na armijo, os 8 núcleos são iguais e todos são usados.
  - No macOS não há fixação de núcleos.
- **Threads da configuração A.** O artigo usou `num_threads=24`, todas as CPUs lógicas do
  Ryzen 9 7900 deles. A configuração A usa, por padrão, uma thread por CPU lógica dos
  núcleos de performance da máquina, e fica presa a essas CPUs. São 16 na armijo e 16 na
  dantzig, onde os núcleos E ficam de fora; `--threads N` escolhe outro número. O número
  entra no nome do diretório (`A-paper-16threads-300s`), para que rodadas com números
  diferentes não se misturem.
- **Comparar máquinas.** A armijo (Zen 5) e a dantzig (Alder Lake) têm o mesmo número de
  CPUs lógicas de performance e de núcleos para as configurações de uma thread. Os tempos
  de uma não valem para a outra; compare execuções da mesma máquina.
- **Carga alheia.** A dantzig é usada por outras pessoas. Os runners avisam no início e
  registram a carga alheia em cada linha. No Mac, programas abertos (iCloud, indexação)
  também pesam.

## Mac do Gabriel

- **Capturado em:** 2026-10-17
- **Modelo:** MacBook Pro (Mac16,8)
- **Sistema:** macOS 26.6.2 (25G83), Darwin 25.6.0, arm64
- **CPU:** Apple M4 Pro
- **Núcleos:** 12 (8 Performance and 4 Efficiency); 12 CPUs lógicas
- **Cache L2:** 16 MiB por cluster P, 4 MiB por cluster E
- **Memória:** 24 GiB LPDDR5
- **Modo de energia:** powermode=0
- **Compiladores:**
  - `c++`, `g++` e `clang++`: Apple clang version 21.0.0 (clang-2100.3.34.2);
  - `g++-16`: Homebrew GCC 16.2.0.
- **Python:** Python 3.14.8, python3.12 presente; uv 0.12.23 (Homebrew)
- **Gurobi instalado:** /Library/gurobi1303; licença presente (~/gurobi.lic)

## dantzig

- **Capturado em:** 2026-10-17
- **Sistema:** Ubuntu 24.04.3 LTS, kernel 6.8.0-147-generic, x86_64, glibc 2.39
- **Placa:** ASUSTeK COMPUTER INC. TUF GAMING B660M-PLUS D4 (memória DDR4)
- **CPU:** 12th Gen Intel(R) Core(TM) i9-12900K
- **Núcleos:** 1 soquete, 16 núcleos (8 P com SMT, 8 E sem SMT); 24 CPUs lógicas; NUMA: 1.
  O `lscpu` diz "16 núcleos × 2 threads", o que não vale para os núcleos E.
- **Híbrido:** núcleos P nas CPUs 0-15, núcleos E nas CPUs 16-23
- **Frequência:** mín. 800 MHz, máx. 5200 MHz
- **Caches:** L1d 640 KiB (16 instâncias); L1i 768 KiB (16 instâncias); L2 14 MiB
  (10 instâncias); L3 30 MiB (1 instância)
- **Memória:** 125,5 GiB; sem swap
- **Frequência dinâmica:** driver intel_pstate, governador powersave, boost ativo, SMT ativo
- **Compiladores:**
  - `c++`, `g++` e `g++-13`: Ubuntu 13.3.0-6ubuntu2~24.04.1;
  - `g++-14`: Ubuntu 14.2.0-4ubuntu2~24.04.1.
- **Python:** Python 3.12.3; uv 0.12.23
- **Gurobi instalado:** /opt/gurobi1303; licença presente (~/gurobi.lic). O solver do
  Fekete usa o Gurobi 12.0.0 que a receita dele baixa; da máquina, só a licença.

## armijo

- **Capturado em:** 2026-10-17
- **Sistema:** Ubuntu 24.04.5 LTS, kernel 7.0.0-31-generic, x86_64, glibc 2.39
- **Placa:** ASUSTeK COMPUTER INC. PRIME X670E-PRO WIFI (memória DDR5)
- **CPU:** AMD Ryzen 7 9700X 8-Core Processor
- **Núcleos:** 1 soquete × 8 núcleos × 2 threads; 16 CPUs lógicas; NUMA: 1
- **Frequência:** mín. 605 MHz, máx. 5582 MHz
- **Caches:** L1d 384 KiB (8 instâncias); L1i 256 KiB (8 instâncias); L2 8 MiB
  (8 instâncias); L3 32 MiB (1 instância)
- **Memória:** 123,5 GiB; swap 8,0 GiB
- **Frequência dinâmica:** driver amd-pstate-epp, governador powersave, boost ativo, SMT ativo
- **Compiladores:** `c++`, `g++` e `g++-13`: Ubuntu 13.3.0-6ubuntu2~24.04.1
- **Python:** Python 3.12.3; uv 0.12.23
- **Gurobi instalado:** /opt/gurobi1302; licença presente (~/gurobi.lic)

## Fekete et al. (máquina do artigo)

Segundo o artigo deles (SoCG 2026, seção de experimentos):

- **Máquina:** estação AMD Ryzen 7900, com 96 GiB de DDR5-5200 e Ubuntu 24.04.
- **Build:** g++ 13.3, Gurobi 12.0, Boost.Geometry 1.83 e CGAL 6.0.1.
- **Execução:** `num_threads=24` nos scripts de experimento, no nó `alggen05` do cluster
  deles, em modo exclusivo.
