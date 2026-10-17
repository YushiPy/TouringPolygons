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
| dantzig | Intel Core i9-12900K | 8 P (com SMT) + 8 E / 24 | a completar | Linux | campanhas longas; compartilhada |
| armijo | AMD Ryzen 7 9700X | 8 / 16 | 128 GB | Linux | campanhas longas |
| Fekete et al. (artigo) | AMD Ryzen 9 7900 (o artigo diz "Ryzen 7900") | 12 / 24 | 96 GiB DDR5-5200 | Ubuntu 24.04 | referência: os tempos do artigo deles |

## Notas para os benchmarks

- **Núcleos de performance.** No Linux, as configurações de uma thread (B e C das
  [medições de referência](README.md#medições-de-referência-o-fekete-original-e-o-nosso-configurações-a-b-e-c),
  `free-order`, `free-compare`) fixam cada execução num núcleo físico, presa aos dois
  hyperthreads dele.
  - Na dantzig, o i9-12900K é híbrido: só os 8 núcleos P são usados.
  - Na armijo, os 8 núcleos são iguais e todos são usados.
  - No macOS não há fixação de núcleos.
- **24 threads.** A configuração A usa as 24 threads do artigo do Fekete, como no
  Ryzen 9 7900 deles (12 núcleos, 24 CPUs lógicas).
  - Na dantzig, 24 é o total de CPUs lógicas, mas 8 dos núcleos são E.
  - Na armijo, 24 threads passam das 16 CPUs lógicas.
  - Em nenhuma das duas o paralelismo é o do artigo.
- **Carga alheia.** A dantzig é usada por outras pessoas. Os runners avisam no início e
  registram a carga alheia em cada linha. No Mac, programas abertos (iCloud, indexação)
  também pesam.

## Mac do Gabriel

- **Capturado em:** 2026-10-16
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

A completar com `./scripts/machine_specs.sh`. Até aqui:

- **CPU:** Intel Core i9-12900K. É híbrida: 8 núcleos P com SMT e 8 núcleos E, 24 CPUs
  lógicas.
- **Software:** g++ 13; Gurobi 13 em `/opt/gurobi1302`. O solver do Fekete usa o
  Gurobi 12.0.0 que a receita dele baixa; da máquina, só a licença.

## armijo

A completar com `./scripts/machine_specs.sh`. Até aqui:

- **CPU:** AMD Ryzen 7 9700X, 8 núcleos com SMT, 16 CPUs lógicas.
- **Frequência:** de 605 a 5582 MHz, com boost ativo.
- **Memória:** 128 GB, conforme as campanhas da Paula de 10/2026.

## Fekete et al. (máquina do artigo)

Segundo o artigo deles (SoCG 2026, seção de experimentos):

- **Máquina:** estação AMD Ryzen 7900, com 96 GiB de DDR5-5200 e Ubuntu 24.04.
- **Build:** g++ 13.3, Gurobi 12.0, Boost.Geometry 1.83 e CGAL 6.0.1.
- **Execução:** `num_threads=24` nos scripts de experimento, no nó `alggen05` do cluster
  deles, em modo exclusivo.
