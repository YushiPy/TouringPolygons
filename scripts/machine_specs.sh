#!/usr/bin/env bash
# Imprime, em Markdown, o hardware e o software desta máquina que importam para os
# benchmarks (CPU, núcleos P/E, frequências, caches, memória, governador, compiladores,
# Python, Gurobi), para colar em benchmarks/machines.md. Não usa sudo e não imprime
# números de série, UUIDs nem o conteúdo de licenças.
set -uo pipefail
export LC_ALL=C  # decimal points and English tool output whatever the session's locale

have() { command -v "$1" >/dev/null 2>&1; }
first_line() { "$@" 2>/dev/null | head -n 1; }
read_file() { [[ -r "$1" ]] && tr -d '\n' < "$1"; }
row() { [[ -n "${2:-}" ]] && printf -- '- **%s:** %s\n' "$1" "$2"; return 0; }

printf '### %s\n\n' "$(hostname -s 2>/dev/null || hostname)"
row "Capturado em" "$(date '+%Y-%m-%d')"

if [[ "$(uname -s)" == "Darwin" ]]; then
	hardware="$(system_profiler SPHardwareDataType 2>/dev/null)"
	field() { printf '%s\n' "$hardware" | sed -n "s/^ *$1: //p" | head -n 1; }
	row "Modelo" "$(field 'Model Name') ($(field 'Model Identifier'))"
	row "Sistema" "macOS $(sw_vers -productVersion) ($(sw_vers -buildVersion)), Darwin $(uname -r), $(uname -m)"
	row "CPU" "$(field 'Chip')"
	row "Núcleos" "$(field 'Total Number of Cores'); $(sysctl -n hw.logicalcpu) CPUs lógicas"
	l2p=$(( $(sysctl -n hw.perflevel0.l2cachesize 2>/dev/null || echo 0) / 1048576 ))
	l2e=$(( $(sysctl -n hw.perflevel1.l2cachesize 2>/dev/null || echo 0) / 1048576 ))
	row "Cache L2" "${l2p} MiB por cluster P, ${l2e} MiB por cluster E"
	memory_type="$(system_profiler SPMemoryDataType 2>/dev/null | sed -n 's/^ *Type: //p' | head -n 1)"
	row "Memória" "$(( $(sysctl -n hw.memsize) / 1073741824 )) GiB ${memory_type}"
	row "Modo de energia" "$(pmset -g 2>/dev/null | awk '/powermode|lowpowermode/ {printf "%s%s=%s", sep, $1, $2; sep=", "}')"
else
	lscpu_field() { lscpu 2>/dev/null | sed -n "s/^$1: *//p" | head -n 1; }
	os="$( . /etc/os-release 2>/dev/null && printf '%s' "${PRETTY_NAME:-}")"
	row "Sistema" "${os:-Linux}, kernel $(uname -r), $(uname -m), $(first_line ldd --version | sed 's/^ldd //')"
	row "Placa" "$(read_file /sys/devices/virtual/dmi/id/board_vendor) $(read_file /sys/devices/virtual/dmi/id/board_name)"
	row "CPU" "$(lscpu_field 'Model name')"
	sockets="$(lscpu_field 'Socket(s)')"; cores="$(lscpu_field 'Core(s) per socket')"; threads="$(lscpu_field 'Thread(s) per core')"
	row "Núcleos" "${sockets:-?} soquete(s) × ${cores:-?} núcleos × ${threads:-?} threads; $(nproc --all) CPUs lógicas; NUMA: $(lscpu_field 'NUMA node(s)')"
	if [[ -r /sys/devices/cpu_core/cpus ]]; then
		row "Híbrido" "núcleos P nas CPUs $(read_file /sys/devices/cpu_core/cpus), núcleos E nas CPUs $(read_file /sys/devices/cpu_atom/cpus)"
	fi
	row "Frequência" "mín. $(lscpu_field 'CPU min MHz') MHz, máx. $(lscpu_field 'CPU max MHz') MHz"
	caches="$(lscpu -C=NAME,ONE-SIZE,ALL-SIZE --noheadings 2>/dev/null | awk '{printf "%s %s por instância (%s no total); ", $1, $2, $3}')"
	[[ -n "$caches" ]] || caches="$(lscpu 2>/dev/null | sed -n 's/^\(L[0-9][di]* cache\): *\(.*\)/\1 \2;/p' | tr '\n' ' ')"
	row "Caches" "${caches%; }"
	row "Memória" "$(awk '/^MemTotal/ {printf "%.1f GiB", $2 / 1048576}' /proc/meminfo); swap $(awk '/^SwapTotal/ {printf "%.1f GiB", $2 / 1048576}' /proc/meminfo)"
	boost="$(read_file /sys/devices/system/cpu/cpufreq/boost)"
	[[ -n "$boost" ]] || { no_turbo="$(read_file /sys/devices/system/cpu/intel_pstate/no_turbo)"; [[ -n "$no_turbo" ]] && boost=$(( 1 - no_turbo )); }
	row "Frequência dinâmica" "driver $(read_file /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver), governador $(read_file /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor), boost ${boost:-?}, SMT $(read_file /sys/devices/system/cpu/smt/control)"
fi

compilers=""
for compiler in c++ g++ g++-16 g++-15 g++-14 g++-13 clang++; do
	have "$compiler" && compilers+="\`$compiler\`: $(first_line "$compiler" --version); "
done
row "Compiladores" "${compilers%; }"
row "Python" "$(first_line python3 --version)$(have python3.12 && printf ', python3.12 presente'); uv: $(have uv && first_line uv --version || printf 'ausente')"
gurobi="$(ls -d /opt/gurobi* /Library/gurobi* 2>/dev/null | tr '\n' ' ')"
license="ausente"
for file in "${GRB_LICENSE_FILE:-}" "$HOME/gurobi.lic" /opt/gurobi/gurobi.lic /opt/gurobi*/gurobi.lic /Library/gurobi/gurobi.lic; do
	if [[ -n "$file" && -f "$file" ]]; then
		shown="${file#"$HOME"}"
		[[ "$shown" == "$file" ]] || shown="~$shown"
		license="presente ($shown)"
		break
	fi
done
row "Gurobi instalado" "${gurobi:-nenhum}; licença: ${license}"
