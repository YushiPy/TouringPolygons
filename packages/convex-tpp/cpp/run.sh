#!/bin/zsh

# alias run='zsh run.sh'

PROJECT_NAME=$(sed -n 's/project(\(.*\))/\1/p' CMakeLists.txt)
BUILD_DIR=build
SILENCE=0
RUN=1
TARGET="main"
SRC_DIR="src"

while [[ $# -gt 0 ]]; do
	case $1 in
		-*) 
			[[ $1 == *s* ]] && SILENCE=1
			[[ $1 == *c* ]] && RUN=0
			shift;;
		*)  TARGET=$1; shift;;
	esac
done

# If TARGET ends with .cpp, remove the extension
if [[ $TARGET == *.cpp ]]; then
	TARGET=${TARGET%.cpp}
fi

# Check if the target source file exists
if [ ! -f "${SRC_DIR}/${TARGET}.cpp" ]; then

	matches=(${SRC_DIR}/${TARGET}*.cpp)

	# Remove non-existing globs
	matches=(${^matches}(N))

	if (( ${#matches[@]} == 1 )); then
		file="${matches[1]}"
		PREVIOUS_TARGET=$TARGET
		TARGET="${file:t:r}"  # basename without extension
		echo "⚠️ Warning: '$PREVIOUS_TARGET' does not exist in '$SRC_DIR', defaulting TARGET to '$TARGET'"
	elif (( ${#matches[@]} == 0 )); then
		echo "Error: no matching .cpp file found for prefix '$TARGET'" >&2
		exit 1
	else
		echo "Error: multiple matching .cpp files found for prefix '$TARGET':" >&2
		for f in "${matches[@]}"; do
			echo "  $f" >&2
		done
		exit 1
	fi
fi

OPENMP_ROOT=$(brew --prefix libomp)

# Gurobi
GUROBI_HOME=$(ls -d /Library/gurobi*/macos* 2>/dev/null | tail -1)

if [ -z "$GUROBI_HOME" ]; then
    echo "Error: could not find Gurobi installation in /Library" >&2
    exit 1
fi

if [ $SILENCE -eq 1 ]; then
    cmake -S . -B $BUILD_DIR -DTARGET=$TARGET -DOpenMP_ROOT=$OPENMP_ROOT -DGUROBI_HOME=$GUROBI_HOME > /dev/null 2>&1
    cmake --build $BUILD_DIR --config Release > /dev/null 2>&1
else
    cmake -S . -B $BUILD_DIR -DTARGET=$TARGET -DOpenMP_ROOT=$OPENMP_ROOT -DGUROBI_HOME=$GUROBI_HOME
    cmake --build $BUILD_DIR --config Release
fi

if [ $? -ne 0 ]; then
	echo "Build failed."
	exit 1
fi

if [ $RUN -eq 1 ]; then
	./$BUILD_DIR/$PROJECT_NAME
fi