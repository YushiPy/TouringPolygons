#!/bin/zsh

# alias run='zsh run.sh'

PROJECT_NAME=$(sed -n 's/project(\(.*\))/\1/p' CMakeLists.txt)
BUILD_DIR=build
SILENCE=0
RUN=1
TARGET="main"

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
if [ ! -f "src/${TARGET}.cpp" ]; then
	echo "Error: Source file src/${TARGET}.cpp not found."
	exit 1
fi

OPENMP_ROOT=$(brew --prefix libomp)

if [ $SILENCE -eq 1 ]; then
	cmake -S . -B $BUILD_DIR -DTARGET=$TARGET -DOpenMP_ROOT=$OPENMP_ROOT > /dev/null 2>&1
	cmake --build $BUILD_DIR --config Release > /dev/null 2>&1
else
	cmake -S . -B $BUILD_DIR -DTARGET=$TARGET -DOpenMP_ROOT=$OPENMP_ROOT
	cmake --build $BUILD_DIR --config Release
fi

if [ $? -ne 0 ]; then
	echo "Build failed."
	exit 1
fi

if [ $RUN -eq 1 ]; then
	./$BUILD_DIR/$PROJECT_NAME
fi