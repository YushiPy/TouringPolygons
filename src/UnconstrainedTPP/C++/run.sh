#!/bin/zsh

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

if [ $SILENCE -eq 1 ]; then
	cmake -S . -B $BUILD_DIR -DTARGET=$TARGET > /dev/null 2>&1
	cmake --build $BUILD_DIR --config Release > /dev/null 2>&1
else
	cmake -S . -B $BUILD_DIR -DTARGET=$TARGET
	cmake --build $BUILD_DIR --config Release
fi

if [ $? -ne 0 ]; then
	echo "Build failed."
	exit 1
fi

if [ $RUN -eq 1 ]; then
	./$BUILD_DIR/$PROJECT_NAME
fi