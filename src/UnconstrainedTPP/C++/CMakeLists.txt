cmake_minimum_required(VERSION 3.20)
project(utpp)

set(CMAKE_CXX_STANDARD 26)

set(TARGET "main" CACHE STRING "Target to build")

file(GLOB_RECURSE SOURCES src/*.cpp)
list(FILTER SOURCES EXCLUDE REGEX "src/main.*\\.cpp")

add_executable(${PROJECT_NAME} ${SOURCES} src/${TARGET}.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE include)

target_compile_options(${PROJECT_NAME} PRIVATE -O3)
