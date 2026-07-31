cmake_minimum_required(VERSION 3.20)
project(tpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 26)

set(TARGET "main" CACHE STRING "Target to build")

if(APPLE)
	set(OpenMP_ROOT /opt/homebrew/opt/libomp)

	set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
	set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")

	set(OpenMP_C_LIB_NAMES "omp")
	set(OpenMP_CXX_LIB_NAMES "omp")

	set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
endif()

file(GLOB_RECURSE SOURCES src/*.cpp)
list(FILTER SOURCES EXCLUDE REGEX "src/main.*\\.cpp")

add_executable(${PROJECT_NAME} ${SOURCES} src/${TARGET}.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE include)

target_compile_options(${PROJECT_NAME} PRIVATE -O3)

find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

find_package(CGAL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE CGAL::CGAL)

target_compile_definitions(${PROJECT_NAME} PRIVATE CGAL_DO_NOT_USE_BOOST_MP)