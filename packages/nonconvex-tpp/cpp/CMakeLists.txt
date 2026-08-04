cmake_minimum_required(VERSION 3.20)
project(tpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 26)

set(TARGET "main" CACHE STRING "Target to build")

if(NOT TARGET tpp_geometry)
	add_subdirectory(../../common-geometry/cpp tpp_geometry)
endif()

if(NOT TARGET tpp_convex)
	add_subdirectory(../../convex-tpp/cpp tpp_convex)
endif()

if(APPLE)
	set(OpenMP_ROOT /opt/homebrew/opt/libomp)
	set(OpenMP_INCLUDE_DIR /opt/homebrew/opt/libomp/include)

	set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
	set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")

	set(OpenMP_C_LIB_NAMES "omp")
	set(OpenMP_CXX_LIB_NAMES "omp")

	set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
endif()

file(GLOB_RECURSE TPP_NONCONVEX_SOURCES src/*.cpp)
list(FILTER TPP_NONCONVEX_SOURCES EXCLUDE REGEX "src/main.*\\.cpp")
list(FILTER TPP_NONCONVEX_SOURCES EXCLUDE REGEX "src/tpp_convex\\.cpp")
list(FILTER TPP_NONCONVEX_SOURCES EXCLUDE REGEX "src/vector2\\.cpp")

add_library(tpp_nonconvex STATIC ${TPP_NONCONVEX_SOURCES})
target_include_directories(tpp_nonconvex PUBLIC include)
target_link_libraries(tpp_nonconvex PUBLIC tpp_geometry tpp_convex)
target_compile_options(tpp_nonconvex PRIVATE -O3)

add_executable(${PROJECT_NAME} src/${TARGET}.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE tpp_nonconvex)

target_compile_options(${PROJECT_NAME} PRIVATE -O3)

find_package(OpenMP REQUIRED)
target_link_libraries(tpp_nonconvex PUBLIC OpenMP::OpenMP_CXX)

if(APPLE)
	target_include_directories(tpp_nonconvex PUBLIC ${OpenMP_INCLUDE_DIR})
endif()

find_package(CGAL REQUIRED)
target_link_libraries(tpp_nonconvex PUBLIC CGAL::CGAL)

target_compile_definitions(tpp_nonconvex PUBLIC CGAL_DO_NOT_USE_BOOST_MP)
