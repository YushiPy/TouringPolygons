cmake_minimum_required(VERSION 3.20)
project(tpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE "Release")
set(TPP_CXX_STANDARD 26 CACHE STRING "C++ standard used by the maintained TPP packages")
set_property(CACHE TPP_CXX_STANDARD PROPERTY STRINGS 23 26)
set(CMAKE_CXX_STANDARD ${TPP_CXX_STANDARD})

set(TARGET "main" CACHE STRING "Target to build")

if(NOT TARGET tpp_geometry)
	add_subdirectory(../../common-geometry/cpp tpp_geometry)
endif()

if(NOT TARGET tpp_convex)
	add_subdirectory(../../convex-tpp/cpp tpp_convex)
endif()

if(NOT TARGET optimal_convex_partition)
	add_subdirectory(../../optimal-convex-partition/cpp optimal_convex_partition)
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

file(GLOB_RECURSE TPP_NONCONVEX_SOURCES CONFIGURE_DEPENDS src/*.cpp)
list(FILTER TPP_NONCONVEX_SOURCES EXCLUDE REGEX "src/main.*\\.cpp")

add_library(tpp_nonconvex STATIC ${TPP_NONCONVEX_SOURCES})
target_include_directories(tpp_nonconvex PUBLIC include)
target_link_libraries(tpp_nonconvex PUBLIC tpp_geometry tpp_convex optimal_convex_partition)
target_compile_options(tpp_nonconvex PRIVATE -O3)

add_executable(${PROJECT_NAME} src/${TARGET}.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE tpp_nonconvex)

target_compile_options(${PROJECT_NAME} PRIVATE -O3)

find_package(Threads REQUIRED)
target_link_libraries(tpp_nonconvex PUBLIC Threads::Threads)

find_package(OpenMP REQUIRED)
target_link_libraries(tpp_nonconvex PUBLIC OpenMP::OpenMP_CXX)

if(APPLE)
	target_include_directories(tpp_nonconvex PUBLIC ${OpenMP_INCLUDE_DIR})
endif()

if("${TARGET}" STREQUAL "main-compare_cgal_partition")
	find_package(CGAL REQUIRED)
	target_link_libraries(${PROJECT_NAME} PRIVATE CGAL::CGAL)
	target_compile_definitions(${PROJECT_NAME} PRIVATE CGAL_DO_NOT_USE_BOOST_MP)
endif()

# Named tools: src/main-foo_bar.cpp also builds on demand as tpp-foo-bar (a
# leading "tpp_" is dropped), so several tools share one build directory
# without reconfiguring TARGET. CGAL is only needed by the parity tool.
file(GLOB TPP_NONCONVEX_TOOL_SOURCES CONFIGURE_DEPENDS src/main-*.cpp)
foreach(source IN LISTS TPP_NONCONVEX_TOOL_SOURCES)
	get_filename_component(stem "${source}" NAME_WE)
	if(stem STREQUAL "main-compare_cgal_partition")
		# CGAL's own setup requires an installed Boost CMake package.
		if(TPP_BOOST_HEADERS_ONLY)
			continue()
		endif()
		find_package(CGAL QUIET)
		if(NOT CGAL_FOUND)
			continue()
		endif()
	endif()
	string(REGEX REPLACE "^main-(tpp_)?" "" tool "${stem}")
	string(REPLACE "_" "-" tool "${tool}")
	add_executable(tpp-${tool} EXCLUDE_FROM_ALL "${source}")
	target_link_libraries(tpp-${tool} PRIVATE tpp_nonconvex)
	target_compile_options(tpp-${tool} PRIVATE -O3)
	set_target_properties(tpp-${tool} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
	if(stem STREQUAL "main-compare_cgal_partition")
		target_link_libraries(tpp-${tool} PRIVATE CGAL::CGAL)
		target_compile_definitions(tpp-${tool} PRIVATE CGAL_DO_NOT_USE_BOOST_MP)
	endif()
endforeach()
