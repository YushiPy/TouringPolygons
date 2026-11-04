cmake_minimum_required(VERSION 3.20)
project(steiner-convex-partition CXX)

option(STEINER_CONVEX_PARTITION_BUILD_TESTS "Build the tests" ON)
# The comparison with Greene's algorithm links the optimal-convex-partition
# package, which needs Boost headers. The library itself has no dependencies.
option(STEINER_CONVEX_PARTITION_BUILD_GREENE_COMPARISON "Build the comparison with optimal_convex_partition" OFF)

add_library(steiner_convex_partition STATIC
	src/assembly.cpp
	src/dynamic_program.cpp
	src/geometry.cpp
	src/loose_x4.cpp
	src/steiner_convex_partition.cpp
	src/superrange.cpp
	src/validate.cpp
)
target_include_directories(steiner_convex_partition PUBLIC include)
target_compile_features(steiner_convex_partition PUBLIC cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	target_compile_options(steiner_convex_partition PRIVATE -Wall -Wextra -O2)
endif()

if(STEINER_CONVEX_PARTITION_BUILD_TESTS)
	enable_testing()
	add_executable(steiner_convex_partition_tests tests/unit_tests.cpp)
	target_link_libraries(steiner_convex_partition_tests PRIVATE steiner_convex_partition)
	add_test(NAME steiner_convex_partition_tests COMMAND steiner_convex_partition_tests)
	add_test(NAME steiner_convex_partition_tests_cubic COMMAND steiner_convex_partition_tests --cubic)
	add_executable(steiner_convex_partition_brute_force tests/brute_force.cpp)
	target_link_libraries(steiner_convex_partition_brute_force PRIVATE steiner_convex_partition)
	add_test(NAME steiner_convex_partition_brute_force COMMAND steiner_convex_partition_brute_force 150)
	add_test(NAME steiner_convex_partition_brute_force_cubic COMMAND steiner_convex_partition_brute_force --cubic 150)
	add_executable(steiner_convex_partition_superrange_tests tests/superrange_tests.cpp)
	target_link_libraries(steiner_convex_partition_superrange_tests PRIVATE steiner_convex_partition)
	add_test(NAME steiner_convex_partition_superrange_tests COMMAND steiner_convex_partition_superrange_tests 200)
	add_executable(steiner_convex_partition_cubic_differential tests/cubic_differential.cpp)
	target_link_libraries(steiner_convex_partition_cubic_differential PRIVATE steiner_convex_partition)
	add_test(NAME steiner_convex_partition_cubic_differential COMMAND steiner_convex_partition_cubic_differential 300)
endif()

if(STEINER_CONVEX_PARTITION_BUILD_GREENE_COMPARISON)
	if(NOT TARGET optimal_convex_partition)
		add_subdirectory(../../optimal-convex-partition/cpp optimal-convex-partition)
	endif()
	add_executable(steiner_convex_partition_compare tests/compare_greene.cpp)
	target_link_libraries(steiner_convex_partition_compare PRIVATE steiner_convex_partition optimal_convex_partition)
endif()
