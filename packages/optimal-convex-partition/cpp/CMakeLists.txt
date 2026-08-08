cmake_minimum_required(VERSION 3.20)
project(optimal-convex-partition)

add_library(optimal_convex_partition STATIC
	src/optimal_convex_partition.cpp
)

target_include_directories(optimal_convex_partition PUBLIC include)
target_compile_features(optimal_convex_partition PUBLIC cxx_std_20)
target_compile_options(optimal_convex_partition PRIVATE -O3)
