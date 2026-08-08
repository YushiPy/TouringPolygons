cmake_minimum_required(VERSION 3.20)
project(optimal-convex-partition)

option(OPTIMAL_CONVEX_PARTITION_BUILD_CGAL_PARITY_TEST
	"Build the CGAL parity test executable"
	OFF
)

add_library(optimal_convex_partition STATIC
	src/optimal_convex_partition.cpp
)

target_include_directories(optimal_convex_partition PUBLIC include)
target_compile_features(optimal_convex_partition PUBLIC cxx_std_20)
target_compile_options(optimal_convex_partition PRIVATE -O3)

if(OPTIMAL_CONVEX_PARTITION_BUILD_CGAL_PARITY_TEST)
	find_package(CGAL REQUIRED)

	add_executable(optimal_convex_partition_cgal_parity
		tests/cgal_parity.cpp
	)

	target_link_libraries(optimal_convex_partition_cgal_parity PRIVATE
		optimal_convex_partition
		CGAL::CGAL
	)

	target_compile_features(optimal_convex_partition_cgal_parity PRIVATE cxx_std_20)
endif()
