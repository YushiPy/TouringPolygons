cmake_minimum_required(VERSION 3.20)
project(TouringPolygons)

option(TPP_BUILD_CONVEX "Build the convex Touring Polygons C++ solver" OFF)
option(TPP_BUILD_NONCONVEX "Build the non-convex Touring Polygons C++ solver" OFF)

if(TPP_BUILD_CONVEX OR TPP_BUILD_NONCONVEX)
    add_subdirectory(packages/common-geometry/cpp)
endif()

if(TPP_BUILD_CONVEX OR TPP_BUILD_NONCONVEX)
    add_subdirectory(packages/convex-tpp/cpp)
endif()

if(TPP_BUILD_NONCONVEX)
    add_subdirectory(packages/optimal-convex-partition/cpp)
endif()

if(TPP_BUILD_NONCONVEX)
    add_subdirectory(packages/nonconvex-tpp/cpp)
endif()
