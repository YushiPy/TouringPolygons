cmake_minimum_required(VERSION 3.20)
project(tpp-geometry)

set(TPP_CXX_STANDARD 26 CACHE STRING "C++ standard used by the maintained TPP packages")
set_property(CACHE TPP_CXX_STANDARD PROPERTY STRINGS 23 26)

add_library(tpp_geometry STATIC src/common.cpp)
target_include_directories(tpp_geometry PUBLIC include)
target_compile_features(tpp_geometry PUBLIC cxx_std_${TPP_CXX_STANDARD})
target_compile_options(tpp_geometry PRIVATE -O3)
