cmake_minimum_required(VERSION 3.20)
project(tpp-geometry)

add_library(tpp_geometry STATIC src/common.cpp)
target_include_directories(tpp_geometry PUBLIC include)
target_compile_features(tpp_geometry PUBLIC cxx_std_26)
target_compile_options(tpp_geometry PRIVATE -O3)
