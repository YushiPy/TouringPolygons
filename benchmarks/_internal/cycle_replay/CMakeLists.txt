cmake_minimum_required(VERSION 3.20)
project(tpp_cycle_replay LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_BUILD_TYPE Release)
get_filename_component(TPP_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(TPP_ENABLE_GUROBI OFF CACHE BOOL "The replay runner does not use Gurobi" FORCE)
set(TARGET main-unordered CACHE STRING "Nonconvex package's existing sample target" FORCE)
add_subdirectory("${TPP_ROOT}/packages/nonconvex-tpp/cpp" nonconvex)

add_executable(tpp-cycle-replay main-cycle_replay.cpp)
target_include_directories(tpp-cycle-replay PRIVATE "${TPP_ROOT}/packages/nonconvex-tpp/cpp/src/solvers")
target_link_libraries(tpp-cycle-replay PRIVATE tpp_nonconvex)
target_compile_options(tpp-cycle-replay PRIVATE -O3)
